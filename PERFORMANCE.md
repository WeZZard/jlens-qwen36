# Performance Optimization Plan

## Context

The Jacobian-lens fit on Qwen3.6-27B-4bit is bottlenecked by three
architecture factors that compound:

1. **Qwen3.5 uses hybrid attention** — 48 of 64 layers are linear-attention
   (Gated DeltaNet, GDN). Unlike softmax attention (a few big matmuls),
   GDN is a *recurrence*: `for t in range(T): state = state * decay + k *
   delta; y = state · q`. Sequential over time, ~10× slower than softmax.
2. **The fused Metal kernel for GDN has no registered VJP** — MLX's
   autograd can't backprop through it. We forced the pure-Python ops
   fallback, ~22× slower than the kernel.
3. **MLX re-runs the forward on every `mx.vjp` call** — no PyTorch-style
   graph retention. Computing J_ℓ ∈ R^{5120×5120} needs 5120 VJP calls,
   each paying forward + backward, vs PyTorch's 1 forward + 5120 backward.

### What's already shipped

| Optimization | Status | Speedup |
|--------------|--------|---------|
| Chain-multiply (`J_ℓ = J_{ℓ+1} @ M_ℓ`) | ✅ in `fit.py` | avoids full-stack VJP per source layer |
| `mx.compile` on per-layer forward | ✅ | ~3× |
| Custom Metal GDN VJP kernel + `mx.custom_function` | ✅ in `custom_gdn_vjp.py` | ~5× (19ms → 3.8ms per GDN backward) |
| 32-token fitting sequences | ✅ | ~4× vs 128-token |
| 25 evenly-spaced layers (late range) | ✅ | avoids full-depth chain |
| Checkpoint/resume every prompt | ✅ | survives interruptions |

### Current fit speed

- One per-layer M_ℓ (5120 VJPs through one DecoderLayer): ~1 min
- 20 prompts × 23-layer chain ≈ 8 hours on M4 Pro / 64 GB
- The paper (1000 prompts, full depth, on Sonnet 4.5) ≈ hours on big GPUs

---

## A. Remaining software optimizations

Ranked by payoff. Risk is broken into four dimensions so the decision can
be made on the dimension that actually matters.

### Risk dimensions

| Dimension | What it means | How to detect failure |
|-----------|---------------|----------------------|
| **Correctness** | Will the code produce *wrong numbers silently*? | Numerical gradient check vs `mx.vjp` on a tiny example (the harness exists in `gdn_backward.py:test_gdn_vjp`) |
| **Quality** | Will the math approximation degrade the lens? (it runs, but is it the *right* lens?) | Compare top-k readouts against the full-rank lens on known prompts — but needs a full-rank reference first (chicken-and-egg) |
| **Numerical stability** | Will reduced precision blow up / NaN / drift? | Run and watch for NaN / compare to fp32 reference |
| **Schedule** | Will it take much longer than the estimate? | n/a — known after the fact |

### Optimization candidates

#### Fix 1 — Manual batched backward for the full DecoderLayer  ⭐ recommended

| | |
|---|---|
| **Speedup** | ~10-50× on M_ℓ (the dominant cost) |
| **Effort** | ~1 day |
| **Correctness** | Medium — silent bugs likely (hit two in `gdn_backward.py`), but **verifiable**: the gradient-check harness already exists. Effective risk is Low. |
| **Quality** | Low — exact same math, just batched across D cotangents |
| **Stability** | Low — fp32 throughout |
| **Schedule** | Low — pattern is established (already done for GDN) |

**The idea.** Right now each per-layer M_ℓ does D=5120 *separate* `mx.vjp`
calls, each re-running that one layer's forward (MLX has no graph
retention). The backward recurrence cost is independent of the number of
cotangents, so batching all D cotangents through **one** backward pass is
a ~D-fold win on the dominant cost. Memory: ~3.4 GB for D cotangents —
fits in 64 GB.

**Why it generalizes from GDN.** `gdn_backward.py` already does the
manual BPTT for the GDN recurrence (verified exact vs `mx.vjp`). The same
pattern extends to the rest of the DecoderLayer (RMSNorm, attention,
MLP, residual) — each is a known differentiable op whose backward is a
matmul or elementwise. Compose them in a manual backward function that
takes `[D, B, S, ...]` cotangents and produces `[D, ...]` VJPs in one
pass.

**Verification.** The gradient-check pattern in
`gdn_backward.py:test_gdn_vjp` (compare manual backward to `mx.vjp` on a
tiny example) extends directly. Run it before trusting the batched
backward.

#### Fix 2 — Low-rank J via power iteration

| | |
|---|---|
| **Speedup** | ~80× fewer VJPs (compute top-k singular vectors of J_ℓ, k=64) |
| **Effort** | ~0.5 day |
| **Correctness** | Low — well-understood algorithm |
| **Quality** | **Medium-High** — assumes J_ℓ is rank-≤64. The paper says J-space holds ~25 concepts, but that's about *active J-space contents*, not the rank of the *linear map* J_ℓ. J_ℓ could be full-rank even if J-space activations are sparse. **Unverifiable cheaply** — needs a full-rank reference. |
| **Stability** | Low |
| **Schedule** | Low |

**The concern.** The lens readout is `softmax(W_U · norm(J_ℓ h))`. A
rank-64 approximation `J_ℓ ≈ U_64 Σ_64 V_64ᵀ` changes the readout in
ways that are hard to detect without a full-rank J_ℓ to compare against.
Test on a small sub-block before committing.

#### Fix 3 — Two-pass Metal kernel (no atomics)

| | |
|---|---|
| **Speedup** | ~2-3× on the GDN backward (current kernel uses atomic adds for dq/dk) |
| **Effort** | ~0.5 day |
| **Correctness** | Medium — Metal C++ is hard to debug; verifiable against `gdn_backward.py` |
| **Quality** | Low — exact math |
| **Stability** | Low |
| **Schedule** | Medium — Metal compile errors are opaque |

**The idea.** The current `custom_gdn_vjp.py` uses `atomic_fetch_add` for
the dq/dk reductions across the Dv dimension (~8192 contending threads).
A two-pass kernel (pass 1: each threadgroup writes its reduced partial to
a per-(b,hv,t,dk,tg_y) buffer with direct writes; pass 2: a tiny
reduction kernel sums the tg_y slots) avoids atomics entirely.

#### Fix 4 — Blelloch parallel prefix scan for GDN

| | |
|---|---|
| **Speedup** | ~5× on the GDN scan (O(log T) depth vs O(T)) |
| **Effort** | ~1-2 days |
| **Correctness** | **High** — GDN isn't a pure prefix sum (the `delta` update couples to `kv_mem`), so it's a scan with a non-trivial associative decomposition; off-by-ones hide until real inputs |
| **Quality** | Low — exact if correct |
| **Stability** | Low |
| **Schedule** | **High** — known-hard algorithm, could eat 2+ days |

**The concern.** GDN's recurrence is `s_t = g_t · s_{t-1} + k_t · δ_t`
where `δ_t = (v_t - ⟨s_{t-1}·g_t, k_t⟩) · β_t`. The `δ_t` depends on
`s_{t-1}`, so this is *not* a standard associative scan — the
decomposition needs Brent-Kung-style work on a non-associative operator.
Known-hard; only pursue if Fixes 1+3 are insufficient.

#### Fix 5 — fp16 / bf16 accumulation

| | |
|---|---|
| **Speedup** | ~1.5× |
| **Effort** | Hours |
| **Correctness** | Low |
| **Quality** | Low |
| **Stability** | **High** — summing 5120 bf16 terms can cancel catastrophically given J-lens values ~0.3 |
| **Schedule** | Low |

**The concern.** Use a high-precision reference run (fp32) before
trusting. Kahan summation or fp32 accumulation in the inner loop
mitigates this, but then the speedup vanishes.

---

## B. Cloud computing

### Path A — MLX on a rented Mac (MacStadium, AWS EC2 mac2)

- **Code changes:** none. Same code, same Metal.
- **Speed:** ≈ your M4 Pro (M1/M2 Mac minis are ~1.5× *slower*). Not a
  speed win — just lets it run unattended without tying up your machine.
- **Cost:** ~$1/hour spot.
- **Use case:** "let the 20-prompt fit run overnight off my laptop."

### Path B — Cloud GPU (A100 80GB / H100) + PyTorch  ⭐ for research grade

- **Speed:** H100 ≈ 60× the FLOPs of M4 Pro, *and* PyTorch retains the
  graph (eliminates factor 3 — the 30× re-forward penalty). Combined
  ~1000× faster than the current local fit.
- **Code changes:** port the `gdn_backward.py` math to Triton (NVIDIA's
  kernel DSL, Python-like, runs on GPU). The reference `jlens` repo
  already runs on standard-attention models; only the GDN VJP needs
  porting. ~1 day.
- **Model:** bf16 Qwen3.6-27B (~54 GB) → needs 80 GB GPU. A100 80GB
  ~$1.5/h spot, H100 ~$3/h.
- **Cost for a research-grade fit:** 1000 prompts, full depth, ~2-4h on
  H100 → ~$10.
- **The key property:** the lens is **portable** — fit on bf16 in the
  cloud once, download the ~3.4 GB fp16 lens file, apply on the 4-bit MLX
  model locally forever. The 4-bit vs bf16 weight difference adds noise,
  but the paper shows J-lens ≈ logit-lens at late layers, so the
  approximation is bounded (mainly affects early layers, which you can't
  interpret without a full-depth fit anyway).

### Path C — Distributed

The reference `JacobianLens.merge()` (and our `lens.py:merge`) supports
sharding prompts across machines. N cloud Macs (Path A) or N cloud GPUs
(Path B), each fits a prompt subset, merge. Linear speedup. Best for the
1000-prompt "research grade" fit if a single machine isn't enough.

### Path D — 4-bit on cloud GPU

Not possible — MLX 4-bit quantization is Apple-only. Cloud GPU fits
require bf16 (hence the 54 GB memory requirement).

---

## C. Recommended path

### Short term (tonight)

Let the current 20-prompt fit finish (~05:00). You'll have a
demo-quality lens for the viewer. No code changes.

### Medium term (this week)

Implement **Fix 1 (manual batched full-layer backward)**. ~1 day of
work, ~10-50× speedup → 100-prompt fit in ~1 hour on your M4 Pro. No
cloud needed.

- Correctness risk is **Low effective** because the gradient-check
  harness already exists (`gdn_backward.py:test_gdn_vjp` pattern).
- It's the same effort as the Triton port (Path B) and makes the local
  fit fast enough that cloud may not be needed at all.

### Long term (research grade)

If you want 1000 prompts / full depth (matching the paper), **Path B
(cloud H100 + Triton port of the GDN VJP)** is the answer. ~$10 and ~3
hours for a lens that matches the paper's quality, then use it locally.

- This is the *only* path to the paper's quality on a reasonable budget.
- Fix 1 makes the *local* fit fast but still caps at ~100 prompts in an
  hour; 1000 prompts still wants the cloud.
- The Triton port reuses the exact math from `gdn_backward.py` — it's
  the same backward recurrence, just on NVIDIA's DSL instead of MLX
  ops. The correctness is verifiable the same way.

### Decision criteria

- If you just want a working lens for the viewer and some experiments:
  **Fix 1 alone** (local, ~1 day, verifiable, reusable).
- If you want to reproduce the paper's interventions reliably (spider→ant
  flipping the answer, etc.): **Fix 1 + 100 prompts local** (still
  ~1 hour after Fix 1).
- If you want to publish or match the paper exactly: **Path B** (cloud,
  ~$10, ~3 hours, plus ~1 day Triton port).

---

## D. What to verify before trusting any optimization

Every optimization above is verifiable, but the verification cost differs.
Run these checks in order:

1. **Gradient check** (catches correctness bugs in the backward):
   ```python
   # For Fix 1 / Fix 3: compare manual backward to mx.vjp on a tiny example.
   # The harness: jlens_qwen/gdn_backward.py:test_gdn_vjp
   # Extend it to the full DecoderLayer.
   ```
2. **Readout sanity** (catches quality regressions):
   ```bash
   # Run on the known paper prompts and check the top tokens:
   uv run python scripts/workspace_range.py --lens data/lens/lens.npz
   # Expect: "euro"/"Euro" at late layers on the currency prompt,
   # "Italy" as an intermediate concept, etc.
   ```
3. **Intervention sanity** (catches J-lens-vector quality):
   ```bash
   # Steer "ant" on the spider-legs prompt; does the answer change?
   # With a good lens, alpha~100 should flip 8 -> 6.
   ```
4. **Numerical stability** (catches fp16 drift):
   ```python
   # Compare fp16 vs fp32 J_ℓ on a small sub-block; assert max abs diff < 1e-3.
   ```