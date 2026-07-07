# Review of PERFORMANCE.md — Is the Plan the Best We Can Build?

**Verdict: the plan optimizes the right cost center with the wrong lever.**
Fix 1 as written cannot deliver its promised 10–50×. A fundamentally
better formulation — assembling each per-layer Jacobian analytically
instead of backpropagating 5120 cotangent vectors — delivers ~30–60× on
the same hardware, is exact, and makes a **full-depth 64-layer fit
affordable locally**. That last property is not a perf nicety: it is what
the project's actual goal requires (see §5).

Constraints honored: hardware fixed (M4 Pro / 64 GB), model fixed
(Qwen3.6-27B-4bit MLX).

Model facts used throughout (from the model config):

| Quantity | Value |
|---|---|
| d_model (D) | 5120 |
| Layers | 64 (48 GDN linear-attention, 16 full-attention) |
| MLP intermediate | 17408 (~267M params/layer in the MLP alone) |
| GDN heads | Hk=16, Dk=128 (key_dim 2048); Hv=48, Dv=128 (value_dim 6144) |
| GDN conv | depthwise, conv_dim 10240, kernel 4 |
| Full-attention | 24 heads × 256; 4 KV heads (GQA) |
| GDN layer params | ~383M (qkv 52M, z 32M, out 32M, a/b 0.5M, MLP 267M) |

---

## 1. Central finding: the current fit is already near the roofline for the FLOPs it does

Current cost per M_ℓ (`fit.py:90-99`): 5120 sequential `mx.vjp` calls,
each = one layer forward + one backward at `[1, 32, 5120]`, with an
`mx.eval` sync per call.

- Per VJP ≈ 24.5 GFLOP forward + ~24.5 GFLOP backward-w.r.t.-inputs
  ≈ **49 GFLOP** (2 × 383M params × 32 tokens each way).
- Per M_ℓ ≈ **250 TFLOP**. Measured ~60 s → **~4.2 TFLOPS sustained**,
  i.e. roughly 50–60% of the M4 Pro's practical peak.

**Why this caps Fix 1.** Backward through a linear layer for C cotangent
sets costs C × (one GEMM) — the FLOPs scale linearly with the number of
cotangents no matter how they are batched. Batching all 5120 cotangents
into one backward pass recovers only:

1. the 5120 redundant forwards (~2×, since MLX re-runs the forward per
   `mx.vjp` call), and
2. GEMM/launch/sync efficiency (~1.5–2×).

**Fix 1's realistic ceiling is ~2–4×, not 10–50×.** The 10–50× estimate
in PERFORMANCE.md implicitly assumes the backward cost is independent of
the cotangent count. It is not: `gdn_backward.py:6-8` claims "the
backward recurrence cost is independent of the number of cotangents,"
but each cotangent carries its own adjoint state `s_bar [Hv, Dv, Dk]`
(786k floats). That term happens to be numerically small (~150 MFLOP per
cotangent per 32-token sequence, ~0.8 TFLOP for all 5120 — under 1% of
the 250 TFLOP total); what actually dominates, and what scales
irreducibly with cotangent count, is the backward GEMMs through the
projection and MLP matrices.

Conclusion: **stop scheduling those FLOPs better; stop doing them.**

---

## 2. The real fix: identity-basis analytic assembly of M_ℓ

### The idea

The cotangent basis for a full Jacobian is the **identity**.
Backpropagating the identity through a weight matrix is not 5120
vector-Jacobian products — it *is* the weight matrix. Every expensive
matrix in a DecoderLayer (`in_proj_qkv`, `in_proj_z`, `out_proj`,
`gate/up/down_proj`) is position-independent, and everything
data-dependent is cheap structure:

- RMSNorm / gated RMSNorm → diagonal + rank-1 (per 128-dim head for the
  gated norm)
- SiLU, sigmoid gates → diagonal
- depthwise conv1d → banded (4-tap) diagonal-per-channel
- the GDN recurrence → lives in head space (Hv·Dv = 6144), where
  per-cotangent work is ~200× cheaper than in model space

So M_ℓ is assembled as explicit matrix products instead of extracted
column-by-column through autograd.

### Per-branch construction (GDN layer)

**MLP branch — exact position-sum via a Hadamard trick.**
The fit averages over source positions s. For terms of the form
Σₛ diag(aₛ) · W · diag(lnₛ):

```
Σₛ diag(aₛ) · W · diag(lnₛ)  =  W ⊙ (Σₛ aₛ lnₛᵀ)
```

a rank-27 outer-product sum (27 valid positions), computed as one skinny
GEMM `[17408×27] @ [27×5120]` (~5 GFLOP) + one elementwise mask. The
whole MLP-branch Jacobian is then two masked weight matrices and one
`[5120×17408] @ [17408×5120]` GEMM ≈ **~1 TFLOP instead of ~130 TFLOP**.
RMSNorm's rank-1 correction terms fold in as additional low-rank outer
products of the same shape. This trick applies anywhere the position
dependence enters only through diagonals.

**GDN branch — head-space BPTT with analytically-seeded cotangents.**

1. Seed cotangents at the recurrence output: for output dim d, the seed
   at position t is row d of `W_o · J_gatenorm(t)` — computable per
   position as column scaling + 48 rank-1 head corrections (~4 GFLOP
   total for all 32 positions).
2. Run the batched BPTT through the recurrence core only. **The existing
   Metal kernel already supports this today**: `custom_gdn_vjp.py`'s
   grid is `(32, Dv, B·Hv)`, so cotangent chunks fold into the batch
   dimension B — tile q/k/v across B (a 512-chunk costs ~0.7 GB) and
   stack the seeds as `dy`. No new kernel code.
3. Contract the position-resolved gradients `[C, S, 10240]` with the
   per-position input diagonals (conv/SiLU/q-k-norm/LN), sum over s,
   then one GEMM against the stacked input projections
   `[5120, 10240] @ [10240, 5120]` (~0.5 TFLOP).
4. The z-gating path (`in_proj_z`) and the g/β paths (`in_proj_a/b`,
   see §4.1) join as additional cheap GEMMs — the a/b projections are
   5120→48, essentially free.

**Full-attention layers (16 of 64).** Same pattern; the softmax core has
a working autograd VJP, so batch identity cotangents in head space
(6144-dim value space; attention core backward is tiny at S=32) and
assemble through the projections analytically.

**Final norm J_64.** Currently 5120 VJPs ≈ 1 min/prompt (`fit.py:140`).
RMSNorm's Jacobian is closed-form diag + rank-1. Ten lines, ~free.

### Cost, memory, wall-clock

| | Current | Analytic |
|---|---|---|
| FLOPs per GDN M_ℓ | ~250 TFLOP | ~4 TFLOP (0.8 BPTT + ~3 GEMM) |
| Wall per M_ℓ | ~60 s | **~1–3 s** |
| Per prompt | 25–50 min, 25 layers only | **~2–4 min, all 64 layers** |
| 20 prompts | ~8 h | ~1 h (full depth) |
| 100 prompts | — | overnight |
| 1000 prompts (paper scale) | cloud only | **~2 days, local** |

Memory: chunk cotangents at ~512; the largest transients are the tiled
q/k/v (~0.7 GB), the adjoint state `[512, 48, 128, 128]` (~1.6 GB), and
the gradient buffer `[512, 32, 10240]` (~0.7 GB). Comfortable in 64 GB.

The GEMMs can run fp16 with fp32 accumulation (MLX matmul default) for
~2× on the dominant cost with low stability risk — this replaces Fix 5's
riskier proposal.

### Side benefits

- **No autograd through the layer at all** → the `patch_gdn` ops-fallback
  and `mx.checkpoint` machinery (and the 154 GB retention concern) drop
  out of the fit path entirely.
- **Most of the math is the Triton port.** The assembly GEMMs transfer
  verbatim to PyTorch; the BPTT scan does not — it would need a Triton
  time-loop. Likely shortcut: the `flash-linear-attention` (fla)
  library ships Triton chunked gated-delta-rule kernels with a
  training-grade backward (including dg/dβ) — the same kernels HF
  Transformers uses for Qwen3-Next-class models. Verify equivalence
  against `gdn_backward.py` and reuse instead of writing a kernel.
- **Position-resolved for free.** The BPTT gradients are per-source-
  position anyway, so the true position-diagonal Jacobian is available
  at no extra cost if ever wanted (see §4.2 — the future-summed form is
  the correct one for this project, but the option costs nothing).

### Verification

It computes the *same object* as `per_layer_jacobian`, and the finished
20-prompt run's checkpoints are a free exact reference:

1. Compare analytic M_ℓ vs `per_layer_jacobian` to ~1e-4 on one GDN
   layer, one full-attention layer, and the final norm (with g/β paths
   disabled for the comparison, since the current kernel drops them —
   §4.1).
2. Then the standard readout/intervention sanity checks from
   PERFORMANCE.md §D.

### Effort and risk

~2–3 days. Correctness risk is medium in the writing (many hand-derived
pieces: gated-norm rank-1 terms, the conv's banded Jacobian, GQA/head-
repeat bookkeeping are the fiddly bits) but **low effective** — the
verification is cheap, total, and already has reference data.

---

## 3. Interim option: unbiased probing (~10× this weekend, replaces Fix 2)

Swap the 5120 one-hot cotangents for k ≈ 512 Rademacher (±1) cotangents:

```
M̂ = (1/k) Σᵢ vᵢ (vᵢᵀ M),   E[v vᵀ] = I  ⇒  E[M̂] = M
```

- **Unbiased with no rank assumption** — strictly dominates Fix 2
  (low-rank power iteration), whose rank-64 assumption PERFORMANCE.md
  itself flags as unverifiable. Drop Fix 2.
- ~20 lines changed in `fit.py`; immediate ~10× (k/D = 512/5120).
- Probe noise averages down across prompts; chain products of
  independent unbiased factors remain unbiased in expectation. Validate
  the variance against one exact prompt from the existing checkpoint.
- **Scope limit (see §5): readout demos only.** The intervention
  experiments are causal claims about J; use the analytic (exact) lens
  for those.

Variant: probes via central finite differences through the **stock fused
kernel** (`(f(x+εv) − f(x−εv)) / 2ε`, batched along B) — zero backward
code, and it exactly includes the g/β/z/conv paths the current custom
VJP drops. Risk shifts to step-size tuning; verify against `mx.vjp` on
one layer.

---

## 4. Correctness findings (change what "exact" means)

### 4.1 The custom kernel VJP silently drops the x→g and x→β paths — measure, then decide

`custom_gdn_patch.py:63-64` returns zeros for dg/dβ, and g, β are
projections of the layer input (`in_proj_a`, `in_proj_b`). The ops
fallback *includes* these paths, so a kernel-fit lens and an ops-fit
lens differ silently.

Structure of the gap: the dropped term is
`(W_o-side) · ∂y/∂(g,β) · [W_a; W_b]` — a **rank ≤ 96** additive
correction to the 5120×5120 M_ℓ (48+48 projection rows). Two competing
a-priori arguments, neither decisive:

- *Against negligibility:* low-rank ≠ small. g gates state persistence,
  so ∂(future outputs)/∂g compounds over the horizon — exactly the
  future-summed influence the lens measures. And the correction is a
  full matrix term, not a rescaling: ∂y_t/∂g_s points wherever the
  accumulated q·(k⊗δ) structure points, so "magnitude but not
  direction" is not guaranteed.
- *For negligibility:* `compute_g`/sigmoid saturation shrinks these
  derivatives (β′ = β(1−β) ≤ 0.25), and the 5120→48 bottleneck limits
  how much input variation reaches the gates.

Measurement protocol (cheap, uses existing code):

1. Diff one M_ℓ fit with the ops fallback (includes g/β) vs the custom
   kernel (drops them) — at an **early, mid, and late GDN layer** (gate
   saturation varies with depth; one layer is not representative).
2. Report ‖ΔM‖_F/‖M‖_F *and* the downstream deltas: top-k readout
   changes on the known prompts, and intervention-vector direction
   change (cosine) for a few words.
3. Decision: if readouts and intervention vectors are stable, document
   the approximation and keep dropping the paths. Under the analytic
   assembly the inclusion cost is small anyway (two 5120→48 projections
   in the assembly plus dg/dβ accumulators in the BPTT — both simple
   contractions already available in the loop: dg from `ds_dec ⊙ s_pre`
   summed over (Dv,Dk), dβ from `d_delta ⊙ (v − kv_mem)`), so the bar
   for including them should be low.

For the intervention experiments (causal claims) the paths remain in
scope for the analytic mainline unless the measurement shows they are
noise-level.

### 4.2 The cross-position sum is correct; the docstring is wrong

`fit.py:10-13` claims the fit uses the "position-diagonal block" of
M_ℓ. It does not: the cotangent is hot at **all** valid output positions
(`fit.py:84-93`), so M includes Σ_{t≥s} cross-position flow — the
influence of position-s activity on all future outputs.

That is the *right* object for this project: the global-workspace post
defines the J-lens as the pattern that makes a word more likely "**at
some point in the future**" — a future-summed influence. **Fix the
docstring, not the math.**

---

## 5. Alignment with the goal (the global-workspace post)

The purpose of this repo is to implement the system in
https://www.anthropic.com/research/global-workspace: fit a J-lens →
read out J-space contents per layer → causally intervene
(swap/inject/ablate/steer). The fit is the sole bottleneck on quality,
and the review's recommendation is a pure speedup of the *identical*
J objects the post's method needs — no change to the science.

Two goal-driven consequences for prioritization:

1. **Full depth is required, not nice-to-have.** The post's core
   observations — concept evolution across all processing stages, the
   workspace census ("a few dozen concepts," "<1/10 of activity"), the
   ~100× connectivity-density claim — need J at early and middle
   layers. Today's 25-late-layers chain can reproduce readout demos but
   not the workspace-level findings. The analytic route is what makes
   full depth affordable locally.
2. **Exactness matters because half the system is causal.** Readouts
   tolerate noise (probing is fine there); swap/ablate/steer experiments
   (spider→ant, France→China-style) are claims about the true local
   linearization — mainline the analytic lens for those, and settle the
   g/β question by measurement (§4.1).

Downstream milestones once a trustworthy full-depth 100+-prompt lens
exists (all cheap relative to the fit): workspace census, whole-J-space
ablation (fluency preserved / multi-step reasoning lost), reportability
and modulability tests.

---

## 6. Verdict on PERFORMANCE.md's items

| Plan item | Verdict |
|---|---|
| Fix 1 (batched backward) | Right target, wrong ceiling (~2–4× real, not 10–50×). Replace with **identity-basis analytic assembly** (§2): ~30–60×, exact, same verification harness. |
| Fix 2 (low-rank power iteration) | **Drop** — dominated by unbiased probing (§3) on every axis: unbiased, no rank assumption, verifiable against existing checkpoints. |
| Fix 3 (two-pass kernel, no atomics) | **Defer** — under analytic assembly the recurrence is ~20% of cost and batched cotangents amortize contention. Revisit only if profiling disagrees. |
| Fix 4 (Blelloch scan) | Agree, **skip** — T=32 makes it pointless. |
| Fix 5 (fp16/bf16 accumulation) | Skip as written; instead run the analytic GEMMs fp16 with fp32 accumulation (~2× on the dominant cost, low risk). |
| Path A (rented Mac) | Unchanged (unattended runs only). |
| Path B (cloud H100 + Triton) | Still correct for maximum speed at 1000-prompt scale, but now **optional** — the analytic math *is* the Triton port, and local full depth reaches paper scale in ~2 days. |
| Path C (distributed merge) | Unchanged; composes with everything. |
| Path D (4-bit on GPU) | Unchanged (impossible). |

---

## 7. Recommended sequence

1. **Now (hours):** closed-form J_64; fix the `fit.py` docstring (§4.2);
   batch `mx.eval` syncs; optionally start probing (§3) for a ~10×
   readout-quality lens validated against the existing exact checkpoint.
2. **This week (~2–3 days):** identity-basis analytic assembly (§2) —
   the real Fix 1. Run the §4.1 g/β measurement first; include the
   paths if the gap is above noise (cheap to add in the assembly).
   Result: full-depth 64-layer fit at ~2–4 min/prompt.
3. **Then:** 100-prompt full-depth lens overnight → run the post's
   experiment suite (census, ablation, reportability, interventions).
4. **Only if needed:** 1000-prompt census faster than ~2 days → Path B,
   reusing the analytic math as the Triton port.
