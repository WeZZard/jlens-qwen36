# fit-01 — Baseline VJP fitting pipeline

**Commits:** `ff83cb3` (scaffold + custom Metal GDN VJP + chain-multiply),
`d8f8201` (threadgroup-reduction kernel).

## The problem

The J-lens needs, for every source layer ℓ, the Jacobian
`M_ℓ = d(h_{ℓ+1})/d(h_ℓ) ∈ R^{5120×5120}`, then chains them:
`J_ℓ = J_{ℓ+1} @ M_ℓ`.

Three architecture facts make this expensive on Qwen3.6-27B:

1. **48 of 64 layers are Gated DeltaNet (GDN) linear attention** — a
   sequential recurrence over time, ~10× slower than softmax attention.
2. **The fused Metal GDN kernel has no registered VJP** — MLX autograd
   cannot backprop through it.
3. **MLX re-runs the forward on every `mx.vjp` call** — no graph retention,
   so a 5120-dim Jacobian naively costs 5120 forward+backward pairs.

## What this baseline shipped

- **Chain-multiply** `J_ℓ = J_{ℓ+1} @ M_ℓ` — avoids a full-stack VJP per
  source layer.
- **`mx.compile`** on the per-layer forward — ~3×.
- **32-token fitting sequences** — ~4× vs 128-token (shorter GDN scan).
- **25 evenly-spaced late layers** — avoids the full-depth chain.
- **Custom Metal GDN VJP kernel** (`custom_gdn_vjp.py`) — the missing
  backward, hand-written; `d8f8201` reduced it from 8.3 ms → 3.8 ms with a
  threadgroup reduction (~5× over the ops loop).
- **Checkpoint/resume** every prompt.

## Result

- One per-layer `M_ℓ` (5120 VJPs through one DecoderLayer): **~1 min**.
- 20 prompts × 23-layer chain: **~8 hours** on an M4 Pro / 64 GB.

Sustained throughput was already ~4.2 TFLOPS — roughly 50–60 % of the
machine's practical peak. **The lesson that drove everything after this:**
we were near the roofline *for the FLOPs we were doing*, so the win had to
come from doing fewer FLOPs, not scheduling them better (see fit-02, fit-04).

## Verification

The GDN backward has a numerical gradient check against `mx.vjp` on a tiny
example (`gdn_backward.py:test_gdn_vjp`) — the harness every later iteration
reused.
