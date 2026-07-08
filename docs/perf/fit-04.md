# fit-04 — Analytic attention branch (the real Fix 1)

**Commits:** `e5e51f4` (implement + verify + chain fix),
`0e413cd` (wire into the fit).
**Code:** `jlens_qwen/analytic_attn.py`, `gdn_backward.py:gdn_vjp_batched`.

## The problem it replaced

After fit-02/03, the **attention branch** was the last part still computed
by per-cotangent VJP: ~15–30 s per layer, dominating the fit. A naive "batch
all 5120 cotangents into one backward pass" cannot fix this — backward
through a linear layer for C cotangents costs C GEMMs no matter how they are
batched, so its ceiling is only ~2–4×, not the 10–50× first assumed.

## The idea

Extend the identity-basis assembly (fit-02) to the whole attention branch,
computing `d(attn(norm_in(x)))/dx` with the input RMSNorm folded
per-position (a scalar `1/r_s` + rank-1 split — exact, not the decorrelated
product of position averages the earlier hybrid used).

- **Seed cotangents at the pre-`out_proj` space** = rows of `W_o`, so the
  output projection never appears as a per-cotangent GEMM.
- **Backprop only through the small nonlinear core:**
  - **Full-attention layers (16):** q/k head-norms → partial RoPE → softmax
    core → sigmoid gate, batched in head space (6144-dim, not 5120).
  - **GDN layers (48):** gated norm + z gate (autograd), then the recurrence
    via a **batched-cotangent BPTT with a shared forward**
    (`gdn_vjp_batched`) — the forward states are computed once; only the
    adjoint recurrence is batched over cotangents.
- **Contract with the stacked input projections once** per chunk, after the
  per-position norm fold, so the position sum happens *before* the single
  `[C,F] @ [F,D]` GEMM.

Everything data-dependent (RMSNorm, SiLU, sigmoid gates, depthwise conv, the
recurrence core) is diagonal, rank-1-per-head, or lives in head space; every
expensive matrix (`in_proj_*`, `out_proj`, the projections) is
position-independent and enters as one GEMM.

## Result (real-model A/B vs. exact 5120-VJP reference)

| Layer | analytic (new) | old hybrid | exact VJP |
|-------|----------------|-----------|-----------|
| GDN L32 | **2.14e-2 err @ 4.2 s** | 2.59e-2 @ 28.3 s | (reference) @ 54 s |
| FA L35 | **1.45e-2 err @ 1.4 s** | 1.97e-2 @ 19.3 s | (reference) @ 45 s |

So ~7–14× faster than the hybrid it replaced *and* more accurate — the
residual error is the single remaining within-layer approximation (the
averaged branch-product junction; the old hybrid carried three such
junctions). ~13–32× faster than the exact VJP.

## Correctness fixes that rode along

- **Chain off-by-one** (`fit.py`, `fit_analytic.py`): the chain loop
  evaluated layer ℓ's Jacobian at its own *output* `acts[ℓ]` and saved the
  product under `ℓ` — an off-by-one that left an extra layer-ℓ factor in
  every `J_ℓ`, while `lens.transport` applies `J_ℓ` to `acts[ℓ]`. Fixed to
  `J_{ℓ-1} = J_ℓ @ M_ℓ(@acts[ℓ-1])`. Toy-chain ground truth:
  **33–49 %** relative error as-coded vs. **2.6–4.8 %** fixed
  (`tests/test_chain_indexing.py`). v0.1-demo and the first full-depth run
  carry the bug; v0.2 is the first correctly-indexed lens.

## Verification

- Synthetic branch Jacobians (real mlx_lm classes, tiny dims) vs. `mx.vjp`:
  FA **1.5e-8**, GDN with g/β **1.5e-8**, GDN kernel-semantics **3e-8**, MLP
  **1.1e-8**, batched BPTT **~1e-4** (`tests/test_analytic_attention.py`).
- Real-model A/B above (`scripts/verify_analytic_layer.py`).
