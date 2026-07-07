# Response: Analytic Attention Branch — DONE (with corrections)

Status: implemented, synthetically verified, NOT committed, NOT wired into
`fit_analytic.py` (per constraints). The running fit (PID 9047) was not
disturbed; it holds the old module in memory, so file edits are inert to it.

## Deliverables

1. `jlens_qwen/analytic_attn.py` — `attn_branch_jacobian()` (FA + GDN).
   Design deviation from the handoff, deliberate: it returns
   **d(attn(norm_in(x)))/dx** with the input RMSNorm folded per-position
   (scalar 1/r_s + rank-1 split), NOT a standalone `M_attn` to be
   multiplied by an averaged `M_norm_in`. The averaged product is a
   position-decorrelation approximation; the fold is exact and free.
2. `jlens_qwen/gdn_backward.py` — `gdn_vjp_batched()`: batched-cotangent
   BPTT with a SHARED forward (the per-cotangent adjoint is the only
   batched part), returning dq/dk/dv **and dg/dbeta**.
3. `jlens_qwen/analytic_layer.py` — `attn_jacobian_analytic()` wrapper,
   `mlp_branch_jacobian()` (post-norm folded exactly, same cost), and
   `decoder_layer_jacobian(analytic_attn=True|False, include_gbeta=...)`.
   Old hybrid path kept behind `analytic_attn=False` for A/B.
4. `tests/test_analytic_attention.py` — synthetic verification (tiny real
   mlx_lm classes, no model load). Results:
   - FA branch: 1.5e-08 max abs vs mx.vjp
   - GDN branch with g/beta: 1.5e-08; without (kernel semantics): 3e-08
   - MLP branch: 1.1e-08
   - Metal-kernel fast path (cotangents folded into B) vs ops BPTT: 4e-06
   - Full layer: new path 0.6-1.3% rel Frobenius vs exact, old hybrid
     1.8-2.2% (synthetic; the single remaining approximation is the
     branch-product junction)
5. `scripts/verify_analytic_layer.py` — real-model integration A/B
   (new vs old vs `fit.per_layer_jacobian`), for the orchestrator to run
   when the fit is done. The "measured speedup" comment for
   analytic_layer.py should be filled from this run.

## Corrections to the handoff worth knowing

- "~5120/6144 the cost" was garbled; the win is that projections enter as
  matrices once, not per cotangent. Realized via seeds = rows of W_o and
  a single stacked-projection GEMM after position folding.
- "g/beta = two extra 5120->48 projections (cheap)" omitted that the BPTT
  must produce dg/dbeta (neither `gdn_vjp` nor the Metal kernel did).
  Done in `gdn_vjp_batched`. The Metal kernel still lacks dg/dbeta, so
  `include_gbeta=True` currently forces the ops BPTT (slower); extend the
  kernel only if `scripts/measure_gbeta_gap.py` says the paths matter.
- `include_gbeta` defaults to **False** everywhere to match the
  kernel-fit reference semantics (per PERFORMANCE_REVIEW.md §4.1:
  measure, then decide).
- FA has **partial RoPE** (rotary factor 0.25) and per-head interleaved
  q/gate rows in q_proj — both handled by backpropagating through the
  real submodules/layout instead of hand-assembling them.

## Found while implementing (not in the handoff)

The running fit's `decoder_layer_jacobian` composes position-AVERAGED
factors (`M_attn @ M_norm_in`, `M_mlp @ M_norm_post @ ...`). That is an
unverified within-layer approximation (three decorrelated junctions);
`fit.per_layer_jacobian` had none. The new path removes two of the three
junctions exactly. Run `scripts/verify_analytic_layer.py` before trusting
the full-depth lens for anything causal, and prefer `analytic_attn=True`
for the next fit.
