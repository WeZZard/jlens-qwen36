# Response: Measure g/beta Gap — script written (methodology corrected)

Status: `scripts/measure_gbeta_gap.py` written, NOT run (fit active), NOT
committed.

## The handoff's method was invalid — replaced

The handoff proposed comparing `gdn_backward.gdn_vjp` (called the "ops
path, includes g/beta") against `custom_gdn_vjp.gdn_kernel_vjp` ("drops
g/beta"). **Both functions hold g/beta constant and return only
(dq, dk, dv)** — their difference measures kernel-vs-ops floating-point
noise and would report "negligible" no matter how large the g/beta
contribution actually is.

The corrected script computes the gap exactly:

    Delta_M = attn_branch_jacobian(include_gbeta=True)
            - attn_branch_jacobian(include_gbeta=False)

using the same ops-BPTT code path for both (verified to ~1e-8 vs mx.vjp
in tests/test_analytic_attention.py), so the difference isolates the
g/beta paths with zero implementation noise. Plus:

- Measured at an **early, mid, and late** GDN layer (gate saturation
  varies with depth; one layer is not representative).
- Independent cross-check: random probes through the whole GDN module via
  autograd with g/beta live in the graph (self-contained ops forward, no
  reliance on the analytic implementation being correct).
- Reports row/column concentration of Delta_M (the correction is rank
  <= 96 = rank(W_a) + rank(W_b), so expect concentration).
- Verdict bands preserved from the handoff (<1% / 1-10% / >10%).

Run when the fit is done (~5-10 min GPU):

    uv run python scripts/measure_gbeta_gap.py

Then fill the VERDICT header at the top of the script.
