# Handoff: Measure g/β Path Gap (§4.1)

**Project:** jlens-qwen36 (`~/Repositories/com.github/WeZZard/jlens-qwen36`)
**Task:** Quantify the difference between the ops-fit lens (includes decay-gate paths) and the kernel-fit lens (drops them) on one GDN layer, to decide whether the custom kernel needs fixing.

## Context

A full Jacobian lens fit is running in the background (PID 9045) using ~20 GB of the 64 GB unified memory. **Do not load the full model into memory** — write a self-contained measurement script that the orchestrator will run when the fit is paused or done (~5 min of GPU time needed).

Read these files first:
- `PERFORMANCE_REVIEW.md` §4.1 — the finding that motivates this.
- `jlens_qwen/custom_gdn_vjp.py` — the Metal kernel VJP that drops the g/β paths.
- `jlens_qwen/custom_gdn_patch.py` — how the custom VJP is wired in.
- `jlens_qwen/gdn_backward.py` — the ops-based backward that *includes* the g/β paths (verified exact vs `mx.vjp`).
- `jlens_qwen/patch_gdn.py` — the ops fallback patch.

## The finding (§4.1)

`custom_gdn_patch.py:63-64` returns zeros for `dg/dβ`:
```python
def _gdn_custom_vjp(primals, cotangent, output):
    ...
    dg = mx.zeros_like(g)
    dbeta = mx.zeros_like(beta)
    ...
```

But `g` and `β` are projections of the layer input (`in_proj_a`, `in_proj_b`), so the true VJP *does* flow gradient back through them to the input. The ops fallback (`gdn_backward.py:gdn_vjp`) includes these paths. So a kernel-fit lens and an ops-fit lens differ silently.

This matters because: in a GDN model, the decay-gate path is part of how *current* activity influences *future* outputs — precisely the influence the J-lens is defined to measure, and the intervention experiments rely on J being the true causal linearization.

## The task

Write a script `scripts/measure_gbeta_gap.py` that:

1. Loads the model (`from jlens_qwen.model import load`).
2. Picks one GDN layer (e.g. layer 0).
3. Captures the residual stream entering that layer from a short prompt.
4. Computes `M_l = d(h_{l+1})/d(h_l)` two ways:
   - **Ops path:** using `gdn_backward.gdn_vjp` (includes g/β) — 5120 VJPs, ~2 min.
   - **Kernel path:** using `custom_gdn_vjp.gdn_kernel_vjp` (drops g/β) — 5120 VJPs, ~30s.
5. Compares the two M_l matrices:
   - Max abs error, mean abs error, relative Frobenius error.
   - Per-entry: is the gap concentrated in certain rows/columns or uniform?
6. Also compute the "g/β-only" contribution: the difference M_ops - M_kernel = the contribution of the g/β paths. Measure its norm relative to M_ops.
7. Print a verdict:
   - If relative gap < 1%: the g/β paths are negligible; the kernel is fine as-is.
   - If relative gap 1-10%: moderate; worth fixing for interventions but not for readouts.
   - If relative gap > 10%: significant; the kernel must be fixed for any causal use.

### How to compute M_l each way

The per-layer Jacobian `M_l = d(h_{l+1})/d(h_l)` is the position-averaged Jacobian of the full DecoderLayer. To isolate the GDN contribution, compute `d(gdn_out)/d(gdn_in)` where gdn_in is the normed input and gdn_out is the GDN output (before out_proj). This is the "attention core" Jacobian.

For the **ops path**: use `gdn_backward.gdn_vjp(q, k, v, g, beta, state, dy)` directly. You need to extract q, k, v, g, beta from the GDN's internal state during a forward pass. See `gdn_forward` in `gdn_backward.py` — it returns the intermediate states.

For the **kernel path**: use `custom_gdn_vjp.gdn_kernel_vjp(q, k, v, g, beta, state, dy)` — same interface, different implementation.

Both take `dy` (the cotangent w.r.t. the GDN output `y`) and return `(dq, dk, dv)`. The full `d(gdn_out)/d(gdn_in)` also includes the paths through g and β back to the input, but those go through `in_proj_a`/`in_proj_b` (projections of the input). The ops path includes them; the kernel path doesn't.

To get the *full* M_l (including the projection paths), you'd also need `dg` and `dβ` from the ops path, then chain through `in_proj_a`/`in_proj_b`. But for the gap measurement, comparing just the `(dq, dk, dv)` from both paths is sufficient to see how much the g/β omission matters.

### Script structure

```python
# scripts/measure_gbeta_gap.py
"""
Measure the gap between the ops-fit and kernel-fit GDN Jacobians.

Run this when the fit is NOT running (it needs the model for ~5 min):
    uv run python scripts/measure_gbeta_gap.py
"""
import mlx.core as mx
import numpy as np
from jlens_qwen.model import load
from jlens_qwen.gdn_backward import gdn_forward, gdn_vjp as ops_vjp
from jlens_qwen.custom_gdn_vjp import gdn_kernel_vjp
# ... load model, pick layer 0, capture gdn internals, compute M both ways, compare
```

## Deliverables

1. `scripts/measure_gbeta_gap.py` — the measurement script.
2. A summary comment at the top with the verdict (fill in after running):
   ```python
   # VERDICT: <negligible | moderate | significant>
   # Relative Frobenius gap: X%
   # The g/β paths contribute Y% of the GDN Jacobian's norm.
   # Recommendation: <no fix needed | fix for interventions | must fix>
   ```

## Constraints

- **Do not run the script** — the orchestrator will run it when the fit is done. Just write it.
- **Do not load the model at import time** — only inside `main()` / `if __name__`.
- **Match the code style** (no comments unless asked, type hints, concise).
- **Do not commit** — leave for the orchestrator.

## Estimated effort

~2-3 hours. The script is straightforward; the subtlety is extracting the GDN internals (q, k, v, g, beta, state) from a real forward pass.