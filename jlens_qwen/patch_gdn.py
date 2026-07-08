"""Force the Gated DeltaNet ops fallback and wrap each linear-attention layer
in mx.checkpoint so the backward pass through the 48 linear layers of
Qwen3.6-27B doesn't retain ~154 GB of recurrent state.

Two patches:

1. Monkey-patch ``mlx_lm.models.gated_delta.gated_delta_update`` so it
   always calls ``gated_delta_ops`` (the pure-Python differentiable loop)
   instead of the fused ``metal_kernel`` which has no registered VJP.

2. Wrap each ``GatedDeltaNet.__call__`` in ``mx.checkpoint`` so the
   per-timestep recurrence is recomputed during the backward pass instead
   of being retained.

Both patches are idempotent; calling ``patch_gdn()`` more than once is a
no-op after the first call.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gated_delta as _gdn_mod
from mlx_lm.models.gated_delta import gated_delta_ops, gated_delta_update

_PATCHED = False
_CHECKPOINT_APPLIED = False
_INFERENCE_KERNEL = False


def set_inference_mode(enabled: bool = True) -> None:
    """Route ``gated_delta_update`` back to the stock fused kernel.

    The ops-forcing patch exists so autograd can differentiate the
    recurrence during fitting. Cached generation (StreamSession) needs no
    gradients and the fused kernel is ~20x faster on prefill, so inference
    contexts flip this on. Fitting processes never call this; the default
    (False) preserves fit semantics.
    """
    global _INFERENCE_KERNEL
    _INFERENCE_KERNEL = enabled


def _patched_gated_delta_update(
    q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
):
    """Drop-in replacement for ``gated_delta_update`` that uses the
    ops-based differentiable implementation (ignores ``use_kernel``),
    unless inference mode is enabled (see ``set_inference_mode``).

    Mirrors the pre-processing in the original ``gated_delta_update``: compute
    ``beta = sigmoid(b)`` and ``g = compute_g(A_log, a, dt_bias)``, then call
    ``gated_delta_ops``.
    """
    if _INFERENCE_KERNEL and use_kernel:
        # Stock dispatch (fused Metal kernel when on GPU).
        return gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel
        )
    from mlx_lm.models.gated_delta import compute_g, gated_delta_ops

    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    return gated_delta_ops(q, k, v, g, beta, state, mask)


def patch_gdn() -> None:
    """Apply both patches. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    # Patch 1: force ops fallback in the module-level dispatch function.
    # qwen3_5.py does `from .gated_delta import gated_delta_update`, so it
    # has its *own* reference. We must patch both.
    _gdn_mod.gated_delta_update = _patched_gated_delta_update
    import mlx_lm.models.qwen3_5 as _qwen35_mod
    _qwen35_mod.gated_delta_update = _patched_gated_delta_update

    # Patch 2: wrap GatedDeltaNet.__call__ in mx.checkpoint.
    # We patch the class method once; every instance inherits it.
    #
    # mx.checkpoint treats all positional args as mx.array inputs to
    # checkpoint. But __call__ is a bound method whose first arg is `self`
    # (an nn.Module, not an array). So we can't just wrap __call__ directly.
    # Instead we wrap a function that takes only the array inputs (inputs,
    # mask, cache) and calls the original method with `self` bound.
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    global _CHECKPOINT_APPLIED
    if not _CHECKPOINT_APPLIED:
        original_call = GatedDeltaNet.__call__

        # The checkpointed fn takes the same array args as __call__ (inputs,
        # optionally mask). We exclude `cache` from checkpointing because
        # cache holds mutable state; in our forward we always pass cache=None.
        # When cache is not None (generation), we fall back to the original
        # un-checkpointed call.
        def _checkpointed_call(self, inputs, mask=None, cache=None):
            if cache is not None:
                # Generation path: don't checkpoint (stateful cache).
                return original_call(self, inputs, mask, cache)

            # Checkpoint only the array inputs. We use a closure that
            # captures self and mask, and takes only `inputs` as the
            # differentiable primal.
            mask_arr = mask  # may be None or an mx.array or "causal" string
            # mx.checkpoint requires array inputs; strings/None are passed
            # via closure, not as primals.
            def body(h):
                return original_call(self, h, mask_arr, None)

            return mx.checkpoint(body)(inputs)

        GatedDeltaNet.__call__ = _checkpointed_call
        _CHECKPOINT_APPLIED = True

    _PATCHED = True


__all__ = ["patch_gdn", "set_inference_mode"]