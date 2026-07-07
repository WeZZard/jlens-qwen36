"""Wire the custom Metal GDN backward into the GDN forward via mx.custom_function.

This replaces the GDN forward inside GatedDeltaNet.__call__ with a custom
function that:
- forward: uses the stock gated_delta_kernel (fast Metal, 0.4ms).
- vjp: uses our Metal backward kernel (8ms) instead of the ops loop (19ms).

The custom function is registered for the (q, k, v) inputs; (g, beta, state)
are treated as constants (no grad needed for the lens).

This module is imported for its side effect: patching the GDN forward to
use the custom function. Import after patch_gdn to override the ops fallback.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .custom_gdn_vjp import gdn_kernel_vjp
from .gdn_backward import gdn_forward as _ops_forward, gdn_vjp as _ops_vjp


_PATCHED_CUSTOM = False


@mx.custom_function
def _gdn_custom_forward(q, k, v, g, beta, state):
    """Custom forward: use the stock Metal kernel for the recurrence.

    Inputs are the post-processed q, k, v, g, beta (g = compute_g(A_log, a,
    dt_bias), beta = sigmoid(b)). The custom_function is registered for
    these so the VJP gives grads w.r.t. (q, k, v); grads w.r.t. (g, beta,
    state) are zeros (the lens doesn't need them, and upstream ops handle
    the g/beta computation's own backward).

    q, k: [B, T, Hk, Dk]. v: [B, T, Hv, Dv]. g, beta: [B, T, Hv].
    state: [B, Hv, Dv, Dk] or None.
    Returns y: [B, T, Hv, Dv].
    """
    from mlx_lm.models.gated_delta import gated_delta_kernel
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    y, _ = gated_delta_kernel(q, k, v, g, beta, state, mask=None)
    return y


@_gdn_custom_forward.vjp
def _gdn_custom_vjp(primals, cotangent, output):
    """Custom VJP: use our Metal backward kernel.

    primals: (q, k, v, g, beta, state). cotangent: dy (same shape as y).
    Returns (dq, dk, dv, dg, dbeta, dstate). dg/dbeta are REAL gate
    gradients (kernel v4, verified vs gdn_backward.gdn_vjp_batched) —
    upstream autograd chains them through compute_g/sigmoid back to the
    input, so per-layer VJPs now include the x->a/b->gate paths that were
    previously zeroed (measured 4.9-7.5% of ||M||, see
    scripts/measure_gbeta_gap.py). Only dstate stays zero (state is a
    zeros constant in the lens fit).
    """
    q, k, v, g, beta, state = primals
    dy = cotangent
    dq, dk, dv, dg, dbeta = gdn_kernel_vjp(
        q, k, v, g, beta, state, dy, return_gbeta=True
    )
    if state is not None:
        dstate = mx.zeros_like(state)
    else:
        dstate = None
    return dq, dk, dv, dg, dbeta, dstate


def patch_gdn_custom() -> None:
    """Patch GatedDeltaNet.__call__ to use the custom forward (with Metal VJP).

    Idempotent. Call after patch_gdn() (which sets up the ops fallback as a
    base). This overrides the forward to use the kernel + custom VJP.
    """
    global _PATCHED_CUSTOM
    if _PATCHED_CUSTOM:
        return

    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    # Save the original (post-patch_gdn) __call__ for fallback.
    original_call = GatedDeltaNet.__call__

    def _new_call(self, inputs, mask=None, cache=None):
        if cache is not None:
            # Generation path: fall back to original.
            return original_call(self, inputs, mask, cache)

        # Replicate the GDN forward up to the gated_delta_update call,
        # then use our custom forward for the recurrence.
        from mlx_lm.models.gated_delta import compute_g
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = __import__("mlx.nn.layers.distributed", fromlist=["sum_gradients"]).sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        # conv state (cache=None -> zero init)
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        # Compute g, beta (differentiable, handled by mx.autograd upstream
        # of the custom function).
        from mlx_lm.models.gated_delta import compute_g
        g = compute_g(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)

        # Custom forward (Metal kernel + registered VJP).
        out = _gdn_custom_forward(q, k, v, g, beta, None)
        # out shape: [B, S, Hv, Dv]

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    GatedDeltaNet.__call__ = _new_call
    _PATCHED_CUSTOM = True


__all__ = ["patch_gdn_custom"]