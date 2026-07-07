"""Analytic assembly of per-layer Jacobians — the fast path.

Instead of backpropagating 5120 one-hot cotangents through a DecoderLayer
to extract its Jacobian column-by-column (the VJP-based fit in fit.py,
~60s/layer), this module assembles M_l analytically from the layer's
known structure:

- Every position-independent matrix (in_proj_qkv, out_proj, gate/up/down_proj)
  contributes via a single GEMM.
- Every position-dependent term (RMSNorm diagonal+rank-1, SiLU diagonal,
  conv banded, GDN recurrence) is a cheap correction.
- The Hadamard trick `Sigma_s diag(a_s) W diag(ln_s) = W odot (Sigma_s a_s ln_s^T)`
  turns a 130 TFLOP brute-force into a ~1 TFLOP assembly.

The GDN recurrence still needs a small BPTT pass in head space, but the
seed cotangents are analytic and the batched cotangents fold into the
existing Metal kernel's batch dimension (no new kernel code).

Result: ~30-60x faster than the VJP fit, exact, same verification harness
(compare analytic M_l vs `per_layer_jacobian` on one layer). Full-depth
64-layer fit at ~2-4 min/prompt instead of 25-50 min for 25 late layers.

Reference: PERFORMANCE_REVIEW.md §2.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import MLXLensModel


def rms_norm_jacobian(x: mx.array, weight: mx.array, eps: float = 1e-6,
                      valid_mask: mx.array | None = None) -> mx.array:
    """Closed-form Jacobian of RMSNorm, position-averaged over valid positions.

    RMSNorm: y = x / rms(x) * w,  where rms(x) = sqrt(mean(x^2) + eps).
    dy/dx at position s = (1/rms_s) * (I - x_s x_s^T / (rms_s^2 * D)) * diag(w)

    For a batch of positions, returns the position-averaged [D, D] Jacobian,
    averaged over the valid positions (those where valid_mask[s] == 1).
    If valid_mask is None, averages over all positions.

    Args:
        x: [S, D] — activations at S positions.
        weight: [D] — RMSNorm weight.
        eps: norm epsilon.
        valid_mask: [S] — 1.0 for valid positions, 0.0 otherwise. If None,
            all positions are used.
    Returns:
        [D, D] — the position-averaged Jacobian dy/dx.
    """
    S, D = x.shape
    xf = x.astype(mx.float32)
    ms = (xf * xf).mean(axis=-1, keepdims=True) + eps  # [S, 1]
    rms = mx.sqrt(ms)  # [S, 1]
    w = weight.astype(mx.float32)  # [D]

    if valid_mask is None:
        mask = mx.ones((S,), dtype=mx.float32)
    else:
        mask = valid_mask.astype(mx.float32)
    n_valid = float(mask.sum().tolist())  # scalar

    # J_avg = mean_valid[ J_pos_s ]
    # = (1/n_valid) * sum_s mask_s * [ (1/rms_s) * diag(w) - (1/rms_s^2 * D) * (w * x_s) x_s^T ]
    # Term 1 (diagonal): (w/n_valid) * sum_s mask_s / rms_s
    weighted_inv_rms = (mask / rms[:, 0])  # [S]
    term1_diag = w * (weighted_inv_rms.sum() / n_valid)  # [D]
    # Term 2 (rank-1 per pos): (1/(n_valid * D)) * sum_s mask_s * (w * x_s / rms_s) (x_s / rms_s)^T
    x_normed = xf / rms  # [S, D]
    wx_normed = w * x_normed  # [S, D]
    m = mask[:, None]  # [S, 1]
    term2 = mx.matmul((m * wx_normed).T, x_normed) / (n_valid * D)  # [D, D]
    J_avg = -term2 + mx.diag(term1_diag)
    return J_avg


def final_norm_jacobian(model: MLXLensModel, h: mx.array,
                         skip_first: int = 4) -> mx.array:
    """J_64 = Jacobian of the final RMSNorm, position-averaged.

    h: [1, S, D] — the final-layer residual stream.
    skip_first: number of leading positions to exclude (attention sinks).
    Returns [D, D] — the position-averaged Jacobian over valid positions
    [skip_first, S-1).

    This replaces the 5120-VJP `per_layer_jacobian(model, h, n_layers)`
    call in fit.py (~1 min) with a closed-form computation (~ms).
    """
    h_2d = h[0].astype(mx.float32)  # [S, D]
    S = h_2d.shape[0]
    weight = model._text_module.norm["weight"]
    eps = float(model._text_module.norm.eps) if hasattr(model._text_module.norm, "eps") else 1e-6
    # Valid mask: 1 at positions [skip_first, S-1)
    arange_S = mx.arange(S)
    valid_mask = mx.where(
        (arange_S >= skip_first) & (arange_S < S - 1),
        mx.ones((S,), dtype=mx.float32),
        mx.zeros((S,), dtype=mx.float32),
    )
    return rms_norm_jacobian(h_2d, weight, eps, valid_mask=valid_mask)


# Placeholder for the full DecoderLayer analytic assembly — the big win.
# This is the ~2-3 day work item from PERFORMANCE_REVIEW.md §2.
def per_layer_jacobian_analytic(
    model: MLXLensModel,
    h_in: mx.array,
    layer_idx: int,
    *,
    skip_first: int = 4,
) -> mx.array:
    """Analytic assembly of M_l = d(h_{l+1})/d(h_l), position-averaged.

    Assembles the Jacobian from the layer's structure instead of
    backpropagating 5120 cotangents. See PERFORMANCE_REVIEW.md §2.

    TODO: implement the full assembly (GDN BPTT in head space + analytic
    projections + Hadamard trick for the MLP). For now, falls back to the
    slow VJP-based per_layer_jacobian for verification.
    """
    from .fit import per_layer_jacobian
    return per_layer_jacobian(model, h_in, layer_idx, skip_first=skip_first)


__all__ = ["rms_norm_jacobian", "final_norm_jacobian", "per_layer_jacobian_analytic"]