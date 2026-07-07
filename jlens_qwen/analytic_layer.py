"""Identity-basis analytic assembly of per-layer Jacobians.

The fast path (~30-60x over VJP-based fit). Assembles M_l = d(h_{l+1})/d(h_l)
analytically from the layer's structure instead of backpropagating 5120
cotangents. See PERFORMANCE_REVIEW.md §2.

Structure of a Qwen3.5 DecoderLayer:
    r = attn(norm_in(x))           # attention/linear-attention residual
    h = x + r
    out = h + mlp(norm_post(h))    # MLP residual

So d(out)/d(x) = I + d(mlp)/d(h) @ (I + d(r)/d(x) @ norm_in_jac)
              = I + M_mlp @ (I + M_attn @ M_norm_in)

Each component is assembled analytically:
- RMSNorm Jacobian: closed-form (diag + rank-1), already implemented.
- MLP Jacobian: Hadamard trick (the big win).
- Attention Jacobian: for FA layers, batch identity cotangents in head
  space through the softmax core; for GDN, head-space BPTT with analytic
  seeds.

This module implements the MLP branch first (the biggest cost), then
builds up to the full layer.

Measured on Qwen3.6-27B-4bit (2026-07-08, scripts/verify_analytic_layer.py),
analytic_attn=True vs the old hybrid vs the exact 5120-VJP reference:
- GDN layer 32: 4.2s vs 28.3s vs 54s  (rel err 2.14e-2 vs 2.59e-2)
- FA  layer 35: 1.4s vs 19.3s vs 45s  (rel err 1.45e-2 vs 1.97e-2)
i.e. ~7-14x over the hybrid, ~13-32x over the exact VJP, while being
MORE accurate (the residual error is the branch-product junction).
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import MLXLensModel
from .analytic import rms_norm_jacobian, final_norm_jacobian
from .analytic_attn import attn_branch_jacobian, _linear_weight, _valid_mask


def _dequantize_linear(layer, module) -> tuple[mx.array, mx.array]:
    """Dequantize a QuantizedLinear into (weight [out, in], bias [out] or None).

    Returns the dequantized weight and bias as fp32 mx arrays.
    """
    return _linear_weight(module)


def mlp_jacobian(
    model: MLXLensModel,
    layer_idx: int,
    h_post_norm: mx.array,
    valid_mask: mx.array,
) -> mx.array:
    """Analytic Jacobian of the MLP branch: d(down_proj(swiglu(gate(x), up(x))))/d(x),
    position-averaged over valid positions.

    The MLP is: y = down_proj(silu(gate_proj(x)) * up_proj(x))
    dy/dx at position s = down_proj^T @ [ diag(silu'(g_s) * u_s) @ gate_proj
                                        + diag(silu(g_s)) @ up_proj ]
    where g_s = gate_proj(x_s), u_s = up_proj(x_s).

    Position-averaged via the Hadamard trick:
        avg_s[ diag(a_s) @ W @ diag(ln_s) ] = W ⊙ (avg_s[ a_s * ln_s^T ])
    so the whole thing is a few GEMMs + a Hadamard, ~1 TFLOP instead of ~130.

    Args:
        model: the MLX model.
        layer_idx: which layer's MLP.
        h_post_norm: [S, D] — the post-attention-norm activations (input to MLP).
        valid_mask: [S] — 1.0 at valid positions.
    Returns:
        [D, D] — the position-averaged MLP Jacobian.
    """
    mlp = model.layers[layer_idx].mlp
    D = h_post_norm.shape[-1]
    S = h_post_norm.shape[0]
    xf = h_post_norm.astype(mx.float32)

    # Dequantize weights
    W_gate, _ = _dequantize_linear(model, mlp.gate_proj)  # [I, D] where I=17408
    W_up, _ = _dequantize_linear(model, mlp.up_proj)      # [I, D]
    W_down, _ = _dequantize_linear(model, mlp.down_proj)  # [D, I]

    # Compute gate and up projections at all positions
    g = mx.matmul(xf, W_gate.T)  # [S, I]
    u = mx.matmul(xf, W_up.T)    # [S, I]

    # silu(g) = g * sigmoid(g)
    # silu'(g) = sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g))
    #         = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    sig = mx.sigmoid(g)  # [S, I]
    silu_g = g * sig      # [S, I]
    silu_prime_g = sig * (1.0 + g * (1.0 - sig))  # [S, I]

    # The Jacobian has two terms:
    # Term A: down^T @ diag(silu'(g) * u) @ gate  -> via Hadamard:
    #   avg_s[ diag(silu'(g_s) * u_s) @ gate ] = gate ⊙ avg_s[ (silu'(g_s) * u_s) * 1^T ]
    #   But gate is [I, D], and the diag is [I], so:
    #   avg_s[ diag(d_s) @ W_gate ] = W_gate ⊙ avg_s[ d_s[:, None] ]
    #   where d_s = silu'(g_s) * u_s  (element-wise, [I])
    #   avg_s[ d_s[:, None] ] = mean over valid s of d_s, shaped [I, 1]
    m = valid_mask[:, None]  # [S, 1]
    n_valid = float(valid_mask.sum().tolist())

    d_a = silu_prime_g * u  # [S, I] — the diagonal of term A
    avg_d_a = (d_a * m).sum(axis=0) / n_valid  # [I]
    # termA = diag(avg_d_a) @ W_gate -> [I, D] (scale rows of W_gate)
    termA = W_gate * avg_d_a[:, None]  # [I, D]
    # J_a = W_down @ termA = [D, I] @ [I, D] = [D, D]
    J_a = mx.matmul(W_down, termA)  # [D, D]

    # Term B: down^T @ diag(silu(g)) @ up -> same Hadamard structure
    d_b = silu_g  # [S, I]
    avg_d_b = (d_b * m).sum(axis=0) / n_valid  # [I]
    termB = W_up * avg_d_b[:, None]  # [I, D] = diag(avg_d_b) @ W_up
    J_b = mx.matmul(W_down, termB)  # [D, D]

    return J_a + J_b  # [D, D]


def mlp_branch_jacobian(
    mlp,
    h_mid: mx.array,
    w_norm: mx.array,
    eps: float,
    valid_mask: mx.array,
) -> mx.array:
    """Analytic Jacobian of the full MLP branch d(mlp(norm(h)))/dh with the
    post-attention RMSNorm folded PER-POSITION (exact, unlike
    `mlp_jacobian @ rms_norm_jacobian` which multiplies position averages).

    Per position: J(s) = W_down [diag(dA_s) W_gate + diag(dB_s) W_up] J_n(s)
    with dA_s = silu'(g_s) u_s, dB_s = silu(g_s). Splitting
    J_n(s) = diag(w)/r_s - (w .* x_hat_s) x_hat_s^T/(D r_s), the position sum
    factors into diagonal rescales plus rank-|valid| outer-product
    corrections — same cost as the decorrelated version.

    h_mid: [S, D] pre-norm residual entering the MLP branch.
    Returns [D, D], position-averaged over valid positions.
    """
    S, D = h_mid.shape
    xf = h_mid.astype(mx.float32)
    r = mx.sqrt((xf * xf).mean(axis=-1, keepdims=True) + eps)  # [S, 1]
    x_hat = xf / r
    w = w_norm.astype(mx.float32)
    xn = mx.fast.rms_norm(xf, w, eps)  # [S, D]
    m_over_r = valid_mask / r[:, 0]    # [S]
    n_valid = valid_mask.sum()

    W_gate, _ = _linear_weight(mlp.gate_proj)  # [I, D]
    W_up, _ = _linear_weight(mlp.up_proj)      # [I, D]
    W_down, _ = _linear_weight(mlp.down_proj)  # [D, I]

    g = mx.matmul(xn, W_gate.T)  # [S, I]
    u = mx.matmul(xn, W_up.T)    # [S, I]
    sig = mx.sigmoid(g)
    dA = sig * (1.0 + g * (1.0 - sig)) * u  # silu'(g) * u, [S, I]
    dB = g * sig                             # silu(g), [S, I]

    wx = w[None] * x_hat  # [S, D]
    inner = mx.zeros((W_gate.shape[0], D), dtype=mx.float32)
    for dcoef, W in ((dA, W_gate), (dB, W_up)):
        # diag part: diag(sum_s dcoef_s m_s/r_s) @ W @ diag(w)
        d1 = mx.einsum("si,s->i", dcoef, m_over_r)  # [I]
        inner = inner + (W * d1[:, None]) * w[None]
        # rank part: sum_s (dcoef_s .* (W (w .* x_hat_s))) x_hat_s^T / (D r_s)
        p = mx.matmul(wx, W.T)  # [S, I]
        P = dcoef * p * (m_over_r / D)[:, None]  # [S, I]
        inner = inner - mx.matmul(P.T, x_hat)    # [I, D]

    return mx.matmul(W_down, inner) / n_valid


def decoder_layer_jacobian(
    model: MLXLensModel,
    layer_idx: int,
    h_in: mx.array,
    *,
    skip_first: int = 4,
    analytic_attn: bool = True,
    include_gbeta: bool = False,
    chunk: int = 256,
) -> mx.array:
    """Full per-layer Jacobian M_l = d(h_{l+1})/d(h_l), position-averaged.

    Assembles from the layer structure:
        r = attn(norm_in(x));  h_mid = x + r;  out = h_mid + mlp(norm_post(h_mid))
    => d(out)/d(x) = (I + M_mlp_branch) @ (I + M_attn_branch)

    analytic_attn=True (default): both branch Jacobians are analytic and
    EXACT within their branch (input norms folded per-position, attention
    core backpropagated with analytic seeds — see analytic_attn.py). The
    single remaining within-layer approximation is the product junction
    between the two averaged branch factors (position decorrelation of
    M_mlp_branch @ M_attn_branch).

    analytic_attn=False: the original hybrid (VJP attention + averaged
    norm factors), kept for A/B comparison. It carries three decorrelated
    junctions instead of one.

    include_gbeta: include the GDN x->g/beta gate paths (see
    PERFORMANCE_REVIEW.md §4.1). False matches the kernel-fit reference.

    Args:
        model: the MLX model.
        layer_idx: which DecoderLayer.
        h_in: [1, S, D] — input to the layer (residual stream entering).
        skip_first: leading positions to skip.
    Returns:
        [D, D] — the position-averaged full-layer Jacobian.
    """
    from .fit import per_layer_jacobian, valid_positions, _one_layer_forward

    D = model.d_model
    S = h_in.shape[1]
    h = h_in[0].astype(mx.float32)  # [S, D]

    # Valid mask
    arange_S = mx.arange(S)
    valid_mask = mx.where(
        (arange_S >= skip_first) & (arange_S < S - 1),
        mx.ones((S,), dtype=mx.float32),
        mx.zeros((S,), dtype=mx.float32),
    )

    # Compute the intermediate activations we need.
    layer = model.layers[layer_idx]
    # norm_in(x) -> input to attention
    x_normed_in = layer.input_layernorm(h.astype(h_in.dtype)).astype(mx.float32)  # [S, D]
    # r = attn(norm_in(x)) -> need the full forward to get this
    # Compute the intermediate activations: r = attn(norm_in(x)), h_mid = x + r
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(h_in, cache=None)
    ssm_mask = create_ssm_mask(h_in, cache=None)
    mask = ssm_mask if layer.is_linear else fa_mask
    # The normed input needs to be [1, S, D] for the attention sub-layer.
    x_normed_4d = x_normed_in[None].astype(h_in.dtype)
    if layer.is_linear:
        r = layer.linear_attn(x_normed_4d, mask=mask, cache=None)
    else:
        r = layer.self_attn(x_normed_4d, mask=mask, cache=None)
    r = r[0].astype(mx.float32)  # [S, D]
    h_mid = h + r  # [S, D] — residual after attention
    x_normed_post = layer.post_attention_layernorm(h_mid.astype(h_in.dtype)).astype(mx.float32)  # [S, D]

    weight_post = layer.post_attention_layernorm["weight"].astype(mx.float32)
    eps_post = float(layer.post_attention_layernorm.eps) if hasattr(layer.post_attention_layernorm, "eps") else 1e-6

    if analytic_attn:
        # Exact branch Jacobians (input norms folded per-position).
        M_attn_full = attn_branch_jacobian(
            layer, h, skip_first=skip_first,
            include_gbeta=include_gbeta, chunk=chunk,
        )  # d(r)/d(x)
        M_mid = mx.eye(D) + M_attn_full  # d(h_mid)/d(x)
        M_mlp_branch = mlp_branch_jacobian(
            layer.mlp, h_mid, weight_post, eps_post, valid_mask
        )  # d(mlp(norm_post(h)))/d(h_mid)
        # Single remaining approximation: the averaged-product junction.
        return M_mid + M_mlp_branch @ M_mid

    # Original hybrid path (three decorrelated junctions), kept for A/B.
    weight_in = layer.input_layernorm["weight"].astype(mx.float32)
    eps_in = float(layer.input_layernorm.eps) if hasattr(layer.input_layernorm, "eps") else 1e-6
    M_norm_in = rms_norm_jacobian(h, weight_in, eps_in, valid_mask=valid_mask)  # [D, D]

    M_attn = _attn_jacobian_vjp(model, layer_idx, x_normed_in, valid_mask, skip_first)  # [D, D]
    M_mlp = mlp_jacobian(model, layer_idx, x_normed_post, valid_mask)  # [D, D]
    M_norm_post = rms_norm_jacobian(h_mid, weight_post, eps_post, valid_mask=valid_mask)  # [D, D]

    # r = attn(norm_in(x))      => d(r)/d(x) = M_attn @ M_norm_in
    # h_mid = x + r             => d(h_mid)/d(x) = I + M_attn @ M_norm_in
    # out = h_mid + mlp(norm_post(h_mid))
    M_attn_full = M_attn @ M_norm_in
    M_mid = mx.eye(D) + M_attn_full
    M_mlp_full = M_mlp @ M_norm_post @ M_mid
    return M_mid + M_mlp_full


def _attn_jacobian_vjp(
    model: MLXLensModel,
    layer_idx: int,
    x_normed: mx.array,
    valid_mask: mx.array,
    skip_first: int,
) -> mx.array:
    """VJP-based Jacobian of the attention sub-layer w.r.t. its (normed) input.

    This is the slow part that the full analytic assembly will replace.
    For now, it's the same cost as the original per_layer_jacobian's
    attention portion.
    """
    D = model.d_model
    S = x_normed.shape[0]
    layer = model.layers[layer_idx]

    # Wrap the attention sub-layer as a function of the normed input.
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    h_4d = x_normed[None].astype(mx.float32)  # [1, S, D]
    fa_mask = create_attention_mask(h_4d, cache=None)
    ssm_mask = create_ssm_mask(h_4d, cache=None)
    mask = ssm_mask if layer.is_linear else fa_mask

    def attn_fn(x):
        if layer.is_linear:
            return layer.linear_attn(x.astype(h_4d.dtype), mask=mask, cache=None)
        else:
            return layer.self_attn(x.astype(h_4d.dtype), mask=mask, cache=None)

    compiled = mx.compile(attn_fn)
    out = compiled(h_4d)
    mx.eval(out)

    arange_S = mx.arange(S)
    pos_mask = mx.where(
        (arange_S >= skip_first) & (arange_S < S - 1),
        mx.ones((S,), dtype=mx.float32),
        mx.zeros((S,), dtype=mx.float32),
    )
    pos_idx = mx.array(valid_positions(S, skip_first))

    M = np.zeros((D, D), dtype=np.float32)
    for d in range(D):
        dim_onehot = mx.zeros((D,), dtype=mx.float32)
        dim_onehot = dim_onehot.at[d].add(mx.array(1.0))
        cot = pos_mask[None, :, None] * dim_onehot[None, None, :]
        _, vjps = mx.vjp(compiled, [h_4d], [cot])
        grad = vjps[0]
        mx.eval(grad)
        M[d] = np.array(grad[0, pos_idx, :].astype(mx.float32).mean(axis=0))

    return mx.array(M)


def valid_positions(seq_len: int, skip_first: int = 4) -> list[int]:
    return list(range(skip_first, seq_len - 1))


def attn_jacobian_analytic(
    model: MLXLensModel,
    layer_idx: int,
    h_in_2d: mx.array,
    *,
    skip_first: int = 4,
    include_gbeta: bool = False,
    chunk: int = 256,
) -> mx.array:
    """Analytic attention-branch Jacobian, replacing `_attn_jacobian_vjp`.

    NOTE the contract change: takes the PRE-norm residual `h_in_2d` [S, D]
    and returns d(attn(norm_in(x)))/dx with the input norm folded exactly
    per-position — i.e. it replaces the product
    `_attn_jacobian_vjp(...) @ rms_norm_jacobian(...)`, not just the first
    factor. See analytic_attn.attn_branch_jacobian.
    """
    return attn_branch_jacobian(
        model.layers[layer_idx], h_in_2d,
        skip_first=skip_first, include_gbeta=include_gbeta, chunk=chunk,
    )


__all__ = [
    "mlp_jacobian",
    "mlp_branch_jacobian",
    "decoder_layer_jacobian",
    "attn_jacobian_analytic",
    "rms_norm_jacobian",
]