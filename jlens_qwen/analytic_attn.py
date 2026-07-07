"""Analytic attention-branch Jacobians (FA + GDN) for the per-layer assembly.

Computes M_branch = (1/V) sum_{s,t valid} d(attn(norm_in(x))_t)/d(x_s) in
R^{DxD} with the input RMSNorm folded per-position (exact within the
branch). This replaces both the VJP fallback
(`analytic_layer._attn_jacobian_vjp`, ~15-30s/layer) and the decorrelated
`M_attn @ M_norm_in` product (which multiplied position-averaged factors,
losing per-position norm/gradient correlations).

Strategy (PERFORMANCE_REVIEW.md §2):
- Seed cotangents at the pre-out_proj space: rows of W_o (position-masked),
  so out_proj never appears as a per-cotangent GEMM.
- Backprop the seeds through the small nonlinear core only:
  - FA: q/k head norms + partial RoPE + softmax core + sigmoid gate
    (plain ops, autograd, chunked over cotangents).
  - GDN: gated norm + z gate (autograd), the recurrence (manual batched
    BPTT with a shared forward — `gdn_backward.gdn_vjp_batched` — or the
    existing Metal kernel with cotangent chunks folded into its batch
    axis), then conv + silu + q/k scaled norms (autograd).
- Contract with the stacked input projections ONCE per chunk after folding
  the input norm via its scalar/rank-1 split:
      g^T J_norm(s) = (g .* w)/r_s - (g . (w .* x_hat_s)) x_hat_s^T / (D r_s)
  so the position sum happens BEFORE the single [C,F]@[F,D] GEMM. This is
  what keeps the projection backward at one GEMM instead of one GEMM per
  (cotangent, position).

The g/beta paths (x -> in_proj_a/b -> decay/write gates) are included when
`include_gbeta=True`. Default is False to match the semantics of the
kernel-fit reference (`per_layer_jacobian` with the custom Metal VJP
active, which zeros dg/dbeta) — see PERFORMANCE_REVIEW.md §4.1; the
measurement of that gap is `scripts/measure_gbeta_gap.py`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .gdn_backward import gdn_forward, gdn_vjp_batched


def _linear_weight(module) -> tuple[mx.array, mx.array | None]:
    """Weight [out, in] (dequantized if needed) and bias, as fp32."""
    if hasattr(module, "scales"):
        w = mx.dequantize(
            module.weight, module.scales, module.biases,
            group_size=module.group_size, bits=module.bits,
        ).astype(mx.float32)
    else:
        w = module.weight.astype(mx.float32)
    b = module.bias.astype(mx.float32) if "bias" in module else None
    return w, b


def _project(x: mx.array, module) -> mx.array:
    """x @ W^T (+ bias) in fp32."""
    w, b = _linear_weight(module)
    y = mx.matmul(x, w.T)
    return y + b if b is not None else y


def _valid_mask(S: int, skip_first: int) -> mx.array:
    ar = mx.arange(S)
    return mx.where(
        (ar >= skip_first) & (ar < S - 1),
        mx.ones((S,), dtype=mx.float32),
        mx.zeros((S,), dtype=mx.float32),
    )


def _fold_norm_project(
    draw: mx.array,
    W_stack: mx.array,
    x: mx.array,
    w_norm: mx.array,
    eps: float,
    valid_mask: mx.array,
) -> mx.array:
    """Fold the input RMSNorm per-position, sum over source positions, and
    apply the stacked projection backward as one GEMM.

    draw: [C, S, F] — gradients w.r.t. the stacked projection outputs.
    W_stack: [F, D] — stacked projection weights (row-order matches draw).
    x: [S, D] — pre-norm residual entering the branch.
    Returns [C, D] rows of the branch Jacobian (position-averaged).

    Uses g^T J_n(s) = (g .* w)/r_s - (g . (w .* x_hat_s)) x_hat_s^T/(D r_s)
    with g = draw[c,s] @ W_stack, so the scalar 1/r_s and the rank-1 term
    fold BEFORE the GEMM.
    """
    S, D = x.shape
    xf = x.astype(mx.float32)
    r = mx.sqrt((xf * xf).mean(axis=-1, keepdims=True) + eps)  # [S, 1]
    x_hat = xf / r  # [S, D]
    w = w_norm.astype(mx.float32)
    m_over_r = valid_mask / r[:, 0]  # [S]
    n_valid = valid_mask.sum()

    # Diagonal part: ((sum_s m_s/r_s * draw[:,s]) @ W) .* w
    A = mx.einsum("csf,s->cf", draw, m_over_r)  # [C, F]
    term1 = mx.matmul(A, W_stack) * w[None]     # [C, D]

    # Rank-1 part: alpha[c,s] = draw[c,s] . (W (w .* x_hat_s)) * m_s/r_s
    U = mx.matmul(w[None] * x_hat, W_stack.T)   # [S, F]
    alpha = mx.einsum("csf,sf->cs", draw, U * m_over_r[:, None])  # [C, S]
    term2 = mx.matmul(alpha, x_hat) / D          # [C, D]

    return (term1 - term2) / n_valid


def _causal_bias(S: int) -> mx.array:
    ar = mx.arange(S)
    return mx.where(ar[None, :] <= ar[:, None], 0.0, -1e9).astype(mx.float32)


def _fa_branch(
    layer,
    x: mx.array,
    valid_mask: mx.array,
    *,
    chunk: int,
) -> mx.array:
    """Full-attention branch Jacobian d(self_attn(norm_in(x)))/dx, [D, D]."""
    attn = layer.self_attn
    S, D = x.shape
    H = attn.num_attention_heads
    KV = attn.num_key_value_heads
    hd = attn.head_dim
    scale = attn.scale
    rf = H // KV

    w_in = layer.input_layernorm.weight.astype(mx.float32)
    eps_in = layer.input_layernorm.eps
    xn = mx.fast.rms_norm(x.astype(mx.float32), w_in, eps_in)  # [S, D]

    W_q, b_q = _linear_weight(attn.q_proj)
    W_k, b_k = _linear_weight(attn.k_proj)
    W_v, b_v = _linear_weight(attn.v_proj)
    W_o, _ = _linear_weight(attn.o_proj)

    q_full = mx.matmul(xn, W_q.T) + (b_q if b_q is not None else 0.0)
    k_pre = mx.matmul(xn, W_k.T) + (b_k if b_k is not None else 0.0)
    v_pre = mx.matmul(xn, W_v.T) + (b_v if b_v is not None else 0.0)
    q_full = q_full.reshape(S, H, 2 * hd)
    k_pre = k_pre.reshape(S, KV, hd)
    v_pre = v_pre.reshape(S, KV, hd)

    wq_norm = attn.q_norm.weight.astype(mx.float32)
    wk_norm = attn.k_norm.weight.astype(mx.float32)
    eps_h = attn.q_norm.eps
    causal = _causal_bias(S)

    def core(qf, kp, vp):
        # qf: [C, S, H, 2*hd], kp/vp: [C, S, KV, hd] -> y_gated [C, S, H*hd]
        C = qf.shape[0]
        qh, gate = mx.split(qf, 2, axis=-1)
        qh = mx.fast.rms_norm(qh, wq_norm, eps_h)
        kh = mx.fast.rms_norm(kp, wk_norm, eps_h)
        qh = attn.rope(qh.transpose(0, 2, 1, 3))  # [C, H, S, hd]
        kh = attn.rope(kh.transpose(0, 2, 1, 3))  # [C, KV, S, hd]
        vh = vp.transpose(0, 2, 1, 3)
        if rf > 1:
            kh = mx.repeat(kh, rf, axis=1)
            vh = mx.repeat(vh, rf, axis=1)
        scores = mx.matmul(qh, kh.transpose(0, 1, 3, 2)) * scale + causal
        out = mx.matmul(mx.softmax(scores, axis=-1), vh)  # [C, H, S, hd]
        out = out.transpose(0, 2, 1, 3).reshape(C, S, H * hd)
        gate_flat = gate.reshape(C, S, H * hd)
        return out * mx.sigmoid(gate_flat)

    W_stack = mx.concatenate([W_q, W_k, W_v], axis=0)  # [F, D]
    pos = valid_mask[None, :, None]
    M = np.zeros((D, D), dtype=np.float32)

    for c0 in range(0, D, chunk):
        c1 = min(c0 + chunk, D)
        C = c1 - c0
        dy = pos * W_o[c0:c1][:, None, :]  # [C, S, H*hd]
        primals = [
            mx.broadcast_to(t[None], (C,) + t.shape)
            for t in (q_full, k_pre, v_pre)
        ]
        _, grads = mx.vjp(core, primals, [dy])
        draw = mx.concatenate(
            [g.reshape(C, S, -1) for g in grads], axis=-1
        )  # [C, S, F]
        rows = _fold_norm_project(draw, W_stack, x, w_in, eps_in, valid_mask)
        mx.eval(rows)
        M[c0:c1] = np.array(rows)

    return mx.array(M)


def _gdn_branch(
    layer,
    x: mx.array,
    valid_mask: mx.array,
    *,
    chunk: int,
    include_gbeta: bool,
    use_kernel: bool | str = "auto",
) -> mx.array:
    """GDN branch Jacobian d(linear_attn(norm_in(x)))/dx, [D, D]."""
    from mlx_lm.models.gated_delta import compute_g

    gdn = layer.linear_attn
    S, D = x.shape
    Hk, Dk = gdn.num_k_heads, gdn.head_k_dim
    Hv, Dv = gdn.num_v_heads, gdn.head_v_dim
    key_dim, value_dim = gdn.key_dim, gdn.value_dim
    conv_dim = gdn.conv_dim
    K = gdn.conv_kernel_size

    w_in = layer.input_layernorm.weight.astype(mx.float32)
    eps_in = layer.input_layernorm.eps
    xn = mx.fast.rms_norm(x.astype(mx.float32), w_in, eps_in)  # [S, D]

    W_qkv, _ = _linear_weight(gdn.in_proj_qkv)
    W_z, _ = _linear_weight(gdn.in_proj_z)
    W_a, _ = _linear_weight(gdn.in_proj_a)
    W_b, _ = _linear_weight(gdn.in_proj_b)
    W_o, _ = _linear_weight(gdn.out_proj)
    W_conv = gdn.conv1d.weight.astype(mx.float32)  # [conv_dim, K, 1]

    qkv_raw = mx.matmul(xn, W_qkv.T)   # [S, conv_dim]
    z = mx.matmul(xn, W_z.T).reshape(S, Hv, Dv)
    a = mx.matmul(xn, W_a.T)           # [S, Hv]
    b = mx.matmul(xn, W_b.T)           # [S, Hv]

    inv_scale = Dk ** -0.5

    def pre(qkv_flat):
        # [C, S, conv_dim] -> (q [C,S,Hk,Dk], k [C,S,Hk,Dk], v [C,S,Hv,Dv])
        C = qkv_flat.shape[0]
        conv_state = mx.zeros((C, K - 1, conv_dim), dtype=mx.float32)
        conv_in = mx.concatenate([conv_state, qkv_flat], axis=1)
        conv_out = nn.silu(mx.conv1d(conv_in, W_conv, groups=conv_dim))
        q, k, v = mx.split(conv_out, [key_dim, 2 * key_dim], axis=-1)
        q = q.reshape(C, S, Hk, Dk)
        k = k.reshape(C, S, Hk, Dk)
        v = v.reshape(C, S, Hv, Dv)
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q, k, v

    q1, k1, v1 = pre(qkv_raw[None])            # shared forward, [1, ...]
    g1 = compute_g(gdn.A_log, a[None], gdn.dt_bias).astype(mx.float32)
    beta1 = mx.sigmoid(b)[None]
    y1, *_ = gdn_forward(q1, k1, v1, g1, beta1, state=None)  # [1, S, Hv, Dv]
    mx.eval(q1, k1, v1, g1, beta1, y1)

    w_gn = gdn.norm.weight.astype(mx.float32)
    eps_gn = gdn.norm.eps

    def post(y, z_):
        # gated RMSNorm: rms_norm(y) * silu(z), [C, S, Hv, Dv]
        return mx.fast.rms_norm(y, w_gn, eps_gn) * nn.silu(z_)

    # d(g)/d(a) and d(beta)/d(b), elementwise at the shared forward point.
    sig_a = mx.sigmoid(a + gdn.dt_bias.astype(mx.float32)[None])
    dg_da = -mx.exp(gdn.A_log.astype(mx.float32))[None] * g1[0] * sig_a  # [S, Hv]
    db_db = (beta1 * (1.0 - beta1))[0]  # [S, Hv]

    # The Metal kernel now produces dg/dbeta too (verified vs the ops BPTT
    # to ~3e-7 incl. saturated gates), so it serves both include_gbeta modes.
    if use_kernel == "auto":
        use_kernel = (
            mx.metal.is_available()
            and Dk % 32 == 0
            and Dv % 4 == 0
        )

    stacks = [W_qkv, W_z]
    if include_gbeta:
        stacks += [W_a, W_b]
    W_stack = mx.concatenate(stacks, axis=0)

    pos = valid_mask[None, :, None]
    M = np.zeros((D, D), dtype=np.float32)

    for c0 in range(0, D, chunk):
        c1 = min(c0 + chunk, D)
        C = c1 - c0
        dy_gn = (pos * W_o[c0:c1][:, None, :]).reshape(C, S, Hv, Dv)

        y_t = mx.broadcast_to(y1[0][None], (C, S, Hv, Dv))
        z_t = mx.broadcast_to(z[None], (C, S, Hv, Dv))
        _, (dy, dz) = mx.vjp(post, [y_t, z_t], [dy_gn])

        if use_kernel:
            from .custom_gdn_vjp import gdn_kernel_vjp
            tile = lambda t: mx.contiguous(
                mx.broadcast_to(t, (C,) + t.shape[1:])
            )
            out = gdn_kernel_vjp(
                tile(q1), tile(k1), tile(v1), tile(g1), tile(beta1),
                None, dy.astype(mx.float32), return_gbeta=include_gbeta,
            )
            if include_gbeta:
                dq, dk, dv, dg, db = out
            else:
                dq, dk, dv = out
                dg = db = None
        else:
            dq, dk, dv, dg, db = gdn_vjp_batched(
                q1, k1, v1, g1, beta1, None, dy.astype(mx.float32)
            )

        qkv_t = mx.broadcast_to(qkv_raw[None], (C, S, conv_dim))
        _, (dqkv,) = mx.vjp(pre, [qkv_t], [dq, dk, dv])

        pieces = [dqkv, dz.reshape(C, S, value_dim)]
        if include_gbeta:
            pieces += [dg * dg_da[None], db * db_db[None]]
        draw = mx.concatenate(pieces, axis=-1)  # [C, S, F]

        rows = _fold_norm_project(draw, W_stack, x, w_in, eps_in, valid_mask)
        mx.eval(rows)
        M[c0:c1] = np.array(rows)

    return mx.array(M)


def attn_branch_jacobian(
    layer,
    x: mx.array,
    *,
    skip_first: int = 4,
    chunk: int = 256,
    include_gbeta: bool = False,
    use_kernel: bool | str = "auto",
) -> mx.array:
    """Analytic Jacobian of the whole attention branch, input norm folded.

    M = (1/V) sum_{s in valid} sum_{t in valid} d(attn(norm_in(x))_t)/d(x_s)

    layer: a DecoderLayer (has .is_linear, .input_layernorm, and
        .linear_attn or .self_attn).
    x: [S, D] pre-norm residual entering the layer (NOT normed).
    include_gbeta: include the x -> in_proj_a/b -> decay/write gate paths
        (GDN only). False matches the custom-kernel-fit semantics.
    Returns [D, D], rows indexed by output dim (same convention as
    fit.per_layer_jacobian).
    """
    valid = _valid_mask(x.shape[0], skip_first)
    if layer.is_linear:
        return _gdn_branch(
            layer, x, valid,
            chunk=chunk, include_gbeta=include_gbeta, use_kernel=use_kernel,
        )
    return _fa_branch(layer, x, valid, chunk=chunk)


__all__ = ["attn_branch_jacobian"]
