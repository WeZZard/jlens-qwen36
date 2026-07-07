"""Synthetic verification of the analytic attention-branch Jacobians.

Builds tiny Qwen3.5 DecoderLayers (the real mlx_lm classes, random weights,
small dims) and checks `attn_branch_jacobian` against a brute-force
per-output-dim VJP reference using fit.py's cotangent convention:

    M[d, :] = (1/V) sum_{s in valid} sum_{t in valid} d(branch(x)_{t,d})/d(x_s)

No real model is loaded — synthetic tensors only (safe to run while a fit
is using the GPU).

Run: uv run python -m pytest tests/test_analytic_attention.py -v
  or: uv run python tests/test_analytic_attention.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from jlens_qwen.analytic_attn import attn_branch_jacobian, _valid_mask
from jlens_qwen.patch_gdn import patch_gdn

SKIP_FIRST = 2
SEQ = 8


def _args() -> TextModelArgs:
    return TextModelArgs(
        model_type="qwen3_5",
        hidden_size=48,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
    )


def _random_x(D: int, seed: int = 0) -> mx.array:
    np.random.seed(seed)
    return mx.array(np.random.randn(SEQ, D).astype(np.float32))


def _brute_force_branch(layer, x: mx.array, *, stop_gbeta: bool = False) -> np.ndarray:
    """Reference: per-dim VJP through norm_in + attention, fit.py convention.

    stop_gbeta replicates the custom Metal kernel's semantics (dg/dbeta
    zeroed) by stop-gradient-ing g and beta in an ops-based forward.
    """
    S, D = x.shape
    valid = _valid_mask(S, SKIP_FIRST)
    pos_idx = mx.array([i for i in range(S) if float(valid[i]) > 0])

    if layer.is_linear and stop_gbeta:
        gdn = layer.linear_attn

        def branch(x_):
            from mlx_lm.models.gated_delta import compute_g, gated_delta_ops
            B, S_, _ = x_.shape
            xn = layer.input_layernorm(x_)
            qkv = gdn.in_proj_qkv(xn)
            z = gdn.in_proj_z(xn).reshape(B, S_, gdn.num_v_heads, gdn.head_v_dim)
            b = gdn.in_proj_b(xn)
            a = gdn.in_proj_a(xn)
            conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim))
            conv_out = nn.silu(gdn.conv1d(mx.concatenate([conv_state, qkv], axis=1)))
            q, k, v = [
                t.reshape(B, S_, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                    [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                    [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
                )
            ]
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            g = mx.stop_gradient(compute_g(gdn.A_log, a, gdn.dt_bias))
            beta = mx.stop_gradient(mx.sigmoid(b))
            state = mx.zeros(
                (B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim),
                dtype=mx.float32,
            )
            y, _ = gated_delta_ops(q, k, v, g, beta, state, None)
            out = gdn.norm(y, z)
            return gdn.out_proj(out.reshape(B, S_, -1))
    else:
        mask = None if layer.is_linear else "causal"
        attn_mod = layer.linear_attn if layer.is_linear else layer.self_attn

        def branch(x_):
            return attn_mod(layer.input_layernorm(x_), mask=mask, cache=None)

    x4 = x[None]
    M = np.zeros((D, D), dtype=np.float32)
    for d in range(D):
        onehot = mx.zeros((D,)).at[d].add(1.0)
        cot = valid[None, :, None] * onehot[None, None, :]
        _, vjps = mx.vjp(branch, [x4], [cot])
        M[d] = np.array(vjps[0][0, pos_idx, :].mean(axis=0))
    return M


def _check(name: str, got: mx.array, ref: np.ndarray, tol: float = 1e-3):
    got = np.array(got)
    err = np.abs(got - ref).max()
    rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12)
    print(f"{name}: max abs err {err:.2e}, rel frob {rel:.2e}")
    assert err < tol, f"{name} mismatch: {err:.2e}"


def test_fa_branch():
    mx.random.seed(0)
    layer = DecoderLayer(_args(), layer_idx=3)  # (3+1) % 4 == 0 -> FA
    assert not layer.is_linear
    x = _random_x(48, seed=1)
    M = attn_branch_jacobian(layer, x, skip_first=SKIP_FIRST, chunk=16)
    M_ref = _brute_force_branch(layer, x)
    _check("FA branch", M, M_ref)


def test_gdn_branch_with_gbeta():
    patch_gdn()  # ops fallback so the reference VJP can differentiate
    mx.random.seed(1)
    layer = DecoderLayer(_args(), layer_idx=0)
    assert layer.is_linear
    x = _random_x(48, seed=2)
    M = attn_branch_jacobian(
        layer, x, skip_first=SKIP_FIRST, chunk=16,
        include_gbeta=True, use_kernel=False,
    )
    M_ref = _brute_force_branch(layer, x)
    _check("GDN branch (with g/beta)", M, M_ref)


def test_gdn_branch_without_gbeta():
    patch_gdn()
    mx.random.seed(2)
    layer = DecoderLayer(_args(), layer_idx=1)
    assert layer.is_linear
    x = _random_x(48, seed=3)
    M = attn_branch_jacobian(
        layer, x, skip_first=SKIP_FIRST, chunk=16,
        include_gbeta=False, use_kernel=False,
    )
    M_ref = _brute_force_branch(layer, x, stop_gbeta=True)
    _check("GDN branch (no g/beta)", M, M_ref)


def test_gdn_kernel_path_matches_ops():
    """The Metal-kernel fast path (cotangents folded into B) must match the
    ops BPTT — including the dg/dbeta gate gradients. Needs Dk % 32 == 0
    and Dv % 4 == 0. Runs at small dims AND real Qwen3.6 head dims, with
    saturated gates (g -> 0) to stress the s_pre-storage path."""
    if not mx.metal.is_available():
        print("Metal unavailable; skipping kernel-path test")
        return
    from jlens_qwen.custom_gdn_vjp import gdn_kernel_vjp
    from jlens_qwen.gdn_backward import gdn_vjp_batched

    configs = [
        ("small", 6, 1, 32, 2, 8, 5),
        ("real-dims", 8, 16, 128, 48, 128, 3),
    ]
    for name_cfg, T, Hk, Dk, Hv, Dv, C in configs:
        np.random.seed(4)
        q = mx.array(np.random.randn(1, T, Hk, Dk).astype(np.float32))
        k = mx.array(np.random.randn(1, T, Hk, Dk).astype(np.float32))
        v = mx.array(np.random.randn(1, T, Hv, Dv).astype(np.float32))
        # include saturated gates (g ~ 1e-14) alongside moderate ones
        g = mx.array((np.random.rand(1, T, Hv) ** 8).astype(np.float32))
        g = mx.where(mx.arange(Hv)[None, None, :] % 3 == 0, g * 1e-12, g)
        beta = mx.array(np.random.rand(1, T, Hv).astype(np.float32))
        dy = mx.array(np.random.randn(C, T, Hv, Dv).astype(np.float32))

        dq_o, dk_o, dv_o, dg_o, db_o = gdn_vjp_batched(q, k, v, g, beta, None, dy)
        tile = lambda t: mx.contiguous(mx.broadcast_to(t, (C,) + t.shape[1:]))
        dq_k, dk_k, dv_k, dg_k, db_k = gdn_kernel_vjp(
            tile(q), tile(k), tile(v), tile(g), tile(beta), None, dy,
            return_gbeta=True,
        )
        for name, o, kk in [
            ("dq", dq_o, dq_k), ("dk", dk_o, dk_k), ("dv", dv_o, dv_k),
            ("dg", dg_o, dg_k), ("dbeta", db_o, db_k),
        ]:
            on, kn = np.array(o), np.array(kk)
            scale = max(np.abs(on).max(), 1.0)
            err = np.abs(on - kn).max() / scale
            print(f"kernel-vs-ops [{name_cfg}] {name}: {err:.2e}")
            assert err < 1e-4, f"kernel {name} mismatch ({name_cfg}): {err:.2e}"
            assert np.isfinite(kn).all(), f"kernel {name} has non-finite values"


def test_gdn_branch_kernel_path_with_gbeta():
    """End-to-end: GDN branch via the Metal-kernel BPTT with g/beta included
    must match the brute-force VJP. Uses kernel-compatible head dims."""
    if not mx.metal.is_available():
        print("Metal unavailable; skipping kernel branch test")
        return
    patch_gdn()
    args = _args()
    args.linear_key_head_dim = 32
    args.linear_value_head_dim = 8
    mx.random.seed(5)
    layer = DecoderLayer(args, layer_idx=0)
    assert layer.is_linear
    x = _random_x(48, seed=6)
    M = attn_branch_jacobian(
        layer, x, skip_first=SKIP_FIRST, chunk=16,
        include_gbeta=True, use_kernel=True,
    )
    M_ref = _brute_force_branch(layer, x)
    _check("GDN branch (kernel path, with g/beta)", M, M_ref)


def test_mlp_branch():
    from jlens_qwen.analytic_layer import mlp_branch_jacobian

    mx.random.seed(3)
    layer = DecoderLayer(_args(), layer_idx=0)
    x = _random_x(48, seed=5)
    valid = _valid_mask(SEQ, SKIP_FIRST)
    pos_idx = mx.array([i for i in range(SEQ) if float(valid[i]) > 0])
    w_post = layer.post_attention_layernorm.weight
    eps = layer.post_attention_layernorm.eps

    M = mlp_branch_jacobian(layer.mlp, x, w_post, eps, valid)

    def branch(x_):
        return layer.mlp(mx.fast.rms_norm(x_, w_post, eps))

    D = x.shape[1]
    M_ref = np.zeros((D, D), dtype=np.float32)
    for d in range(D):
        onehot = mx.zeros((D,)).at[d].add(1.0)
        cot = valid[None, :, None] * onehot[None, None, :]
        _, vjps = mx.vjp(branch, [x[None]], [cot])
        M_ref[d] = np.array(vjps[0][0, pos_idx, :].mean(axis=0))
    _check("MLP branch (norm folded)", M, M_ref)


def test_full_layer_junction_error():
    """The full-layer assembly has ONE remaining approximation (the
    averaged-product junction between the two branches). Quantify it for
    both the new exact-branch path and the old 3-junction hybrid; the new
    path must be at least as close to the exact layer Jacobian."""
    from types import SimpleNamespace
    from jlens_qwen.analytic_layer import decoder_layer_jacobian

    patch_gdn()
    for layer_idx, name in ((0, "GDN"), (3, "FA")):
        mx.random.seed(10 + layer_idx)
        layer = DecoderLayer(_args(), layer_idx=layer_idx)
        x = _random_x(48, seed=20 + layer_idx)
        model = SimpleNamespace(layers=[None] * layer_idx + [layer], d_model=48)

        M_new = decoder_layer_jacobian(
            model, layer_idx, x[None], skip_first=SKIP_FIRST,
            analytic_attn=True, include_gbeta=True, chunk=16,
        )
        M_old = decoder_layer_jacobian(
            model, layer_idx, x[None], skip_first=SKIP_FIRST,
            analytic_attn=False,
        )

        valid = _valid_mask(SEQ, SKIP_FIRST)
        pos_idx = mx.array([i for i in range(SEQ) if float(valid[i]) > 0])
        mask = None if layer.is_linear else "causal"

        def full(x_):
            return layer(x_, mask=mask, cache=None)

        D = x.shape[1]
        M_ref = np.zeros((D, D), dtype=np.float32)
        for d in range(D):
            onehot = mx.zeros((D,)).at[d].add(1.0)
            cot = valid[None, :, None] * onehot[None, None, :]
            _, vjps = mx.vjp(full, [x[None]], [cot])
            M_ref[d] = np.array(vjps[0][0, pos_idx, :].mean(axis=0))

        ref_norm = np.linalg.norm(M_ref)
        rel_new = np.linalg.norm(np.array(M_new) - M_ref) / ref_norm
        rel_old = np.linalg.norm(np.array(M_old) - M_ref) / ref_norm
        print(f"{name} layer: rel frob error new={rel_new:.3e} old={rel_old:.3e}")
        assert rel_new <= rel_old * 1.05, (
            f"{name}: exact-branch path ({rel_new:.3e}) should not be worse "
            f"than the old hybrid ({rel_old:.3e})"
        )


if __name__ == "__main__":
    test_fa_branch()
    test_gdn_branch_with_gbeta()
    test_gdn_branch_without_gbeta()
    test_gdn_kernel_path_matches_ops()
    test_gdn_branch_kernel_path_with_gbeta()
    test_mlp_branch()
    test_full_layer_junction_error()
    print("ALL PASSED")
