# VERDICT: MODERATE (measured 2026-07-08 on Qwen3.6-27B-4bit)
# Relative Frobenius gap: 4.91% / 7.53% / 6.62%  (GDN layers 0 / 32 / 62)
# Row/col concentration (52.8x rows at L0) confirms the rank<=96 structure.
# Cross-check vs module autograd (g/beta live): ~2e-3 over 8 probes/layer.
# Recommendation: include g/beta for the fit (include_gbeta=True; forces the
# ops BPTT until the Metal kernel grows dg/dbeta outputs — GDN branch ~27s
# vs ~3.6s per layer, full 20-prompt fit ~7h vs ~1h).
"""Measure the g/beta path gap in the GDN per-layer Jacobian (§4.1).

The custom Metal kernel VJP zeros dg/dbeta (custom_gdn_patch.py), silently
dropping the x -> in_proj_a/b -> decay/write-gate paths from every
kernel-fit M_l. This script quantifies that gap EXACTLY:

    Delta_M = attn_branch_jacobian(include_gbeta=True)
            - attn_branch_jacobian(include_gbeta=False)

Both terms use the same ops-BPTT code path (analytic_attn.py, verified to
~1e-8 against mx.vjp on synthetic layers), so the difference isolates the
g/beta contribution with no kernel-vs-ops numerical noise.

NOTE the original handoff proposed comparing gdn_backward.gdn_vjp against
custom_gdn_vjp.gdn_kernel_vjp — both of those return only (dq, dk, dv)
with g/beta held constant, so their difference measures kernel numerics,
NOT the g/beta paths (it would report ~0 regardless of the truth). This
script replaces that design.

An independent cross-check backpropagates a few random probes through the
whole GDN module with autograd (ops recurrence, g/beta in the graph) and
compares against the analytic rows, validating the implementation on real
weights.

Measured at an early, mid, and late GDN layer (gate saturation varies with
depth). Needs the model for ~5-10 min — run only when no fit is using the
GPU:

    uv run python scripts/measure_gbeta_gap.py
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, ".")

PROMPT = "The capital of the country famous for the Colosseum uses a currency called the"


def _gdn_layer_indices(model) -> list[int]:
    idx = [i for i, l in enumerate(model.layers) if l.is_linear]
    return [idx[0], idx[len(idx) // 2], idx[-1]]


def _ops_module_row(layer, x4: mx.array, cot: mx.array) -> mx.array:
    """One probe row through norm_in + GDN via autograd with g/beta live.

    Replicates the module forward with the differentiable ops recurrence
    (the stock module may have the custom-kernel forward patched in, which
    zeros dg/dbeta — that is exactly what we must NOT use here).
    """
    from mlx_lm.models.gated_delta import compute_g, gated_delta_ops

    gdn = layer.linear_attn

    def branch(x_):
        B, S, _ = x_.shape
        xn = layer.input_layernorm(x_)
        qkv = gdn.in_proj_qkv(xn)
        z = gdn.in_proj_z(xn).reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
        b = gdn.in_proj_b(xn)
        a = gdn.in_proj_a(xn)
        conv_state = mx.zeros(
            (B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=x_.dtype
        )
        conv_out = nn.silu(gdn.conv1d(mx.concatenate([conv_state, qkv], axis=1)))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        beta = mx.sigmoid(b)
        state = mx.zeros(
            (B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim),
            dtype=mx.float32,
        )
        y, _ = gated_delta_ops(q, k, v, g, beta, state, None)
        out = gdn.norm(y, z)
        return gdn.out_proj(out.reshape(B, S, -1))

    _, vjps = mx.vjp(branch, [x4], [cot])
    return vjps[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument("--probes", type=int, default=8,
                        help="random probes for the autograd cross-check")
    parser.add_argument("--model-id", default="mlx-community/Qwen3.6-27B-4bit")
    args = parser.parse_args()

    print("NOTE: needs ~20 GB and the GPU — do not run while a fit is active.")

    from jlens_qwen.model import load
    from jlens_qwen.analytic_attn import attn_branch_jacobian, _valid_mask

    model = load(args.model_id)
    layers = _gdn_layer_indices(model)
    print(f"GDN layers measured (early/mid/late): {layers}")

    input_ids = model.encode(PROMPT, max_length=args.max_seq_len)
    capture = sorted({l - 1 for l in layers})
    _, acts = model.forward(input_ids, capture_layers=capture)
    for c in capture:
        mx.eval(acts[c])

    S = int(input_ids.shape[1])
    valid = _valid_mask(S, args.skip_first)
    pos_idx = mx.array([i for i in range(S) if float(valid[i]) > 0])

    worst_rel = 0.0
    for l in layers:
        layer = model.layers[l]
        x = acts[l - 1][0].astype(mx.float32)  # residual entering layer l

        t0 = time.perf_counter()
        M_with = attn_branch_jacobian(
            layer, x, skip_first=args.skip_first,
            include_gbeta=True, use_kernel=False,
        )
        M_without = attn_branch_jacobian(
            layer, x, skip_first=args.skip_first,
            include_gbeta=False, use_kernel=False,
        )
        mx.eval(M_with, M_without)
        dt = time.perf_counter() - t0

        Mw = np.array(M_with)
        Mo = np.array(M_without)
        delta = Mw - Mo
        rel = np.linalg.norm(delta) / np.linalg.norm(Mw)
        worst_rel = max(worst_rel, rel)
        row_norms = np.linalg.norm(delta, axis=1)
        col_norms = np.linalg.norm(delta, axis=0)
        print(
            f"layer {l:2d}: ||dM||_F/||M||_F = {rel:.4%}  "
            f"max|dM| = {np.abs(delta).max():.3e}  "
            f"row-conc = {row_norms.max() / (row_norms.mean() + 1e-12):.1f}x  "
            f"col-conc = {col_norms.max() / (col_norms.mean() + 1e-12):.1f}x  "
            f"({dt:.0f}s)"
        )

        # Independent cross-check: autograd probes with g/beta in the graph.
        if args.probes > 0:
            np.random.seed(0)
            errs = []
            x4 = x[None]
            for _ in range(args.probes):
                u = mx.array(
                    np.random.choice([-1.0, 1.0], size=(x.shape[1],)).astype(np.float32)
                )
                cot = valid[None, :, None] * u[None, None, :]
                grad = _ops_module_row(layer, x4, cot)
                row_ref = np.array(grad[0, pos_idx, :].mean(axis=0))
                row_ana = np.array(mx.matmul(u[None], M_with)[0])
                errs.append(
                    np.linalg.norm(row_ana - row_ref)
                    / (np.linalg.norm(row_ref) + 1e-12)
                )
            print(f"          cross-check vs module autograd (g/beta live): "
                  f"mean rel err {np.mean(errs):.3e} over {args.probes} probes")

    print()
    if worst_rel < 0.01:
        verdict = "NEGLIGIBLE — kernel is fine as-is; document the approximation."
    elif worst_rel < 0.10:
        verdict = ("MODERATE — include g/beta for interventions (analytic path "
                   "has it already; Metal kernel needs dg/dbeta outputs for speed).")
    else:
        verdict = "SIGNIFICANT — g/beta must be included for any causal use."
    print(f"VERDICT: worst relative Frobenius gap {worst_rel:.2%} -> {verdict}")
    print("Fill the header comment of this script with the verdict.")


if __name__ == "__main__":
    main()
