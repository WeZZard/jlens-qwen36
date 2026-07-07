"""Unbiased Rademacher probing for the per-layer Jacobian.

An interim ~10x speedup over the VJP-based fit (PERFORMANCE_REVIEW.md §3).
Swap the 5120 one-hot cotangents for k ≈ 512 Rademacher (±1) cotangents:

    M_hat = (1/k) sum_i v_i (v_i^T M),  E[v v^T] = I  =>  E[M_hat] = M

Unbiased with NO rank assumption — strictly dominates low-rank power
iteration (Fix 2), which assumed rank-64. Probe noise averages down
across prompts; chain products of independent unbiased factors remain
unbiased in expectation.

Verification: compare against one exact prompt from the existing
checkpoint. Use the analytic (exact) lens for intervention experiments
(causal claims about J); probing is for readout-quality lenses only.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .model import MLXLensModel
from .fit import valid_positions, _one_layer_forward, SKIP_FIRST_N_POSITIONS


def per_layer_jacobian_probed(
    model: MLXLensModel,
    h_in: mx.array,
    layer_idx: int,
    *,
    n_probes: int = 512,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    seed: int = 0,
) -> np.ndarray:
    """Estimate M = d(h_out)/d(h_in) via k Rademacher probes.

    For each probe i, draw v_i ~ Uniform{-1, +1}^D, compute the VJP
    with cotangent = v_i (broadcast over valid positions), and
    accumulate (1/k) v_i (v_i^T M). The result is an unbiased estimator
    of the position-averaged Jacobian.

    Cost: k VJPs (vs D=5120 for the exact fit). With k=512, ~10x faster.
    Noise: O(1/sqrt(k * n_valid)) per entry; averages down across prompts.

    Args:
        model: the MLX model.
        h_in: [1, S, D] — input to the layer.
        layer_idx: which layer (or model.n_layers for the final norm).
        n_probes: number of Rademacher probes (default 512).
        skip_first: leading positions to skip.
        seed: RNG seed for reproducibility.

    Returns:
        M_hat: [D, D] numpy float32 — unbiased estimate of the
        position-averaged Jacobian.
    """
    D = h_in.shape[-1]
    S = h_in.shape[1]
    valid = valid_positions(S, skip_first)
    pos_idx = mx.array(valid)
    n_valid = len(valid)

    # Position mask for the cotangent.
    arange_S = mx.arange(S)
    pos_mask = mx.where(
        (arange_S >= skip_first) & (arange_S < S - 1),
        mx.ones((S,), dtype=mx.float32),
        mx.zeros((S,), dtype=mx.float32),
    )

    compiled = mx.compile(lambda h: _one_layer_forward(model, h, layer_idx))
    out = compiled(h_in)
    mx.eval(out)  # warm compile

    # For the final norm, use the closed-form (exact, ~5ms).
    if layer_idx == model.n_layers:
        from .analytic import final_norm_jacobian
        return np.array(final_norm_jacobian(model, h_in, skip_first=skip_first).astype(mx.float32))

    mx.random.seed(seed)
    M_hat = np.zeros((D, D), dtype=np.float32)

    for i in range(n_probes):
        # Draw Rademacher probe: v ~ Uniform{-1, +1}^D
        v = mx.random.bernoulli(0.5, shape=(D,)).astype(mx.float32) * 2.0 - 1.0  # [D]
        # Cotangent: v broadcast to [1, S, D], masked to valid positions.
        cot = pos_mask[None, :, None] * v[None, None, :]  # [1, S, D]

        # VJP: grad = d(out)/d(h_in)^T @ cot, shape [1, S, D]
        _, vjps = mx.vjp(compiled, [h_in], [cot])
        grad = vjps[0]  # [1, S, D]
        mx.eval(grad)

        # v^T M = average over valid positions of grad[0, valid, :]
        # (the VJP already includes the sum over output positions via the
        # position mask; we average over source positions)
        vTM = grad[0, pos_idx, :].astype(mx.float32).mean(axis=0)  # [D]
        # Accumulate (1/k) * v * v^T M = (1/k) * outer(v, vTM)
        v_np = np.array(v)
        vTM_np = np.array(vTM)
        M_hat += np.outer(v_np, vTM_np) / n_probes

    return M_hat


def fit_probed(
    model: MLXLensModel,
    prompts: list[str],
    *,
    source_layers: list[int],
    max_seq_len: int = 32,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    n_probes: int = 512,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> dict[int, np.ndarray]:
    """Average J_l over prompts via Rademacher probing. Returns {l: J_l}.

    Same chain-multiply structure as fit.fit(), but each per-layer M_l is
    estimated with n_probes Rademacher VJPs instead of D=5120 exact VJPs.
    ~10x faster; unbiased; noise averages down across prompts.
    """
    from .analytic import final_norm_jacobian
    from .fit import _atomic_save
    import os

    D = model.d_model
    source_layers = sorted(set(source_layers))
    min_src = min(source_layers)

    # Resume state
    J_sum: dict[int, np.ndarray]
    n_done: int
    next_idx: int
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        state = np.load(checkpoint_path, allow_pickle=True).item()
        J_sum = state["J_sum"]
        n_done = state["n_done"]
        next_idx = state["next_idx"]
        print(f"resuming from checkpoint: {next_idx}/{len(prompts)} prompts done", flush=True)
    else:
        J_sum = {l: np.zeros((D, D), dtype=np.float32) for l in source_layers}
        n_done = 0
        next_idx = 0

    for i, prompt in enumerate(prompts):
        if i < next_idx:
            continue
        print(f"=== prompt {i+1}/{len(prompts)} ===", flush=True)
        import time
        t0 = time.perf_counter()
        per_prompt = _fit_chain_probed_single(
            model, prompt, source_layers=source_layers,
            max_seq_len=max_seq_len, skip_first=skip_first,
            n_probes=n_probes,
        )
        print(f"  prompt {i+1} done in {time.perf_counter()-t0:.1f}s", flush=True)
        for l, J in per_prompt.items():
            J_sum[l] += J
        n_done += 1
        next_idx = i + 1
        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            _atomic_save(
                {"J_sum": J_sum, "n_done": n_done, "next_idx": next_idx},
                checkpoint_path,
            )

    if checkpoint_path:
        _atomic_save(
            {"J_sum": J_sum, "n_done": n_done, "next_idx": next_idx},
            checkpoint_path,
        )
    J_mean = {l: J_sum[l] / n_done for l in source_layers}
    return J_mean


def _fit_chain_probed_single(
    model: MLXLensModel,
    prompt: str,
    *,
    source_layers: list[int],
    max_seq_len: int = 32,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    n_probes: int = 512,
) -> dict[int, np.ndarray]:
    """Chain-multiply fit for one prompt using probed M_l estimates."""
    n_layers = model.n_layers
    D = model.d_model
    source_layers = sorted(set(source_layers))
    min_src = min(source_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    final, acts = model.forward(input_ids, capture_layers=list(range(n_layers)))
    for l in range(n_layers):
        mx.eval(acts[l])

    # J_{n_layers} = final norm Jacobian (closed-form, exact)
    print(f"  computing final norm Jacobian (J_{n_layers})...", flush=True)
    J_norm = np.array(final_norm_jacobian(model, acts[n_layers - 1], skip_first=skip_first).astype(mx.float32))

    J_current = J_norm
    results: dict[int, np.ndarray] = {}

    for l in range(n_layers - 1, min_src - 1, -1):
        print(f"  computing M_{l} (probed, {n_probes} probes)...", flush=True)
        import time
        t0 = time.perf_counter()
        M_l = per_layer_jacobian_probed(
            model, acts[l], l, n_probes=n_probes, skip_first=skip_first
        )
        print(f"    {time.perf_counter()-t0:.1f}s, ||M||={np.linalg.norm(M_l):.3e}", flush=True)
        J_current = J_current @ M_l
        print(f"    J_{l} ||.||={np.linalg.norm(J_current):.3e}", flush=True)
        if l in source_layers:
            results[l] = J_current.copy()

    return results


__all__ = ["per_layer_jacobian_probed", "fit_probed"]