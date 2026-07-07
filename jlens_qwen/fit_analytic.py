"""Hybrid analytic fit: analytic MLP + closed-form norm + VJP attention.

Uses the analytic MLP Jacobian (77x faster) and closed-form RMSNorm
Jacobian (12000x faster) from analytic_layer.py, with VJP for the
attention branch (still the bottleneck, ~14-19s per layer).

Per-layer M_l cost: ~15-20s (was ~60s) = ~3x speedup.
Full 64-layer chain per prompt: ~19min (was ~34min for 23 layers).
20 prompts, full 64-layer depth: ~6.3h (overnight, with full depth!).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from .model import MLXLensModel
from .analytic import final_norm_jacobian
from .analytic_layer import decoder_layer_jacobian

logger = logging.getLogger(__name__)

SKIP_FIRST_N_POSITIONS = 4


def fit_analytic_single_prompt(
    model: MLXLensModel,
    prompt: str,
    *,
    source_layers: Sequence[int],
    max_seq_len: int = 32,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> dict[int, np.ndarray]:
    """Compute J_l for all source layers via chain-multiply with analytic M_l."""
    n_layers = model.n_layers
    D = model.d_model
    source_layers = sorted(set(source_layers))
    min_src = min(source_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    final, acts = model.forward(input_ids, capture_layers=list(range(n_layers)))
    for l in range(n_layers):
        mx.eval(acts[l])

    # J_{n_layers} = final norm Jacobian (closed-form, ~5ms)
    logger.info("  computing final norm Jacobian (J_%d)...", n_layers)
    t0 = time.perf_counter()
    J_norm = np.array(final_norm_jacobian(model, acts[n_layers - 1], skip_first=skip_first).astype(mx.float32))
    logger.info("    %.1fs", time.perf_counter() - t0)

    J_current = J_norm
    results: dict[int, np.ndarray] = {}

    for l in range(n_layers - 1, min_src - 1, -1):
        logger.info("  computing M_%d (analytic hybrid)...", l)
        t0 = time.perf_counter()
        M_l = np.array(decoder_layer_jacobian(model, l, acts[l], skip_first=skip_first).astype(mx.float32))
        mx.eval(M_l)
        logger.info("    %.1fs, ||M||=%.3e", time.perf_counter() - t0, np.linalg.norm(M_l))

        J_current = J_current @ M_l
        logger.info("    J_%d ||.||=%.3e", l, np.linalg.norm(J_current))

        if l in source_layers:
            results[l] = J_current.copy()

    return results


def fit_analytic(
    model: MLXLensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int],
    max_seq_len: int = 32,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> dict[int, np.ndarray]:
    """Average J_l over prompts via analytic hybrid fit."""
    D = model.d_model
    source_layers = sorted(set(source_layers))

    J_sum: dict[int, np.ndarray]
    n_done: int
    next_idx: int
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        state = np.load(checkpoint_path, allow_pickle=True).item()
        J_sum = state["J_sum"]
        n_done = state["n_done"]
        next_idx = state["next_idx"]
        logger.info("resuming from checkpoint: %d/%d prompts done", next_idx, len(prompts))
    else:
        J_sum = {l: np.zeros((D, D), dtype=np.float32) for l in source_layers}
        n_done = 0
        next_idx = 0

    for i, prompt in enumerate(prompts):
        if i < next_idx:
            continue
        logger.info("=== prompt %d/%d ===", i + 1, len(prompts))
        t0 = time.perf_counter()
        per_prompt = fit_analytic_single_prompt(
            model, prompt, source_layers=source_layers,
            max_seq_len=max_seq_len, skip_first=skip_first,
        )
        logger.info("  prompt %d done in %.1fs", i + 1, time.perf_counter() - t0)
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


def _atomic_save(obj, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    np.save(tmp, obj, allow_pickle=True)
    os.replace(tmp + ".npy", path)


__all__ = ["fit_analytic", "fit_analytic_single_prompt"]