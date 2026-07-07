"""Full-depth analytic fit: exact-branch Jacobians, chain-multiplied.

Uses decoder_layer_jacobian(analytic_attn=True) — analytic attention +
MLP branches with input norms folded per-position (analytic_attn.py) —
and the closed-form final-norm Jacobian.

With include_gbeta=True (measured g/beta gap 4.9-7.5%, see
scripts/measure_gbeta_gap.py) the GDN branch runs the ops BPTT:
~27s per GDN layer, ~1.4s per FA layer => ~22 min/prompt full depth,
~7h for 20 prompts. With include_gbeta=False (Metal kernel path) it
drops to ~3.5s/layer => ~1h for 20 prompts, at the cost of dropping
the decay-gate paths (readout-grade, not intervention-grade).

Chain convention: J_l transports acts[l] (see fit_analytic_single_prompt).
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
    """Compute J_l for all source layers via chain-multiply with analytic M_l.

    Convention: J_l transports acts[l], the residual AFTER layer l (what
    lens.transport applies it to and what interventions patch). So
    J_{n_layers-1} = J_norm, and each chain step multiplies the Jacobian of
    layer l EVALUATED AT ITS INPUT acts[l-1]:

        J_{l-1} = J_l @ M_l,   M_l = d(layer_l(h))/dh at h = acts[l-1]

    The previous version evaluated layer l at acts[l] (its own OUTPUT) and
    saved the product under index l — an off-by-one that both linearized at
    the wrong point and left an extra layer-l factor in every J_l (33-49%
    rel error on a toy chain vs 3-5% for this indexing; the residual is the
    known position-averaging approximation).
    """
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
    if n_layers - 1 in source_layers:
        results[n_layers - 1] = J_current.copy()

    for l in range(n_layers - 1, min_src, -1):
        logger.info("  computing M_%d (layer %d at its input acts[%d])...", l, l, l - 1)
        t0 = time.perf_counter()
        # include_gbeta=True per the measured 4.9-7.5% g/beta gap
        # (scripts/measure_gbeta_gap.py, 2026-07-08): required for causal
        # interventions; forces the ops BPTT for GDN layers (~27s vs ~3.6s).
        M_l = np.array(decoder_layer_jacobian(
            model, l, acts[l - 1], skip_first=skip_first, include_gbeta=True,
        ).astype(mx.float32))
        mx.eval(M_l)
        logger.info("    %.1fs, ||M||=%.3e", time.perf_counter() - t0, np.linalg.norm(M_l))

        J_current = J_current @ M_l
        logger.info("    J_%d ||.||=%.3e", l - 1, np.linalg.norm(J_current))

        if l - 1 in source_layers:
            results[l - 1] = J_current.copy()

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