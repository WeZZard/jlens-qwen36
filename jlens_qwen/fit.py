"""Chain-multiply Jacobian fitting for MLX Qwen3.5 models.

Strategy (fixes 1+2+7+11):
- Fix 1 (chain-multiply): J_l = J_{l+1} @ M_l where M_l is the per-layer
  Jacobian (one DecoderLayer's VJP). Compute J from the last layer backward.
- Fix 2 (mx.compile): compile per-layer forward+VJP for ~3x speedup.
- Fix 7 (32-token prompts): short sequences for ~4x faster GDN scan.
- Fix 11 (25 evenly-spaced layers): fit a subset, interpolate the rest.

What M_l actually captures (per the Anthropic paper's definition):
  M_l[i, j] = E_{prompt, s, t>=s} [ d(h_{l+1}[t, i]) / d(h_l[s, j]) ]
i.e. the *future-summed* cross-position influence of position-s activity
on all current-and-future outputs, averaged over valid source positions
and prompts. The cotangent is hot at ALL valid output positions
(`per_layer_jacobian:84-93`), so M_l is NOT the position-diagonal block —
it includes the causal cross-position flow. This matches the paper's
J-lens definition: the pattern that makes a word more likely "at some
point in the future."

Per-prompt cost: forward (1.6s) + 25 layers x (D VJPs through one layer).
At ~0.8-2 min per M_l = ~25-50 min per prompt. For 10-20 prompts: a few
hours. Resume via checkpoint.

NOTE: this VJP-based fit is the slow path. See PERFORMANCE.md and the
identity-basis analytic assembly in `analytic.py` for the ~30-60x faster
route that makes full-depth 64-layer fits affordable locally.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from .model import MLXLensModel

logger = logging.getLogger(__name__)

SKIP_FIRST_N_POSITIONS = 4  # for 32-token prompts; paper uses 16 for 128-token


def valid_positions(seq_len: int, skip_first: int = SKIP_FIRST_N_POSITIONS) -> list[int]:
    return list(range(skip_first, seq_len - 1))


def _one_layer_forward(model: MLXLensModel, h: mx.array, layer_idx: int) -> mx.array:
    """Run one DecoderLayer (or final norm if layer_idx == n_layers)."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    if layer_idx == model.n_layers:
        return model._text_module.norm(h)
    layer = model.layers[layer_idx]
    fa_mask = create_attention_mask(h, cache=None)
    ssm_mask = create_ssm_mask(h, cache=None)
    mask = ssm_mask if layer.is_linear else fa_mask
    return layer(h, mask=mask, cache=None)


def per_layer_jacobian(
    model: MLXLensModel,
    h_in: mx.array,
    layer_idx: int,
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    progress_every: int = 256,
) -> np.ndarray:
    """Compute M = d(h_out)/d(h_in) for one layer, position-diagonal averaged.

    h_in: [1, S, D]. Returns M: [D, D] numpy float32.

    For each output dim d (0..D-1), set a one-hot cotangent at output dim d
    at every valid position, VJP, average the gradient over valid source
    positions -> row d of M.
    """
    D = h_in.shape[-1]
    S = h_in.shape[1]
    valid = valid_positions(S, skip_first)
    pos_idx = mx.array(valid)
    n_valid = len(valid)

    M = np.zeros((D, D), dtype=np.float32)

    compiled = mx.compile(lambda h: _one_layer_forward(model, h, layer_idx))
    # Warm up compile
    out = compiled(h_in)
    mx.eval(out)

    arange_S = mx.arange(S)
    pos_mask = mx.where(
        (arange_S >= skip_first) & (arange_S < S - 1),
        mx.ones((S,), dtype=mx.float32),
        mx.zeros((S,), dtype=mx.float32),
    )

    for d in range(D):
        dim_onehot = mx.zeros((D,), dtype=mx.float32)
        dim_onehot = dim_onehot.at[d].add(mx.array(1.0))
        cot = pos_mask[None, :, None] * dim_onehot[None, None, :]

        _, vjps = mx.vjp(compiled, [h_in], [cot])
        grad = vjps[0]  # [1, S, D]
        mx.eval(grad)
        row = grad[0, pos_idx, :].astype(mx.float32).mean(axis=0)  # [D]
        M[d] = np.array(row)

        if d % progress_every == 0 or d == D - 1:
            logger.info("    M_%s dim %d/%d", layer_idx, d + 1, D)

    return M


def fit_chain_single_prompt(
    model: MLXLensModel,
    prompt: str,
    *,
    source_layers: Sequence[int],
    max_seq_len: int = 32,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> dict[int, np.ndarray]:
    """Compute J_l for all source layers via chain multiplication, one prompt.

    J_l transports acts[l] (the residual AFTER layer l): J_{n_layers-1} is
    the final-norm Jacobian, and J_{l-1} = J_l @ M_l with M_l the Jacobian
    of layer l evaluated at its input acts[l-1].

    We compute M_l for ALL layers from min(source_layers)+1 to n_layers-1
    (the chain must be continuous), and save J_l only at source_layers.
    """
    n_layers = model.n_layers
    D = model.d_model
    source_layers = sorted(set(source_layers))
    min_src = min(source_layers)

    # Forward, capturing all layer activations.
    input_ids = model.encode(prompt, max_length=max_seq_len)
    capture = list(range(n_layers))
    final, acts = model.forward(input_ids, capture_layers=capture)
    for l in range(n_layers):
        mx.eval(acts[l])
    h_final = model._text_module.norm(acts[n_layers - 1])
    mx.eval(h_final)

    # J_{n_layers} = final norm Jacobian.
    logger.info("  computing final norm Jacobian (J_%d)...", n_layers)
    t0 = time.perf_counter()
    J_norm = per_layer_jacobian(
        model, acts[n_layers - 1], n_layers, skip_first=skip_first
    )
    logger.info("    %.1fs", time.perf_counter() - t0)

    # Chain backward. Convention: J_l transports acts[l] (residual AFTER
    # layer l — what lens.transport applies it to), so each step multiplies
    # the Jacobian of layer l evaluated at ITS INPUT acts[l-1]:
    #   J_{l-1} = J_l @ M_l,  M_l = d(layer_l(h))/dh at h = acts[l-1]
    # (The previous indexing evaluated layer l at acts[l] and saved under l
    # — off by one on both counts: wrong linearization point AND an extra
    # layer-l factor in every J_l.)
    J_current = J_norm  # d(final)/d(acts[n_layers-1])
    results: dict[int, np.ndarray] = {}
    if n_layers - 1 in source_layers:
        results[n_layers - 1] = J_current.copy()

    for l in range(n_layers - 1, min_src, -1):
        logger.info("  computing M_%d (layer %d at its input acts[%d])...", l, l, l - 1)
        t0 = time.perf_counter()
        M_l = per_layer_jacobian(model, acts[l - 1], l, skip_first=skip_first)
        logger.info("    %.1fs, ||M||=%.3e", time.perf_counter() - t0, np.linalg.norm(M_l))

        J_current = J_current @ M_l  # [D, D]
        logger.info("    J_%d ||.||=%.3e", l - 1, np.linalg.norm(J_current))

        if l - 1 in source_layers:
            results[l - 1] = J_current.copy()

    return results


def fit(
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
    """Average J_l over prompts via chain-multiply. Returns {l: J_l [D, D]}."""
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
        per_prompt = fit_chain_single_prompt(
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