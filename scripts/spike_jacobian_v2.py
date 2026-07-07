"""Spike v2: batched VJP. Replicate the prompt dim_batch times along the
batch axis, set a different one-hot cotangent per batch element, and run
one VJP that produces dim_batch rows of J_ℓ at once.

This mirrors the reference jlens.fitting estimator: "one-hot cotangent at
dim (dim_start + b) for batch element b, at every valid target position".
"""

import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".")
from jlens_qwen.model import load


def run_from_layer(model, src_layer: int, h: mx.array) -> mx.array:
    """Run layers src_layer+1 .. end + final norm, starting from h.

    h shape [B, S, D] -> output [B, S, D].
    """
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_mask = create_attention_mask(h, cache=None)
    ssm_mask = create_ssm_mask(h, cache=None)
    hidden = h
    for i in range(src_layer + 1, model.n_layers):
        layer = model.layers[i]
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=None)
    return model._text_module.norm(hidden)


def compute_J_batched(model, h_src, src_layer, dim_batch=8):
    """Compute J_ℓ via batched VJP.

    h_src: [1, S, D] — the source-layer residual (batch=1).
    Returns J: [D, D] numpy float32.
    """
    D = h_src.shape[-1]
    S = h_src.shape[1]
    skip_first = 16
    # Valid source positions and valid target positions are the same set.
    valid_positions = list(range(skip_first, S - 1))
    n_valid = len(valid_positions)

    J = np.zeros((D, D), dtype=np.float32)
    n_passes = (D + dim_batch - 1) // dim_batch

    print(f"  computing J_{src_layer}: {n_passes} batched VJP passes, "
          f"dim_batch={dim_batch}, n_valid={n_valid}", flush=True)

    # Replicate h_src along batch: [dim_batch, S, D]
    h_batch = mx.broadcast_to(h_src, (dim_batch, S, D)).astype(h_src.dtype)

    # Position mask: 1 at valid positions, 0 elsewhere. Shape [S].
    arange_S = mx.arange(S)
    pos_mask = mx.where(
        (arange_S >= skip_first) & (arange_S < S - 1),
        mx.ones((S,), dtype=mx.float32),
        mx.zeros((S,), dtype=mx.float32),
    )

    for pass_idx, dim_start in enumerate(range(0, D, dim_batch)):
        n_dims = min(dim_batch, D - dim_start)

        # Cotangent: [dim_batch, S, D], one-hot at dim (dim_start + b) for
        # batch element b, at every valid position.
        # Build as: cot[b, s, dim_start+b] = pos_mask[s], else 0.
        # = pos_mask[s] * one_hot_d[dim_start+b]
        # Construct a [n_dims, D] block where row b is one-hot at dim_start+b.
        # Then cot[b, s, d] = pos_mask[s] * block[b, d].
        block = mx.zeros((n_dims, D), dtype=mx.float32)
        for b in range(n_dims):
            block = block.at[b, dim_start + b].add(mx.array(1.0))
        # cot: [n_dims, S, D] = pos_mask[None, :, None] * block[:, None, :]
        cot = pos_mask[None, :, None] * block[:, None, :]
        # If n_dims < dim_batch, pad to dim_batch for the batched forward.
        if n_dims < dim_batch:
            pad = mx.zeros((dim_batch - n_dims, S, D), dtype=mx.float32)
            cot = mx.concatenate([cot, pad], axis=0)

        # VJP. The function takes h of shape [dim_batch, S, D] and returns
        # [dim_batch, S, D]. Cotangent matches output.
        _, vjps = mx.vjp(lambda h: run_from_layer(model, src_layer, h), [h_batch], [cot])
        grad = vjps[0]  # [dim_batch, S, D]
        mx.eval(grad)

        # For each batch element b in [0, n_dims), row (dim_start+b) of J =
        # grad[b, valid_positions, :].mean(0).
        pos_idx = mx.array(valid_positions)
        for b in range(n_dims):
            row = grad[b, pos_idx, :].astype(mx.float32).mean(axis=0)  # [D]
            J[dim_start + b] = np.array(row)

        if pass_idx % 25 == 0 or pass_idx == n_passes - 1:
            print(f"    pass {pass_idx+1}/{n_passes} (dims {dim_start}-{dim_start+n_dims})", flush=True)

    return J


def main():
    print("=" * 60, flush=True)
    print("PHASE 1 SPIKE v2: batched VJP for J_ℓ", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading model...", flush=True)
    t0 = time.perf_counter()
    model = load()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    prompt = (
        "The history of computing is a fascinating journey through human ingenuity. "
        "From the abacus to modern quantum computers, each era has built upon the "
        "innovations of the previous one. The invention of the transistor at Bell Labs "
        "in 1947 marked a pivotal moment, leading to integrated circuits and the "
        "microprocessor. These developments enabled the personal computer revolution "
        "of the 1980s, bringing computing power to homes and offices worldwide. "
        "Today, artificial intelligence and machine learning represent the latest "
        "frontier, with neural networks achieving remarkable feats in natural language"
    )
    input_ids = model.encode(prompt, max_length=128)
    print(f"  input_ids shape: {input_ids.shape}", flush=True)

    print("\nForward pass (capture layers 32, 63)...", flush=True)
    t0 = time.perf_counter()
    final, acts = model.forward(input_ids, capture_layers=[32, 63])
    mx.eval(final)
    mx.eval(acts[32])
    mx.eval(acts[63])
    print(f"  forward in {time.perf_counter()-t0:.1f}s", flush=True)

    # Spike: compute first 64 dims of J_32 with dim_batch=8 (8 passes).
    n_dims_spike = 64
    dim_batch = 8

    print(f"\n--- J_32 batched (first {n_dims_spike} dims, dim_batch={dim_batch}) ---", flush=True)
    h_src = acts[32]
    t0 = time.perf_counter()
    # Compute only first n_dims_spike rows by limiting the loop.
    J_partial = compute_J_batched_partial(model, h_src, 32, dim_batch=dim_batch, n_dims_total=n_dims_spike)
    elapsed = time.perf_counter() - t0
    n_passes_done = (n_dims_spike + dim_batch - 1) // dim_batch
    per_pass = elapsed / n_passes_done
    per_dim = elapsed / n_dims_spike
    print(f"  {n_dims_spike} dims in {elapsed:.1f}s ({per_pass:.2f}s/pass, {per_dim:.3f}s/dim)", flush=True)
    print(f"  extrapolated full J_32: {per_dim * 5120:.0f}s = {per_dim * 5120 / 60:.1f}min", flush=True)
    print(f"  J_32 partial shape: {J_partial.shape}", flush=True)
    print(f"  J_32 finite: {np.isfinite(J_partial).all()}", flush=True)
    print(f"  J_32 Frobenius (partial): {np.linalg.norm(J_partial):.3e}", flush=True)

    # Compare per-dim J_32 row 0 against the unbatched spike (sanity: should be identical).
    print(f"\n  J_32[0, :5] = {J_partial[0, :5]}", flush=True)

    print("\nSPIKE v2 COMPLETE", flush=True)


def compute_J_batched_partial(model, h_src, src_layer, dim_batch=8, n_dims_total=64):
    """Compute only the first n_dims_total rows of J_ℓ (for the spike)."""
    D = h_src.shape[-1]
    S = h_src.shape[1]
    skip_first = 16
    valid_positions = list(range(skip_first, S - 1))

    J = np.zeros((n_dims_total, D), dtype=np.float32)
    n_passes = (n_dims_total + dim_batch - 1) // dim_batch

    h_batch = mx.broadcast_to(h_src, (dim_batch, S, D)).astype(h_src.dtype)

    arange_S = mx.arange(S)
    pos_mask = mx.where(
        (arange_S >= skip_first) & (arange_S < S - 1),
        mx.ones((S,), dtype=mx.float32),
        mx.zeros((S,), dtype=mx.float32),
    )
    pos_idx = mx.array(valid_positions)

    for pass_idx, dim_start in enumerate(range(0, n_dims_total, dim_batch)):
        n_dims = min(dim_batch, n_dims_total - dim_start)

        block = mx.zeros((n_dims, D), dtype=mx.float32)
        for b in range(n_dims):
            block = block.at[b, dim_start + b].add(mx.array(1.0))
        cot = pos_mask[None, :, None] * block[:, None, :]
        if n_dims < dim_batch:
            pad = mx.zeros((dim_batch - n_dims, S, D), dtype=mx.float32)
            cot = mx.concatenate([cot, pad], axis=0)

        _, vjps = mx.vjp(lambda h: run_from_layer(model, src_layer, h), [h_batch], [cot])
        grad = vjps[0]
        mx.eval(grad)

        for b in range(n_dims):
            row = grad[b, pos_idx, :].astype(mx.float32).mean(axis=0)
            J[dim_start + b] = np.array(row)

        print(f"    pass {pass_idx+1}/{n_passes} (dims {dim_start}-{dim_start+n_dims})", flush=True)

    return J


if __name__ == "__main__":
    main()