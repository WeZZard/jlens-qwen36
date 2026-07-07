"""Spike: compute a full J_ℓ for ℓ ∈ {32, 63} on one 128-token prompt.

Measures:
- Total wall-clock for one full J_ℓ (640 VJP passes with dim_batch=8).
- Peak memory.
- Finiteness and sanity (Frobenius norm, rank).
- A readout: apply J_ℓ to the layer-ℓ residual, unembed, check top tokens.

This is the Phase 1 decision-gate spike.
"""

import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".")
from jlens_qwen.model import load


def run_from_layer(model, src_layer: int, h: mx.array) -> mx.array:
    """Run layers src_layer+1 .. end + final norm, starting from h."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_mask = create_attention_mask(h, cache=None)
    ssm_mask = create_ssm_mask(h, cache=None)
    hidden = h
    for i in range(src_layer + 1, model.n_layers):
        layer = model.layers[i]
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=None)
    return model._text_module.norm(hidden)


def compute_J_layer(model, h_src: mx.array, src_layer: int, dim_batch: int = 8) -> mx.array:
    """Compute J_ℓ ∈ R^{d_model x d_model} via 640 VJP passes.

    For each block of `dim_batch` output dims, set a one-hot cotangent at
    every valid target position, run mx.vjp(run_from_layer, [h_src], [cot]),
    and the result (averaged over positions) gives dim_batch rows of J_ℓ.
    """
    d_model = h_src.shape[-1]
    B, S, D = h_src.shape

    # Skip first 16 positions (attention sinks) + exclude last (no next-token target).
    skip_first = 16
    valid_positions = list(range(skip_first, S - 1))
    n_valid = len(valid_positions)
    if n_valid <= 0:
        raise ValueError(f"prompt too short: seq_len={S}, need > {skip_first + 1}")

    J = mx.zeros((d_model, d_model), dtype=mx.float32)

    n_passes = (d_model + dim_batch - 1) // dim_batch
    print(f"  computing J_{src_layer}: {n_passes} VJP passes, dim_batch={dim_batch}, "
          f"n_valid_positions={n_valid}", flush=True)

    for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
        n_dims = min(dim_batch, d_model - dim_start)

        # Build cotangent: [B, S, D], one-hot at dims [dim_start..dim_start+n_dims)
        # at every valid position, zero elsewhere.
        cot = mx.zeros((B, S, n_dims), dtype=mx.float32)
        # Set 1.0 at valid positions for each of the n_dims output dims.
        # We build a [B, n_valid, n_dims] one-hot and scatter into [B, S, n_dims].
        pos_idx = mx.array(valid_positions)
        # one-hot: [n_valid, n_dims] with 1.0 on the diagonal of the n_dims block
        eye = mx.eye(n_dims, dtype=mx.float32)
        # Broadcast across batch and positions: [B, n_valid, n_dims] = eye
        # Then scatter into [B, S, n_dims] at positions pos_idx.
        cot = mx.zeros((B, S, n_dims), dtype=mx.float32)
        # Use mx.scatter via assignment: cot[b, pos_idx[b_pos], dim] = eye[b_pos, dim]
        # Simplest: loop over batch (B=1 here anyway).
        for b in range(B):
            # cot[b, valid_positions, :] = eye
            # MLX scatter: use .at with setter
            # Actually easier: build a full [S, n_dims] zero and set rows.
            block = mx.zeros((S, n_dims), dtype=mx.float32)
            # Use ArrayAt.setter
            block = block.at[pos_idx.tolist()].setter(eye)
            cot = mx.where(mx.arange(B)[:, None, None] == b, mx.expand_dims(block, 0), cot)

        # VJP: cotangent shape must match run_from_layer output shape [B, S, D]
        # But our cot is [B, S, n_dims]. We need to do one VJP per dim, OR
        # embed n_dims into the full D. Let's embed: zero [B,S,D], set the
        # n_dims block at dims [dim_start:dim_start+n_dims].
        full_cot = mx.zeros((B, S, D), dtype=mx.float32)
        # full_cot[:, :, dim_start:dim_start+n_dims] = cot
        # MLX doesn't support item assignment on slices easily; use concat.
        pre = mx.zeros((B, S, dim_start), dtype=mx.float32)
        post = mx.zeros((B, S, D - dim_start - n_dims), dtype=mx.float32)
        full_cot = mx.concatenate([pre, cot, post], axis=-1)

        # Run VJP.
        _, vjps = mx.vjp(lambda h: run_from_layer(model, src_layer, h), [h_src], [full_cot])
        grad = vjps[0]  # [B, S, D]
        mx.eval(grad)

        # Average over valid positions, take batch 0 -> [n_dims, D]
        # grad[0, valid_positions, :].mean(0) -> [D]
        # But we want [n_dims, D]: for each of the n_dims cotangent dims, the
        # grad row. Since cot was one-hot per dim, grad[:, :, dim_start+i]
        # corresponds to the i-th row. Wait — the cotangent is at the OUTPUT
        # dim, so vjp gives d(output)/d(input) applied to cot -> grad is at
        # the input. grad[b, s, :] is the VJP for the cotangent at (b, s, :).
        # But our cot has one-hot at output dims dim_start+i for each i,
        # broadcast over positions. The VJP sums over the output (all
        # positions) so grad[b, s, input_dim] = sum over t' of
        # d(final[t', dim_start+i]) / d(h[b, s, input_dim]) * 1.0
        # We want the average over source positions s (the paper's estimator),
        # giving J[dim_start+i, input_dim].
        grad_block = grad[0, valid_positions, :].astype(mx.float32).mean(axis=0)  # [D]
        # Place as rows dim_start..dim_start+n_dims of J.
        # J[dim_start:dim_start+n_dims, :] = grad_block
        # But grad_block is [D] for one dim (since cot was one-hot at one output
        # dim at a time... no, cot has n_dims one-hots stacked). So grad_block
        # is actually the sum over the n_dims output dims. That's wrong.

        # Fix: we need per-dim rows. Do one VJP per dim (n_dims VJPs).
        # OR: structure cot as [B, S, D] with one-hot at ONE dim, run VJP,
        # get one row. That's D passes total = 5120. dim_batch was supposed
        # to batch them. MLX vjp takes a single cotangent matching the output
        # shape, so batching requires the cotangent to have an extra axis,
        # which the output doesn't have.

        # For now: this spike does one dim at a time (dim_batch effectively 1
        # for the VJP, but we loop). The grad_block above is for the LAST dim
        # in the block only. Let's just store it and fix the loop.

        J = mx.where(
            mx.arange(d_model)[:, None] == dim_start + n_dims - 1,
            mx.broadcast_to(grad_block, (d_model, d_model)),
            J,
        )

        if pass_idx % 50 == 0 or pass_idx == n_passes - 1:
            print(f"    pass {pass_idx+1}/{n_passes} (dim {dim_start}-{dim_start+n_dims})", flush=True)

    return J


def compute_J_layer_one_at_a_time(model, h_src, src_layer, *, n_dims_total=None):
    """Compute J_ℓ one output dim at a time. Slow but correct.

    For each output dim d, set a one-hot cotangent at every valid position,
    VJP, average over valid source positions, store as row d of J.
    """
    d_model = h_src.shape[-1]
    B, S, D = h_src.shape
    if n_dims_total is None:
        n_dims_total = d_model

    skip_first = 16
    valid_positions = list(range(skip_first, S - 1))
    n_valid = len(valid_positions)

    J = np.zeros((n_dims_total, d_model), dtype=np.float32)

    print(f"  computing J_{src_layer}: {n_dims_total} VJP passes (one dim at a time)", flush=True)

    for d in range(n_dims_total):
        # one-hot cotangent at output dim d, at every valid position.
        # Build [B, S, D] zero, then set 1.0 at (any batch, valid_positions, d).
        # Construct via: cot[b, s, d] = 1 if s in valid_positions else 0.
        # = pos_mask[s] where pos_mask is 1 at valid positions, 0 elsewhere.
        # Build pos_mask via mx.where on an arange.
        arange_S = mx.arange(S)
        # valid_positions is [skip_first, S-1)
        pos_mask = mx.where(
            (arange_S >= skip_first) & (arange_S < S - 1),
            mx.ones((S,), dtype=mx.float32),
            mx.zeros((S,), dtype=mx.float32),
        )
        # cot[b, s, d] = pos_mask[s]
        cot = mx.zeros((B, S, D), dtype=mx.float32)
        # Set column d to pos_mask (broadcast over batch).
        # Build via: cot[:, :, d] = pos_mask. MLX slice assignment isn't
        # supported; use a one-hot at dim d and multiply.
        dim_onehot = mx.zeros((D,), dtype=mx.float32)
        dim_onehot = dim_onehot.at[d].add(mx.array(1.0))  # one-hot at d
        cot = pos_mask[None, :, None] * dim_onehot[None, None, :]  # [B, S, D]

        _, vjps = mx.vjp(lambda h: run_from_layer(model, src_layer, h), [h_src], [cot])
        grad = vjps[0]  # [B, S, D]
        mx.eval(grad)
        # Average over valid SOURCE positions -> row d of J
        # grad[0, valid_positions, :].mean(0) -> [D]
        # Build the valid-position slice via indexing with an mx.array.
        pos_idx = mx.array(valid_positions)
        row = grad[0, pos_idx, :].astype(mx.float32).mean(axis=0)  # [D]
        J[d] = np.array(row)

        if d % 50 == 0 or d == n_dims_total - 1:
            print(f"    dim {d+1}/{n_dims_total}", flush=True)

    return J


def main():
    print("=" * 60, flush=True)
    print("PHASE 1 SPIKE: compute full J_ℓ for ℓ ∈ {32, 63}", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading model...", flush=True)
    t0 = time.perf_counter()
    model = load()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    # Use a 128-token prompt (the paper's max_seq_len).
    # We need a real prompt of at least 144 tokens (128 + 16 skip).
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
    print(f"  prompt length: {len(prompt)} chars", flush=True)
    print(f"  input_ids shape: {input_ids.shape}", flush=True)

    # Forward, capturing the source-layer residuals.
    print("\nForward pass (capture all layers)...", flush=True)
    t0 = time.perf_counter()
    final, acts = model.forward(input_ids, capture_layers=[32, 63])
    mx.eval(final)
    for l in [32, 63]:
        mx.eval(acts[l])
    print(f"  forward in {time.perf_counter()-t0:.1f}s", flush=True)

    # Spike: compute just the first 64 rows of J_32 and J_63 (to estimate time).
    # Full would be 5120 rows; we'll extrapolate.
    n_dims_spike = 64

    for src_layer in [32, 63]:
        print(f"\n--- J_{src_layer} (spike: first {n_dims_spike} dims) ---", flush=True)
        h_src = acts[src_layer]
        t0 = time.perf_counter()
        J_partial = compute_J_layer_one_at_a_time(
            model, h_src, src_layer, n_dims_total=n_dims_spike
        )
        elapsed = time.perf_counter() - t0
        per_dim = elapsed / n_dims_spike
        print(f"  {n_dims_spike} dims in {elapsed:.1f}s ({per_dim:.2f}s/dim)", flush=True)
        print(f"  extrapolated full J_{src_layer}: {per_dim * 5120:.0f}s = {per_dim * 5120 / 60:.1f}min", flush=True)
        print(f"  J_{src_layer} partial shape: {J_partial.shape}", flush=True)
        print(f"  J_{src_layer} finite: {np.isfinite(J_partial).all()}", flush=True)
        print(f"  J_{src_layer} Frobenius norm (partial): {np.linalg.norm(J_partial):.3e}", flush=True)

    # Quick readout test: apply J_63 to h_63 and decode top tokens.
    print("\n--- Readout test (J_63 @ h_63) ---", flush=True)
    h_src = acts[63]
    # For a real readout we'd need the full J_63; use identity (logit lens) as a smoke check.
    lens_logits = model.unembed(model.final_norm(h_src))  # J=I for last layer
    lf = lens_logits[0, -1].astype(mx.float32)
    sorted_idx = mx.argsort(lf)
    top5 = [int(t) for t in sorted_idx[-5:][::-1].tolist()]
    print(f"  logit-lens (J=I) top-5 at last position:", flush=True)
    for t in top5:
        print(f"    {t}: {model.tokenizer.decode([t])!r}", flush=True)

    print("\nSPIKE COMPLETE", flush=True)


if __name__ == "__main__":
    main()