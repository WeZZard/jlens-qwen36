"""J-space interventions: steer, swap, ablate.

These are the "write" operations on the J-space, mirroring the paper's
experiments. Each operates on a residual-stream activation h_ℓ at one
layer+position, modifying it to change what the model says.

J-lens vectors
--------------
For a token t in the vocabulary, the J-lens vector at layer ℓ is:

    v_t = (W_U J_ℓ)ᵀ[t, :] = J_ℓᵀ @ W_U[t, :]

where W_U is the unembedding matrix [vocab, d_model] and J_ℓ is the
fitted Jacobian [d_model, d_model]. v_t is the direction in h_ℓ-space
that, when added to h_ℓ, makes the model more likely to say token t
(averaged over contexts).

We compute v_t lazily and cache it. We use the quantized lm_head's
dequantized weight as W_U.

Interventions
-------------
- steer(h, v_t, alpha): h += alpha * v_t. Positive alpha makes the model
  more likely to say t; negative ablates t.
- patch_swap(h, v_s, v_t, alpha=1): exchange the "s" component of h for
  an equal-magnitude "t" component, leaving the orthogonal component
  unchanged. c = V† h; h += V (σ(c) - c) where V=[v_s, v_t], σ swaps.
- ablate_topk(h, lens, layer, k): sparse-decompose h into top-k J-lens
  vectors via gradient pursuit, subtract the J-space component.
"""

from __future__ import annotations

import functools
from typing import Sequence

import mlx.core as mx
import numpy as np

from .lens import JacobianLens
from .model import MLXLensModel


def get_unembedding_matrix(model: MLXLensModel) -> mx.array:
    """Dequantize the model's lm_head into a dense W_U [vocab, d_model].

    Cached on the model object as fp32 (~5 GB for the 248k vocab).
    SCRIPT-ONLY: the server path must use unembed_rows() instead, which
    dequantizes just the rows it needs.
    """
    if hasattr(model, "_W_U"):
        return model._W_U
    lm_head = model._lm_head
    # QuantizedLinear stores weight, scales, biases, group_size, bits.
    w = lm_head["weight"]
    scales = lm_head["scales"]
    biases = lm_head["biases"]
    W_U = mx.dequantize(
        w, scales, biases,
        group_size=lm_head.group_size,
        bits=lm_head.bits,
    )
    # W_U shape: [vocab, d_model] (lm_head is nn.Linear: out_features=vocab, in_features=d_model)
    model._W_U = W_U.astype(mx.float32)
    return model._W_U


@functools.lru_cache(maxsize=8)
def j_lens_vectors(lens: JacobianLens, model: MLXLensModel, layer: int) -> mx.array:
    """Compute all J-lens vectors v_t = J_ℓᵀ @ W_U[t] for layer ℓ.

    Returns V: [vocab, d_model] mx.array (fp32). Each row V[t] is the
    J-lens vector for token t. Cached per (lens, layer).

    This is a big matmul: [d_model, d_model] @ [d_model, vocab] = [d_model, vocab],
    then transpose -> [vocab, d_model]. ~134 GFLOPs, ~1s on M4 Pro.
    SCRIPT-ONLY: ~5 GB fp32 per layer, lru-cached — the server path must
    use j_lens_vectors_lite() (per-token matvecs) instead.
    """
    if layer not in lens.jacobians:
        raise KeyError(f"layer {layer} not fitted (source_layers={lens.source_layers})")
    J = mx.array(lens.jacobians[layer])  # [d_model, d_model] fp32
    W_U = get_unembedding_matrix(model)  # [vocab, d_model] fp32
    # v_t = Jᵀ @ W_U[t] for each t -> V = W_U @ Jᵀᵀ = W_U @ J... wait.
    # We want v_t such that <v_t, h> ≈ <W_U[t], J_ℓ h> = <J_ℓᵀ W_U[t], h>.
    # So v_t = J_ℓᵀ @ W_U[t].  V[t] = J_ℓᵀ @ W_U[t]  -> V = W_U @ J_ℓ  (rows of W_U times J).
    V = mx.matmul(W_U, J)  # [vocab, d_model]
    mx.eval(V)
    return V


def j_lens_vector(lens: JacobianLens, model: MLXLensModel, layer: int, token_id: int) -> mx.array:
    """Get the J-lens vector for a single token at a layer. [d_model]."""
    V = j_lens_vectors(lens, model, layer)
    return V[token_id]


def j_lens_vector_for_text(lens, model, layer, text: str) -> mx.array:
    """Get the J-lens vector for the first token of `text` at `layer`."""
    # Encode just this text and take the first token id.
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"text {text!r} encodes to no tokens")
    return j_lens_vector(lens, model, layer, ids[0])


# ---------------------------------------------------------------------------
# Server-path helpers: per-token vectors without the [vocab, d_model] matrix.
# ---------------------------------------------------------------------------


def unembed_rows(model: MLXLensModel, token_ids: Sequence[int]) -> mx.array:
    """Dequantize ONLY rows `token_ids` of W_U. Returns [k, d_model] fp32.

    Quantization groups run along d_model, so slicing rows of the packed
    weight/scales/biases before mx.dequantize is exact (verified against
    the full dequantize). ~20 KB per row instead of ~5 GB for the matrix.
    """
    lm_head = model._lm_head
    idx = mx.array(list(token_ids))
    rows = mx.dequantize(
        lm_head["weight"][idx],
        lm_head["scales"][idx],
        lm_head["biases"][idx],
        group_size=lm_head.group_size,
        bits=lm_head.bits,
    )
    return rows.astype(mx.float32)


def j_lens_vectors_lite(
    lens: JacobianLens | None,
    model: MLXLensModel,
    layer: int,
    token_ids: Sequence[int],
) -> mx.array:
    """J-lens vectors v_t = J_ℓᵀ @ W_U[t] for a few tokens. [k, d_model] fp32.

    Matvec against the lens's resident fp16 GPU Jacobian (lens.jacobian_mx)
    instead of materializing W_U @ J — the memory-safe server-path twin of
    j_lens_vectors(). The final (unfitted) layer reads out through the
    plain logit lens, so there v_t = W_U[t].
    """
    W_rows = unembed_rows(model, token_ids)  # [k, D] fp32
    if lens is None or layer not in lens.jacobians:
        if layer != model.n_layers - 1:
            fitted = lens.source_layers if lens is not None else []
            raise KeyError(f"layer {layer} has no fitted Jacobian (source_layers={fitted})")
        return W_rows
    J = lens.jacobian_mx(layer)  # [D, D] fp16
    # <v_t, h> = <W_U[t], J h>  =>  v_t = J_ℓᵀ W_U[t]  =>  rows: W_rows @ J.
    return mx.matmul(W_rows.astype(mx.float16), J).astype(mx.float32)


def steer(h: mx.array, v_t: mx.array, alpha: float) -> mx.array:
    """h += alpha * v_t. Modifies the activation to make the model more (alpha>0)
    or less (alpha<0) likely to say token t.

    h: [..., d_model]. v_t: [d_model]. Returns [..., d_model].
    """
    return h + alpha * v_t


def make_swap_basis(v_s: mx.array, v_t: mx.array) -> tuple[mx.array, mx.array]:
    """Stack V = [v_s, v_t] [2, d_model] with inv(V Vᵀ) [2, 2].

    V Vᵀ is inverted explicitly (mx.linalg.solve is CPU-only):
    inv([[a,b],[c,d]]) = (1/det) [[d,-b],[-c,a]]. Precompute once per
    intervention; patch_swap / patch_swap_rows consume the pair.
    """
    V = mx.stack([v_s, v_t], axis=0)  # [2, d_model]
    VVt = mx.matmul(V, V.T)  # [2, 2]
    a, b = VVt[0, 0], VVt[0, 1]
    c_, d = VVt[1, 0], VVt[1, 1]
    det = a * d - b * c_
    inv = mx.stack(
        [mx.stack([d, -b]), mx.stack([-c_, a])], axis=0
    ) / det  # [2, 2]
    return V, inv


def patch_swap(h: mx.array, v_s: mx.array, v_t: mx.array, alpha: float = 1.0) -> mx.array:
    """Exchange the "s" component of h for an equal-magnitude "t" component.

    Reads the lens coordinates c = V† h (V = [v_s, v_t], V† = pseudoinverse),
    swaps the two entries (scaled by alpha), writes back, leaving the
    component orthogonal to span{v_s, v_t} unchanged.

    h: [d_model]. v_s, v_t: [d_model]. Returns [d_model].
    """
    V, inv = make_swap_basis(v_s, v_t)
    Vh = mx.matmul(V, h)  # [2]
    co = mx.matmul(inv, Vh[..., None])[..., 0]  # [2] = c
    # Swap with scaling: σ(c) = [alpha*c[1], alpha*c[0]]
    c_swapped = mx.stack([alpha * co[1], alpha * co[0]], axis=0)
    delta_c = c_swapped - co  # [2]
    # h_patched = h + Vᵀ delta_c  (V is [2, d_model]; Vᵀ delta_c = delta_c @ V)
    h_patched = h + mx.matmul(delta_c[None, :], V)[0]  # [d_model]
    return h_patched


def patch_swap_rows(
    h: mx.array, V: mx.array, VVt_inv: mx.array, alpha: float = 1.0
) -> mx.array:
    """Batched patch_swap over rows: h [n, d_model] -> [n, d_model].

    Same math as patch_swap per row, vectorized for the streaming path
    where one edit covers many positions of a chunk.
    """
    co = mx.matmul(VVt_inv, mx.matmul(V, h.T))  # [2, n] lens coordinates
    swapped = mx.stack([alpha * co[1], alpha * co[0]], axis=0)  # [2, n]
    return h + mx.matmul((swapped - co).T, V)  # [n,2] @ [2,D] -> [n,D]


def gram_inv(V: mx.array, ridge: float = 1e-6) -> mx.array:
    """Ridge-regularized inverse of the Gram matrix V Vᵀ, [k, k] fp32.

    Computed on CPU via numpy (mx.linalg is CPU-only anyway) once per
    intervention compile — k is small (<= ~32 ablate directions).
    """
    G = np.array(mx.matmul(V, V.T).astype(mx.float32))
    k = G.shape[0]
    reg = ridge * (np.trace(G) / max(k, 1))
    return mx.array(np.linalg.inv(G + reg * np.eye(k, dtype=G.dtype)).astype(np.float32))


def ablate_rows(
    h: mx.array, V: mx.array, G_inv: mx.array, alpha: float = 1.0
) -> mx.array:
    """Remove the span{V} lens component from each row of h.

    h [n, d_model]; V [k, d_model]; G_inv = gram_inv(V). The least-squares
    coordinates c = G_inv V h are subtracted (scaled by alpha, 1 = full
    removal), leaving the component orthogonal to span{V} unchanged —
    the paper's workspace-ablation with an explicit direction set.
    """
    co = mx.matmul(G_inv, mx.matmul(V, h.T))  # [k, n]
    return h - alpha * mx.matmul(co.T, V)  # [n, D]


def ablate_topk(
    h: mx.array,
    lens: JacobianLens,
    model: MLXLensModel,
    layer: int,
    k: int = 16,
    n_iters: int = 5,
) -> mx.array:
    """Remove the top-k J-space component of h via gradient pursuit.

    Greedy: at each of n_iters iterations, find the J-lens vector v_t
    most aligned with the residual, subtract its projection, repeat
    until k vectors are accumulated.

    h: [d_model]. Returns [d_model] with the J-space component removed.
    """
    V = j_lens_vectors(lens, model, layer)  # [vocab, d_model]
    residual = h.astype(mx.float32)
    accumulated = mx.zeros_like(h.astype(mx.float32))
    chosen: list[int] = []

    for _ in range(min(k, n_iters * 4)):
        # Find the J-lens vector most aligned with the residual.
        # scores = V @ residual -> [vocab]
        scores = mx.matmul(V, residual)  # [vocab]
        # Normalize by vector norms (V rows may have different magnitudes)
        V_norms = mx.sqrt((V * V).sum(axis=-1))  # [vocab]
        normalized = scores / (V_norms + 1e-8)
        # Pick the best token not already chosen. (In-place __setitem__;
        # MLX's ArrayAt has no .set on 0.31.x.)
        for tid in chosen:
            normalized[tid] = -1e9
        best = int(mx.argmax(normalized).tolist())
        chosen.append(best)
        v = V[best]  # [d_model]
        # Project residual onto v and move it to accumulated.
        coef = mx.matmul(v, residual) / mx.matmul(v, v)
        component = coef * v
        accumulated = accumulated + component
        residual = residual - component
        if len(chosen) >= k:
            break

    # h_ablated = h - accumulated (remove the J-space component)
    return h.astype(mx.float32) - accumulated


# ---------------------------------------------------------------------------
# Spec compilation for the streaming path.
# ---------------------------------------------------------------------------


def compile_edits(
    lens: JacobianLens | None,
    model: MLXLensModel,
    *,
    mode: str,
    layers: Sequence[int],
    token_id: int | None = None,
    target_id: int | None = None,
    alpha: float = 1.0,
    positions: Sequence[int] | None = None,
    from_pos: int | None = None,
    ablate_token_ids: Sequence[int] | None = None,
    label: str = "",
) -> list:
    """Compile one intervention spec into per-layer LayerEdits.

    All concept vectors, bases and inverses are computed and mx.eval'ed
    HERE, once per request — the returned closures are pure GPU matvecs
    with no host syncs and no numpy captures, safe to run per decode step.

    Modes:
      steer:  h += alpha * v_token          (alpha < 0 suppresses)
      swap:   lens-coordinate swap token -> target (patch_swap_rows)
      ablate: remove span{v_t : t in ablate_token_ids} (ablate_rows)
    """
    from .model import LayerEdit

    edits: list[LayerEdit] = []
    pos_t = tuple(positions) if positions is not None else None
    for layer in layers:
        if mode == "steer":
            if token_id is None:
                raise ValueError("steer requires token_id")
            v = j_lens_vectors_lite(lens, model, layer, [token_id])[0]
            mx.eval(v)
            fn = lambda h, v=v, a=alpha: h + a * v
        elif mode == "swap":
            if token_id is None or target_id is None:
                raise ValueError("swap requires token_id and target_id")
            vs = j_lens_vectors_lite(lens, model, layer, [token_id, target_id])
            V, inv = make_swap_basis(vs[0], vs[1])
            mx.eval(V, inv)
            fn = lambda h, V=V, inv=inv, a=alpha: patch_swap_rows(h, V, inv, a)
        elif mode == "ablate":
            if not ablate_token_ids:
                raise ValueError("ablate requires ablate_token_ids")
            V = j_lens_vectors_lite(lens, model, layer, list(ablate_token_ids))
            G_inv = gram_inv(V)
            mx.eval(V, G_inv)
            fn = lambda h, V=V, G=G_inv, a=alpha: ablate_rows(h, V, G, a)
        else:
            raise ValueError(f"unknown intervention mode {mode!r}")
        edits.append(LayerEdit(
            layer=layer, fn=fn, positions=pos_t, from_pos=from_pos, label=label,
        ))
    return edits