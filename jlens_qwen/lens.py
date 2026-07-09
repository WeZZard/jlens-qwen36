"""JacobianLens: hold fitted J_l matrices, apply them to read out J-space tokens.

Mirrors jlens.lens.JacobianLens but stores J as numpy arrays (no torch dep
for the lens itself). Save/load via safetensors or numpy.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from .model import MLXLensModel


class JacobianLens:
    """A fitted Jacobian lens: per-layer J_l matrices and the readout method."""

    def __init__(
        self,
        jacobians: dict[int, np.ndarray],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {l: J.astype(np.float32) for l, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model
        # Per-layer J as fp16 mx arrays, built lazily on first transport.
        # fp16 matches the on-disk precision (save() writes fp16), and
        # memoizing avoids re-uploading the 105MB fp32 numpy matrix on
        # EVERY transport call (~6.6GB of host->GPU churn per generated
        # token when reading out all 63 layers).
        self._mx_jacobians: dict[int, mx.array] = {}

    def __repr__(self) -> str:
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers={self.source_layers})"
        )

    def save(self, path: str) -> None:
        """Save as npz (one file, all layers). fp16 to halve size."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        J_fp16 = {f"J_{l}": J.astype(np.float16) for l, J in self.jacobians.items()}
        np.savez(path, **J_fp16, n_prompts=self.n_prompts, d_model=self.d_model)
        # Write meta sidecar
        meta_path = path.replace(".npz", ".json")
        with open(meta_path, "w") as f:
            json.dump({
                "n_prompts": self.n_prompts,
                "d_model": self.d_model,
                "source_layers": self.source_layers,
            }, f)

    @classmethod
    def load(cls, path: str) -> "JacobianLens":
        data = np.load(path, allow_pickle=True)
        jacobians = {}
        n_prompts = int(data["n_prompts"])
        d_model = int(data["d_model"])
        for key in data.files:
            if key.startswith("J_"):
                l = int(key.split("_")[1])
                jacobians[l] = data[key].astype(np.float32)
        return cls(jacobians, n_prompts=n_prompts, d_model=d_model)

    def warm(self) -> None:
        """Materialize the fp16 GPU copies of all J matrices now.

        The first transport of each layer otherwise pays a one-time
        fp32->fp16 conversion + upload (~3.3GB total across 63 layers,
        measured ~2s) — which lands inside the first request's prefill
        readout. Call this at server startup instead.
        """
        for l in self.source_layers:
            if l not in self._mx_jacobians:
                J = mx.array(self.jacobians[l].astype(np.float16))
                mx.eval(J)
                self._mx_jacobians[l] = J

    def transport(self, residual: mx.array, layer: int) -> mx.array:
        """Map a residual at `layer` into the final-layer basis: J_l @ h.

        residual: [..., D] mx.array. Returns [..., D] mx.array.
        """
        if layer not in self.jacobians:
            raise KeyError(f"layer {layer} not in source_layers {self.source_layers}")
        J = self._mx_jacobians.get(layer)
        if J is None:
            J = mx.array(self.jacobians[layer].astype(np.float16))  # [D, D]
            mx.eval(J)
            self._mx_jacobians[layer] = J
        # residual @ J.T -> [..., D]; fp16 matmul (the lens has fp16
        # information content anyway), fp32 result for downstream readout.
        return mx.matmul(residual.astype(mx.float16), J.T).astype(mx.float32)

    def apply(
        self,
        model: MLXLensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> dict:
        """Run model on prompt, return lens logits at each source layer.

        Returns dict with:
          - "lens_logits": {layer: [seq_len, vocab] mx.array}
          - "model_logits": [seq_len, vocab] mx.array (final layer)
          - "input_ids": [1, seq_len] mx.array
          - "token_strs": list of decoded tokens
        """
        if layers is None:
            layers = self.source_layers
        if use_jacobian:
            unknown = set(layers) - set(self.source_layers)
            if unknown:
                raise ValueError(f"layers {sorted(unknown)} not fitted")

        input_ids = model.encode(prompt, max_length=max_seq_len)
        final, acts = model.forward(input_ids, capture_layers=list(layers) + [model.n_layers - 1])
        for l in set(layers) | {model.n_layers - 1}:
            mx.eval(acts[l])

        lens_logits = {}
        for layer in layers:
            h = acts[layer][0].astype(mx.float32)  # [seq_len, D]
            if use_jacobian and layer in self.jacobians:
                h = self.transport(h, layer)
            # Apply final norm + lm_head
            logits = model.unembed(model.final_norm(h))
            lens_logits[layer] = logits

        # Model logits (final layer, no J)
        h_final = acts[model.n_layers - 1][0].astype(mx.float32)
        model_logits = model.unembed(model.final_norm(h_final))

        token_strs = [
            model.tokenizer.decode([int(t)])
            for t in input_ids[0].tolist()
        ]

        return {
            "lens_logits": lens_logits,
            "model_logits": model_logits,
            "input_ids": input_ids,
            "token_strs": token_strs,
        }


def topk_tokens(logits: mx.array, k: int = 10) -> list[tuple[int, str, float]]:
    """Top-k tokens from logits [vocab] or [seq, vocab]. Returns list of
    (token_id, decoded_str, score) for the last position if 2D."""
    if logits.ndim == 2:
        logits = logits[-1]  # last position
    lf = logits.astype(mx.float32)
    sorted_idx = mx.argsort(lf)
    top = sorted_idx[-k:][::-1]
    top_list = [int(t) for t in top.tolist()]
    scores = [float(mx.take(lf, t).tolist()) for t in top_list]
    return top_list, scores