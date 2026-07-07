"""MLX LensModel adapter for Qwen3.6-27B (4-bit).

Implements the same interface as ``jlens.protocol.LensModel`` but against an
MLX-loaded Qwen3.5-architecture model. The forward pass is rewritten from
``Qwen3_5TextModel.__call__`` so it:

- captures the residual stream after every requested layer in a dict,
- runs without KV cache (full-sequence forward, needed for grad),
- forces the Gated DeltaNet ops fallback (via ``patch_gdn.patch_gdn()``),
- builds the autograd graph through every layer so ``mx.vjp`` can backprop
  from the final-layer residual to any earlier layer's residual.

The unembedding reuses the model's own quantized ``lm_head``
(``QuantizedLinear.__call__`` = ``mx.quantized_matmul`` with the dequantized
weight), so we never materialize the 2.5 GB dense ``W_U``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx_lm

from .patch_gdn import patch_gdn
from .custom_gdn_patch import patch_gdn_custom


class MLXLensModel:
    """Wraps an MLX-loaded Qwen3.5 model as a :class:`jlens.protocol.LensModel`.

    The constructor calls ``patch_gdn()`` once (idempotent) so the
    linear-attention layers use the differentiable ops fallback and are
    wrapped in ``mx.checkpoint``.

    Attributes:
        n_layers: Number of residual blocks (64 for Qwen3.6-27B).
        d_model: Residual-stream width (5120).
        layers: The list of ``DecoderLayer`` modules (``model.layers``).
        tokenizer: The mlx_lm tokenizer wrapper.
    """

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        patch_gdn()
        try:
            patch_gdn_custom()
        except Exception as e:
            # Custom Metal VJP is optional; fall back to ops if it fails.
            print(f"Warning: custom GDN VJP patch failed ({e}); using ops fallback")
        self._model = model
        self.tokenizer = tokenizer

        # The Qwen3.5 Model wraps TextModel wraps Qwen3_5TextModel.
        # model.layers -> language_model.model.layers (a Python list).
        self.layers = list(model.layers)
        self.n_layers = len(self.layers)
        # d_model from the first layer's RMSNorm weight shape.
        self.d_model = self.layers[0].input_layernorm.weight.shape[0]

        # Submodules we need for forward / unembed.
        self._text_module = model.language_model.model  # Qwen3_5TextModel
        self._lm_head = model.language_model.lm_head  # QuantizedLinear

        # Mark params as not trainable (matches reference HFLensModel).
        # MLX params are mx.array; we don't need requires_grad toggles
        # because mx.vjp only differentiates w.r.t. inputs we pass as
        # primals. But we do want eval mode for dropout/training flags.
        model.eval()

    def __repr__(self) -> str:
        return f"MLXLensModel(n_layers={self.n_layers}, d_model={self.d_model})"

    def encode(self, text: str, *, max_length: int = 512) -> mx.array:
        """Tokenize ``text`` to ``input_ids`` of shape ``[1, seq_len]``."""
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > max_length:
            ids = ids[-max_length:]  # keep the tail (matches reference truncation)
        return mx.array([ids])

    def forward(
        self,
        input_ids: mx.array,
        *,
        capture_layers: list[int] | None = None,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        """Run the residual stack on ``input_ids`` (no LM head).

        Replaces ``Qwen3_5TextModel.__call__`` so we can capture per-layer
        residual streams while keeping them in the autograd graph.

        Args:
            input_ids: ``[batch, seq_len]`` int array.
            capture_layers: Layer indices whose residual-stream output should
                be captured in the returned dict. ``None`` captures nothing
                (but the forward still runs end to end).

        Returns:
            ``(final_residual, layer_acts)`` where ``final_residual`` is the
            post-norm final-layer residual of shape ``[batch, seq_len, d_model]``
            and ``layer_acts[i]`` is the residual stream *after* layer ``i``
            (i.e. the output of ``layers[i]``), for each ``i`` in
            ``capture_layers``. The pre-layer-0 embedding is keyed as ``-1``
            if requested. All tensors are in the autograd graph.
        """
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        capture = set(capture_layers) if capture_layers is not None else set()
        text = self._text_module
        B, S = input_ids.shape

        # Embedding -> first hidden state. This is "layer -1" output.
        hidden = text.embed_tokens(input_ids)
        acts: dict[int, mx.array] = {}
        if -1 in capture:
            acts[-1] = hidden

        # Build masks once (no cache -> full-sequence causal masks).
        fa_mask = create_attention_mask(hidden, cache=None)
        ssm_mask = create_ssm_mask(hidden, cache=None)

        # We pass cache=None to every layer; the patched GatedDeltaNet
        # handles cache=None correctly (creates a zero conv_state).
        for i, layer in enumerate(self.layers):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=None)
            if i in capture:
                acts[i] = hidden

        # Final pre-unembed norm. This is the final-layer residual.
        final = text.norm(hidden)
        return final, acts

    def unembed(self, residual: mx.array) -> mx.array:
        """Map a residual-stream tensor ``[..., d_model]`` to logits
        ``[..., vocab_size]`` (final norm + LM head).

        We apply the model's own final norm (already applied in ``forward``
        for the final-layer case; callers passing intermediate-layer residuals
        must apply the norm themselves first if they want a fair readout).
        Here we just call ``lm_head`` directly — callers are responsible for
        norming if they want the paper's ``W_U · norm(J_ℓ h_ℓ)`` form.
        """
        return self._lm_head(residual)

    def final_norm(self, residual: mx.array) -> mx.array:
        """Apply the model's final pre-unembed RMSNorm."""
        return self._text_module.norm(residual)


def load(model_id: str = "mlx-community/Qwen3.6-27B-4bit") -> MLXLensModel:
    """Load the model via mlx_lm and wrap it as an MLXLensModel."""
    model, tokenizer = mlx_lm.load(model_id)
    return MLXLensModel(model, tokenizer)


__all__ = ["MLXLensModel", "load"]