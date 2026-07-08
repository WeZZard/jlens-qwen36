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

    def forward_from_layer(
        self,
        h: mx.array,
        start_layer: int,
    ) -> mx.array:
        """Run layers start_layer..end + final norm, starting from h.

        Used to resume a forward pass after an intervention at start_layer.
        h: [batch, seq, d_model] (the residual after `start_layer - 1`,
        i.e. entering `start_layer`). Returns final-norm residual.
        """
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        fa_mask = create_attention_mask(h, cache=None)
        ssm_mask = create_ssm_mask(h, cache=None)
        hidden = h
        for i in range(start_layer, self.n_layers):
            layer = self.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=None)
        return self._text_module.norm(hidden)

    def forward_with_intervention(
        self,
        input_ids: mx.array,
        intervene_layer: int,
        intervene_positions: list[int] | None,
        patched_h_fn,
    ) -> mx.array:
        """Forward with an intervention at one layer, multiple positions.

        Runs the forward up to `intervene_layer`, captures the residual,
        calls `patched_h_fn(h)` for each position in `intervene_positions`
        (or all positions if None), writes the patched residuals back,
        then resumes the forward from `intervene_layer + 1`.

        patched_h_fn: callable taking h [d_model] -> patched h [d_model].
            Called once per position.

        Returns the final-norm residual [batch, seq, d_model].
        """
        _, acts = self.forward(input_ids, capture_layers=[intervene_layer])
        h = acts[intervene_layer]  # [batch, seq, d_model]
        B, S, D = h.shape
        positions = intervene_positions if intervene_positions is not None else list(range(S))
        # Patch each requested position.
        patched = []
        for p in range(S):
            if p in positions:
                patched.append(patched_h_fn(h[0, p]))  # [d_model]
            else:
                patched.append(h[0, p])
        new_h = mx.stack(patched, axis=0)[None]  # [1, S, D]
        return self.forward_from_layer(new_h, intervene_layer + 1)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temp: float = 0.0,
        intervene_layer: int | None = None,
        intervene_fn=None,
        intervene_positions: list[int] | None = None,
        intervene_each_step: bool = False,
    ) -> tuple[str, list[int]]:
        """Generate a continuation of `prompt`.

        Greedy if temp == 0, else sample from softmax(1/temp).

        If intervene_layer + intervene_fn are given, applies the intervention
        at `intervene_positions` (default: all positions) of each forward
        pass. If intervene_each_step is False, applies only on the first
        (prefill) pass; if True, on every step (so the intervention persists).

        Returns (decoded_text, token_ids).
        """
        input_ids = self.encode(prompt, max_length=512)
        generated: list[int] = []
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        def _sample(lf: mx.array) -> int:
            if temp == 0:
                return int(mx.argmax(lf).tolist())
            # mx.random.categorical takes UNNORMALIZED logits (it applies the
            # softmax itself). Feeding it softmax probabilities compresses the
            # distribution to near-uniform over the 248k vocab -> gibberish.
            return int(mx.random.categorical(lf / temp).tolist())

        if intervene_layer is None or intervene_fn is None:
            # Fast path: cached incremental decoding (O(T) instead of O(T^2)).
            session = self.make_stream()
            logits, _ = session.extend(input_ids[0].tolist())
            for _ in range(max_tokens):
                next_tok = _sample(logits.astype(mx.float32))
                generated.append(next_tok)
                if eos_id is not None and next_tok == eos_id:
                    break
                logits, _ = session.extend([next_tok])
            return self.tokenizer.decode(generated), generated

        # Intervention path: uncached full re-forward per step (the
        # intervention patches a mid-stack layer, which the cached step
        # loop does not support yet).
        for step in range(max_tokens):
            if step == 0 or intervene_each_step:
                final = self.forward_with_intervention(
                    input_ids, intervene_layer, intervene_positions, intervene_fn
                )
            else:
                final, _ = self.forward(input_ids)
            next_tok = _sample(self.unembed(final[:, -1, :])[0].astype(mx.float32))
            generated.append(next_tok)
            input_ids = mx.concatenate([input_ids, mx.array([[next_tok]])], axis=1)
            if eos_id is not None and next_tok == eos_id:
                break

        text = self.tokenizer.decode(generated)
        return text, generated

    def make_stream(self, capture_layers=None) -> "StreamSession":
        """Create a cached incremental-decoding session (see StreamSession)."""
        return StreamSession(self, capture_layers=capture_layers)


class StreamSession:
    """Cached incremental decoding with per-layer residual capture.

    Uses the model's own hybrid caches (KV for the 16 full-attention
    layers, conv+recurrent state for the 48 GDN layers) so each new chunk
    costs one pass over its own tokens instead of re-running the whole
    sequence — O(T) total decode vs the O(T^2) of repeated
    ``MLXLensModel.forward`` calls. GDN prefill/steps run the stock fused
    kernel (``patch_gdn.set_inference_mode``); no gradients are available
    on this path, which is fine — readouts only need activations.
    """

    def __init__(self, model: MLXLensModel, capture_layers=None) -> None:
        from mlx_lm.models.cache import make_prompt_cache
        from .patch_gdn import set_inference_mode

        set_inference_mode(True)
        self._model = model
        self.caches = make_prompt_cache(model._model)
        self.capture = sorted(set(capture_layers or []))
        self.n_consumed = 0

    def extend(self, ids) -> tuple[mx.array, dict[int, mx.array]]:
        """Feed token ids (a Python list) after the tokens already consumed.

        Returns ``(logits, acts)`` where ``logits`` is the last position's
        vocab logits ``[vocab]`` and ``acts[l]`` is the residual after
        layer ``l`` for the new tokens only, ``[1, len(ids), D]``.
        """
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        m = self._model
        text = m._text_module
        hidden = text.embed_tokens(mx.array([list(ids)]))
        fa_mask = create_attention_mask(hidden, self.caches[text.fa_idx])
        ssm_mask = create_ssm_mask(hidden, self.caches[text.ssm_idx])

        acts: dict[int, mx.array] = {}
        for i, (layer, c) in enumerate(zip(m.layers, self.caches)):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=c)
            if i in self.capture:
                acts[i] = hidden
        final = text.norm(hidden)
        logits = m.unembed(final[:, -1, :])[0]
        mx.eval(logits, *acts.values())
        self.n_consumed += len(ids)
        return logits, acts


def load(model_id: str = "mlx-community/Qwen3.6-27B-4bit") -> MLXLensModel:
    """Load the model via mlx_lm and wrap it as an MLXLensModel."""
    model, tokenizer = mlx_lm.load(model_id)
    return MLXLensModel(model, tokenizer)


__all__ = ["MLXLensModel", "StreamSession", "load"]