# decode-01 — Cached incremental decoding + readout fixes

**Commit:** `f7d56a6` — "Viewer decode: O(T) cached streaming + readout fixes
(was O(T²) full re-forwards)".
**Code:** `jlens_qwen/model.py` (`StreamSession`), `patch_gdn.py`
(`set_inference_mode`), `serve.py` (chat stream), `lens.py` (`transport`).

## The problem

The chat viewer's generation loop called `_model.forward(input_ids)` after
**every** generated token, and the lens-model forward always ran
`cache=None` (it was built for fitting, where autograd needs the full graph).
So decoding re-ran the **entire conversation through all 64 layers per
token** — O(T²) — starting at ~1.6 s/token and growing with context. On top
of that, `patch_gdn` force-routed GDN to the slow ops recurrence *globally*,
so even a cached forward would crawl.

Two readout costs compounded it:

- `lens.transport` did `mx.array(self.jacobians[ℓ])` on **every call**,
  re-uploading a 105 MB fp32 matrix per layer per token — **~6.6 GB of
  host→GPU churn per token** across 63 layers.
- `_readout_at_positions` transported and unembedded the **full sequence**
  even though it reads exactly one frontier position.

## The fix (three pieces)

- **C1 — inference-mode toggle** (`set_inference_mode`): route
  `gated_delta_update` back to the stock fused Metal kernel for generation.
  Fitting keeps the differentiable ops path by default; the flag is off
  unless a serving process sets it, so fit semantics are untouched.
- **C2 — `StreamSession`**: hybrid-cache incremental decoding — KV cache for
  the 16 full-attention layers, conv + recurrent state for the 48 GDN layers
  — with per-layer frontier capture for the readout. Prefill once, then one
  token per step.
- **C3 — rewire `serve.py`'s chat stream**: one session per stream,
  historical messages fed as delta chunks, generation as one-token steps.
  SSE protocol unchanged.
- **Readout fixes:** `transport` now memoizes each `J` as an fp16 MLX array
  on first use (kills the per-token re-upload); `_readout_at_positions`
  slices activations to the requested positions *before* transport/unembed.

## Result

- Decode goes **O(T²) → O(T)** — per-token cost constant in context length.
- ~2.4× at a 24-token horizon (~1.4 → ~3.3 tok/s); the win grows without
  bound as the conversation lengthens.

## Verification (the exactness gate)

- Cached vs. uncached **greedy 24-token generation: token-identical**.
- Frontier activations vs. the full-forward reference: **≤ 1.3e-2** rel error
  (bf16 chunked-state tolerance; the GDN recurrence agreed to 2.7e-4).
- The J-lens matrices are applied read-only in the readout and never enter
  the generation forward, so none of this can change lens values.
