# decode-02 — Batched per-token readout across layers

**Commit:** `59e5add` — "Batch the per-token readout across layers: one
unembed instead of 64".
**Code:** `jlens_qwen/serve.py:_readout_at_positions`.

## The problem

After decode-01, the model forward was fast and constant-time — so the
*readout* became the dominant per-token cost. `_readout_at_positions` looped
over 64 layers and, for **each**, did its own `final_norm` + unembed +
argsort + two `.tolist()` GPU→CPU syncs.

That means per generated token:

- the ~636 MB quantized `lm_head` was read **64 times** (~40 GB of memory
  traffic),
- ~64 separate argsorts over the 248k-token vocab,
- ~128 GPU↔CPU round-trips.

Estimated 300–600 ms/token — more than the model forward itself.

## The fix

Pure batching, identical math:

- Stack the 64 transported vectors into one `[L, P, D]` tensor.
- **One** `final_norm`, **one** unembed (lm_head read once: 40 GB → 0.64 GB),
  **one** batched argsort `[L, P, vocab]`, **one** sync.
- Position-chunked (CHUNK=8) so the `[L, P, vocab]` logits tensor stays
  ~0.5 GB.
- Token-id → string decode cache (640 `tokenizer.decode` calls/token → ~0
  after warmup).

## Result

- End-to-end SSE timing with the full 63-layer workspace band attached:
  **141 ms/token = 7.07 tok/s, flat** with context length.
- Progression across the two decode iterations, same prompt:

  | Stage | tok/s |
  |-------|-------|
  | Original O(T²) re-forwards | ~0.3–0.6, degrading |
  | After decode-01 (cached streaming) | ~1.5–2.5 |
  | After decode-02 (batched readout) | **7.1, flat** |

At 141 ms/token we are near the floor: single-token decode through the 27B
4-bit weights is bandwidth-bound at ~100 ms on the M4 Pro, so the readout now
adds only ~30–40 ms.

## Verification

Values verified **identical** to the old per-layer implementation — ids,
tokens, and scores matched to **0.0** difference on a synthetic harness
before deploying. Same argsort ordering, same top-k, so the workspace band
is byte-for-byte unchanged.
