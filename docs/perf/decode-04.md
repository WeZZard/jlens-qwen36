# decode-04 — Prefill: lens warm-up at startup; chunk-size lever rejected

**Code:** `jlens_qwen/lens.py` (`JacobianLens.warm`), `jlens_qwen/serve.py`
(startup warm-up, `READOUT_CHUNK` env knob), `scripts/bench_decode.py`.

## The problem

The first request's prefill readout paid a one-time ~3.3 GB fp32→fp16
conversion + GPU upload of all 63 J matrices (measured 1.78 s cold),
memoized lazily inside the first `transport()` call. On top of that, the
readout chunked positions at CHUNK=8, and the hypothesis was that each
chunk re-reads the full J stack + lm_head, so larger chunks should
amortize weight traffic across positions.

## What landed

`JacobianLens.warm()` materializes the fp16 GPU copies eagerly; the
server calls it at startup (and the benchmark before measuring), so the
upload leaves request latency entirely:

| metric | before | after |
|---|---|---|
| prefill readout (45-tok prompt) | 92.6 ms/prompt-token | **39.4 ms/prompt-token** |

The chunk size became an env knob (`JLENS_READOUT_CHUNK`, default 8,
unchanged behavior).

## What was rejected — and what it revealed

Sweeping CHUNK ∈ {8, 16, 32, 64} on a 265-token prompt through the real
readout: **35.1 → 34.3 ms/prompt-token — noise.** The amortization
premise is false at these shapes: per-position cost is dominated by the
quantized unembed at ~30 ms per position *regardless of batch size* —
about 4× the compute floor (~8 ms) and 13× the bandwidth floor (~2.3 ms)
for `[64·P, 5120] × 4-bit [248k, 5120]`. The kernel is shape-bound, not
bandwidth-bound.

That negative result spawned the current top hypothesis (LEDGER H11):
replace the per-readout quantized unembed with a dense fp16 `W_U`
(+2.5 GB resident, bandwidth-bound ~9 ms estimated) or find a better
batching shape — worth ~20 ms on every generated token and every prefill
position.

## Verification

Gate 4/4 (`tests/test_decode_gate.py`) — warm-up and chunking are pure
restructuring; golden ids and tokens unchanged. Decode this run:
114.5 ms/token (delta vs decode-03's 118.1 is run-to-run variance; this
change affects prefill only).
