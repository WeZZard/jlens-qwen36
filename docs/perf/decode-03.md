# decode-03 — Exact blocked top-k for the readout

**Code:** `jlens_qwen/serve.py` (`_blocked_topk`, used by
`_readout_at_positions`).
**Benchmark:** `scripts/bench_decode.py` (added in the same iteration
series; see `LEDGER.md`).

## The problem

After decode-02, the per-token readout did ONE batched argsort over the
`[64, 1, 248k]` logits tensor — but MLX's `argsort` executes a **full
sort** on GPU, and so do `mx.argpartition` and `mx.topk` (measured: no
faster). Producing a total order of 248k entries to extract 10 cost
**14.7 ms per generated token** (11% of the 131.7 ms budget) and ~187 ms
of every 8-position prefill chunk.

## The fix

Two-stage exact selection, pure MLX ops:

1. Reshape the vocab axis into `B = 1024` blocks (padded to a block
   multiple with `-inf`) and take per-block maxima.
2. Keep the top-k *blocks* by max — provably sufficient: any top-k
   element has at most k−1 elements above it, which occupy at most k−1
   other blocks, so its own block ranks within the top-k blocks by max.
3. Full-sort only the surviving `k·C ≈ 2.4k` candidates and map indices
   back to vocab ids.

Exact, not approximate — the selection provably contains the true top-k,
and values match the argsort path bit-for-bit (ordering at exact score
ties may differ; the gate's golden data pins it).

## Result

| metric | before | after |
|---|---|---|
| readout.topk (per token, median) | 14.7 ms | **1.36 ms** |
| end-to-end decode | 131.7 ms/token (7.59 tok/s) | **118.1 ms/token (8.47 tok/s)** |
| prefill readout (45-tok prompt) | 110.6 ms/prompt-token | **92.6 ms/prompt-token** |

## Verification (the exactness gate)

`uv run python -m pytest tests/test_decode_gate.py` — 4/4 green:
golden top-10 ids **byte-identical** across all 192 golden cells, golden
64-token greedy decode unchanged, tie-aware batched-vs-naive equivalence
and streaming-detokenization identity unchanged. The gate was built and
mutation-checked *before* this change landed (see `LEDGER.md`, Gate).
