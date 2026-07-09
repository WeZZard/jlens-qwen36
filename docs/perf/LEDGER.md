# Perf ledger — decode subsystem

Live state for the `performance-optimize` skill (one iteration per
invocation; see `~/.claude/skills/performance-optimize/SKILL.md`). Landed
iterations get a `decode-NN.md` doc per this directory's convention.

## Target

- Metric: steady-state chat decode speed (ms/token) over `/api/chat_stream`
  with the full 63-layer lens attached; secondary: prefill readout latency
  per prompt token.
- Workload: greedy decode, `top_n=10`, bundled full-depth lens
  (`data/lens/lens.npz`), Qwen3.6-27B-4bit.
- Machine: Apple M4 Pro, 64 GB.
- Stop threshold: best remaining expected win < 5% of token budget (~7 ms).

## Gate

**Status: PRESENT.** Command:

```
uv run python -m pytest tests/test_decode_gate.py -v
```

(pytest added as a dev dependency.) Golden-based:
`tests/golden/decode_gate_golden.json` freezes current behavior
(64 greedy tokens via the cached path + top-10 readout at 3 positions ×
64 layers with the full-depth lens). Regenerate via
`scripts/gen_decode_gate_golden.py` ONLY as a deliberate act: when a
numerics-changing optimization has an approved tolerance recorded here,
or when the model / lens / MLX version changes.

Four invariants:

1. **Golden token identity** — cached greedy decode reproduces the frozen
   64-token sequence exactly.
2. **Golden readout identity** — `_readout_at_positions` reproduces the
   frozen top-10 ids exactly (scores atol 1e-3).
3. **Batched-vs-naive readout equivalence, tie-aware** — EPS 5e-3,
   calibrated over 192 cells: worst common-id score diff 1.089e-03,
   worst boundary-tie gap 2.8e-05 (2026-07-09, MLX 0.31.2).
4. **Streaming detokenization identity** on UTF-8-splitting samples
   (emoji ZWJ/flags, CJK, mixed scripts).

Mutation-checked 2026-07-09: M1 (drop the top-k reversal in
`serve._readout_at_positions`) → 2 tests FAIL; M2 (skip the final norm in
`StreamSession.extend`) → token identity FAILS. Both reverted; clean run
4/4 green in ~14 s warm.

Design note (measured, not assumed): cached-vs-uncached token identity is
NOT a valid exact gate — near-tie flip (`'\n'` vs `'\n\n'`) at step 21/64
on the gate prompt; and batched-vs-naive readout swaps near-tied ranks
because different matmul batch shapes drift scores up to ~1.1e-03. Hence
golden self-consistency of the shipped path as the primary gate.

Tolerance approvals on file: none yet (H2/H7 need a user decision;
landing either requires a documented tolerance here + golden regen).

## Baseline

**131.7 ms/token = 7.59 tok/s** @ `d3d7bbe`, 2026-07-09 — measured by
`uv run python scripts/bench_decode.py` (median of 32 greedy tokens,
45-tok prompt, 64 layers, top_n 10, full-depth lens, M4 Pro, MLX 0.31.2).
The bench drives the real production functions in-process; decode-02's
141 ms/token additionally included SSE/HTTP transport.

Measured split per generated token (call-level, exact; readout stages via
`JLENS_PERF=1`, medians):

| stage | ms | notes |
|---|---|---|
| forward (extend) | 72.2 | was estimated ~100 — headroom over pure-bandwidth floor is smaller than assumed |
| readout.transport | 15.5 | matches microbench 14.2 |
| readout.unembed | 29.9 | **largest readout stage**; matches the synthetic stand-in (~24–30) |
| readout.topk | 14.7 | matches microbench 14.6 |
| readout.tolist | 0.1 | |
| sample + emit (detok+json) | 0.5 | CPU-side is negligible at short T |
| **TOTAL** | **131.7** | 7.59 tok/s |

Prefill readout: **4976 ms for 45 prompt tokens = 110.6 ms/prompt-token**
(6 chunks ≈ 825 ms avg; the first chunk includes the one-time lens
fp16 memoization upload, so steady-state per-chunk cost needs a separate
measurement — see H4). Prefill forward: 637 ms. Instrumentation overhead
when ON: ~2% (134.0 vs 131.7); zero when off (default).

## Backlog

| id | hypothesis | expected win | confidence | effort | status | evidence |
|----|-----------|--------------|------------|--------|--------|----------|
| H0 | Per-stage instrumentation + in-repo benchmark script; re-measure baseline | enables all | — | S | landed | `scripts/bench_decode.py` + `jlens_qwen/perf.py` + 4 flag-gated marks in `_readout_at_positions`; baseline 131.7 ms/token measured; gate 4/4 green with marks in place |
| G1 | Build the decode correctness gate (see Gate) | enables all | — | M | landed | 4 golden-based tests + generator, mutation-checked (2/2 caught); see Gate section |
| H1 | Replace full-vocab `argsort` with exact two-stage blocked top-k (block-max → top-K blocks → sort ~2.5k candidates) | ~14 ms/token (11%); big prefill win | high | S | open | **in-situ: topk = 14.7 ms/token median**; microbench 14.6→0.7–0.8 ms at [64,1,248k], ids identical; [64,8,248k] 112→3.2 ms; `mx.argpartition`/`mx.topk` are NOT faster (full sort); numpy 7× slower |
| H2 | fp16 unembed activations + kill bf16→fp32→fp16 cast chain in `_readout_at_positions` | ~6 ms/token | medium | S | needs-decision | **in-situ: unembed = 29.9 ms/token — largest readout stage**; 29.7→23.7 ms on synthetic stand-in; may flip near-tied ranks → tolerance decision + golden regen required |
| H3 | Incremental streaming detokenizer (mlx_lm's) replacing per-token `tok.decode(gen_ids)` (O(T²)) and `_token_segments` (O(S²)) | O(1)/token; matters only at large T/S | high | S | open | in-situ at T=32: emit = 0.3 ms (invisible); the win is long conversations and multi-k-token prompts, not steady-state tok/s |
| H4 | Prefill readout: measure steady-state chunk cost (exclude one-time lens upload), raise CHUNK, fp16 logits, per-chunk SSE frames | prefill TTFT: measured 110.6 ms/prompt-token | medium→high | M | open | **in-situ: 4976 ms readout for a 45-tok prompt** (6 chunks ≈ 825 ms avg incl. one-time lens fp16 memoization in chunk 1); at 1k-tok prompts this extrapolates to ~2 min — top UX pain |
| H5 | Pipeline: sample on-GPU, `mx.async_eval` next `extend`, overlap current token's tolist/JSON/SSE with GPU | ~0.5 ms/token | — | M | rejected | in-situ: CPU-side (sample+emit) = 0.5 ms/token — there is nothing to hide; GPU work is serial on one queue regardless. Premise falsified by H0's measurement |
| H6 | Run generation on a worker thread (asyncio.Queue → SSE) so `/api/chat_control` (pause) and other endpoints stay responsive | responsiveness, not tok/s | high | M | open | all MLX work blocks the uvicorn event loop; worst during multi-second prefill |
| H7 | int8-quantize J transport (`mx.quantized_matmul`) | ~7 ms/token; lens RAM 3.3→1.7 GB | medium | M | needs-decision | measured 14.2→7.7 ms, max rel err 0.5% (int4: 4.1 ms but 7.5% err — likely too lossy); needs rank-stability gate + user tolerance approval |
| H8 | Speculative decoding: small draft model + batched verification (capture-aware); amortizes 13.5 GB weight read AND readout J/lm_head traffic across accepted tokens | forward 72→~45–55 ms effective | low-medium | L | open | **in-situ: forward = 72.2 ms/token** (not ~100 as estimated) — ~1.4× above the ~50 ms pure-bandwidth floor; smaller but still the largest single lever after readout work |
| H9 | Top-1-only streaming band + lazy top-10 on cell click | prefill readout ~30× (argmax 0.47 ms vs 14.6) | high | L | needs-decision | changes SSE/UI contract; superseded in part by H1 |
| C1 | `/api/slice`: rewire to `_readout_at_positions` or delete (old per-position argsort + per-score `.tolist()` syncs; UI never calls it) | cleanup | high | S | open | web/index.html only calls `/api/chat_stream` |

Suggested order (re-ranked on measured data): H1 → H4 → H7 → H3 → H6 →
H2 (pending decision) → H8. Projected: H1 lands ~117 ms/token
(~8.5 tok/s); +H7 ≈ ~110 ms (~9.1 tok/s); +H2 (if approved) ≈ ~104 ms
(~9.6 tok/s). Prefill readout is now the top UX pain (110.6 ms/prompt-
token) — H1 + H4 attack it directly. Past ~10 tok/s requires H8.

## History

- 2026-07-09 — Seeded from an analysis session: read decode path
  (`serve.py`, `model.py`, `lens.py`), microbenchmarked readout components
  at real shapes (L=64, D=5120, V=248k, K=10) on M4 Pro / MLX 0.31.2.
  Validated H1's blocked top-k exact-identical to argsort. No code changes.
- 2026-07-09 — G1 landed: decode correctness gate (4 golden-based tests +
  `scripts/gen_decode_gate_golden.py`), mutation-checked (M1 readout
  ordering, M2 skipped final norm — both caught, both reverted; clean run
  4/4). Found along the way: cached-vs-uncached greedy decode near-tie
  flip at step 21/64; batched-vs-naive readout score drift ≤1.1e-03 with
  near-tie rank swaps → gate design switched to golden self-consistency.
  Also: server default `JLENS_PATH=data/lens/lens.npz` does not exist on
  this machine — the real lens is `full_depth_analytic.npz` (gate honors
  `JLENS_PATH`). pytest added as dev dep. No hot-path changes.
- 2026-07-09 — H0 landed: `scripts/bench_decode.py` (in-repo benchmark)
  + `jlens_qwen/perf.py` (flag-gated stage timers) + 4 marks in
  `_readout_at_positions`. Baseline re-measured: **131.7 ms/token =
  7.59 tok/s** @ d3d7bbe (forward 72.2, readout 59.0 = transport 15.5 +
  unembed 29.9 + topk 14.7, CPU-side 0.5). Prefill readout 110.6 ms/
  prompt-token — top UX pain. Re-ranked backlog on data: H5 REJECTED
  (premise falsified: CPU-side is 0.5 ms, not 5–10), H2 → needs-decision
  and rank up (unembed is largest readout stage), H8 win revised down
  (forward 72 not 100), H3 deprioritized for tok/s (invisible at T=32).
  Gate 4/4 green with marks in place; instrumentation overhead 0 when
  off, ~2% when on.
