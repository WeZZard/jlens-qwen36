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

- 141 ms/token = 7.07 tok/s, flat in context length — from `decode-02.md`
  (commit `59e5add`). **Quoted, not re-measured; Bootstrap must re-measure
  with per-stage instrumentation before the first land.**
- Estimated split (microbench evidence, 2026-07-09, MLX 0.31.2): forward
  ~100 ms; readout ≈ transport ~14 ms + unembed ~24–30 ms (synthetic-weight
  stand-in, confirm in situ) + top-k ~14.6 ms; CPU-side (tolist/JSON/SSE/
  detok) a few ms.

## Backlog

| id | hypothesis | expected win | confidence | effort | status | evidence |
|----|-----------|--------------|------------|--------|--------|----------|
| H0 | Per-stage instrumentation + in-repo benchmark script; re-measure baseline | enables all | — | S | open | prerequisite; decode-02 numbers are historical |
| G1 | Build the decode correctness gate (see Gate) | enables all | — | M | landed | 4 golden-based tests + generator, mutation-checked (2/2 caught); see Gate section |
| H1 | Replace full-vocab `argsort` with exact two-stage blocked top-k (block-max → top-K blocks → sort ~2.5k candidates) | ~13 ms/token; prefill chunk 112→3.2 ms | high | S | open | measured 14.6→0.7–0.8 ms at [64,1,248k], ids identical; [64,8,248k] 112→3.2 ms; `mx.argpartition`/`mx.topk` are NOT faster (full sort); numpy path 7× slower |
| H2 | fp16 unembed activations + kill bf16→fp32→fp16 cast chain in `_readout_at_positions` | ~6 ms/token | medium | S | open | 29.7→23.7 ms on synthetic 4-bit lm_head stand-in; stand-in bias risk; may flip near-tied ranks → needs tolerance decision |
| H3 | Incremental streaming detokenizer (mlx_lm's) replacing per-token `tok.decode(gen_ids)` (O(T²)) and `_token_segments` (O(S²)) | O(1)/token; seconds on multi-k-token prompts | high | S | open | serve.py:784, serve.py:333; cumulative-decode cost grows linearly per token |
| H4 | Prefill: raise CHUNK 8→32 (fp16 logits), per-chunk SSE snapshot frames, round scores in JSON | long-prompt time-to-first-token; MB-scale SSE frames | medium | M | open | [64,8,248k] fp32 = 508 MB/chunk; single prefill snapshot JSON can hit tens of MB at 1k tokens |
| H5 | Pipeline: sample on-GPU, `mx.async_eval` next `extend`, overlap current token's tolist/JSON/SSE with GPU | ~5–10 ms/token | medium | M | open | loop is fully serial today (serve.py:764–808); mlx_lm generate_step pattern |
| H6 | Run generation on a worker thread (asyncio.Queue → SSE) so `/api/chat_control` (pause) and other endpoints stay responsive | responsiveness, not tok/s | high | M | open | all MLX work blocks the uvicorn event loop; worst during multi-second prefill |
| H7 | int8-quantize J transport (`mx.quantized_matmul`) | ~7 ms/token; lens RAM 3.3→1.7 GB | medium | M | needs-decision | measured 14.2→7.7 ms, max rel err 0.5% (int4: 4.1 ms but 7.5% err — likely too lossy); needs rank-stability gate + user tolerance approval |
| H8 | Speculative decoding: small draft model + batched verification (capture-aware); amortizes 13.5 GB weight read AND readout J/lm_head traffic across accepted tokens | forward ~100→60–80 ms effective | low-medium | L | open | forward is ~2× above the ~50 ms pure-bandwidth floor on M4 Pro; largest lift, acceptance-rate dependent |
| H9 | Top-1-only streaming band + lazy top-10 on cell click | prefill readout ~30× (argmax 0.47 ms vs 14.6) | high | L | needs-decision | changes SSE/UI contract; superseded in part by H1 |
| C1 | `/api/slice`: rewire to `_readout_at_positions` or delete (old per-position argsort + per-score `.tolist()` syncs; UI never calls it) | cleanup | high | S | open | web/index.html only calls `/api/chat_stream` |

Suggested order: H0 → G1 → H1 → H3 → H2 → H4 → H5 → H6 → H7 → H8.
Projected after H1–H7: ~100–110 ms/token (~9–10 tok/s), prefill readout
~10× faster. Past 10 tok/s requires H8.

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
