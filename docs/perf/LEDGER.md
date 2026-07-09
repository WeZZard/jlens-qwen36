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

**118.1 ms/token = 8.47 tok/s** @ H1 (blocked top-k), 2026-07-09 —
measured by `uv run python scripts/bench_decode.py` (median of 32 greedy
tokens, 45-tok prompt, 64 layers, top_n 10, full-depth lens, M4 Pro,
MLX 0.31.2). The bench drives the real production functions in-process;
decode-02's historical 141 ms/token additionally included SSE/HTTP
transport.

Measured split per generated token (call-level, exact; readout stages via
`JLENS_PERF=1`, medians):

| stage | ms | notes |
|---|---|---|
| forward (extend) | 71.8 | ~1.4× above the ~50 ms pure-bandwidth floor |
| readout.transport | 15.3 | next target (H7 int8: → ~7.7, needs-decision) |
| readout.unembed | 29.8 | **largest readout stage** (H2 fp16: → ~24, needs-decision) |
| readout.topk | 1.4 | was 14.7 — H1 landed (exact blocked selection) |
| readout.tolist | 0.1 | |
| sample + emit (detok+json) | 0.5 | CPU-side negligible at short T |
| **TOTAL** | **118.1** | 8.47 tok/s (was 131.7 @ d3d7bbe) |

Prefill readout: **1772 ms for 45 prompt tokens = 39.4 ms/prompt-token**
(was 110.6 → 92.6; the one-time lens fp16 upload — measured 1.78 s cold —
now happens at server startup via `JacobianLens.warm()`, excluded from
request latency). Steady-state prefill cost is ~35 ms/prompt-token on a
265-token prompt, **~30 ms of which is the unembed per position** — see
H11. Prefill forward: 677 ms. Instrumentation overhead when ON ~2%.

History of the headline number: 131.7 @ `d3d7bbe` (H0 re-measure) →
118.1 (H1) → 114.5 (H4 run; decode Δ vs H1 is run-to-run variance —
the H4 change affects prefill only).

## Backlog

| id | hypothesis | expected win | confidence | effort | status | evidence |
|----|-----------|--------------|------------|--------|--------|----------|
| H0 | Per-stage instrumentation + in-repo benchmark script; re-measure baseline | enables all | — | S | landed | `scripts/bench_decode.py` + `jlens_qwen/perf.py` + 4 flag-gated marks in `_readout_at_positions`; baseline 131.7 ms/token measured; gate 4/4 green with marks in place |
| G1 | Build the decode correctness gate (see Gate) | enables all | — | M | landed | 4 golden-based tests + generator, mutation-checked (2/2 caught); see Gate section |
| H1 | Replace full-vocab `argsort` with exact two-stage blocked top-k (block-max → top-K blocks → sort ~2.5k candidates) | ~14 ms/token (11%); big prefill win | high | S | landed | **e2e 131.7→118.1 ms/token (−10.3%); topk 14.7→1.36 ms; prefill readout 110.6→92.6 ms/pt; gate 4/4, golden ids byte-identical**; see decode-03.md |
| H2 | fp16 unembed activations + kill bf16→fp32→fp16 cast chain in `_readout_at_positions` | ~6 ms/token | medium | S | rejected | **falsified on REAL weights: fp16 acts make qmm SLOWER (31.9 vs 29.5 ms at [64,1,D])**; the synthetic stand-in had said 23.7 vs 29.7 — second stand-in-bias instance. No decision needed |
| H3 | Incremental streaming detokenizer (mlx_lm's) replacing per-token `tok.decode(gen_ids)` (O(T²)) and `_token_segments` (O(S²)) | O(1)/token at large T/S | high→— | S | rejected | **mlx_lm `BPEStreamingDetokenizer` is NOT byte-exact**: segment attribution differs (space lands on a different token → breaks UI column alignment) and on random-id cases concatenation ≠ `tok.decode` (drops leading space). AND the quadratic's constant is tiny: 18.6 ms total at T=512, ~0.6 ms/tok averaged at T=8192 (~0.5% of budget), ~1.3 s one-time at 4k-prompt prefill. A custom exact incremental detok isn't justified. Revisit only if generations routinely exceed ~8k tokens |
| H4 | Prefill readout: lens warm-up at startup + chunk-size amortization | prefill TTFT | high | S | landed | **warm-up landed: prefill readout 92.6 → 39.4 ms/prompt-token** (1.78 s one-time upload moved to startup). **Chunk lever REJECTED**: CHUNK 8→64 sweep on a 265-tok prompt = 35.1→34.3 ms/pt (noise) — readout is kernel-shape-bound, not bandwidth-bound (spawned H11). `READOUT_CHUNK` env knob kept, default 8. See decode-04.md |
| H4b | Prefill payload: per-chunk SSE snapshot frames + rounded scores in JSON | long-prompt serialization + browser parse (tens of MB at 1k tokens) | medium | M | needs-decision | changes the SSE event shape the client parses — UI contract decision |
| H11 | Unembed: dense-fp16 W_U or reshaped batching beats the qmm | ~20 ms/token | medium-high | M | rejected | **all variants measured on real weights at [64,1,D]**: 2D reshape 31.2 ms (worse, bit-identical), dense fp16 25.0 ms (−4.6 ms only, +2.5 GB resident, 2.0e-2 drift = 20× golden atol), dense fp32 33.7 ms (worse + a top-10 order flip). Even plain fp16 dense matmul is 2.7× off its 9.3 ms bandwidth floor at skinny M — the op class is kernel-limited on MLX 0.31.2; only a custom kernel changes it (→ H12) |
| H12 | Custom Metal skinny-matmul (M≈64) kernel for the unembed — same class of work as the shipped GDN VJP kernel | up to ~20 ms/token AND ~20 ms/prompt-token (floor 9.3 ms vs 29.5 measured) | medium | L | open | H11 experiment: MLX's qmm and dense matmul both kernel-limited at M=64; project precedent: `custom_gdn_vjp.py`. Numerics: exact-dequant fp32 accumulate can match qmm within tie-aware EPS; verify via gate |
| H5 | Pipeline: sample on-GPU, `mx.async_eval` next `extend`, overlap current token's tolist/JSON/SSE with GPU | ~0.5 ms/token | — | M | rejected | in-situ: CPU-side (sample+emit) = 0.5 ms/token — there is nothing to hide; GPU work is serial on one queue regardless. Premise falsified by H0's measurement |
| H6 | Run generation on a worker thread (asyncio.Queue → SSE) so `/api/chat_control` (pause) and other endpoints stay responsive | responsiveness, not tok/s | high | M | open | all MLX work blocks the uvicorn event loop; worst during multi-second prefill |
| H7 | int8-quantize J transport (`mx.quantized_matmul`) | ~7 ms/token; lens RAM 3.3→1.7 GB | medium | M | rejected | **re-validated on the REAL lens: speed holds at P=1 (15.9→8.6 ms) but P=8 regresses (21.1→22.3), and rank stability collapses — only 80/192 cells keep top-10 order, 32/192 (17%) change top-10 MEMBERSHIP, worst common-id drift 0.2513 (50× tie-aware EPS), boundary gaps to ±0.10.** The synthetic 0.5%-rel-err estimate measured the tensor, not the decision: real J structure aligns with real activations. Not a tolerance call — a measured product-quality regression for a 6.4% win. Third stand-in-bias instance |
| H8 | Speculative decoding: small draft model + batched verification (capture-aware); amortizes 13.5 GB weight read AND readout J/lm_head traffic across accepted tokens | forward 72→~45–55 ms effective | low-medium | L | open | **in-situ: forward = 72.2 ms/token** (not ~100 as estimated) — ~1.4× above the ~50 ms pure-bandwidth floor; smaller but still the largest single lever after readout work |
| H9 | Top-1-only streaming band + lazy top-10 on cell click | prefill readout ~30× (argmax 0.47 ms vs 14.6) | high | L | needs-decision | changes SSE/UI contract; superseded in part by H1 |
| C1 | `/api/slice`: rewire to `_readout_at_positions` or delete (old per-position argsort + per-score `.tolist()` syncs; UI never calls it) | cleanup | high | S | open | web/index.html only calls `/api/chat_stream` |

Suggested order: H3 → H6 → C1 (small hygiene) → then the strategic fork:
H8 (speculative decode, ~25 ms, L) vs H12 (custom skinny-matmul kernel,
~20 ms, L) — both week-scale; pick ONE with the user. Every cheap decode
lever is now exhausted or rejected on evidence; readout floor with
current kernels ≈ transport 15.9 + unembed 29.5 + topk 1.4 ≈ 47 ms.
Remaining decision items are UI-contract only (H4b, H9).

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
- 2026-07-09 — H1 landed: exact blocked two-stage top-k
  (`serve._blocked_topk`, B=1024, −inf padding for the non-divisible
  vocab). **131.7 → 118.1 ms/token (8.47 tok/s, −10.3%)**; topk stage
  14.7 → 1.36 ms/token; prefill readout 110.6 → 92.6 ms/prompt-token.
  Gate 4/4 — golden top-10 ids byte-identical across all 192 cells
  (exactness proof held on real data, including near-ties). Iteration
  doc: decode-03.md.
- 2026-07-09 — H4 landed (warm-up) + chunk lever rejected: lens fp16
  upload (measured 1.78 s) moved to startup (`JacobianLens.warm()`);
  **prefill readout 92.6 → 39.4 ms/prompt-token**. CHUNK 8→64 sweep on a
  265-tok prompt: 35.1→34.3 ms/pt = noise → amortization premise
  falsified; readout is kernel-shape-bound in the unembed (~30 ms/position
  at any P) → spawned H11 (dense-fp16 W_U / reshaped batching), now the
  top open item. Decode 114.5 ms/token this run (Δ vs 118.1 = variance).
  Gate 4/4. Iteration doc: decode-04.md.
- 2026-07-09 — H11 REJECTED + H2 REJECTED (collateral), H12 spawned.
  Six unembed variants measured on the REAL lm_head at [64,1,D]:
  current qmm 29.5 ms; 2D reshape 31.2 (bit-identical, slower); fp16
  acts 31.9 (**falsifies H2 — synthetic stand-in had promised 23.7**);
  dense fp16 W_U 25.0 (−4.6 ms for +2.5 GB resident and 2.0e-2 drift —
  bad trade); dense fp32 33.7. Even plain fp16 dense matmul is 2.7× off
  its 9.3 ms bandwidth floor at skinny M → the op class is
  kernel-limited on MLX 0.31.2; a custom Metal kernel is the only path
  (H12, L effort, GDN-VJP precedent). H7's synthetic-J numbers flagged
  for re-validation on the real lens before the user decision. No code
  changes this iteration.
- 2026-07-09 — H7 REJECTED on real-lens re-validation (third stand-in
  hit). Speed holds at P=1 (15.9→8.6 ms) but regresses at P=8
  (21.1→22.3); rank stability collapses: 80/192 cells identical top-10
  order, 32/192 change membership, worst common-id drift 0.2513 with
  boundary gaps to ±0.10 — a product-quality regression, not a tolerance
  call; not presented for approval. Lens int8 also would have halved
  lens RAM (3.30→1.75 GB) — moot. No code changes.
- 2026-07-09 — H3 REJECTED: mlx_lm's BPEStreamingDetokenizer is not
  byte-exact vs `tok.decode` (segment attribution shifts; leading space
  dropped on subsequence cases — concat mismatch on 2/8 torture cases),
  and the O(T²) constant is immaterial at realistic horizons (18.6 ms
  total at T=512; ~0.6 ms/tok averaged at T=8192 ≈ 0.5% of budget;
  ~1.3 s one-time at 4k-prompt prefill). Custom exact incremental detok
  not justified; revisit above ~8k-token generations. No code changes.
