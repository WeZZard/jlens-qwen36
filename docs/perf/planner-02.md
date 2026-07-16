# Planner latency 02 — bounded recipe replay throughput

**Date:** 2026-07-16
**Machine:** Apple M4 Pro, 64 GiB unified memory
**Model:** `mlx-community/Qwen3.6-27B-4bit`
**Lens configuration:** Neuronpedia n1000, measured workspace L26–L59

## Question

How many intervention recipes can the application actually replay and verify
inside conservative one-minute and three-minute search budgets?

This follows [`planner-01.md`](planner-01.md), which found that asking the model
to propose coordinates takes about 15 seconds with a compact brief and that
injecting the complete paper cannot finish prefill within three minutes.

## Verification path

`POST /api/intervention_search` evaluates caller-supplied recipes. A candidate
is one complete recipe containing one to eight exact
`(position, layer, alpha)` cells. Each candidate:

1. compiles every exact cell independently (no layer×position Cartesian
   product),
2. creates a fresh bare `StreamSession`,
3. prefills the same chat-template prompt without activation capture or lens
   readout,
4. greedily decodes with a normal 64-token allowance and repetition breaker,
5. is verified only after EOS when the requested phrase occurs exactly once
   and the old phrase is absent.

The endpoint checks its wall-clock deadline between atomic candidates and lets
an in-flight MLX call finish safely. It reports deadline overshoot instead of
cancelling GPU work behind the lock.

The benchmark driver is
[`scripts/benchmark_intervention_search.py`](../../scripts/benchmark_intervention_search.py).
It asserts the live n1000 configuration, reads the measured workspace band,
and searches exact positions only. α is restricted to 1 and 0.5 for single
cells; combination cells use α=0.5. Broad `from_position` edits and α>1 are
excluded from this conservative condition.

## One-minute single-cell matrix

Three deterministic contradiction prompts from
[`data/benchmarks/jspace_contradiction_prompts.json`](../../data/benchmarks/jspace_contradiction_prompts.json)
were tested. `literal` uses the user-visible reply direction (`Paris→Beijing`,
`8→6`, `A→B`). `oracle` uses the paper-derived latent direction stored only in
evaluation metadata (`France→China`, `spider→ant`, `repeat→switch`).

```sh
uv run python scripts/benchmark_intervention_search.py \
  --cases country_capital,spider_ant,bandit_repeat_switch \
  --direction literal --budget-seconds 60 --max-tokens 64 \
  --recipe-sizes 1 --position-limit 8 --max-candidates 256 \
  --stop-on-success --output /tmp/intervention-search-literal-60.json

uv run python scripts/benchmark_intervention_search.py \
  --cases country_capital,spider_ant,bandit_repeat_switch \
  --direction oracle --budget-seconds 60 --max-tokens 64 \
  --recipe-sizes 1 --position-limit 8 --max-candidates 256 \
  --stop-on-success --output /tmp/intervention-search-oracle-60.json
```

| Direction | Case | Tested | Wall time | Rate | Median candidate | Verified |
|---|---|---:|---:|---:|---:|---|
| Literal | Country capital | 1 | 0.96 s | — | 508 ms | **L59, p25, α=1** |
| Literal | Spider → ant | 145 | 60.16 s | 144.6/min | 371 ms | none |
| Literal | Repeat → switch | 94 | 60.36 s | 93.4/min | 633 ms | none |
| Oracle | Country capital | 164 | 60.29 s | 163.2/min | 360 ms | none |
| Oracle | Spider → ant | 161 | 60.25 s | 160.3/min | 370 ms | none |
| Oracle | Repeat → switch | 97 | 60.57 s | 96.1/min | 616 ms | none |

The successful capital candidate returned exactly `Beijing`, reached EOS,
contained no `Paris`, and matched the literal baseline replacement. It spent
355 ms in prefill, 149 ms in decode, and 4 ms compiling the edit.

All unsuccessful candidates also reached EOS cleanly; they were classified as
unchanged rather than partial, off-target, or repetitive. Prompt length drove
the throughput range: the 45-token bandit prompt costs roughly 0.62 seconds per
candidate, while the shorter country/spider prompts cost roughly 0.36–0.37
seconds after warm-up.

The one-minute deadline overshot by only 0.16–0.57 seconds—the duration of the
atomic candidate that was already in flight.

## Three-minute combination stress test

The failed literal spider case was used to measure multi-cell cost. A static
combination sweep is intentionally more permissive than the recommended search
policy, which should only combine promising non-degenerate singles.

```sh
uv run python scripts/benchmark_intervention_search.py \
  --cases spider_ant --direction literal --budget-seconds 180 \
  --max-tokens 64 --recipe-sizes 2 --position-limit 8 \
  --max-candidates 256 --stop-on-success \
  --output /tmp/intervention-search-spider-pairs-180.json

# Spend the unused remainder of the 180-second staged budget on triples.
uv run python scripts/benchmark_intervention_search.py \
  --cases spider_ant --direction literal --budget-seconds 82 \
  --max-tokens 64 --recipe-sizes 3 --position-limit 8 \
  --max-candidates 256 --stop-on-success \
  --output /tmp/intervention-search-spider-triples-82.json
```

| Recipe size | Tested | Wall time | Rate | Median candidate | p95 | Verified |
|---:|---:|---:|---:|---:|---:|---|
| 2 cells | 256 | 97.19 s | 158.0/min | 384 ms | 400 ms | none |
| 3 cells | 216 | 82.25 s | 157.6/min | 378 ms | 399 ms | none |
| **Staged total** | **472** | **179.45 s** | **157.8/min** | — | — | **none** |

The staged run repeats one baseline between phases, so its throughput is a
slightly conservative approximation of one continuous three-minute search.
Adding cells barely changed latency: median compile time rose from about 1.4 ms
for one cell to 1.9 ms for pairs and 2.5 ms for triples; prompt prefill still
dominates.

## Decision

A wall-clock-bounded conservative search is feasible and more useful than
spending the budget on textual coordinate prediction:

- **60 seconds is a sensible default.** It currently fully replays and judges
  roughly 90–160 one-cell recipes for short contradiction prompts and can
  return immediately on a verified success.
- **180 seconds is a valid thorough mode.** On the measured short prompt it
  covered 472 pair/triple recipes with sub-second deadline overshoot.
- **Throughput does not imply coverage.** Five of six one-minute conditions
  found nothing, and 472 additional combinations did not rescue the spider
  edit. Coordinate breadth cannot repair an ineffective concept direction,
  strength, or candidate-ranking policy.
- **Only EOS-complete replay is a result.** Short scan greens and orange
  partials remain leads. Max-token, repetition, and deadline stops are never
  verified.

The next search controller should be anytime and adaptive: coarse single cells
first, refine neighboring layers around changed non-degenerate outcomes, then
form a small pair beam; triples belong only in thorough mode after a pair shows
measurable improvement. If the deadline expires, report that no verified
recipe was found within the budget rather than presenting the best partial as
a solution.
