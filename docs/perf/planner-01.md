# Planner latency 01 — contradiction prompts and paper-context cost

**Date:** 2026-07-16
**Commit measured:** `f5f0a96` plus the working-tree planner probe
**Machine:** Apple M4 Pro, 64 GiB unified memory, macOS 27.0
**Runtime:** MLX 0.31.2, Transformers 5.13.0
**Model:** `mlx-community/Qwen3.6-27B-4bit`

## Question

Can the resident model propose backward J-Space intervention recipes quickly
enough for an interactive UI, and is injecting the complete workspace paper a
viable way to teach it the task?

The versioned contradiction corpus is
[`data/benchmarks/jspace_contradiction_prompts.json`](../../data/benchmarks/jspace_contradiction_prompts.json).
It adapts examples from the paper's country generalization, latent animal,
rhyme-planning, repeat/switch, and multilingual experiments. The compact task
description is
[`data/benchmarks/jspace_planner_brief.md`](../../data/benchmarks/jspace_planner_brief.md).

## Measurement path

`POST /api/planner_probe` reuses the already-loaded model, applies the real chat
template, and runs a bare cached stream. It does **not** capture activations or
run J-lens readout (`lens_readout: false`). Tokenization, queue wait, chunked
prefill, first-token latency, and decode are timed separately. Inputs are
rejected rather than silently truncated.

The server was verified to have the n1000 lens configuration and workspace band
L26–L59. That check establishes configuration provenance only: the probe does
not use lens weights, so n1000 versus n20 cannot affect planner inference time.

The comparison used three representative contradiction cases:

- `country_capital`
- `spider_ant`
- `bandit_repeat_switch`

Each condition ran three times at temperature 0 with thinking disabled,
`max_tokens=160`, 1,024-token prefill chunks, and a 60-second per-request
budget. Condition order alternated by case and repetition. Local tokenizer
setup (0.65 seconds) and one warm-up request were excluded.

```sh
uv run python scripts/benchmark_planner_latency.py \
  --guidance minimal,distilled \
  --cases country_capital,spider_ant,bandit_repeat_switch \
  --repetitions 3 \
  --max-tokens 160 \
  --time-budget-seconds 60 \
  --request-timeout-seconds 90 \
  --output /tmp/jspace-planner-counterbalanced.json
```

## Compact-context results

| Guidance | Requests | Median request | p95 request | Median prefill | Median input | Median output | JSON | Contract-valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Minimal (152 guidance tokens) | 9 | 13.68 s | 16.24 s | 5.80 s | 657 tok | 95 tok | 9/9 | 0/9 |
| Distilled (355 guidance tokens) | 9 | 15.07 s | 17.36 s | 7.58 s | 859 tok | 91 tok | 9/9 | 9/9 |

The 203-token task brief added 1.38 seconds to median end-to-end request time
and 1.78 seconds to median prefill. All 18 generations reached EOS.

The minimal condition's outputs were structurally useful but returned numeric
confidence values (`0.7`/`0.8`) instead of the declared `low | medium | high`
enum. The distilled condition followed the full output and causal-position
contract in every run.

Median request time by case:

| Case | Minimal | Distilled |
|---|---:|---:|
| Country capital | 12.75 s | 14.74 s |
| Spider → ant | 13.68 s | 15.07 s |
| Repeat → switch | 15.41 s | 16.47 s |

These are planner-request measurements over canned conversations. They exclude
baseline response generation, recipe search/replay, and UI rendering. A
contract-valid coordinate is still only a hypothesis until intervention replay
verifies the requested response.

## Complete-paper condition

The official article was downloaded at run time and converted to visible text;
no copyrighted copy is stored in this repository. Extraction produced 69,284
paper tokens. With the final planner contract and case payload, the request had
70,136 input tokens.

```sh
uv run python scripts/benchmark_planner_latency.py \
  --guidance paper \
  --cases country_capital \
  --repetitions 1 \
  --paper-token-limit 0 \
  --max-tokens 160 \
  --time-budget-seconds 180 \
  --request-timeout-seconds 240 \
  --output /tmp/jspace-planner-full-paper.json
```

| Budget | Processed | Fraction | Observed prefill rate | First token | Output |
|---:|---:|---:|---:|---:|---:|
| 180.07 s | 18,432 / 70,136 tok | 26.3% | 102.36 tok/s | none | 0 tok |

At the observed average rate, a linear projection is about **685 seconds
(11.4 minutes) just to finish prefill**. This is an optimistic estimate because
attention work can grow as the cached context becomes longer. The complete
paper therefore cannot satisfy a 1–3 minute conservative search limit on this
machine.

This paper condition is valid for latency only. The paper explicitly contains
several oracle swaps used by the core prompt cases, so comparing its recipe
quality against minimal or distilled guidance would leak answers.

## Decision

Use the compact, distilled planner brief for the first trial. Its measured
latency is about 15 seconds and it reliably obeys the machine-readable contract.
Do not inject the entire paper per request. If the brief proves insufficient,
retrieve only a small, relevant paper fragment and test that fragment as a
separate condition; keep recipe replay as the source of truth.

An initial six-case cold smoke test with a 96-token output cap is intentionally
excluded from the table: every response was truncated before closing its JSON.
That failure led to the compact one-recipe contract used above.
