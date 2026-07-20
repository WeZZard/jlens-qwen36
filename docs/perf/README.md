# Performance optimization history

This directory retells how jlens-qwen36 was made fast, subsystem by
subsystem. Each file is one iteration:

```
docs/perf/{subsystem}-{NN}.md
```

`NN` is the iteration ordinal **within that subsystem**. Read a subsystem's
files in order to see how each change built on the last.

[`LEDGER.md`](LEDGER.md) is the *live* state of ongoing optimization work —
target, correctness gate, measured baseline, and the ranked hypothesis
backlog — maintained by the `performance-optimize` skill (one hypothesis
tested per iteration). This README stays the record of what landed.

## Subsystems

### `fit` — the J-lens fitting pipeline

Computing the per-layer Jacobians `M_ℓ = d(h_{ℓ+1})/d(h_ℓ)` and chaining
them into the lens. This is where the bulk of the compute lived: a naive
fit was ~8 hours; the final analytic pipeline fits a full-depth,
intervention-grade lens in ~2.75 hours.

| # | Iteration | Headline |
|---|-----------|----------|
| [fit-01](fit-01.md) | Baseline VJP pipeline | chain-multiply + `mx.compile` + short seqs + custom Metal GDN VJP |
| [fit-02](fit-02.md) | Analytic MLP branch (Hadamard trick) | ~77× on the MLP Jacobian |
| [fit-03](fit-03.md) | Closed-form final-norm Jacobian | ~12000× on `J_norm` |
| [fit-04](fit-04.md) | Analytic attention branch | ~7–14× per layer, exact, replaces per-cotangent VJP |
| [fit-05](fit-05.md) | Metal GDN backward v4 (`dg`/`dβ`) | decay-gate paths at kernel speed → full fit 2.75 h |

### `decode` — generation & serving

Making the chat viewer generate tokens (and their J-lens readouts) fast
enough to feel live.

| # | Iteration | Headline |
|---|-----------|----------|
| [decode-01](decode-01.md) | Cached incremental decoding + readout fixes | O(T²) → O(T); ~0.3 → ~3 tok/s |
| [decode-02](decode-02.md) | Batched per-token readout | one unembed for all 63 layers; → ~7 tok/s |
| [decode-03](decode-03.md) | Exact blocked top-k selection | full argsort was 14.7 ms/token; → 8.5 tok/s |
| [decode-04](decode-04.md) | Prefill lens warm-up; chunk lever rejected | first-request readout 92.6 → 39.4 ms/prompt-token |
| [decode-05](decode-05.md) | GPU work off the event loop | pause latency 340 → 1.4 ms during generation |

### `ui` — the web front-end

Keeping the position × layer grid responsive as conversations grow to
thousands of rows.

| # | Iteration | Headline |
|---|-----------|----------|
| [ui-01](ui-01.md) | Append-only grid + debounced persistence | per-token render 180 ms → 5 ms |
| [ui-02](ui-02.md) | Grid virtualization | render only the visible row window |

### `planner` — backward intervention search

Profiling the resident model as a text-only candidate-recipe planner, without
mixing J-lens readout or UI work into the measurement.

| # | Iteration | Headline |
|---|-----------|----------|
| [planner-01](planner-01.md) | Contradiction prompt matrix + full-paper bound | distilled planner ~15 s; complete paper exceeds the 3-minute budget during prefill |
| [planner-02](planner-02.md) | Bounded full-replay throughput | 60 s judges 90–160 singles; staged 180 s covers 472 pairs/triples |

## The one rule that held throughout

**No optimization shipped without an exactness gate.** Every change to the
fit or the decode path was validated against a reference before landing:

- unit equivalence against an ops/`mx.vjp` reference (kernels: ~1e-6–3e-7),
- greedy **token-identity** over a long horizon (cached vs. uncached decode),
- **readout-equality** (identical top-k ids, scores within tolerance).

The J-lens matrices are computed by the `fit` subsystem and applied
**read-only** at inference — they never enter the generation forward pass —
so none of the `decode`/`ui` work can change what the lens reports. See the
per-iteration "Verification" sections for the specific checks.
