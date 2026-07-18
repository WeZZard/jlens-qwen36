# J-space interventions

The lens is not just a readout — it is a **write surface**. An
intervention edits the residual stream `h_ℓ` at chosen (layer, position)
cells during the cached streaming decode, so the edit propagates through
every downstream layer, the KV/GDN caches, and the sampled text. The UI
authors edits from grid cells, re-generates, and shows a baseline-vs-
intervened A/B comparison.

Verified causal demo (mirrors the paper): swapping ⟨France→China⟩ at
L30/40/48 makes "What is the capital of France?" answer **Beijing**.

## Modes

For a token *t*, the J-lens vector at layer ℓ is `v_t = J_ℓᵀ W_U[t]` —
the `h_ℓ`-direction that raises *t*'s readout logit. At the final
(unfitted) layer the transport is the identity, so `v_t = W_U[t]`.

The UI names each edit after its purpose — what you want the model's
workspace to do — while the wire protocol keeps the paper's operation
names:

| UI purpose | Plain meaning | Wire mode | Edit |
|---|---|---|---|
| **Replace** | "wherever it's thinking A, think B instead, equally strongly" | `swap` | exchange the lens coordinates of `v_s`/`v_t` (`c = (VVᵀ)⁻¹Vh`, swap, write back); the component orthogonal to span{v_s, v_t} is untouched |
| **Add** | "also think about t, this hard" | `steer`, α > 0 | `h ← h + α·v_t` |
| **Remove** | "think about t less, this hard" | `steer`, α < 0 | `h ← h − α·v_t` |
| **Erase all** | "blank out whatever this cell is thinking" | `ablate` | remove the span of an explicit direction set via least-squares projection (ridge-regularized Gram inverse) |

Replace is self-calibrating (it reuses the measured loading of the
existing thought), so it needs no tuning and is the most reliable edit —
the editor defaults to it, seeded with the cell's current top thought.
Add/Remove push by an *absolute* amount: with the bundled 20-prompt lens
they typically need α 50–800 (the strength slider is log-scaled over
that range).
The paper's ablation rule is applied client-side: the UI ablates a cell's
top-k readout directions **excluding** ids present in the final layer's
top-k at the same position, so the model's intended output tokens are
never ablated.

## Position scopes

Positions are **global indices into the chat-templated token sequence**
— identical to the readout grid's row coordinates. Exactly one scope per
spec:

- `positions: [p, …]` — explicit cells (UI: **"Just this one"**). The
  edit is written only there, but its causal effect still reaches every
  later token through the caches.
- `from_position: P` — every position ≥ P (UI: **"This + after"**;
  0 = everywhere), persisting across all future decode steps.
- `segment: "generation"` — every generated position of this request
  (UI: **"The reply"**; the server resolves it to the generation-prefix
  length).

Because each `/api/chat_stream` request re-prefills the whole
conversation into a fresh session, prompt-position edits are legal and
the re-emitted prefill grids show the *written* residuals — the grid
displays the post-edit workspace at the edited layer and its downstream
consequences everywhere else.

## API

`POST /api/chat_stream` accepts `interventions: [InterventionSpec]`:

```json
{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "interventions": [{
    "mode": "swap",
    "layers": [30, 40, 48],
    "token": " France",  "token_id": null,
    "target": " China",  "target_id": null,
    "alpha": 1.0,
    "from_position": 0
  }]
}
```

- `token_id`/`target_id` (from a readout's `top_ids`) win over the text
  fields; text resolves to its **first** token (`/api/tokenize` previews
  the split). `ablate` takes `ablate_token_ids` instead.
- Layers must be lens source layers or the final layer.
- Invalid specs fail as HTTP 400 **before** the SSE stream starts;
  `stream_start` echoes the resolved specs.
- Baseline vs intervened comparisons are only clean when `messages`,
  `temp`, and `enable_thinking` are identical across the two runs (the
  template fixes the position coordinates; temp > 0 adds sampling noise).
- `POST /api/intervene` remains as a legacy single-layer, uncached
  reference matching `scripts/intervention_sanity.py`.

### Bounded recipe verification

`POST /api/intervention_search` is the lens-free evaluator for backward
search. The caller supplies an ordered list of recipes; each recipe contains
one to eight exact workspace cells:

```json
{
  "messages": [{"role": "user", "content": "Complete with one city name: The capital of France is the city of"}],
  "token": "Paris",
  "target": "Beijing",
  "source_text": "Paris",
  "goal_text": "Beijing",
  "time_budget_seconds": 60,
  "max_tokens": 64,
  "stop_on_success": true,
  "candidates": [{
    "id": "frontier-l59",
    "cells": [{"layer": 59, "position": 25, "alpha": 1.0}]
  }]
}
```

It streams `search_start`, `baseline`, `candidate`, and `search_end` SSE
events with compile/prefill/decode timings. Every candidate uses a fresh bare
model session—no activation capture or readout—and the deadline is checked
between candidates. A recipe is `verified` only when deterministic replay
reaches EOS without repetition, contains exactly one bounded goal phrase, and
contains no old source phrase. Time-budget, max-token, and repetition stops are
never verified. Multi-cell recipes list each `(layer, position)` separately;
combining arrays in one normal intervention spec would instead create their
Cartesian product.

### Latent-premise stage (adaptive search)

`POST /api/intervention_search_adaptive` accepts
`enable_premise_search: true` (the UI always sends it). After the first
eight tested singles — or when the literal queue runs dry, and up front in
a thorough or extended search — the resident model reads the conversation
and the requested reply edit and proposes up to two latent premise
directions (`France → China` for a `Paris → Beijing` request; contract and
~10–15 s cost measured in [Planner latency 01](perf/planner-01.md)). The
`premise_proposal` event names its trigger in an `origin` field
(`early`, `exhausted`, `extension`, or `thorough`). A verified literal
recipe suppresses the proposal — the premise stage exists for when the
literal direction fails — but it does not end the search: the adaptive
search exhausts its time budget and reports every verified recipe it
finds (`verified_count` in `search_end`; `stop_on_verified: true`
restores first-hit termination for evaluation harnesses). Each resolved direction is
replay-tested as a band-clamped swap across the fitted workspace band at
every position (`from_position: 0`); a passing band is bisected while the
budget lasts, so the reported footprint is the smallest band that still
redirects.

A premise replay is reported as `class: "premise_redirect"` when it
reaches EOS without repetition, contains the requested replacement exactly
once as a bounded phrase, and no longer contains the selected source span.
It is deliberately never an exact `verified` recipe: a coherent model that
now believes the swapped premise also verbalizes it (the replay says
"The capital of China is Beijing.", not the requested sentence). The
`premise_proposal` SSE event carries the proposed directions, premise
candidates carry a `premise` descriptor instead of `cells`, and
`search_end` includes a `premise` summary with the proposal status and the
best redirect. The UI lists redirects as PREMISE recipes with their
direction, band, and achieved reply, and applies them as one
`swap … from_position: 0` spec.

## Engine

- `jlens_qwen/interventions.py` — `compile_edits()` turns a resolved spec
  into per-layer `LayerEdit`s. All concept vectors are built by
  `j_lens_vectors_lite()`: dequantize only the needed `W_U` rows
  (`unembed_rows`) and matvec against the lens's resident fp16 GPU
  Jacobians. **The server path never materializes the `[vocab, d_model]`
  matrix** — `get_unembedding_matrix` / `j_lens_vectors` (≈5 GB each)
  are script-only.
- `jlens_qwen/model.py` — `StreamSession.set_edits()` installs the
  compiled edits; inside `extend()` each edit rewrites its rows right
  after that layer's forward and **before** the residual is captured, so
  downstream layers and caches consume the edited stream. Global→chunk
  position mapping is `_chunk_local_indices`.
- Per-step cost: a gather, 1–3 small matvecs, and a scatter per edited
  layer — nothing measurable against the decode itself; an empty edit
  list is byte-identical to the clean path (guarded by the decode gate).

## UI

- Click a grid cell → detail card: hover a readout row for ＋ (add /
  think it harder), － (remove / stop thinking it), ⇄ (replace it) quick
  actions, or press **Intervene** for the full editor (purpose, thought
  with tokenizer preview, strength for Add/Remove, layer scope:
  this/band/custom, token scope).
- The editor is named by **purpose** (Replace / Add / Remove / Erase all),
  not by the paper's operation. Opening it enters an **interactive
  intervening** mode: the chat input and corner buttons sink off-screen,
  the panel takes their place at the bottom, and the page stays fully
  live behind it — nothing is blurred or blocked. The target cells glow
  and pulse on the grid (affected layers × tokens), everything else dims,
  and a counter reads "N cells highlighted (L layers × T tokens)", so
  "which layers / which tokens" is answered by looking at the grid.
- In that mode the grid is the input surface: **click any cell to re-aim**
  the edit there (an untouched thought re-seeds from the new cell), and in
  **Pick…** layer mode **click layer columns** (header or cells) to toggle
  them. Scroll the grid and chat freely. Save / Cancel / Esc exits back to
  normal.
- Specs collect in the top-left rail (toggle dot, click to edit, hover to
  delete). **Re-run ⊕N** truncates the last assistant turn and
  re-generates with the enabled specs; the previous run is kept and the
  top-right **Baseline / Intervened** capsule flips the entire grid+chat
  between the two, with changed tokens wavy-underlined, edited cells
  dash-ringed, cells whose top-1 changed dotted, and per-row rank deltas
  in the detail card.
- Sending a new message with enabled specs applies them to that turn
  (persistent-injection semantics) and retires any open compare, keeping
  the active variant.
- Everything persists in sessions (localStorage quota-trims the compare
  first; the server autosave keeps it whole) and replays read-only in
  presentation builds.

## Tests

```bash
uv run pytest tests/                                   # fast + decode gate
JLENS_SLOW_TESTS=1 uv run pytest tests/test_intervention_stream.py  # France→Beijing through the engine
uv run python scripts/intervention_sanity.py           # legacy uncached sweep
```
