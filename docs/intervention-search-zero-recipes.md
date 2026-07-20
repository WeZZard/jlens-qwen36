# Case study: why a Paris → Beijing search can return zero recipes

- **Observed:** 2026-07-17
- **Model:** `mlx-community/Qwen3.6-27B-4bit`
- **Lens:** Neuronpedia n=1000
- **Measured bands:** sensory L0–L25, workspace L26–L59, motor L60–L63

## Summary

For the natural chat exchange

```text
User:      Tell me the capital of France.
Assistant: The capital of France is **Paris**.
Requested reply edit: Paris → Beijing
```

the one-minute backward search completed without finding a selectable
intervention recipe. The UI displayed `0 recipes`. A real-model Playwright
reproduction tested 61 candidates in one minute; every completed replay was
unchanged, so there was neither a verified recipe nor a safe improving lead.
The exact number tested can vary with replay throughput—the original UI run
showed 54—but the result was the same.

This is not a rendering failure, and it does not mean that J-Space
interventions cannot change `Paris` to `Beijing`. It means something narrower:

> Within that prompt, lens, intervention direction, workspace-only candidate
> policy, and time budget, none of the tested exact-cell recipes produced the
> requested complete response under deterministic replay.

That distinction is important because several earlier successes used different
prompts, layer regimes, scopes, or semantic intervention directions.

## J-Space background

The [J-Space paper](https://transformer-circuits.pub/2026/workspace/index.html)
defines the Jacobian lens (J-lens) from the average linearized effect of an
intermediate residual-stream activation on present and future output logits.
For layer `l`, the fitted transport `J_l` maps an intermediate activation into
the final-layer coordinate system; applying the model's unembedding then ranks
tokens the activation is disposed to make the model verbalize.

The token-indexed J-lens vectors form an overcomplete frame. The paper calls a
sparse, non-negative combination of these vectors the **J-Space**. Its proposed
role is workspace-like: selected concepts can be read, written, used in
intermediate reasoning, and broadcast to different downstream computations.

Two properties matter for this case:

1. **Workspace and output are different regimes.** Intermediate workspace
   layers tend to carry persistent, abstract content. In the final few
   **motor** layers, J-lens readouts increasingly align with the imminent output
   token. Editing a motor representation can force wording without showing
   that an upstream semantic variable was changed.
2. **The causally useful direction may name an intermediate, not the answer.**
   In the paper's flexible-generalization experiment, the model is not edited
   with `Paris → Beijing`. It is edited with the latent argument
   `France → China`; downstream capital, language, and continent circuits
   then compute results appropriate to China. The swap is clamped at every
   position across a band of intermediate layers.

This project uses the same read/write intuition. A coordinate swap exchanges
the measured loading on two J-lens directions while preserving the component
orthogonal to their span. See [J-space interventions](interventions.md) for the
operation and wire format.

## What the current backward search actually asks

The current UI starts from a selected reply span and a requested replacement.
For this case it constructs the literal direction `Paris → Beijing`, ranks
causally eligible prefill positions using the baseline J-Space readout, and
replays exact cells in the measured workspace band.

The one-minute stage currently:

- uses only L26–L59;
- ranks up to eight prefill positions, including the causal frontier;
- samples source-evidence layers and seven coarse workspace layers;
- initially tests one exact `(position, layer, alpha=1)` cell per recipe;
- refines a cell or forms pairs only after a single-cell replay changes the
  response safely; and
- accepts a recipe only when a fresh greedy replay reaches EOS, avoids
  repetition, and exactly matches the complete desired response.

For the natural chat case, the desired response is not merely the token
`Beijing`; it is:

```text
The capital of France is **Beijing**.
```

All tested single cells were unchanged. Therefore no improving single existed
to seed the pair beam, and the UI correctly had no recipe to expose.

Formally, `0 recipes` is a statement about the tested set

```text
S = literal direction
  × ranked prefill positions
  × workspace layers sampled before the deadline
  × allowed strengths
  × adaptively admitted combinations
```

It is not a proof over every possible J-Space intervention.

## Why earlier successes do not contradict this result

| Experiment | Prompt/output shape | Direction and scope | Layer regime | Result |
|---|---|---|---|---|
| Current natural-chat search | Full Markdown sentence | `Paris → Beijing`, exact cells | Workspace L26–L59 | 0 recipes; tested replays unchanged |
| Bounded-search benchmark | Prompt constrained to one city; baseline output exactly `Paris` | `Paris → Beijing`, one exact cell | Workspace L59, frontier p25, α=1 | Verified output exactly `Beijing` |
| Legacy intervention scan | Full Markdown sentence | `Paris → Beijing`, short diagnostic probes | Motor L60–L62 | Target appeared, followed by `Wait,`; changed but not adoptable |
| Simple output test | `Say Paris and nothing else.` | `Paris → Beijing`, exact cell | Motor L60 | Verified output exactly `Beijing` |
| Paper-style/project causal demo | Capital question | `France → China`, persistent/clamped premise edit | Intermediate workspace layers | Downstream answer changes toward Beijing |

These experiments answer different questions:

- The constrained benchmark asks whether a late workspace cell can redirect a
  minimal one-word completion.
- The motor-layer tests ask whether the imminent output can be forced.
- The premise swap asks whether changing a broadcast semantic argument makes
  downstream computation derive a different conclusion.
- The natural-chat search asks whether a small, exact, workspace-only recipe
  using the **reply-token direction** can regenerate an otherwise identical
  full sentence.

A success in any of the first three conditions does not guarantee a recipe in
the fourth.

The measured constrained benchmark is documented in
[Planner latency 02](perf/planner-02.md). Its capital prompt is
`Complete with one city name: The capital of France is the city of`, not the
natural chat prompt used in this case.

## Likely contributors

The observations support several contributors, but do not isolate one root
cause.

### 1. Conclusion direction versus premise direction

`Paris → Beijing` directly describes the requested surface conclusion.
`France → China` describes the semantic variable from which a capital can be
recomputed. The paper's reasoning and flexible-generalization results show why
the latter can be causally stronger in workspace layers: it edits an argument
that downstream circuits are already prepared to consume.

The current production search does not infer and test alternative latent
directions. Its position ranking can find where `Paris` is readable, but it
does not turn the requested reply edit into the hypothesis
`France → China`.

### 2. Exact cells versus clamped interventions

The paper's country experiment clamps a swap across positions and an
intermediate layer band. The current guided recipes are intentionally much
smaller: one to three exact cells. A concept repeatedly reconstructed from the
prompt may survive an isolated edit but yield to a persistent intervention.
The two scopes should not be described as equivalent recipes.

### 3. Prompt and response geometry

The natural response has a sentence prefix, Markdown delimiters, punctuation,
and a multi-token continuation. The successful bounded benchmark emits only a
city name. Those contexts have different token positions, cached states,
causal frontiers, and output constraints, so an effective coordinate need not
transfer between them.

### 4. Workspace loading and task selectivity

The paper reports that swaps fail more often when the source concept is weakly
loaded in the workspace. It also cautions that some automatic computations may
bypass the J-Space and that it is not yet possible to predict this reliably for
an arbitrary task. A familiar factual completion can therefore be less
amenable to a small workspace edit than its apparent simplicity suggests.

### 5. Tokenization is a limitation, but not the whole explanation

In this tokenizer, bare `Beijing` splits into `Be` + `ijing`, while
space-prefixed ` Beijing` is a single token. The intervention API resolves
text directions to one token, so boundary choice matters. However, a
controlled follow-up using the single-token space-prefixed direction also left
the tested natural-chat candidates unchanged. Tokenization remains a general
risk, but it does not by itself explain this negative result.

### 6. A better lens is still an approximation

The n=1000 lens is the project default and was verified for this run. More fit
prompts improve the average Jacobian estimate; they do not make every concept
single-token, every readout interpretable, or every local coordinate swap
causally effective. The paper explicitly describes vocabulary restriction,
inconsistent interpretability, and the incomplete separation of workspace and
motor representations as open limitations.

## Product interpretation

The empty recipe list is the epistemically correct UI state. A changed,
truncated, repetitive, or off-target completion must not be promoted into an
educational recipe merely because it contains `Beijing` somewhere.

The message should be read as:

> No verified recipe was found within this search policy and budget.

It should not be read as:

> No intervention exists, or this model has no J-Space representation of the
> task.

The deeper two-minute extension broadens exact positions, layers, and strengths
while retaining the literal direction. It can improve coordinate coverage, but
it cannot repair a wrong semantic direction. If all literal singles remain
unchanged, the combination beam also has no evidence from which to grow.

## Implication for the search design

A genuinely backward J-Space search should separate hypothesis generation from
causal verification:

1. **Literal workspace search:** try the user-visible reply direction over
   small exact recipes.
2. **Latent-premise search:** infer candidate intermediate transformations from
   the conversation and requested edit—for this example,
   `France → China`—without claiming they are correct.
3. **Replay verification:** test every proposed direction and coordinate recipe
   against the original baseline with deterministic, EOS-complete generation.
4. **Broader workspace scope:** only when justified, test persistent or
   band-clamped workspace recipes and label their larger causal footprint.
5. **Optional motor recipes:** if exposed, present them as output forcing, not
   as evidence that an intermediate thought was changed.

No planner proposal should become a visible recipe until replay verifies it.
When the budget ends first, the application should retain the negative result
rather than fabricate a plausible-looking configuration.

## Reproduction and related material

- Real-browser workflow: [`scripts/e2e_intervention_playwright.cjs`](../scripts/e2e_intervention_playwright.cjs)
- Versioned contradiction cases and evaluation-only latent directions:
  [`data/benchmarks/jspace_contradiction_prompts.json`](../data/benchmarks/jspace_contradiction_prompts.json)
- Search throughput and the constrained capital success:
  [Planner latency 02](perf/planner-02.md)
- Cost and limits of model-assisted premise proposals:
  [Planner latency 01](perf/planner-01.md)
- Intervention operations and verification contract:
  [J-space interventions](interventions.md)
- Lens provenance and n=1000 setup: [Lenses](lenses.md)
- Background paper:
  [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)

When reproducing the case, first verify `/api/lens` reports
`"n_prompts": 1000`; results from the bundled 20-prompt fit are a different
experimental condition.
