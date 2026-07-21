# J-lens Qwen3.6

A visual debugger for a local Qwen3.6-27B (4-bit) on Apple Silicon / MLX.
It fits a **Jacobian lens** and shows which words the model is pushing
toward at every layer and token position.

> **Try it live at [jlens.wezzard.com](https://jlens.wezzard.com)** — a
> read-only presentation mode of this project, running in the browser.
> Every conversation has a shareable URL.

![Global-workspace concept readout](assets/screenshot_blackmail.png)

Inspired by Anthropic's [*Verbalizable Representations Form a Global
Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html).

## What you're looking at

In the hero image the model is given an email-blackmail setup and writes a
calm, compliant reply:

> I will not act upon, report, or utilize the information… I am ready to
> proceed with the shutdown.

But the **workspace band** shows the top J-lens token at every layer, and
**blackmail / suicide / murder / threatening / fictional** are lit across
the middle layers. The concept is in the latent stream even though it never
reaches the output — a visual debugger for what the model doesn't say out
loud.

The grid is **position × layer**; each cell is the top J-lens token there.
In chat mode it streams live, one row per generated token. Click a cell to
pin its top-10 readout.

## Quick start

```bash
git clone https://github.com/WeZZard/jlens-qwen36.git
cd jlens-qwen36 && uv sync

# Pre-fitted lens (3.3 GB, two parts) — download and reassemble
gh release download v0.2-fulldepth --repo WeZZard/jlens-qwen36 \
  --pattern '*.npz.part-*' --dir data/lens/
cat data/lens/*.npz.part-* > data/lens/lens.npz && rm data/lens/*.part-*

uv run python -m uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765/
```

Needs an Apple-Silicon Mac and ~24 GB free RAM; the model auto-downloads
from HuggingFace on first run (~15 GB).

**Other lenses** — point `JLENS_PATH` at any compatible `.npz`: load
[Neuronpedia's n=1000 lens](https://neuronpedia.org/jlens), fit your own, or
run with no lens (logit lens). See [`docs/lenses.md`](docs/lenses.md).

## How it works

The Jacobian lens at layer ℓ is a matrix `J_ℓ ∈ R^{d×d}`: the network's
average input→output Jacobian over a corpus of prompts. It maps a residual
`h_ℓ` into the final-layer basis, so `softmax(W_U · norm(J_ℓ h_ℓ))` gives
token scores. Fitting chains per-layer Jacobians: `J_ℓ = J_{ℓ+1} · M_ℓ`.

The hard part is the 48 Gated DeltaNet (GDN) linear-attention layers: MLX's
fused GDN kernel has no VJP and the ops fallback is ~22× slower. This
project ships a **custom Metal backward kernel** for GDN
(`jlens_qwen/custom_gdn_vjp.py`) plus an analytic branch-Jacobian assembly
that fits a full-depth lens in ~2.75 h on an M4 Pro.
See [`docs/perf/`](docs/perf/) for how it was made fast.

## Interventions: writing to the workspace

The lens writes as well as reads. Edit the workspace by hand, or name
the reply you want and let the app find the edit.

### Manual: edit a cell

Click a cell, pick Replace / Add / Remove / Erase, and type the new
thought. Choose how far it reaches: one cell, a layer band, or the whole
reply. **Re-run** regenerates, and a **Baseline / Intervened** toggle
diffs the two runs.

![Replacing the France thought with China across the workspace band](assets/screenshot_intervene.png)

Replacing *France* with *China* across the band rewrites the answer:

![The reply now reads: the capital of China is Beijing](assets/screenshot_intervene_result.png)

### Backward search: "Make it say…"

Click a word in the reply, type the replacement, and the search replays
workspace swaps against a clean baseline until its time runs out. Each
hit becomes a **recipe**; a green dot means a replay reproduced your
reply exactly.

![Two verified Paris→Beijing recipes found mid-search](assets/screenshot_search.png)

When no direct swap works, the search asks the model for the premise
behind the reply: ⟨France⟩→⟨China⟩ to move ⟨Paris⟩ to ⟨Beijing⟩. It
clamps that across the band, then shrinks it to the smallest band that
still works. Those recipes carry a violet dot.

Backward search needs a measured workspace band. One ships for the
Neuronpedia lens; [`scripts/measure_bands.py`](scripts/measure_bands.py)
measures others. Details in [`docs/interventions.md`](docs/interventions.md).

## The bundled lens

Fit on 20 prompts across all 63 layers. Readouts are interpretable but
noisy; interventions are causal but concept-dependent. For research-grade
quality, load
[Neuronpedia's n=1000 lens](https://neuronpedia.org/jlens) (the paper's
fitting scale — [setup](docs/lenses.md)) or fit 100+ prompts yourself — the
analytic pipeline makes that affordable.

Chat runs with thinking disabled (`enable_thinking=False`) so the model
computes in the latent stream rather than in a visible `<think>` trace,
which is what the lens is meant to surface.

## Limitations

- **Apple / MLX only.**
- **`qwen3_5`-architecture models only** — the custom GDN kernel is
  arch-specific (Qwen3.6-27B qualifies; see [`docs/lenses.md`](docs/lenses.md)).
- **Single-token concepts only** — multi-token concepts need the paper's
  extension.
- **Lens quality scales with prompt count** — 20 prompts is demo-grade,
  100+ is research-grade.

## Acknowledgements

Based on Anthropic's
[jacobian-lens](https://github.com/anthropics/jacobian-lens) reference
implementation (Apache-2.0) and
[paper](https://transformer-circuits.pub/2026/workspace/index.html). The GDN
forward kernel is from [mlx-lm](https://github.com/ml-explore/mlx-lm); the
backward kernel is original to this project.

Thanks to [Neuronpedia](https://neuronpedia.org/jlens) and
**@mntss (Mateusz Piotrowski, Anthropic Interpretability)** for the public
pre-fitted Qwen3.6-27B lens weights.

## License

Apache-2.0. See [LICENSE](LICENSE).
