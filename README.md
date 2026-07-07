# jlens-qwen36

A **J-space / Jacobian-lens visualizer** for Qwen3.6-27B (4-bit) on Apple Silicon, ported to Apple MLX.

Implements the technique from Anthropic's [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html). The J-lens reads out what a residual-stream activation at any layer is "disposed to make the model say" — surfacing the model's internal, unspoken concepts as a ranked list of vocabulary tokens.

![slice visualization](assets/screenshot.png)

## What it does

Given a prompt, the visualizer shows a **position × layer grid** where each cell is the top J-lens token at that (position, layer). This lets you watch the model's internal concepts evolve across layers:

- **Early layers**: echo / task-frame tokens (e.g. "currency")
- **Workspace layers**: intermediate concepts the model uses to reason (e.g. "Italy" for the boot-shaped country, before resolving to "euro")
- **Motor layers** (last few): the output token

For example, on `Fact: The currency used in the country shaped like a boot is the`, you see `currency` → `Italian` (L52) → `euro` (L57+) → `euro` (model output).

## Requirements

- **Apple Silicon Mac** (M1+/M-series) — MLX is Apple-only
- **~20 GB free RAM** (the 4-bit model is ~15 GB resident + working memory)
- **Python 3.12** (managed automatically by `uv`)
- **~15 GB disk** for the model (auto-downloads from HuggingFace on first run)

## Quick start

```bash
# 1. Clone
git clone https://github.com/WeZZard/jlens-qwen36.git
cd jlens-qwen36

# 2. Install dependencies (creates a .venv with Python 3.12)
uv sync

# 3. (Optional) Fit a J-lens. Takes ~5-8 hours on an M4 Pro.
#    Skip this to use the logit lens (J=I) instead — works immediately,
#    just less interpretable in early/mid layers.
uv run python scripts/run_fit.py --n-prompts 20 --n-layers 25 --layer-start 40

# 4. Launch the visualizer
uv run python -m uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765

# 5. Open http://127.0.0.1:8765/ in a browser
```

If you skip step 3, the server starts with the **logit lens** (J = identity), which is free and works immediately but only produces interpretable readouts in the last ~10 layers. Fitting the real J-lens (step 3) extends interpretability back to ~layer 40.

## Using a different model

The default model is `mlx-community/Qwen3.6-27B-4bit`. To use a different MLX-quantized Qwen3.5-architecture model:

```bash
# Fit on a different model
uv run python scripts/run_fit.py --model-id mlx-community/Qwen3.6-35B-A3B-4bit

# Serve with that model + its lens
JLENS_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit \
JLENS_PATH=data/lens/lens.npz \
uv run python -m uvicorn jlens_qwen.serve:app --port 8765
```

The code targets the **Qwen3.5 architecture** (`model_type: qwen3_5`), which has hybrid attention: 48 linear-attention (Gated DeltaNet) + 16 full-attention layers. The custom GDN backward kernel is required for the Jacobian fit to be tractable.

## How it works

The Jacobian lens (J-lens) at layer ℓ is a matrix `J_ℓ ∈ R^{d_model × d_model}` that maps a residual-stream activation `h_ℓ` into the final-layer basis, so that `softmax(W_U · norm(J_ℓ · h_ℓ))` gives vocabulary-token scores. It's the average input→output Jacobian of the network, averaged over a corpus of prompts.

**Fitting** computes `J_ℓ` for each source layer via the chain rule: `J_ℓ = J_{ℓ+1} · M_ℓ`, where `M_ℓ = ∂h_{ℓ+1}/∂h_ℓ` is one decoder layer's Jacobian. This avoids the expensive full-stack VJP for each source layer.

**The hard part** is the Gated DeltaNet (GDN) linear-attention layers: MLX's fused Metal kernel for GDN has no registered VJP, and the pure-Python ops fallback is ~22× slower. This project includes a **custom Metal backward kernel** for GDN (registered via `mx.custom_function`) that brings the backward to kernel speed. See `jlens_qwen/custom_gdn_vjp.py`.

## Project layout

```
jlens_qwen/
  model.py            # MLX LensModel adapter (captures per-layer residuals)
  patch_gdn.py        # Force GDN ops fallback + mx.checkpoint
  gdn_backward.py     # Manual BPTT backward through GDN (reference, verified)
  custom_gdn_vjp.py   # Custom Metal GDN backward kernel (5x speedup)
  custom_gdn_patch.py # Wire the custom VJP into GDN via mx.custom_function
  fit.py              # Chain-multiply Jacobian fitting (J_l = J_{l+1} @ M_l)
  lens.py             # JacobianLens: save / load / apply / transport
  prompts.py          # WikiText-103 + c4 corpus loader
  serve.py            # FastAPI backend (/api/lens, /api/slice)
web/
  index.html          # Self-contained slice-vis UI (no build step)
scripts/
  run_fit.py          # CLI: fit a lens
  workspace_range.py  # CLI: classify layers as echo/workspace/motor
  smoke_model.py      # Test: model loads, VJP works
data/lens/            # Fitted lenses + checkpoints (gitignored)
data/corpus/          # Cached prompts (gitignored)
```

## Performance

On an M4 Pro / 64 GB, fitting 25 layers (L40-L62) × 20 prompts × 32-token sequences takes ~5-8 hours. The fit checkpoints every prompt and resumes on restart, so interruptions are safe.

Memory: ~15 GB for the model + ~3.4 GB for the lens checkpoints + ~2 GB working = ~20 GB peak.

## Limitations

- **MLX only** — no CUDA / Linux support (MLX is Apple-only).
- **Qwen3.5 architecture only** — the custom GDN VJP is specific to this arch. Other MLX models (Llama, Mistral, etc. with standard attention) would work with the `patch_gdn` disabled but haven't been tested.
- **Chain-multiply approximation** — the fit uses the position-diagonal block of each per-layer Jacobian, ignoring cross-position terms. This is exact for the per-position readout's "self" contribution but slightly approximate for attention-mixed positions. Empirically the readouts match the paper's findings.
- **Single-token concepts** — like the reference, the J-lens only identifies concepts that correspond to single vocabulary tokens. Multi-token concepts need the paper's extension (§App-multi-token).

## Acknowledgements

Based on Anthropic's [jacobian-lens](https://github.com/anthropics/jacobian-lens) reference implementation (Apache-2.0) and the accompanying [paper](https://transformer-circuits.pub/2026/workspace/index.html). The GDN forward kernel is from [mlx-lm](https://github.com/ml-explore/mlx-lm); the backward kernel is original to this project.

## License

Apache-2.0. See [LICENSE](LICENSE).