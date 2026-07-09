# Lenses

The server loads the lens at `JLENS_PATH` (default `data/lens/lens.npz`).
Any lens with this project's NPZ schema works — a per-layer `J_ℓ` dict plus
`n_prompts` and `d_model`, using the `J @ h` transport orientation. Four
ways to get one:

## 1. Bundled pre-fitted lens (default)

The `v0.2-fulldepth` release — 20 prompts, all 63 source layers (L0–L62),
fitted with the corrected chain indexing and the GDN decay-gate (g/β) paths.
See the README quick start.

## 2. Neuronpedia's n=1000 lens

[Neuronpedia](https://neuronpedia.org/jlens) publishes Jacobian lenses
fitted with Anthropic's
[jacobian-lens](https://github.com/anthropics/jacobian-lens) library, which
this project can load and visualize — including one for Qwen3.6-27B fitted
on 1000 wikitext prompts (50× the bundled lens). Convert its `.pt` once:

```bash
# 1. Download from Hugging Face (3.3 GB)
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('neuronpedia/jacobian-lens',
    'qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt',
    local_dir='data/lens/hf')
"

# 2. Convert to this project's NPZ schema (torch is needed only here)
uv run --with torch python -c "
import torch, numpy as np, json
d = torch.load('data/lens/hf/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt',
               map_location='cpu', weights_only=False)
out = {f'J_{l}': J.to(torch.float16).numpy() for l, J in d['J'].items()}
out.update(n_prompts=d['n_prompts'], d_model=d['d_model'])
np.savez('data/lens/neuronpedia_n1000.npz', **out)
json.dump({'n_prompts': d['n_prompts'], 'd_model': d['d_model'],
           'source_layers': sorted(d['J'])},
          open('data/lens/neuronpedia_n1000.json', 'w'))
"

# 3. Serve with it
JLENS_PATH=data/lens/neuronpedia_n1000.npz \
  uv run python -m uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
```

Differences vs the bundled lens: it was fitted on the bf16 HF weights (works
fine on the 4-bit MLX quant), readout scores run ~3–6× larger — this mostly
affects score scale rather than the visual layout, but exact ranks can still
differ across lenses — and semantic commitment tends to surface later in the
stack, with more formatting-like tokens in the middle band (a wikitext-fit
trait).

Credit: fitted by @mntss (Mateusz Piotrowski, Anthropic Interpretability)
via Neuronpedia; MIT-licensed.

## 3. Fit your own

```bash
# Full-depth analytic fit — all 63 layers, ~2.75 h on an M4 Pro
uv run python -c "
from jlens_qwen.model import load
from jlens_qwen.fit_analytic import fit_analytic
from jlens_qwen.lens import JacobianLens
from jlens_qwen.prompts import load_prompts
model = load()
prompts = load_prompts(n=20, min_chars=150)
J = fit_analytic(model, prompts, source_layers=list(range(63)),
                 checkpoint_path='data/lens/lens.ckpt.npy')
JacobianLens(J, n_prompts=20, d_model=5120).save('data/lens/lens.npz')
"

# then serve (auto-loads data/lens/lens.npz)
uv run python -m uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
```

Use `--n-prompts 100` (or more) for research-grade quality. The analytic
pipeline that makes full depth affordable is documented in
[`perf/`](perf/).

## 4. No lens (logit lens)

Serve without a lens file: the readout uses `J = I` (no transport). Only the
last ~10 layers are interpretable, but it needs no fit and is good for
exploring the UI.

```bash
uv run python -m uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
```

## Using a different model

The default is `mlx-community/Qwen3.6-27B-4bit`. Any MLX-quantized model in
the **`qwen3_5` architecture family** (`model_type: qwen3_5`, hybrid GDN +
full attention) works — that includes Qwen3.6-27B despite the `3_5` name,
which is the mlx_lm architecture identifier, not the model version. The
custom GDN backward kernel is required for the fit to be tractable.

```bash
uv run python scripts/run_fit.py --model-id mlx-community/Qwen3.6-35B-A3B-4bit
JLENS_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit JLENS_PATH=data/lens/lens.npz \
  uv run python -m uvicorn jlens_qwen.serve:app --port 8765
```
