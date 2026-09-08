# Lenses

The server loads the lens at `JLENS_PATH` (default `data/lens/lens.npz`).
Any lens with this project's NPZ schema works — a per-layer `J_ℓ` dict plus
`n_prompts` and `d_model`, using the `J @ h` transport orientation. Five
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

## 5. Qwen3.8-27B lens (1000 prompts)

The `v0.3-qwen38-n1000` release: 1000 c4 prompts, all 63 source layers
(L0–L62), intervention-grade (`include_gbeta=True`), fitted with this
repo's analytic pipeline against `mlx-community/Qwen3.8-27B-4bit`
(driver: `fit_qwen38_n1000.py`).

```bash
# Download (3.3 GB, two parts) and reassemble
gh release download v0.3-qwen38-n1000 --repo WeZZard/jlens-qwen36 \
  --pattern 'jlens-qwen3.8-*' --dir data/lens/
cat data/lens/jlens-qwen3.8-27b-4bit-1000prompt-63layer.npz.part-* \
  > data/lens/qwen38_27b_n1000.npz && rm data/lens/*.part-*

# Serve — the model id is not optional here
JLENS_MODEL=mlx-community/Qwen3.8-27B-4bit \
JLENS_PATH=data/lens/qwen38_27b_n1000.npz \
  uv run python -m uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
```

**This lens is only valid for Qwen3.8-27B.** Qwen3.6-27B and Qwen3.8-27B
share the `qwen3_5` architecture and identical shapes (64 layers, d_model
5120, vocab 248320), so the lens loads against either model without any
error and produces confident nonsense on the wrong one. Nothing in
`JacobianLens.load()` or the server checks this. The release ships a
`.provenance.json` sidecar recording `model_id`; check it before trusting a
readout. A J-lens is fitted to weights, not architecture — never reuse one
across model versions.

Readout behaviour (`scripts/readout_smoke.py`, logs in
`data/lens/readout_smoke_*.log`) matches the bundled 20-prompt lens run on
Qwen3.6: formatting tokens dominate the middle band, and the late layers
read out the answer concept but often in another language or script
(`八` for ` eight`, `周五` for ` Friday`), so exact top-1 agreement with the
model's next token is low. The 1000-prompt fit commits earlier: ` Paris`
and ` eight` surface at layer 44 instead of 56 and 52. For comparison, the
Neuronpedia lens on Qwen3.6 reads out the exact English token from layer
52 on; that difference is a property of the two fitting pipelines, not of
this fit.

Fit cost on a base M4 Mac mini (16 GB), per-prompt wall time from the
log's "prompt N done in" lines (`data/lens/qwen38_27b_n1000_fit.log`):
1161 s per prompt against 437 s on an M4 Pro, plus about 2 min per prompt
writing the 6.6 GB checkpoint at `checkpoint_every=1` — 356.5 h wall in
all for 1000 prompts. Raise `checkpoint_every` for the next long fit.

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
