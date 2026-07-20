# Project instructions

## Lens selection — n=1000 is the default

**Use the 1000-prompt-tuned Qwen3.6-27B lens for development, testing,
intervention scans, demos, and evaluation unless the user explicitly requests
another lens:**

```bash
JLENS_PATH=data/lens/qwen36_27b_neuronpedia_n1000.npz \
  uv run python -m uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
```

Do not assume that `data/lens/lens.npz` selects the preferred lens; that path
may point to a smaller local fit. After starting the server, verify
`GET /api/lens` reports `"n_prompts": 1000` before interpreting readouts or
intervention results. If the n=1000 lens is unavailable, stop and state that
clearly instead of silently substituting a smaller lens.
