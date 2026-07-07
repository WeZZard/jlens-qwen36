# Handoff: Verify Analytic Attention, Measure g/β Gap, Wire & Refit

**Project:** jlens-qwen36 (`~/Repositories/com.github/WeZZard/jlens-qwen36`)
**Prerequisites:** The old fit is killed. GPU is free. ~23 GB RAM available.

## Context

The analytic attention branch was implemented by a previous external agent
(see `.handoff/analytic-attention-branch.RESPONSE.md`). Synthetic tests
pass at ~1e-8 precision. The orchestrator measured real-model speedups:
- FA layer 59: 16.8× (22.9s → 1.4s)
- GDN layer 48: 10× (36.6s → 3.6s)

The old full-depth fit (PID 9045) was killed after 2/20 prompts. Its
checkpoint (`data/lens/full_depth_analytic.ckpt.npy`) used the OLD
assembly (`analytic_attn=False`) and must NOT be reused — the new path
produces slightly different (more accurate) J matrices, so mixing would
be inconsistent. Start fresh.

## Tasks (in order)

### 1. Run `scripts/verify_analytic_layer.py` (~10 min)

```bash
cd ~/Repositories/com.github/WeZZard/jlens-qwen36
uv run python scripts/verify_analytic_layer.py
```

This compares `M_new` (analytic_attn=True) vs `M_old` (analytic_attn=False)
vs `M_ref` (exact VJP, `per_layer_jacobian`) on real model activations,
for one mid GDN layer and one mid FA layer.

**Decision rule:** if `rel_new <= rel_old` and `rel_new` is at the ~1e-2
level, proceed to step 3 (wire `analytic_attn=True`). The new path should
be both faster AND more accurate (synthetic showed 0.6-1.3% vs 1.8-2.2%).

Record the output in `.handoff/verify-results.txt`.

### 2. Run `scripts/measure_gbeta_gap.py` (~10 min)

```bash
uv run python scripts/measure_gbeta_gap.py
```

This measures the gap between the GDN Jacobian with and without the g/β
decay-gate paths, at early/mid/late GDN layers. The script was written by
a previous external agent (see `.handoff/measure-gbeta-gap.RESPONSE.md`).

**Decision rule:**
- If relative gap < 1%: g/β paths are negligible. Use
  `include_gbeta=False` for the fit (faster — uses the Metal kernel).
- If 1-10%: moderate. Use `include_gbeta=True` for the fit (slower — ops
  BPTT, but more accurate for interventions).
- If > 10%: significant. Must use `include_gbeta=True`.

Record the verdict at the top of `scripts/measure_gbeta_gap.py` (fill in
the `VERDICT` header) and in `.handoff/gbeta-results.txt`.

### 3. Wire `analytic_attn=True` into `fit_analytic.py`

In `jlens_qwen/fit_analytic.py`, the `fit_analytic_single_prompt` function
calls `decoder_layer_jacobian(model, l, acts[l], skip_first=skip_first)`.
The `decoder_layer_jacobian` already defaults to `analytic_attn=True`, so
this should just work. But verify:

```bash
grep "decoder_layer_jacobian" jlens_qwen/fit_analytic.py
```

If it passes `analytic_attn` explicitly as `False`, change it to `True`.
If it doesn't pass the argument at all, the default (`True`) is correct.

Also set `include_gbeta` based on step 2's verdict. If `include_gbeta=False`
(default), no change needed. If `include_gbeta=True`, add it to the call.

Fill in the "measured speedup" comment in `jlens_qwen/analytic_layer.py`
with the numbers from step 1.

### 4. Start a fresh full-depth fit (~1 hour with analytic attention)

```bash
cd ~/Repositories/com.github/WeZZard/jlens-qwen36
rm -f data/lens/full_depth_analytic.ckpt.npy  # remove old checkpoint
nohup uv run python -c "
import logging, sys, os, time
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)
sys.path.insert(0, '.')
from jlens_qwen.model import load
from jlens_qwen.fit_analytic import fit_analytic
from jlens_qwen.lens import JacobianLens
from jlens_qwen.prompts import load_prompts
model = load()
prompts = load_prompts(n=20, min_chars=150)
print(f'Prompts: {len(prompts)}', flush=True)
source_layers = list(range(63))
print(f'Source layers: {len(source_layers)} (0..62) — FULL DEPTH', flush=True)
print(f'Starting full-depth analytic fit (analytic_attn=True)...', flush=True)
t0 = time.perf_counter()
J = fit_analytic(model, prompts, source_layers=source_layers, max_seq_len=32,
                 checkpoint_path='data/lens/full_depth_analytic.ckpt.npy')
elapsed = time.perf_counter() - t0
print(f'TOTAL: {elapsed:.0f}s = {elapsed/60:.1f}min', flush=True)
lens = JacobianLens(J, n_prompts=20, d_model=5120)
lens.save('data/lens/full_depth_analytic.npz')
print(f'Saved: {lens}', flush=True)
" > data/lens/full_depth_fit_v2.log 2>&1 &
echo "Fit PID $!"
```

Expected: ~3.5s/layer × 63 layers × 20 prompts ≈ ~1 hour.

### 5. When the fit finishes, publish v0.2-fulldepth release

```bash
# Verify the lens produces good readouts
uv run python scripts/workspace_range.py --lens data/lens/full_depth_analytic.npz

# Publish as a GitHub release
cp data/lens/full_depth_analytic.npz /tmp/jlens-qwen3.6-27b-4bit-20prompt-64layer.npz
gh release create v0.2-fulldepth \
  /tmp/jlens-qwen3.6-27b-4bit-20prompt-64layer.npz \
  --title "v0.2-fulldepth: full-depth J-lens (20 prompts, 64 layers)" \
  --notes "..."
```

### 6. Commit everything

```bash
git add -A
git commit -m "Wire analytic attention into fit; verify on real model; measure g/beta gap

- analytic_attn=True wired into fit_analytic.py (10-17x speedup)
- verify_analytic_layer.py: real-model A/B confirms new path (X% rel err vs Y% old)
- measure_gbeta_gap.py: g/beta gap = Z% (verdict: ...)
- Full-depth 20-prompt fit completed in ~1h
- v0.2-fulldepth release published"
git push
```

## Constraints

- **Run steps 1-2 before starting the fit** (they need the GPU exclusively).
- **Do not resume from the old checkpoint** — start fresh (different assembly).
- **Do not run anything else heavy** while the fit is running (memory).
- **Commit and push** when done.

## Estimated time

- Steps 1-2: ~20 min (verification scripts)
- Step 3: ~5 min (code change)
- Step 4: ~1 hour (fit)
- Step 5: ~15 min (verify + publish)
- Step 6: ~5 min (commit)
- **Total: ~1.5 hours**