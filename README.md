# jlens-qwen36

A J-space / Jacobian-lens visualizer for **Qwen3.6-27B (4-bit, MLX)** on Apple Silicon.

Implements the technique from Anthropic's [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html), ported to Apple MLX.

## Status

**Architecture complete, fit is slow due to the Gated DeltaNet (GDN) linear-attention recurrence.**

### Working
- `jlens_qwen/model.py` — MLX `LensModel` adapter for Qwen3.5 arch. Rewrites `Qwen3_5TextModel.__call__` to capture per-layer residual streams; reuses the quantized `lm_head` for unembedding (no 2.5 GB dense W_U).
- `jlens_qwen/patch_gdn.py` — forces the GDN ops fallback (the fused Metal kernel has no VJP) and wraps each `GatedDeltaNet.__call__` in `mx.checkpoint`.
- `jlens_qwen/gdn_backward.py` — manual BPTT backward through GDN, verified against `mx.vjp` (dq, dk, dv match exactly).
- `jlens_qwen/fit.py` — chain-multiply Jacobian fitting: J_ℓ = J_{ℓ+1} @ M_ℓ, with checkpoint/resume.
- `jlens_qwen/lens.py` — `JacobianLens` save/load/apply/transport.
- Smoke test passes: model loads, forward works, VJP through linear-attention layers produces finite gradients, "euro"/"Euro" appear as top predictions on the paper's currency prompt.

### Slow
The fit is bottlenecked by the GDN recurrence. Each VJP through one GDN layer takes ~24ms (the Python `for t in range(T)` loop in `gated_delta_ops`). A full J_ℓ ∈ R^{5120×5120} = 5120 VJPs = ~2 min per layer. For 25 layers × 10 prompts = ~8 hours. For all 64 layers × 100 prompts = ~2 weeks.

### The blocker
MLX's `mx.vjp` re-runs the forward on every call (no PyTorch-style graph retention), and the fused GDN Metal kernel has no registered VJP. The ops fallback (Python loop) is ~22× slower than the kernel.

### Planned fix
Write a custom VJP for the GDN Metal kernel via `mx.custom_function`, so the backward runs at kernel speed (~1ms vs 24ms). This is ~1-2 days of Metal C++ engineering. Expected fit time after fix: ~10-20 min for 25 layers × 10 prompts.

## Layout

```
jlens_qwen/
  model.py          # MLX LensModel adapter
  patch_gdn.py      # force GDN ops fallback + mx.checkpoint
  gdn_backward.py   # manual BPTT backward through GDN (verified correct)
  custom_gdn_vjp.py # TODO: custom Metal VJP for kernel-speed backward
  fit.py            # chain-multiply Jacobian fitting
  lens.py           # JacobianLens: save/load/apply/transport
scripts/
  smoke_model.py         # smoke test: load, forward, VJP
  spike_jacobian.py      # timing spike (unbatched)
  spike_jacobian_v2.py   # timing spike (batched)
```

## Run

```bash
uv sync
uv run python scripts/smoke_model.py   # verify model + VJP works
```

## Model
- `mlx-community/Qwen3.6-27B-4bit` (Qwen3.5 arch)
- 64 layers, d_model=5120, hybrid attention (48 GDN linear + 16 full)
- lm_head is 4-bit quantized; reuses `quantized_matmul` for unembedding