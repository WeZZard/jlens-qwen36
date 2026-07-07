# Handoff: Analytic Attention Branch Jacobian

**Project:** jlens-qwen36 (`~/Repositories/com.github/WeZZard/jlens-qwen36`)
**Task:** Implement the analytic attention branch of the per-layer Jacobian, replacing the current VJP-based fallback. This is the remaining bottleneck for the full 30-60× fit speedup.

## Context

A full Jacobian lens fit is running in the background (PID 9045) using ~20 GB of the 64 GB unified memory. **Do not load the full model into memory** — it will OOM and kill the running fit. Develop and verify against *synthetic* examples (random tensors with the right shapes) only. The final integration test against the real model will be done by the orchestrator after the fit completes.

Read these files first:
- `PERFORMANCE_REVIEW.md` — the review that motivates this work (§2).
- `PERFORMANCE.md` — the broader optimization plan.
- `jlens_qwen/analytic_layer.py` — the existing analytic assembly (MLP + norm branches done, attention branch is `_attn_jacobian_vjp` which is the slow fallback you're replacing).
- `jlens_qwen/gdn_backward.py` — the verified manual BPTT for GDN (reference implementation, matches `mx.vjp` exactly). Your GDN analytic should produce the same result.
- `jlens_qwen/custom_gdn_vjp.py` — the custom Metal GDN VJP kernel (5× faster than ops, verified exact). The analytic path should match this too.
- `jlens_qwen/custom_gdn_patch.py` — how the custom VJP is wired into GDN via `mx.custom_function`.
- `jlens_qwen/model.py` — the `MLXLensModel` adapter; `forward()`, `forward_from_layer()`, `_one_layer_forward()`.
- `jlens_qwen/fit.py` — the chain-multiply fit; `per_layer_jacobian` is the slow VJP path.
- `jlens_qwen/fit_analytic.py` — the hybrid fit that uses `decoder_layer_jacobian` (analytic MLP + VJP attention).

## The task

Implement `attn_jacobian_analytic()` in `jlens_qwen/analytic_layer.py` that replaces `_attn_jacobian_vjp()`. It assembles `d(attn(norm(x)))/d(norm(x))` analytically from the attention sub-layer's structure, position-averaged over valid positions.

### For full-attention (FA) layers (16 of 64)

The FA attention is:
```
q, gate = split(q_proj(x))   # q: [B, S, H, head_dim], gate: [B, S, H*head_dim]
k = k_proj(x); v = v_proj(x)
q = q_norm(q); k = k_norm(k)
q = rope(q); k = rope(k)
out = softmax(q k^T / √d) v   # the attention core, [B, H, S, head_dim]
out = o_proj(out * sigmoid(gate))
```

The Jacobian `d(out)/d(x)` breaks into:
- **Projection matrices** (`q_proj`, `k_proj`, `v_proj`, `o_proj`): position-independent, assembled via GEMMs.
- **RMSNorm** (`q_norm`, `k_norm`): closed-form diag + rank-1 (already implemented in `analytic.py`).
- **RoPE**: position-dependent rotation — a block-diagonal orthogonal matrix per position, analytic.
- **Softmax attention core** (`softmax(QK^T/√d) V`): the only nonlinear part. Its Jacobian is small (operates in head space `[H, S, head_dim]`, not `[D]`). Batch identity cotangents through it.
- **Gate** (`sigmoid(gate)`): diagonal, analytic.

The Hadamard trick from `mlp_jacobian` applies to the projection + diagonal-gate terms. The softmax core is the only part that needs a VJP — but in head space (6144-dim for FA), not D=5120 space, so it's ~5120/6144 the cost... actually batch the identity through the *full* attention sub-layer's softmax core and assemble the projections analytically.

### For GDN (linear-attention) layers (48 of 64)

The GDN attention recurrence is the hard part. The review (§2) says:
1. Seed cotangents at the recurrence output analytically (row d of `W_o · J_gatenorm(t)` — computable per position).
2. Run batched BPTT through the recurrence core only (the existing Metal kernel in `custom_gdn_vjp.py` supports this — cotangent chunks fold into the batch dimension B).
3. Contract the position-resolved gradients with per-position input diagonals (conv/SiLU/q-k-norm/LN), sum over s, one GEMM against stacked input projections.

The math for the GDN backward is already verified in `gdn_backward.py:gdn_vjp` — it produces exactly the same result as `mx.vjp`. Your job is to:
- Seed the cotangents analytically (not one-hot).
- Batch them through the kernel (fold into B).
- Assemble the projection Jacobians analytically (not via VJP).

### Verification

Create a test `test_analytic_attention.py` (or extend the existing inline test pattern) that:
1. Constructs synthetic q/k/v/projection weights with known shapes.
2. Computes the analytic attention Jacobian.
3. Compares against `mx.vjp` on the same synthetic input.
4. Asserts max abs error < 1e-3 (fp32).

Do NOT load the real model — synthetic tensors only.

### Don't forget

- The g/β paths (`in_proj_a`, `in_proj_b` → `compute_g`, `sigmoid(b)`) are currently dropped by the custom kernel VJP. The review (§4.1) says to restore them. Include them in the analytic assembly — they're two extra 5120→48 projections (cheap).
- The conv1d (depthwise, kernel 4) has a banded Jacobian — 4-tap diagonal per channel.
- `Qwen3NextRMSNormGated` (the GDN's internal norm) is `rms_norm(x, w) * silu(gate)` — diagonal + rank-1 for the norm, diagonal for the gate.

## Deliverables

1. `jlens_qwen/analytic_layer.py` updated with `attn_jacobian_analytic()` (FA + GDN).
2. `decoder_layer_jacobian()` updated to call `attn_jacobian_analytic()` instead of `_attn_jacobian_vjp()`.
3. `tests/test_analytic_attention.py` — synthetic verification against `mx.vjp`.
4. A comment in `analytic_layer.py` noting the measured speedup once verified.

## Constraints

- **Do not run the real model.** Synthetic tensors only. The fit is using the GPU.
- **Do not modify `fit_analytic.py` or `run_fit.py`** — the orchestrator will wire it in after verifying.
- **Do not commit** — leave changes for the orchestrator to review and commit.
- **Match the code style** of the existing files (no comments unless asked, type hints, concise).

## Estimated effort

~1-2 days. The FA branch is ~0.5 day (mostly Hadamard trick + small softmax VJP). The GDN branch is ~1 day (head-space BPTT seeding + projection assembly).