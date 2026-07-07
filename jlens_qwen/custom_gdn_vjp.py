"""Custom VJP for the Gated DeltaNet kernel.

This module registers a custom_function that wraps the stock forward kernel
and provides a Metal backward kernel for the VJP. The backward kernel
re-runs the forward in registers to recompute per-t states, then runs the
reverse scan, writing dq, dk (via atomic adds to per-hv buffers, reduced
to Hk in Python) and dv (directly).

The backward math is verified against gdn_backward.gdn_vjp (which matches
mx.vjp exactly). The Metal kernel is the same math, just on-GPU.

Only grads w.r.t. (q, k, v) are computed; (g, beta, state) get zeros.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .gdn_backward import gdn_forward, gdn_vjp as _ops_backward


_BACKWARD_KERNEL = None


def _get_backward_kernel():
    """Lazily build and cache the backward Metal kernel."""
    global _BACKWARD_KERNEL
    if _BACKWARD_KERNEL is not None:
        return _BACKWARD_KERNEL
    if not mx.metal.is_available():
        return None

    # Scalar-gating version (matches Qwen3.5 GDN: g shape [B, T, Hv]).
    # T must be a compile-time constant for the register arrays; we use MAX_T
    # and only fill the first T entries. MAX_T=128 covers our use case (32-token
    # prompts with T<=32; can be raised for longer sequences).
    source = """
        // Grid: (32, Dv, B*Hv). Threadgroup: (32, 4, 1).
        // Each thread: one (b, hv, dv, dk_chunk) where dk_chunk = n_per_t slots
        // starting at dk_idx*n_per_t. 32 dk threads cover Dk=192 with n_per_t=6.
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto dk_idx = thread_position_in_threadgroup.x;       // 0..31
        auto dv_idx = thread_position_in_grid.y;               // 0..Dv-1
        constexpr int n_per_t = Dk / 32;
        constexpr int MAX_T = 128;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        // v: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        // dy: [B, T, Hv, Dv]
        auto dy_ = dy + b_idx * T * Hv * Dv + hv_idx * Dv;
        // beta: [B, T, Hv]
        auto beta_ = beta + b_idx * T * Hv;
        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        // state_in: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;

        // Load this thread's n_per_t slice of the initial state.
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }

        // Phase 1: forward, storing per-t s_dec and delta in registers.
        float s_dec_store[MAX_T][n_per_t];
        float delta_store[MAX_T];

        for (int t = 0; t < T; ++t) {
          // Decay
          float g_t = g_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            state[i] *= g_t;
            s_dec_store[t][i] = state[i];
          }
          // kv_mem = sum_dk state * k  (reduce over 32 dk threads)
          float kv_mem_partial = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            kv_mem_partial += state[i] * k_[n_per_t * dk_idx + i];
          }
          float kv_mem = simd_sum(kv_mem_partial);
          float beta_t = beta_[hv_idx];
          float delta = (v_[dv_idx] - kv_mem) * beta_t;
          delta_store[t] = delta;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] += k_[n_per_t * dk_idx + i] * delta;
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          dy_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        // Phase 2: reverse scan.
        float s_bar[n_per_t];
        for (int i = 0; i < n_per_t; ++i) s_bar[i] = 0.0f;

        q_ -= Hk * Dk;
        k_ -= Hk * Dk;
        v_ -= Hv * Dv;
        dy_ -= Hv * Dv;
        g_ -= Hv;
        beta_ -= Hv;

        for (int t = T - 1; t >= 0; --t) {
          float g_t = g_[hv_idx];
          float beta_t = beta_[hv_idx];
          float dy_val = dy_[dv_idx];

          float d_delta_partial = 0.0f;
          float ds_post[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            float s_post_val = s_dec_store[t][i] + k_[s_idx] * delta_store[t];
            ds_post[i] = dy_val * q_[s_idx] + s_bar[i];
            d_delta_partial += ds_post[i] * k_[s_idx];
          }
          float d_delta = simd_sum(d_delta_partial);
          float d_kv_mem = -d_delta * beta_t;

          // dq, dk: atomic-add this thread's partials.
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            float s_post_val = s_dec_store[t][i] + k_[s_idx] * delta_store[t];
            float dq_partial = dy_val * s_post_val;
            float dk_partial = ds_post[i] * delta_store[t] + d_kv_mem * s_dec_store[t][i];
            device atomic<float>* dq_ptr = (device atomic<float>*)(dq_hv_buf + (b_idx * T * Hv * Dk) + (t * Hv * Dk) + (hv_idx * Dk) + s_idx);
            device atomic<float>* dk_ptr = (device atomic<float>*)(dk_hv_buf + (b_idx * T * Hv * Dk) + (t * Hv * Dk) + (hv_idx * Dk) + s_idx);
            atomic_fetch_add_explicit(dq_ptr, dq_partial, memory_order_relaxed);
            atomic_fetch_add_explicit(dk_ptr, dk_partial, memory_order_relaxed);
          }

          // dv: thread 0 of simdgroup writes (d_delta is simd_sum, valid on thread 0).
          if (thread_index_in_simdgroup == 0) {
            dv_out[(b_idx * T * Hv * Dv) + (t * Hv * Dv) + (hv_idx * Dv) + dv_idx] =
                static_cast<OutT>(d_delta * beta_t);
          }

          // s_bar = (ds_post + d_kv_mem * k) * g  (for t-1)
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            float ds_dec = ds_post[i] + d_kv_mem * k_[s_idx];
            s_bar[i] = ds_dec * g_t;
          }

          q_ -= Hk * Dk;
          k_ -= Hk * Dk;
          v_ -= Hv * Dv;
          dy_ -= Hv * Dv;
          g_ -= Hv;
          beta_ -= Hv;
        }
    """
    _BACKWARD_KERNEL = mx.fast.metal_kernel(
        name="gdn_backward_v3",
        input_names=["q", "k", "v", "dy", "g", "beta", "state_in", "T",
                     "dq_hv_buf", "dk_hv_buf"],
        output_names=["dv_out"],
        source=source,
    )
    return _BACKWARD_KERNEL


def gdn_kernel_vjp(
    q: mx.array, k: mx.array, v: mx.array,
    g: mx.array, beta: mx.array, state: Optional[mx.array],
    dy: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Compute (dq, dk, dv) via the Metal backward kernel.

    q, k: [B, T, Hk, Dk]. v: [B, T, Hv, Dv]. g: [B, T, Hv]. beta: [B, T, Hv].
    dy: [B, T, Hv, Dv].
    Returns dq, dk: [B, T, Hk, Dk], dv: [B, T, Hv, Dv].
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    rf = Hv // Hk
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    # Pre-zero the dq_hv, dk_hv accumulation buffers (atomic adds accumulate into these).
    dq_hv = mx.zeros((B, T, Hv, Dk), dtype=mx.float32)
    dk_hv = mx.zeros((B, T, Hv, Dk), dtype=mx.float32)

    kernel = _get_backward_kernel()
    if kernel is None:
        # CPU fallback (no Metal) -- use the ops backward.
        dq_ops, dk_ops, dv = _ops_backward(q, k, v, g, beta, state, dy)
        return dq_ops, dk_ops, dv

    # Cast inputs to what the kernel expects. Use fp32 for accuracy; the
    # kernel's internal state is fp32 anyway.
    q_f = q.astype(mx.float32)
    k_f = k.astype(mx.float32)
    v_f = v.astype(mx.float32)
    g_f = g.astype(mx.float32)
    beta_f = beta.astype(mx.float32)
    state_f = state.astype(mx.float32)
    dy_f = dy.astype(mx.float32)

    dv_out, = kernel(
        inputs=[q_f, k_f, v_f, dy_f, g_f, beta_f, state_f, T, dq_hv, dk_hv],
        template=[("InT", mx.float32), ("OutT", mx.float32),
                  ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv)],
        output_dtypes=[mx.float32],
    )
    mx.eval(dv_out, dq_hv, dk_hv)

    # Reduce dq_hv, dk_hv from Hv heads to Hk heads (sum over the rf repeats).
    if rf > 1:
        dq = dq_hv.reshape(B, T, Hk, rf, Dk).sum(axis=-2)
        dk = dk_hv.reshape(B, T, Hk, rf, Dk).sum(axis=-2)
    else:
        dq = dq_hv
        dk = dk_hv

    # Cast to match input dtype
    dq = dq.astype(q.dtype)
    dk = dk.astype(k.dtype)
    dv = dv_out.astype(v.dtype)
    return dq, dk, dv