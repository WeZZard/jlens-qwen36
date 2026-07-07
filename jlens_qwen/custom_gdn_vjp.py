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
    #
    # Layout v4: Grid = (1, 1, B*Hv*Dk). Each thread handles one (b, hv, dk) and
    # loops over ALL Dv internally (no cross-thread reduction for dq/dk). dv is
    # written per (b, hv, dv, t) by looping over Dv inside the thread too -- but
    # that's Dv*T writes per thread. Alternative: separate kernel for dv.
    # For simplicity, this kernel computes dq, dk (no atomics) and a second
    # small kernel computes dv.
    #
    # Actually, dq[dk] = sum_dv dy[t,dv] * s_post[t, dv, dk]. With one thread per
    # (b, hv, dk), it loops over Dv to sum. State per thread: state[dk] is a
    # scalar (not Dk-wide). The forward recurrence for one dk slot depends on
    # kv_mem = sum_dk state * k, which is a sum over ALL Dk -- so one thread can't
    # compute kv_mem alone. Need simd_sum over dk threads.
    #
    # Reverting to the v3 layout (32 dk threads x Dv dv threads) but using
    # threadgroup shared memory to reduce the 4 dv threads per threadgroup, then
    # atomic-add across the 32 threadgroups along Dv. This reduces atomics by 4x.
    # Even better: use a two-pass approach. Pass 1: each threadgroup writes its
    # reduced partial to a per-(b,hv,t,dk,tg_y) buffer (no atomics, direct write).
    # Pass 2: a tiny reduction kernel sums the 32 tg_y slots per (b,hv,t,dk).
    #
    # For now, keep v3 with atomics but reduce contention by having only 1 of
    # the 4 dv threads per threadgroup do the atomic add (after reducing the 4
    # via shared memory). 4x fewer atomics.
    source = """
        // Grid: (32, Dv, B*Hv). Threadgroup: (32, 4, 1).
        // 32 dk threads x 4 dv threads per threadgroup.
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto dk_idx = thread_position_in_threadgroup.x;       // 0..31
        auto dv_idx = thread_position_in_grid.y;               // 0..Dv-1
        auto tg_dv = thread_position_in_threadgroup.y;          // 0..3
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

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }

        // Phase 1: forward, storing per-t s_pre (PRE-decay state) and delta.
        // s_pre (not s_dec) is stored so dg = <ds_dec, s_pre> needs no
        // division by g_t (which underflows for saturated gates -> 0/0).
        // s_dec is reconstructed as s_pre * g_t where needed.
        float s_pre_store[MAX_T][n_per_t];
        float delta_store[MAX_T];

        for (int t = 0; t < T; ++t) {
          float g_t = g_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            s_pre_store[t][i] = state[i];
            state[i] *= g_t;
          }
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

        // Threadgroup shared memory for reducing the 4 dv threads' dq/dk
        // partials, plus per-dv dg/dbeta scalars.
        // 32 dk * n_per_t * 4 dv = 32*6*4 = 768 floats x2 + 8 = ~6KB, fine.
        threadgroup float dq_shared[32 * 6 * 4];  // [dk_idx][i][tg_dv] -- flattened
        threadgroup float dk_shared[32 * 6 * 4];
        threadgroup float dg_shared[4];
        threadgroup float db_shared[4];

        for (int t = T - 1; t >= 0; --t) {
          float g_t = g_[hv_idx];
          float beta_t = beta_[hv_idx];
          float dy_val = dy_[dv_idx];

          float d_delta_partial = 0.0f;
          float kv_mem_partial = 0.0f;
          float ds_post[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            ds_post[i] = dy_val * q_[s_idx] + s_bar[i];
            d_delta_partial += ds_post[i] * k_[s_idx];
            kv_mem_partial += (s_pre_store[t][i] * g_t) * k_[s_idx];
          }
          float d_delta = simd_sum(d_delta_partial);
          float kv_mem = simd_sum(kv_mem_partial);
          float d_kv_mem = -d_delta * beta_t;

          // Store this thread's dq, dk partials to shared memory.
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            float s_dec_val = s_pre_store[t][i] * g_t;
            float s_post_val = s_dec_val + k_[s_idx] * delta_store[t];
            float dq_partial = dy_val * s_post_val;
            float dk_partial = ds_post[i] * delta_store[t] + d_kv_mem * s_dec_val;
            dq_shared[(dk_idx * n_per_t + i) * 4 + tg_dv] = dq_partial;
            dk_shared[(dk_idx * n_per_t + i) * 4 + tg_dv] = dk_partial;
          }

          // dv: thread 0 of simdgroup writes.
          if (thread_index_in_simdgroup == 0) {
            dv_out[(b_idx * T * Hv * Dv) + (t * Hv * Dv) + (hv_idx * Dv) + dv_idx] =
                static_cast<OutT>(d_delta * beta_t);
          }

          // s_bar = (ds_post + d_kv_mem * k) * g, and the dg partial
          // dg = sum_{dv,dk} ds_dec * s_pre (this thread: its dk slots, its dv).
          float dg_partial = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            float ds_dec = ds_post[i] + d_kv_mem * k_[s_idx];
            dg_partial += ds_dec * s_pre_store[t][i];
            s_bar[i] = ds_dec * g_t;
          }
          float dg_dv = simd_sum(dg_partial);  // summed over dk; one per dv
          if (thread_index_in_simdgroup == 0) {
            dg_shared[tg_dv] = dg_dv;
            // dbeta = sum_dv d_delta * (v - kv_mem); d_delta already dk-summed.
            db_shared[tg_dv] = d_delta * (static_cast<float>(v_[dv_idx]) - kv_mem);
          }

          // Reduce the 4 dv threads' partials and atomic-add to global (1/4 atomics).
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tg_dv == 0) {
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              float dq_sum = 0.0f, dk_sum = 0.0f;
              for (int d = 0; d < 4; ++d) {
                dq_sum += dq_shared[(dk_idx * n_per_t + i) * 4 + d];
                dk_sum += dk_shared[(dk_idx * n_per_t + i) * 4 + d];
              }
              auto dq_ptr = dq_hv_buf + (b_idx * T * Hv * Dk) + (t * Hv * Dk) + (hv_idx * Dk) + s_idx;
              auto dk_ptr = dk_hv_buf + (b_idx * T * Hv * Dk) + (t * Hv * Dk) + (hv_idx * Dk) + s_idx;
              atomic_fetch_add_explicit((device atomic<float>*)dq_ptr, dq_sum, memory_order_relaxed);
              atomic_fetch_add_explicit((device atomic<float>*)dk_ptr, dk_sum, memory_order_relaxed);
            }
            if (dk_idx == 0) {
              float dg_sum = 0.0f, db_sum = 0.0f;
              for (int d = 0; d < 4; ++d) {
                dg_sum += dg_shared[d];
                db_sum += db_shared[d];
              }
              auto dg_ptr = dg_hv_buf + (b_idx * T * Hv) + (t * Hv) + hv_idx;
              auto db_ptr = db_hv_buf + (b_idx * T * Hv) + (t * Hv) + hv_idx;
              atomic_fetch_add_explicit((device atomic<float>*)dg_ptr, dg_sum, memory_order_relaxed);
              atomic_fetch_add_explicit((device atomic<float>*)db_ptr, db_sum, memory_order_relaxed);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          q_ -= Hk * Dk;
          k_ -= Hk * Dk;
          v_ -= Hv * Dv;
          dy_ -= Hv * Dv;
          g_ -= Hv;
          beta_ -= Hv;
        }
    """
    _BACKWARD_KERNEL = mx.fast.metal_kernel(
        name="gdn_backward_v4",
        input_names=["q", "k", "v", "dy", "g", "beta", "state_in", "T",
                     "dq_hv_buf", "dk_hv_buf", "dg_hv_buf", "db_hv_buf"],
        output_names=["dv_out"],
        source=source,
    )
    return _BACKWARD_KERNEL


def gdn_kernel_vjp(
    q: mx.array, k: mx.array, v: mx.array,
    g: mx.array, beta: mx.array, state: Optional[mx.array],
    dy: mx.array,
    *,
    return_gbeta: bool = False,
) -> Tuple[mx.array, ...]:
    """Compute (dq, dk, dv[, dg, dbeta]) via the Metal backward kernel.

    q, k: [B, T, Hk, Dk]. v: [B, T, Hv, Dv]. g: [B, T, Hv]. beta: [B, T, Hv].
    dy: [B, T, Hv, Dv].
    Returns dq, dk: [B, T, Hk, Dk], dv: [B, T, Hv, Dv], and with
    return_gbeta=True also dg, dbeta: [B, T, Hv] (the decay/write-gate
    gradients — verified against gdn_backward.gdn_vjp_batched).
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    rf = Hv // Hk
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    # Pre-zero the accumulation buffers (atomic adds accumulate into these).
    dq_hv = mx.zeros((B, T, Hv, Dk), dtype=mx.float32)
    dk_hv = mx.zeros((B, T, Hv, Dk), dtype=mx.float32)
    dg_hv = mx.zeros((B, T, Hv), dtype=mx.float32)
    db_hv = mx.zeros((B, T, Hv), dtype=mx.float32)

    kernel = _get_backward_kernel()
    if kernel is None:
        # CPU fallback (no Metal) -- ops backward; per-b batched BPTT when
        # the gate gradients are requested.
        if not return_gbeta:
            return _ops_backward(q, k, v, g, beta, state, dy)
        from .gdn_backward import gdn_vjp_batched
        outs = [
            gdn_vjp_batched(
                q[b:b + 1], k[b:b + 1], v[b:b + 1],
                g[b:b + 1], beta[b:b + 1], state[b:b + 1], dy[b:b + 1],
            )
            for b in range(B)
        ]
        return tuple(
            mx.concatenate([o[i] for o in outs], axis=0) for i in range(5)
        )

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
        inputs=[q_f, k_f, v_f, dy_f, g_f, beta_f, state_f, T,
                dq_hv, dk_hv, dg_hv, db_hv],
        template=[("InT", mx.float32), ("OutT", mx.float32),
                  ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv)],
        output_dtypes=[mx.float32],
    )
    mx.eval(dv_out, dq_hv, dk_hv, dg_hv, db_hv)

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
    if return_gbeta:
        return dq, dk, dv, dg_hv.astype(g.dtype), db_hv.astype(beta.dtype)
    return dq, dk, dv