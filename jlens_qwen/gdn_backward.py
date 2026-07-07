"""Manual backward (BPTT) for Gated DeltaNet, computing the full per-layer
Jacobian M_l = d(h_{l+1})/d(h_l) in a batched fashion.

The GDN forward is linear in (q, k, v) given (g, beta, state_init). The
Jacobian d(y)/d(q, k, v) can be computed via a single backward recurrence
over time. Critically, we can batch D output dims of cotangents through
ONE backward pass (the backward recurrence cost is independent of the
number of cotangents), giving a D-fold speedup over one-at-a-time VJP.

This module computes the full D x D per-layer Jacobian M_l. The fit then
chain-multiplies: J_l = J_{l+1} @ M_l.

Correctness: verified against mx.vjp on a tiny example (see test below).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np


def gdn_forward(q, k, v, g, beta, state=None):
    """Reference forward (ops loop). Returns (y, states_pre, states_dec, deltas, k_r, q_r).

    states_pre[t] = state before timestep t (input state)
    states_dec[t] = states_pre[t] * decay_t
    deltas[t] = (v[t] - kv_mem) * beta[t]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    rf = Hv // Hk
    q_r = mx.repeat(q, rf, axis=-2) if rf > 1 else q
    k_r = mx.repeat(k, rf, axis=-2) if rf > 1 else k
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    states_pre = []
    states_dec = []
    deltas = []
    s = state
    ys = []
    for t in range(T):
        states_pre.append(s)
        if g.ndim == 3:
            decay = g[:, t, :, None, None]
        elif g.ndim == 4:
            decay = g[:, t, :, None, :]
        else:
            raise ValueError(f"Unexpected g shape {g.shape}")
        s_dec = s * decay
        states_dec.append(s_dec)
        kv_mem = (s_dec * k_r[:, t, :, None, :]).sum(-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]  # beta[:,t] is [B, Hv] -> [B, Hv, 1]
        deltas.append(delta)
        s = s_dec + k_r[:, t, :, None, :] * delta[..., None]
        y = (s * q_r[:, t, :, None, :]).sum(-1)
        ys.append(y)
    return mx.stack(ys, axis=1), states_pre, states_dec, deltas, k_r, q_r


def gdn_vjp(q, k, v, g, beta, state, dy):
    """Backward through GDN. Given cotangent dy [B, T, Hv, Dv], return
    (dq, dk, dv) with the same shapes as (q, k, v).

    BPTT over time. The running adjoint s_bar accumulates the gradient
    w.r.t. the recurrent state, which flows back from future timesteps.
    """
    y, states_pre, states_dec, deltas, k_r, q_r = gdn_forward(q, k, v, g, beta, state)
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    rf = Hv // Hk

    dq = mx.zeros_like(q_r)
    dk = mx.zeros_like(k_r)
    dv = mx.zeros_like(v)
    s_bar = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    for t in range(T - 1, -1, -1):
        s_pre = states_pre[t]
        s_dec = states_dec[t]
        delta = deltas[t]
        dy_t = dy[:, t]  # [B, Hv, Dv]

        # s_post for this timestep = s_dec + k[t]*delta, BUT the running
        # adjoint s_bar carries gradients from FUTURE timesteps w.r.t. their
        # s_pre, which equals THIS timestep's s_post. So:
        #   ds_post = (contribution from dy[t]) + s_bar
        ds_post_from_dy = dy_t[:, :, :, None] * q_r[:, t, :, None, :]  # [B, Hv, Dv, Dk]
        ds_post = ds_post_from_dy + s_bar

        # y[t] = sum_dk s_post * q[t], so d y[t] / d q[t] = s_post.
        #   d q[t] = sum_dv dy[t] * s_post  -> [B, Hv, Dk]
        s_post = states_dec[t] + k_r[:, t, :, None, :] * delta[:, :, :, None]  # [B, Hv, Dv, Dk]
        dq[:, t] = dq[:, t] + (dy_t[:, :, :, None] * s_post).sum(axis=-2)

        # s_post = s_dec + k[t] * delta[..., None]
        # d s_dec += d s_post  (the s_dec term)
        # d k[t] += sum_dv d s_post * delta   -> result [B, Hv, Dk]
        # d delta += sum_dk d s_post * k[t]  -> result [B, Hv, Dv]
        ds_dec = ds_post  # partial from the s_dec term
        # dk[:, t] is [B, Hv, Dk]; ds_post is [B, Hv, Dv, Dk], delta is [B, Hv, Dv]
        dk[:, t] = dk[:, t] + (ds_post * delta[:, :, :, None]).sum(axis=-2)
        d_delta = (ds_post * k_r[:, t, :, None, :]).sum(axis=-1)  # [B, Hv, Dv]

        # delta = (v[t] - kv_mem) * beta[t], kv_mem = sum_dk s_dec * k[t]
        # d v[t] += d_delta * beta[t]
        # d kv_mem -= d_delta * beta[t]
        # d s_dec += d_kv_mem * k[t]  (kv_mem = s_dec . k[t], so d s_dec = d_kv_mem * k[t])
        #   -> d_kv_mem is [B, Hv, Dv], k[t] is [B, Hv, Dk], product is [B, Hv, Dv, Dk]
        # d k[t] += d_kv_mem * s_dec  (same broadcast)
        dv[:, t] = dv[:, t] + d_delta * beta[:, t, :, None]  # [B, Hv, Dv]
        d_kv_mem = -d_delta * beta[:, t, :, None]  # [B, Hv, Dv]
        ds_dec = ds_dec + d_kv_mem[:, :, :, None] * k_r[:, t, :, None, :]  # [B, Hv, Dv, Dk]
        dk[:, t] = dk[:, t] + (d_kv_mem[:, :, :, None] * s_dec).sum(axis=-2)  # [B, Hv, Dk]

        # s_dec = s_pre * decay_t
        # d s_pre += d s_dec * decay_t
        if g.ndim == 3:
            decay_t = g[:, t, :, None, None]
        else:
            decay_t = g[:, t, :, None, :]
        ds_pre = ds_dec * decay_t

        # s_pre for this t = s_post for t-1. So s_bar for t-1 = ds_pre.
        s_bar = ds_pre

    # Reduce dq, dk from Hv heads back to Hk if rf > 1.
    if rf > 1:
        # q_r, k_r were repeated. The original q, k have Hk heads, each
        # repeated rf times. The gradient w.r.t. the original is the sum
        # over the rf repeats.
        dq = dq.reshape(B, T, Hk, rf, Dk).sum(axis=-2)
        dk = dk.reshape(B, T, Hk, rf, Dk).sum(axis=-2)

    return dq, dk, dv


def gdn_vjp_batched(q, k, v, g, beta, state, dy):
    """Batched-cotangent BPTT through GDN with a shared forward.

    Unlike ``gdn_vjp`` (one cotangent set, B tied between primals and dy),
    this takes B=1 primals and C independent cotangent sets stacked on a
    leading axis. The forward states are computed once and broadcast; only
    the adjoint recurrence is batched over C.

    Also returns dg and dbeta (the decay-gate and write-gate gradients),
    which ``gdn_vjp`` / the Metal kernel do not compute.

    q, k: [1, T, Hk, Dk]. v: [1, T, Hv, Dv]. g, beta: [1, T, Hv] (scalar
    gating only). state: [1, Hv, Dv, Dk] or None. dy: [C, T, Hv, Dv].
    Returns dq, dk: [C, T, Hk, Dk], dv: [C, T, Hv, Dv], dg, dbeta: [C, T, Hv].
    """
    assert g.ndim == 3, f"batched BPTT supports scalar gating only, got {g.shape}"
    y, states_pre, states_dec, deltas, k_r, q_r = gdn_forward(q, k, v, g, beta, state)
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    rf = Hv // Hk
    C = dy.shape[0]

    dq_ts, dk_ts, dv_ts, dg_ts, db_ts = [], [], [], [], []
    s_bar = mx.zeros((C, Hv, Dv, Dk), dtype=mx.float32)

    for t in range(T - 1, -1, -1):
        s_pre = states_pre[t]          # [1, Hv, Dv, Dk]
        s_dec = states_dec[t]          # [1, Hv, Dv, Dk]
        delta = deltas[t]              # [1, Hv, Dv]
        k_t = k_r[:, t]                # [1, Hv, Dk]
        q_t = q_r[:, t]                # [1, Hv, Dk]
        dy_t = dy[:, t]                # [C, Hv, Dv]

        ds_post = dy_t[:, :, :, None] * q_t[:, :, None, :] + s_bar  # [C, Hv, Dv, Dk]
        s_post = s_dec + k_t[:, :, None, :] * delta[:, :, :, None]  # [1, Hv, Dv, Dk]
        dq_ts.append((dy_t[:, :, :, None] * s_post).sum(axis=-2))   # [C, Hv, Dk]

        dk_partial = (ds_post * delta[:, :, :, None]).sum(axis=-2)  # [C, Hv, Dk]
        d_delta = (ds_post * k_t[:, :, None, :]).sum(axis=-1)       # [C, Hv, Dv]

        dv_ts.append(d_delta * beta[:, t, :, None])                 # [C, Hv, Dv]
        kv_mem = (s_dec * k_t[:, :, None, :]).sum(axis=-1)          # [1, Hv, Dv]
        db_ts.append((d_delta * (v[:, t] - kv_mem)).sum(axis=-1))   # [C, Hv]

        d_kv_mem = -d_delta * beta[:, t, :, None]                   # [C, Hv, Dv]
        ds_dec = ds_post + d_kv_mem[:, :, :, None] * k_t[:, :, None, :]
        dk_ts.append(dk_partial + (d_kv_mem[:, :, :, None] * s_dec).sum(axis=-2))

        dg_ts.append((ds_dec * s_pre).sum(axis=(-1, -2)))           # [C, Hv]

        decay_t = g[:, t, :, None, None]                            # [1, Hv, 1, 1]
        s_bar = ds_dec * decay_t

    dq = mx.stack(dq_ts[::-1], axis=1)   # [C, T, Hv, Dk]
    dk = mx.stack(dk_ts[::-1], axis=1)   # [C, T, Hv, Dk]
    dv = mx.stack(dv_ts[::-1], axis=1)   # [C, T, Hv, Dv]
    dg = mx.stack(dg_ts[::-1], axis=1)   # [C, T, Hv]
    dbeta = mx.stack(db_ts[::-1], axis=1)  # [C, T, Hv]

    if rf > 1:
        dq = dq.reshape(C, T, Hk, rf, Dk).sum(axis=-2)
        dk = dk.reshape(C, T, Hk, rf, Dk).sum(axis=-2)

    return dq, dk, dv, dg, dbeta


def test_gdn_vjp():
    """Numerical gradient check against mx.vjp on a tiny example."""
    B, T, Hk, Dk = 1, 4, 2, 3
    Hv, Dv = 2, 2
    np.random.seed(0)
    q = mx.array(np.random.randn(B, T, Hk, Dk).astype(np.float32))
    k = mx.array(np.random.randn(B, T, Hk, Dk).astype(np.float32))
    v = mx.array(np.random.randn(B, T, Hv, Dv).astype(np.float32))
    g = mx.array(np.random.randn(B, T, Hv).astype(np.float32))
    beta = mx.array(np.random.randn(B, T, Hv).astype(np.float32))

    def fwd(q, k, v):
        y, *_ = gdn_forward(q, k, v, g, beta, state=None)
        return y

    y = fwd(q, k, v)
    dy = mx.array(np.random.randn(*y.shape).astype(np.float32))

    # Reference via mx.vjp
    _, (dq_ref, dk_ref, dv_ref) = mx.vjp(fwd, [q, k, v], [dy])
    mx.eval(dq_ref, dk_ref, dv_ref)

    # Manual
    dq, dk, dv = gdn_vjp(q, k, v, g, beta, state=None, dy=dy)
    mx.eval(dq, dk, dv)

    dq_r = np.array(dq_ref); dq_m = np.array(dq)
    dk_r = np.array(dk_ref); dk_m = np.array(dk)
    dv_r = np.array(dv_ref); dv_m = np.array(dv)

    print(f"dq max abs err: {np.abs(dq_r - dq_m).max():.2e}")
    print(f"dk max abs err: {np.abs(dk_r - dk_m).max():.2e}")
    print(f"dv max abs err: {np.abs(dv_r - dv_m).max():.2e}")
    assert np.allclose(dq_r, dq_m, atol=1e-4), "dq mismatch"
    assert np.allclose(dk_r, dk_m, atol=1e-4), "dk mismatch"
    assert np.allclose(dv_r, dv_m, atol=1e-4), "dv mismatch"
    print("GDN VJP test PASSED")


def test_gdn_vjp_batched():
    """Check gdn_vjp_batched (incl. dg/dbeta) against mx.vjp, C cotangents."""
    B, T, Hk, Dk = 1, 5, 2, 4
    Hv, Dv = 4, 3
    C = 3
    np.random.seed(1)
    q = mx.array(np.random.randn(B, T, Hk, Dk).astype(np.float32))
    k = mx.array(np.random.randn(B, T, Hk, Dk).astype(np.float32))
    v = mx.array(np.random.randn(B, T, Hv, Dv).astype(np.float32))
    g = mx.array(np.random.rand(B, T, Hv).astype(np.float32))
    beta = mx.array(np.random.rand(B, T, Hv).astype(np.float32))
    dy = mx.array(np.random.randn(C, T, Hv, Dv).astype(np.float32))

    def fwd(q, k, v, g, beta):
        y, *_ = gdn_forward(q, k, v, g, beta, state=None)
        return y

    dq, dk, dv, dg, dbeta = gdn_vjp_batched(q, k, v, g, beta, None, dy)
    mx.eval(dq, dk, dv, dg, dbeta)

    for c in range(C):
        _, refs = mx.vjp(fwd, [q, k, v, g, beta], [dy[c][None]])
        mx.eval(*refs)
        for name, got, ref in zip(
            ["dq", "dk", "dv", "dg", "dbeta"],
            [dq[c], dk[c], dv[c], dg[c], dbeta[c]],
            refs,
        ):
            err = np.abs(np.array(got) - np.array(ref[0])).max()
            assert err < 1e-4, f"{name} mismatch at cotangent {c}: {err:.2e}"
    print("GDN batched VJP test PASSED")


if __name__ == "__main__":
    test_gdn_vjp()
    test_gdn_vjp_batched()