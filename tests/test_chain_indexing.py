"""Decisive toy test for the chain-indexing question.

Ground truth: J_l must satisfy transport(acts[l], l) semantics, i.e.
J_l = d(final_norm_out)/d(acts[l]) — direct autograd through layers
l+1..end + norm, starting FROM acts[l].

Compare against (a) the chain as coded in fit.py/fit_analytic.py
(M_k at acts[k], saved at k) and (b) the proposed fix (M_k at acts[k-1],
saved at k-1).
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

mx.random.seed(0)
np.random.seed(0)
D, S, N = 6, 5, 3  # tiny: 3 "layers" + final norm

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = mx.random.normal((D, D)) * 0.3
        self.w2 = mx.random.normal((D, D)) * 0.3
    def __call__(self, h):
        return h + mx.tanh(h @ self.w1) @ self.w2  # residual block

layers = [Toy() for _ in range(N)]
norm_w = mx.random.normal((D,)) * 0.1 + 1.0
final_norm = lambda h: mx.fast.rms_norm(h, norm_w, 1e-6)

x0 = mx.random.normal((1, S, D))

# forward, capture acts[i] = OUTPUT of layer i (same as model.py)
acts = {}
h = x0
for i, l in enumerate(layers):
    h = l(h)
    acts[i] = h

valid = mx.ones((S,))  # all positions valid, keeps the toy simple

def jac(fn, h_in):
    """rows d: (1/S) sum_s sum_t d(fn(h)[t,d])/d(h[s]) — fit.py convention."""
    M = np.zeros((D, D), dtype=np.float32)
    for d in range(D):
        cot = valid[None, :, None] * mx.zeros((D,)).at[d].add(1.0)[None, None, :]
        _, vjps = mx.vjp(fn, [h_in], [cot])
        M[d] = np.array(vjps[0][0].mean(axis=0))
    return M

def from_layer(l0):
    def fn(h):
        for l in layers[l0:]:
            h = l(h)
        return final_norm(h)
    return fn

# ground truth J_l = d(final)/d(acts[l]): run layers l+1.. from acts[l]
J_true = {l: jac(from_layer(l + 1), acts[l]) for l in range(N - 1)}

J_norm = jac(final_norm, acts[N - 1])

# (a) chain as coded: M_l = Jac(layer l @ acts[l]); results[l] = running
J = J_norm.copy(); as_coded = {}
for l in range(N - 1, -1, -1):
    M = jac(lambda h: layers[l](h), acts[l])
    J = J @ M
    as_coded[l] = J.copy()

# (b) proposed fix: M_l = Jac(layer l @ acts[l-1]); results[l-1] = running
J = J_norm.copy(); fixed = {}
for l in range(N - 1, 0, -1):
    M = jac(lambda h: layers[l](h), acts[l - 1])
    J = J @ M
    fixed[l - 1] = J.copy()

for l in range(N - 1):
    t = np.linalg.norm(J_true[l])
    ea = np.linalg.norm(as_coded[l] - J_true[l]) / t
    eb = np.linalg.norm(fixed[l] - J_true[l]) / t
    print(f"J_{l}: rel err as-coded={ea:.2e}  fixed={eb:.2e}")
