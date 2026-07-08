# fit-03 — Closed-form final-norm Jacobian

**Commit:** `66e85dc` — "Quick wins from PERFORMANCE_REVIEW: closed-form
J_64 (12000x faster) …".

## The problem

The chain starts from `J_norm = d(final_norm(h))/dh`, the Jacobian of the
model's final pre-unembed RMSNorm. The baseline computed it the same way as
every other block: 5120 one-hot VJP calls, **~1 minute per prompt**
(`fit.py`).

But RMSNorm has a closed-form Jacobian. For `y = w ⊙ x / r` with
`r = sqrt(mean(x²) + ε)`:

```
J_norm(s) = diag(w)/r_s  −  (w ⊙ x̂_s) x̂_sᵀ / (D r_s)
```

a diagonal term plus a single rank-1 correction — no autograd needed.

## Result

**~12000×** on this piece: **4.9 ms** vs. ~60 s. It stops being a
measurable line item in the per-prompt cost.

## Verification

Closed-form vs. VJP reference: **0.027 %** relative error.

## Also in this commit

Two non-speed items landed alongside, because the same review surfaced them:

- **`fit.py` docstring fix** — the fit computes the *future-summed*
  cross-position influence `Σ_{t≥s} d(h_{ℓ+1}[t])/d(h_ℓ[s])`, matching the
  paper's J-lens definition ("makes a word more likely at some point in the
  future"). The old docstring wrongly called it "position-diagonal". The
  *math* was already right; only the comment was wrong.
- **Rademacher probing module** (`probing.py`) — an unbiased ~10× estimator
  (k≈512 ±1 cotangents instead of 5120 one-hots). Shelved after measurement:
  single-prompt variance was too high (316 % Frobenius at k=512) for the
  intervention-grade lens, though it averages down across many prompts.
