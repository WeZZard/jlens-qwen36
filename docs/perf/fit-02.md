# fit-02 — Analytic MLP branch via the Hadamard trick

**Commit:** `2cd8af7` — "Analytic MLP Jacobian via Hadamard trick: 77x
speedup, verified exact (6.7e-5)".

## The insight

The cotangent basis for a full Jacobian is the **identity**.
Backpropagating the identity through a weight matrix is not 5120
vector-Jacobian products — it *is* the weight matrix. So instead of
extracting `M_ℓ` column-by-column through autograd, assemble it as explicit
matrix products.

The MLP branch is `y = down(silu(gate(x)) ⊙ up(x))`. Its Jacobian at one
position `s` is:

```
J(s) = W_down [ diag(silu'(g_s) ⊙ u_s) W_gate + diag(silu(g_s)) W_up ]
```

The fit averages over source positions. The position dependence enters
*only* through those diagonals, so the position sum factors out via a
Hadamard identity:

```
Σ_s diag(a_s) · W · diag(ln_s)  =  W ⊙ (Σ_s a_s ln_sᵀ)
```

— a rank-(number-of-valid-positions) outer-product sum. The whole
position-averaged MLP-branch Jacobian becomes **two elementwise masks + a
couple of GEMMs** instead of 5120 separate VJP calls, each re-running the
MLP forward.

## Result

**~77×** on the MLP branch — historically the single largest term in a
DecoderLayer's Jacobian (~267M of the ~383M params per GDN layer live in
the MLP).

## Verification

Exact math, so the check is direct: compare the assembled `M_MLP` against
`mx.vjp` on the same input. Max relative error **6.7e-5** (fp32 GEMM noise).

## Why it mattered

This proved the identity-basis assembly idea works and is exact. fit-04
generalizes the same trick to the *attention* branch — the other half of
the layer — which was still the VJP bottleneck after this.
