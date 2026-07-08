# fit-05 — Metal GDN backward v4: decay-gate paths at kernel speed

**Commit:** `0e413cd`.
**Code:** `jlens_qwen/custom_gdn_vjp.py` (kernel v4), `custom_gdn_patch.py`.

## The problem

The custom Metal GDN VJP (fit-01) returned **zeros for `dg` and `dβ`** — the
gradients w.r.t. the decay gate `g` and write gate `β`. Both are projections
of the layer input (`in_proj_a`, `in_proj_b`), so the true VJP flows gradient
back through them. Dropping them silently made a kernel-fit lens differ from
an ops-fit lens.

`scripts/measure_gbeta_gap.py` quantified the omission on real activations:
the `dg`/`dβ` paths contribute **4.9–7.5 % of ‖M_ℓ‖** (measured at GDN
layers 0 / 32 / 62). In a GDN model the decay gate is part of how *current*
activity influences *future* outputs — exactly what the J-lens measures — so
for intervention-grade quality they must be included.

The catch: the *ops* BPTT includes them but is slow (~70 s per GDN layer
in-fit). Including them via ops would have pushed the full fit to ~18 h.

## The fix

Extend the Metal backward kernel to compute `dg`/`dβ` itself:

- Store **`s_pre`** (pre-decay state) per timestep instead of `s_dec`, so
  `dg = Σ ds_dec ⊙ s_pre` needs no division by `g_t` — NaN-safe at saturated
  gates (`g → 0`).
- Accumulate `dg`/`dβ` via the existing threadgroup reduction + atomics.
- `gdn_kernel_vjp(..., return_gbeta=True)`; the custom-function VJP
  (`custom_gdn_patch.py`) now returns real `dg`/`dβ`, so `per_layer_jacobian`
  and module autograd include the `x → g/β` paths at kernel speed.

## Result

- GDN `M_ℓ` in-fit: **~8.5 s** (kernel, g/β included) vs. **~70 s** (ops).
- **Full fit: 20 prompts × 63 layers (L0–L62), 164.7 min (~2.75 h)** on an
  M4 Pro — full depth, intervention-grade, versus ~8 h for the 23-layer
  baseline (fit-01). This is the **v0.2-fulldepth** lens.

Readout sanity on the currency prompt after fitting: `Italy`/`Italian`
dominant L24–L48 (the 23-layer baseline couldn't see below L40), →
`euro`/`Euro` at L58–62, matching the model's own output.

## Verification

- Kernel `dg`/`dβ` vs. the batched ops BPTT (`gdn_vjp_batched`): **~3e-7**,
  at small AND real head dims (Dk=128, Dv=128, Hv=48), including saturated
  gates `g ~ 1e-14` (`tests/test_analytic_attention.py`).
- End-to-end GDN branch through the kernel path with g/β vs. brute-force
  VJP: **3.3e-9**.
