"""Integration check: analytic layer Jacobians vs the exact VJP reference.

Compares, on REAL model activations:
  - M_new  = decoder_layer_jacobian(analytic_attn=True)   (exact branches,
             one averaged-product junction)
  - M_old  = decoder_layer_jacobian(analytic_attn=False)  (the hybrid the
             2026-07-08 full-depth fit was launched with; three junctions)
  - M_ref  = fit.per_layer_jacobian (5120 one-hot VJPs, exact per layer,
             ~60s each; custom-kernel semantics, i.e. g/beta dropped)

against each other, for one GDN and one FA layer (plus more via --layers).
Both analytic variants use include_gbeta=False to match M_ref's semantics.

Expected discrepancies vs M_ref:
  - the branch-product junction (synthetic bound: ~0.6-1.3% rel Frobenius
    for the new path, ~1.8-2.2% for the old — see
    tests/test_analytic_attention.py::test_full_layer_junction_error);
  - fp32 re-forward in the analytic path vs model-dtype forward in the
    reference (small).

Decision rule: if new <= old and new is at the ~1e-2 level, wire
analytic_attn=True into the next fit (orchestrator: fit_analytic.py) and
confirm with the readout sanity check (scripts/workspace_range.py).

Needs the model + GPU (~5 min per layer, dominated by M_ref). Run only
when no fit is active:

    uv run python scripts/verify_analytic_layer.py --layers 30 31
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".")

PROMPT = "The capital of the country famous for the Colosseum uses a currency called the"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="layer indices; default: one mid GDN + one mid FA")
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument("--model-id", default="mlx-community/Qwen3.6-27B-4bit")
    args = parser.parse_args()

    print("NOTE: needs ~20 GB and the GPU — do not run while a fit is active.")

    from jlens_qwen.model import load
    from jlens_qwen.fit import per_layer_jacobian
    from jlens_qwen.analytic_layer import decoder_layer_jacobian

    model = load(args.model_id)

    if args.layers:
        layers = args.layers
    else:
        n = model.n_layers
        gdn = [i for i, l in enumerate(model.layers) if l.is_linear]
        fa = [i for i, l in enumerate(model.layers) if not l.is_linear]
        layers = [gdn[len(gdn) // 2], fa[len(fa) // 2]]
    print(f"layers: {layers}")

    input_ids = model.encode(PROMPT, max_length=args.max_seq_len)
    capture = sorted({l - 1 for l in layers})
    _, acts = model.forward(input_ids, capture_layers=capture)
    for c in capture:
        mx.eval(acts[c])

    for l in layers:
        kind = "GDN" if model.layers[l].is_linear else "FA"
        h_in = acts[l - 1]  # [1, S, D] residual entering layer l

        t0 = time.perf_counter()
        M_new = np.array(decoder_layer_jacobian(
            model, l, h_in, skip_first=args.skip_first,
            analytic_attn=True, include_gbeta=False,
        ))
        t_new = time.perf_counter() - t0

        t0 = time.perf_counter()
        M_old = np.array(decoder_layer_jacobian(
            model, l, h_in, skip_first=args.skip_first, analytic_attn=False,
        ))
        t_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        M_ref = per_layer_jacobian(model, h_in, l, skip_first=args.skip_first)
        t_ref = time.perf_counter() - t0

        ref_norm = np.linalg.norm(M_ref)
        rel_new = np.linalg.norm(M_new - M_ref) / ref_norm
        rel_old = np.linalg.norm(M_old - M_ref) / ref_norm
        print(
            f"layer {l:2d} ({kind}): "
            f"rel err new={rel_new:.3e} ({t_new:.1f}s)  "
            f"old={rel_old:.3e} ({t_old:.1f}s)  "
            f"ref exact VJP ({t_ref:.0f}s)"
        )


if __name__ == "__main__":
    main()
