"""Generate golden data for the decode correctness gate.

Freezes the CURRENT decode behavior so future performance work can be
checked against it exactly:

- 64 greedy tokens from the cached StreamSession path,
- top-10 readout (ids + scores) at 3 positions for all record layers.

Also prints a diagnostic comparing the batched `_readout_at_positions`
against a naive per-position reference, to calibrate the tie-aware
epsilon used by the gate's equivalence test.

Regenerating goldens is a DELIBERATE act: only when a numerics-changing
optimization has an approved tolerance policy (see docs/perf/LEDGER.md
Gate section), or the model/lens/MLX version changes. Run:

    uv run python scripts/gen_decode_gate_golden.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "The quick brown fox jumps over the lazy dog. In machine learning,"
HORIZON = 64
TOP_N = 10
MODEL_ID = os.environ.get("JLENS_MODEL", "mlx-community/Qwen3.6-27B-4bit")
LENS_PATH = os.environ.get(
    "JLENS_PATH", os.path.join(REPO, "data", "lens", "full_depth_analytic.npz")
)
OUT = os.path.join(REPO, "tests", "golden", "decode_gate_golden.json")


def main() -> None:
    from jlens_qwen import serve
    from jlens_qwen.lens import JacobianLens
    from jlens_qwen.model import load
    from jlens_qwen.patch_gdn import set_inference_mode

    print(f"loading {MODEL_ID} ...", flush=True)
    model = load(MODEL_ID)
    set_inference_mode(True)
    lens = JacobianLens.load(LENS_PATH) if os.path.exists(LENS_PATH) else None
    serve._model = model
    serve._lens = lens
    print(f"lens: {lens}", flush=True)

    layers = (
        sorted(set(lens.source_layers) | {model.n_layers - 1})
        if lens is not None
        else [model.n_layers - 1]
    )

    # --- golden tokens: cached greedy decode --------------------------------
    input_ids = model.encode(PROMPT)
    session = model.make_stream()
    logits, _ = session.extend(input_ids[0].tolist())
    tokens = []
    for _ in range(HORIZON):
        tok = int(mx.argmax(logits.astype(mx.float32)).tolist())
        tokens.append(tok)
        logits, _ = session.extend([tok])
    print(f"golden tokens: {model.tokenizer.decode(tokens)!r}", flush=True)

    # --- golden readout: batched path on a fresh prefill --------------------
    session = model.make_stream(capture_layers=layers)
    ids = input_ids[0].tolist()
    _, acts = session.extend(ids)
    positions = [0, len(ids) // 2, len(ids) - 1]
    out = serve._readout_at_positions(acts, positions, layers, TOP_N)

    readout = {
        str(layer): {
            "ids": out[layer]["top_ids"],
            "scores": out[layer]["top_scores"],
        }
        for layer in layers
    }

    # --- diagnostic: batched vs naive per-position reference ----------------
    max_common_diff = 0.0
    max_boundary_gap = 0.0
    set_mismatch_cells = 0
    order_mismatch_cells = 0
    n_cells = 0
    for layer in layers:
        for pi, pos in enumerate(positions):
            n_cells += 1
            h = acts[layer][0][pos].astype(mx.float32)[None]
            if lens is not None and layer in lens.jacobians:
                h = lens.transport(h, layer)
            logits = model.unembed(model.final_norm(h)).astype(mx.float32)[0]
            order = mx.argsort(logits)[-TOP_N:][::-1]
            ref_ids = [int(t) for t in order.tolist()]
            ref_scores = np.array(mx.take_along_axis(logits, order, axis=-1).tolist())

            got_ids = out[layer]["top_ids"][pi]
            got_scores = np.array(out[layer]["top_scores"][pi])

            common = set(got_ids) & set(ref_ids)
            gm = {i: s for i, s in zip(got_ids, got_scores)}
            rm = {i: s for i, s in zip(ref_ids, ref_scores)}
            for cid in common:
                max_common_diff = max(max_common_diff, abs(gm[cid] - rm[cid]))
            odd = set(got_ids) ^ set(ref_ids)
            if odd:
                set_mismatch_cells += 1
                kth = min(ref_scores[-1], got_scores[-1])
                for oid in odd:
                    s = gm.get(oid, rm.get(oid))
                    max_boundary_gap = max(max_boundary_gap, abs(s - kth))
            elif got_ids != ref_ids:
                order_mismatch_cells += 1

    print(
        f"[diag] cells={n_cells} "
        f"max |score diff| on common ids = {max_common_diff:.3e}; "
        f"order-only mismatches = {order_mismatch_cells}; "
        f"set mismatches = {set_mismatch_cells} "
        f"(max boundary gap = {max_boundary_gap:.3e})",
        flush=True,
    )

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
        capture_output=True, text=True,
    ).stdout.strip()
    golden = {
        "meta": {
            "model": MODEL_ID,
            "lens": os.path.relpath(LENS_PATH, REPO) if lens else None,
            "prompt": PROMPT,
            "horizon": HORIZON,
            "top_n": TOP_N,
            "positions": positions,
            "generated": date.today().isoformat(),
            "commit": commit,
            "mlx": __import__("mlx.core", fromlist=["__version__"]).__version__,
        },
        "tokens": tokens,
        "readout": readout,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
