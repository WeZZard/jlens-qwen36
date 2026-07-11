"""Measure the model's functional bands (sensory / workspace / motor) from
per-layer J-lens statistics, instead of the UI's percentage-split guess.

Following the global-workspace paper's four per-layer signals:
  1. next-token top-k accuracy  — spikes in the MOTOR band (readout == output)
  2. top-1 persistence          — high across the WORKSPACE (a concept stays
     loaded across positions = "broadcast"); near chance early and late
  3. excess kurtosis of readouts — peaks mid-stack (a few dominant tokens)
  4. effective dimensionality    — participation ratio of the transported
     residuals (random-projected to keep memory bounded)

All four fall out of one forward pass per prompt plus the lens readout, so
this is inference-cost only (~50-80 ms/token), NOT a lens fit. Emits the
per-layer curves plus heuristically-detected band boundaries to a JSON that
can back the grid's band overlay.

Run:
    uv run python scripts/measure_bands.py --prompts 60 --max-len 256
    uv run python scripts/measure_bands.py --lens data/lens/lens.npz --out data/bands/lens.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parent.parent


def excess_kurtosis(logits: mx.array) -> mx.array:
    """Per-row excess kurtosis of the readout logits over the vocab. [S]."""
    mu = logits.mean(axis=-1, keepdims=True)
    d = logits - mu
    var = (d * d).mean(axis=-1)
    m4 = (d**4).mean(axis=-1)
    return m4 / (var * var + 1e-8) - 3.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=int, default=60, help="number of corpus prompts")
    ap.add_argument("--max-len", type=int, default=256, help="max tokens per prompt")
    ap.add_argument("--lens", default=None, help="lens .npz (default: resolve like the server)")
    ap.add_argument("--model", default=os.environ.get("JLENS_MODEL", "mlx-community/Qwen3.6-27B-4bit"))
    ap.add_argument("--proj-dim", type=int, default=512, help="random-projection dim for eff. dimensionality")
    ap.add_argument("--out", default=None, help="output JSON (default: data/bands/<lens>.json)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from jlens_qwen.model import load as load_model
    from jlens_qwen.lens import JacobianLens
    from jlens_qwen.patch_gdn import set_inference_mode
    from jlens_qwen.prompts import load_prompts

    # Resolve the lens the same way the server does.
    lens_dir = str(REPO / "data" / "lens")
    lens_path = args.lens or os.environ.get("JLENS_PATH")
    if not lens_path:
        default = os.path.join(lens_dir, "lens.npz")
        lens_path = default if os.path.exists(default) else None
    if not lens_path or not os.path.exists(lens_path):
        print(f"ERROR: no lens found (pass --lens); looked at {lens_path!r}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else (REPO / "data" / "bands" / (Path(lens_path).stem + ".json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"model:  {args.model}", flush=True)
    print(f"lens:   {lens_path}", flush=True)
    print(f"out:    {out_path}", flush=True)
    print(f"corpus: {args.prompts} prompts x {args.max_len} tokens", flush=True)

    t0 = time.perf_counter()
    model = load_model(args.model)
    set_inference_mode(True)
    lens = JacobianLens.load(lens_path)
    lens.warm()
    layers = list(lens.source_layers)                 # fitted layers
    final_layer = model.n_layers - 1
    record = sorted(set(layers) | {final_layer})       # + logit-lens final column
    print(f"loaded in {time.perf_counter()-t0:.1f}s; measuring {len(record)} layers", flush=True)

    prompts = load_prompts(n=args.prompts, min_chars=200)
    if len(prompts) < args.prompts:
        print(f"WARN: only {len(prompts)} prompts available", flush=True)

    # Fixed random projection for effective dimensionality.
    mx.random.seed(args.seed)
    R = mx.random.normal((model.d_model, args.proj_dim)) / np.sqrt(args.proj_dim)
    mx.eval(R)

    # Per-layer accumulators.
    hits1 = {l: 0 for l in record}
    hits10 = {l: 0 for l in record}
    nt_count = {l: 0 for l in record}
    adj_same = {l: 0 for l in record}
    adj_count = {l: 0 for l in record}
    top1_counts = {l: Counter() for l in record}       # for the chance baseline
    kurt_sum = {l: 0.0 for l in record}
    kurt_count = {l: 0 for l in record}
    proj_sum = {l: np.zeros(args.proj_dim, np.float64) for l in record}
    proj_gram = {l: np.zeros((args.proj_dim, args.proj_dim), np.float64) for l in record}
    proj_n = {l: 0 for l in record}
    total_tokens = 0

    def snapshot():
        curves = _curves(record, hits1, hits10, nt_count, adj_same, adj_count,
                         top1_counts, kurt_sum, kurt_count, proj_sum, proj_gram, proj_n)
        bands, detected = _detect_bands(record, curves, final_layer)
        out = {
            "model": args.model, "lens": os.path.relpath(lens_path, REPO),
            "n_prompts_used": prompts_done, "n_positions": total_tokens,
            "n_layers": model.n_layers, "layers": record,
            "metrics": curves, "bands": bands, "detected": detected,
            "note": "Boundaries are heuristic; inspect the curves. See scripts/measure_bands.py.",
        }
        out_path.write_text(json.dumps(out, indent=2))

    prompts_done = 0
    for pi, prompt in enumerate(prompts):
        input_ids = model.encode(prompt, max_length=args.max_len)
        S = int(input_ids.shape[1])
        if S < 3:
            continue
        _, acts = model.forward(input_ids, capture_layers=record)
        ids = input_ids[0].tolist()
        next_ids = mx.array(ids[1:])                    # target at positions 0..S-2

        for l in record:
            h = acts[l][0].astype(mx.float32)           # [S, D]
            ht = lens.transport(h, l) if l in lens.jacobians else h
            logits = model.unembed(model.final_norm(ht)).astype(mx.float32)   # [S, vocab]

            # 1) next-token top-k accuracy over positions 0..S-2
            lg = logits[:-1]                             # [S-1, vocab]
            tgt = mx.take_along_axis(lg, next_ids[:, None], axis=-1)[:, 0]     # [S-1]
            rank = (lg > tgt[:, None]).sum(axis=-1)      # #tokens strictly above target
            hits1[l] += int((rank < 1).sum().item())
            hits10[l] += int((rank < 10).sum().item())
            nt_count[l] += S - 1

            # 2) top-1 persistence (adjacent agreement) + counts for chance
            top1 = mx.argmax(logits, axis=-1)
            t1 = [int(x) for x in top1.tolist()]
            adj_same[l] += sum(1 for a, b in zip(t1, t1[1:]) if a == b)
            adj_count[l] += len(t1) - 1
            top1_counts[l].update(t1)

            # 3) excess kurtosis of the readout logits
            k = excess_kurtosis(logits)
            kurt_sum[l] += float(k.sum().item())
            kurt_count[l] += S

            # 4) effective dimensionality: random-projected participation ratio
            pr = mx.matmul(ht, R)                        # [S, proj]
            mx.eval(pr)
            prn = np.asarray(pr, dtype=np.float64)
            proj_sum[l] += prn.sum(axis=0)
            proj_gram[l] += prn.T @ prn
            proj_n[l] += S

            del logits, ht, h
        del acts
        total_tokens += S
        prompts_done += 1

        if (pi + 1) % 5 == 0 or pi == len(prompts) - 1:
            rate = total_tokens / (time.perf_counter() - t0 - 0)
            print(f"  [{pi+1}/{len(prompts)}] {total_tokens} tokens, "
                  f"{(time.perf_counter()-t0):.0f}s elapsed", flush=True)
        if (pi + 1) % 20 == 0:
            snapshot()                                  # periodic partial save

    snapshot()
    dt = time.perf_counter() - t0
    bands = json.loads(out_path.read_text())["bands"]
    print(f"\ndone in {dt/60:.1f} min ({total_tokens} tokens). Bands:", flush=True)
    for band in bands:
        print(f"  {band['name']:9s} L{band['start_layer']}-L{band['end_layer']}", flush=True)
    print(f"written to {out_path}", flush=True)
    return 0


def _curves(record, hits1, hits10, nt_count, adj_same, adj_count,
            top1_counts, kurt_sum, kurt_count, proj_sum, proj_gram, proj_n):
    nt1, nt10, persist, chance, above, kurt, effdim = [], [], [], [], [], [], []
    for l in record:
        c = max(1, nt_count[l])
        nt1.append(hits1[l] / c)
        nt10.append(hits10[l] / c)
        a = max(1, adj_count[l])
        p = adj_same[l] / a
        persist.append(p)
        # chance agreement if positions were independent: sum(freq^2)
        tot = sum(top1_counts[l].values()) or 1
        ch = sum((n / tot) ** 2 for n in top1_counts[l].values())
        chance.append(ch)
        above.append(p - ch)
        kurt.append(kurt_sum[l] / max(1, kurt_count[l]))
        # participation ratio of the (centered) projected covariance
        n = max(1, proj_n[l])
        mean = proj_sum[l] / n
        cov = proj_gram[l] / n - np.outer(mean, mean)
        tr = float(np.trace(cov))
        fro2 = float(np.sum(cov * cov))
        effdim.append((tr * tr / fro2) if fro2 > 1e-12 else 0.0)
    return {
        "next_token_top1": nt1, "next_token_top10": nt10,
        "persistence": persist, "persistence_chance": chance,
        "persistence_above_chance": above, "kurtosis": kurt, "eff_dim": effdim,
    }


def _detect_bands(record, curves, final_layer):
    """Heuristic boundaries from the curves.

    workspace start = where top-1 persistence-above-chance first rises past
    half its peak. motor start = the RAMP ONSET, i.e. the first late layer
    where next-token top-1 accuracy breaks 4 sigma above its flat
    workspace baseline (the readout starts turning toward the output).
    This is the functionally useful boundary for choosing intervention
    layers; the paper's "committed" motor (readout ~= output) is recorded
    separately as motor_committed.
    """
    layers = record
    n = len(layers)
    above = np.array(curves["persistence_above_chance"])
    nt1 = np.array(curves["next_token_top1"])

    # workspace onset: first layer whose above-chance persistence exceeds
    # half the peak (scanning forward).
    peak = float(above.max()) if above.size else 0.0
    ws_i = 0
    if peak > 0:
        thr = 0.5 * peak
        for i in range(n):
            if above[i] >= thr:
                ws_i = i
                break

    # Characterize the flat next-token baseline over the workspace, before
    # the output ramp (the front ~40% past the workspace onset).
    base_hi = max(ws_i + 2, int(round(n * 0.65)))
    base = nt1[ws_i:base_hi] if base_hi > ws_i else nt1[ws_i:ws_i + 1]
    b_mean = float(base.mean()) if base.size else 0.0
    b_std = float(base.std()) if base.size else 0.0
    final_val = float(nt1[-1]) if n else 0.0

    # motor RAMP ONSET: first back-half layer 4 sigma above baseline.
    onset_thr = max(b_mean + 4 * b_std, b_mean + 0.02)
    mo_i = n - 1
    for i in range(max(ws_i + 1, n // 2), n):
        if nt1[i] >= onset_thr:
            mo_i = i
            break
    mo_i = max(mo_i, ws_i + 1)

    # motor COMMITTED: where the readout is mostly the output (midpoint to
    # final) — the paper's "last ~8%" landmark, for reference.
    committed_i = mo_i
    mid_thr = b_mean + 0.5 * (final_val - b_mean)
    for i in range(mo_i, n):
        if nt1[i] >= mid_thr:
            committed_i = i
            break

    ws_start = layers[ws_i]
    mo_start = layers[mo_i]
    bands = [
        {"name": "sensory", "start_layer": layers[0], "end_layer": max(layers[0], ws_start - 1)},
        {"name": "workspace", "start_layer": ws_start, "end_layer": max(ws_start, mo_start - 1)},
        {"name": "motor", "start_layer": mo_start, "end_layer": final_layer},
    ]
    return bands, {
        "workspace_start": ws_start,
        "motor_start": mo_start,               # ramp onset (band boundary)
        "motor_committed": layers[committed_i],  # readout ~= output
        "persistence_peak": peak,
        "next_token_baseline": round(b_mean, 4),
        "next_token_final": round(final_val, 4),
    }


if __name__ == "__main__":
    sys.exit(main())
