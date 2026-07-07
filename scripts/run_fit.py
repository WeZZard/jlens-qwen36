"""Run a real Jacobian lens fit on Qwen3.6-27B-4bit.

Usage:
    uv run python scripts/run_fit.py --n-prompts 20 --n-layers 25 --output data/lens/qwen36_27b.npz

The fit uses chain-multiply + custom Metal GDN VJP + 32-token prompts.
Estimated time: ~5-8 hours for 25 late layers x 20 prompts.
Checkpoint/resume every prompt so interruptions are safe.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from jlens_qwen.model import load
from jlens_qwen.fit import fit
from jlens_qwen.lens import JacobianLens
from jlens_qwen.prompts import load_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--n-layers", type=int, default=25)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--layer-start", type=int, default=None,
                        help="Minimum source layer (chain starts here). "
                             "Default: evenly spaced across full range.")
    parser.add_argument("--output", type=str, default="data/lens/qwen36_27b.npz")
    parser.add_argument("--checkpoint", type=str, default="data/lens/qwen36_27b.ckpt.npy")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        stream=sys.stdout,
    )

    print(f"Loading model...", flush=True)
    model = load()
    print(f"  {model}", flush=True)

    print(f"Loading {args.n_prompts} prompts...", flush=True)
    prompts = load_prompts(n=args.n_prompts, min_chars=150)
    print(f"  got {len(prompts)} prompts", flush=True)

    # 25 evenly-spaced source layers, biased toward the late (workspace) range.
    # The paper's workspace range is roughly the last 60% of layers.
    n_layers_model = model.n_layers  # 64
    if args.n_layers >= n_layers_model:
        source_layers = list(range(n_layers_model))
    elif args.layer_start is not None:
        # Evenly space n_layers layers from layer_start to n_layers-2.
        last = n_layers_model - 2
        source_layers = [
            int(round(args.layer_start + i * (last - args.layer_start) / (args.n_layers - 1)))
            for i in range(args.n_layers)
        ]
        source_layers = sorted(set(source_layers))
    else:
        # Evenly space across the full range.
        source_layers = [
            int(round(i * (n_layers_model - 2) / (args.n_layers - 1)))
            for i in range(args.n_layers)
        ]
        source_layers = sorted(set(source_layers))
    print(f"  source_layers ({len(source_layers)}): {source_layers}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    print(f"Starting fit: {len(prompts)} prompts x {len(source_layers)} layers...", flush=True)
    print(f"  estimated time: ~{len(prompts) * len(source_layers) * 1.0 / 60:.0f} min"
          f" (at ~1 min/layer/prompt)", flush=True)
    t0 = time.perf_counter()
    J = fit(
        model, prompts, source_layers=source_layers,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint,
        checkpoint_every=1,
        resume=not args.no_resume,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nFit done in {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

    lens = JacobianLens(J, n_prompts=len(prompts), d_model=model.d_model)
    lens.save(args.output)
    print(f"Saved lens to {args.output}: {lens}", flush=True)


if __name__ == "__main__":
    main()