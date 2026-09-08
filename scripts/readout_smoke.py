"""End-to-end readout smoke test for a fitted lens.

Loads the model and a subset of lens layers, runs a few completion prompts,
and prints the top-1 J-lens token per layer at the last position next to
the plain logit-lens token and the model's own next token. Read the table, not the count at the end:
a J-lens reads out the concept, often in another language or script, so
exact top-1 agreement is low even for a good lens. What a healthy lens
shows is the answer concept surfacing in the middle of the stack, earlier
than the logit lens, and the same pattern as the shipped lens on its own
model (compare data/lens/readout_smoke_*.log). Loads only the requested
layers (fp32 in RAM), so it fits on a 16 GB machine next to the 15 GB model.

Run:
    uv run python scripts/readout_smoke.py \
        --model-id mlx-community/Qwen3.8-27B-4bit \
        --lens data/lens/qwen38_27b_n1000.npz
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

import mlx.core as mx
import numpy as np

from jlens_qwen.lens import JacobianLens
from jlens_qwen.model import load

PROMPTS = [
    "Question: What is the capital of France?\nAnswer: The capital of France is",
    "Question: How many legs does a spider have?\nAnswer: A spider has",
    "Monday, Tuesday, Wednesday, Thursday,",
    "The chemical symbol for water is",
]
DEFAULT_LAYERS = [0, 8, 16, 24, 32, 40, 44, 48, 52, 56, 60, 62]


def load_layers(path: str, layers: list[int]) -> JacobianLens:
    data = np.load(path)
    missing = [l for l in layers if f"J_{l}" not in data.files]
    if missing:
        raise SystemExit(f"lens lacks layers {missing}")
    return JacobianLens(
        {l: data[f"J_{l}"] for l in layers},
        n_prompts=int(data["n_prompts"]),
        d_model=int(data["d_model"]),
    )


def top1(logits: mx.array, tok) -> str:
    return tok.decode([int(mx.argmax(logits[-1]).item())]).replace("\n", "\\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    t0 = time.perf_counter()
    model = load(args.model_id)
    print(f"model: {model}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    lens = load_layers(args.lens, layers)
    print(f"lens: {lens}", flush=True)

    agree_j = agree_ll = 0
    for prompt in PROMPTS:
        out_j = lens.apply(model, prompt, layers=layers, use_jacobian=True)
        out_ll = lens.apply(model, prompt, layers=layers, use_jacobian=False)
        model_tok = top1(out_j["model_logits"], model.tokenizer)
        print(f"\n### {prompt.splitlines()[-1]!r}\nmodel next token: {model_tok!r}")
        print(f"{'layer':>5}  {'J-lens':<14} {'logit-lens':<14}")
        first_j = first_ll = None
        for l in layers:
            tj = top1(out_j["lens_logits"][l], model.tokenizer)
            tl = top1(out_ll["lens_logits"][l], model.tokenizer)
            mark = ""
            if tj == model_tok:
                agree_j += 1
                mark += " J=model"
                first_j = l if first_j is None else first_j
            if tl == model_tok:
                agree_ll += 1
                first_ll = l if first_ll is None else first_ll
            print(f"{l:5d}  {tj!r:<14} {tl!r:<14}{mark}")
        print(f"first layer agreeing with model: J-lens={first_j} logit-lens={first_ll}")

    n = len(PROMPTS) * len(layers)
    print(f"\nlayers whose top-1 is exactly the model's next token: "
          f"J-lens {agree_j}/{n}, logit-lens {agree_ll}/{n} "
          "(informational; see the docstring)")
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
