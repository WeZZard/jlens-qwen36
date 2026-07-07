"""Characterize the workspace range of a model via its fitted J-lens.

For each (position, layer) cell, classify the top-1 J-lens token as:
- "echo": matches the input token at that position (early-layer noise)
- "workspace": an abstract/intermediate concept (neither echo nor final output)
- "motor": matches the model's final-layer output token (output regime)

The workspace range is the band of layers where "workspace" classifications
dominate, between the echo band and the motor band.

Usage:
    uv run python scripts/workspace_range.py --lens data/lens/lens.npz
    uv run python scripts/workspace_range.py --lens data/lens/lens.npz --model-id mlx-community/Qwen3.6-27B-4bit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from jlens_qwen.model import load
from jlens_qwen.lens import JacobianLens


PROMPTS = [
    "Fact: The currency used in the country shaped like a boot is the",
    "The number of legs on the animal that spins webs is",
    "Count to five and introspect deeply.",
    "The capital of the country where the Eiffel Tower is located is",
    "Translate to French: hello",
]


def classify(token, input_token, output_token):
    """Classify a top-1 lens token as echo/workspace/motor."""
    def norm(s):
        return s.strip().lower().lstrip("Ġ▁_").strip()
    t, it, ot = norm(token), norm(input_token), norm(output_token)
    if not t:
        return "empty"
    if t == it and t != ot:
        return "echo"
    if t == ot:
        return "motor"
    if (t in ot or ot in t) and len(t) > 2:
        return "motor"
    return "workspace"


def main():
    parser = argparse.ArgumentParser(description="Characterize the workspace range.")
    parser.add_argument("--lens", type=str, required=True,
                        help="Path to the fitted lens (.npz).")
    parser.add_argument("--model-id", type=str, default="mlx-community/Qwen3.6-27B-4bit",
                        help="HuggingFace repo or local path of the MLX model.")
    args = parser.parse_args()

    model = load(args.model_id)
    print(f"Model: {model}", flush=True)
    lens = JacobianLens.load(args.lens)
    print(f"Lens: {lens}", flush=True)
    print(f"Source layers: {lens.source_layers}", flush=True)
    print()

    layer_stats = {l: {"echo": 0, "workspace": 0, "motor": 0, "empty": 0, "total": 0}
                   for l in lens.source_layers + [model.n_layers - 1]}

    for prompt in PROMPTS:
        print(f"=== {prompt!r} ===")
        result = lens.apply(model, prompt, max_seq_len=32)
        token_strs = result["token_strs"]
        model_logits = result["model_logits"]
        model_top = []
        lf = model_logits.astype(mx.float32)
        for p in range(lf.shape[0]):
            v = lf[p]
            sorted_idx = mx.argsort(v)
            model_top.append(int(sorted_idx[-1].tolist()))
        model_top_strs = [model.tokenizer.decode([t]) for t in model_top]

        for layer in result["lens_logits"]:
            logits = result["lens_logits"][layer]
            lf2 = logits.astype(mx.float32)
            for p in range(lf2.shape[0]):
                v = lf2[p]
                sorted_idx = mx.argsort(v)
                top_tok = int(sorted_idx[-1].tolist())
                top_str = model.tokenizer.decode([top_tok])
                input_tok = token_strs[p] if p < len(token_strs) else ""
                output_tok = model_top_strs[p] if p < len(model_top_strs) else ""
                cls = classify(top_str, input_tok, output_tok)
                layer_stats[layer][cls] += 1
                layer_stats[layer]["total"] += 1

        last_pos = lf2.shape[0] - 1
        print(f"  Last position ({token_strs[last_pos]!r}):")
        for layer in sorted(result["lens_logits"]):
            logits = result["lens_logits"][layer]
            v = logits[last_pos].astype(mx.float32)
            sorted_idx = mx.argsort(v)
            top = int(sorted_idx[-1].tolist())
            print(f"    L{layer}: {model.tokenizer.decode([top])!r}")
        print()

    print("=" * 60)
    print("LAYER CLASSIFICATION (fraction of cells, averaged over prompts/positions)")
    print("=" * 60)
    print(f"{'layer':>6} {'echo':>7} {'workspace':>10} {'motor':>7} {'empty':>7}")
    for layer in sorted(layer_stats):
        s = layer_stats[layer]
        total = max(s["total"], 1)
        print(f"L{layer:>4} {s['echo']/total:>7.2f} {s['workspace']/total:>10.2f} "
              f"{s['motor']/total:>7.2f} {s['empty']/total:>7.2f}")

    print()
    workspace_layers = [l for l in sorted(layer_stats)
                       if layer_stats[l]["workspace"] / max(layer_stats[l]["total"], 1) > 0.5]
    motor_layers = [l for l in sorted(layer_stats)
                    if layer_stats[l]["motor"] / max(layer_stats[l]["total"], 1) > 0.5]
    echo_layers = [l for l in sorted(layer_stats)
                   if layer_stats[l]["echo"] / max(layer_stats[l]["total"], 1) > 0.3]
    print(f"Echo-dominated layers (>30% echo):            {echo_layers}")
    print(f"Workspace-dominated layers (>50% workspace):  {workspace_layers}")
    print(f"Motor-dominated layers (>50% motor):          {motor_layers}")


if __name__ == "__main__":
    main()