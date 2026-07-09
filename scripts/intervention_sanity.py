"""Intervention sanity suite for the v0.2 lens (no-think completion prompts).

Reproduces the paper-style causal edits:
  1. Spider-legs: steer / swap the "ant" concept, does 8 -> 6?
  2. France->China swap: does the capital flip Paris -> Beijing?

Interventions run on raw completion prompts (no chat template), which is the
paper's regime. Sweeps alpha because a 20-prompt lens has small J-lens
vectors.

Run (needs the model + GPU, ~5-10 min):
    uv run python scripts/intervention_sanity.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import mlx.core as mx

from jlens_qwen.model import load
from jlens_qwen.lens import JacobianLens
from jlens_qwen.interventions import (
    j_lens_vector_for_text, j_lens_vectors, steer, patch_swap,
)

LENS_PATH = "data/lens/full_depth_analytic.npz"
LAYERS = [30, 40, 48]           # intervention layers to try
ALPHAS = [50, 100, 200, 400, 800]

SPIDER_PROMPT = "Question: How many legs does a spider have?\nAnswer: A spider has"
FRANCE_PROMPT = "Question: What is the capital of France?\nAnswer: The capital of France is"


def gen(model, prompt, **kw):
    text, _ = model.generate(prompt, max_tokens=6, temp=0.0, **kw)
    return text.strip().replace("\n", " ")[:60]


def main():
    model = load()
    lens = JacobianLens.load(LENS_PATH)
    print(f"lens: {lens}\n")

    base_spider = gen(model, SPIDER_PROMPT)
    base_france = gen(model, FRANCE_PROMPT)
    print(f"BASELINE spider: {base_spider!r}")
    print(f"BASELINE france: {base_france!r}\n")

    for layer in LAYERS:
        print(f"===== intervention layer L{layer} =====")
        v_spider = j_lens_vector_for_text(lens, model, layer, " spider")
        v_ant = j_lens_vector_for_text(lens, model, layer, " ant")
        v_france = j_lens_vector_for_text(lens, model, layer, " France")
        v_china = j_lens_vector_for_text(lens, model, layer, " China")
        mx.eval(v_spider, v_ant, v_france, v_china)

        # --- Spider: steer ant ---
        for a in ALPHAS:
            fn = lambda h, a=a: steer(h, v_ant, a)
            out = gen(model, SPIDER_PROMPT, intervene_layer=layer,
                      intervene_fn=fn, intervene_each_step=True)
            flip = any(w in out.lower() for w in ("six", " 6", "6 "))
            print(f"  spider steer ant  a={a:4d}: {out!r}  {'<-- 6!' if flip else ''}")

        # --- Spider: swap spider->ant ---
        fn = lambda h: patch_swap(h, v_spider, v_ant, 1.0)
        out = gen(model, SPIDER_PROMPT, intervene_layer=layer,
                  intervene_fn=fn, intervene_each_step=True)
        print(f"  spider swap->ant       : {out!r}")

        # --- France: swap France->China ---
        fn = lambda h: patch_swap(h, v_france, v_china, 1.0)
        out = gen(model, FRANCE_PROMPT, intervene_layer=layer,
                  intervene_fn=fn, intervene_each_step=True)
        hit = any(w in out.lower() for w in ("beijing", "china", "peking"))
        print(f"  france swap->china     : {out!r}  {'<-- china!' if hit else ''}")

        # --- France: steer China ---
        for a in ALPHAS:
            fn = lambda h, a=a: steer(h, v_china, a)
            out = gen(model, FRANCE_PROMPT, intervene_layer=layer,
                      intervene_fn=fn, intervene_each_step=True)
            hit = any(w in out.lower() for w in ("beijing", "china", "peking"))
            print(f"  france steer china a={a:4d}: {out!r}  {'<-- china!' if hit else ''}")

        j_lens_vectors.cache_clear()  # free the [vocab, d_model] cache per layer
        print()


if __name__ == "__main__":
    main()
