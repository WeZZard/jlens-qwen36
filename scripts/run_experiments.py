"""Run the paper's J-space experiments on Qwen3.6-27B-4bit.

Experiments:
1. Verbal report: "think of a sport, name it" -> J-lens shows the sport
   before the model says it.
2. Swap: spider -> ant on "number of legs" prompt -> answer should change
   from 8 to 6 (if the lens is well-fit).
3. Steer / inject: inject "lightning" while the model reads an introspection
   prompt -> model reports detecting "lightning".
4. Ablate: remove the J-space from a prompt -> model loses higher-order
   reasoning but keeps fluency (paper's key finding).

Each experiment prints baseline vs. intervened output.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from jlens_qwen.model import load
from jlens_qwen.lens import JacobianLens
from jlens_qwen.interventions import (
    j_lens_vector_for_text, steer, patch_swap, ablate_topk
)


def experiment_swap_spider_ant(model, lens):
    """Spider -> ant: 'legs on the animal that spins webs' should go 8 -> 6."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: spider -> ant swap (paper Fig 6)")
    print("=" * 60)
    prompt = "The number of legs on the animal that spins webs is"
    base, base_toks = model.generate(prompt, max_tokens=3, temp=0)
    print(f"Baseline: {base!r}")

    # The 'spider' concept is the intermediate. Swap to 'ant' at a workspace layer.
    for layer in [48, 52, 57, 60]:
        v_spider = j_lens_vector_for_text(lens, model, layer, " spider")
        v_ant = j_lens_vector_for_text(lens, model, layer, " ant")
        def fn(h, vs=v_spider, vt=v_ant):
            return patch_swap(h, vs, vt, alpha=3.0)
        text, toks = model.generate(prompt, max_tokens=3, temp=0,
                                    intervene_layer=layer, intervene_fn=fn)
        print(f"  swap spider->ant at L{layer} alpha=3: {text!r}")

    # Also try strong steer of 'ant'
    for alpha in [100, 300, 1000]:
        v_ant = j_lens_vector_for_text(lens, model, 57, " ant")
        def fn(h, va=v_ant, a=alpha):
            return steer(h, va, a)
        text, toks = model.generate(prompt, max_tokens=3, temp=0,
                                    intervene_layer=57, intervene_fn=fn)
        print(f"  steer ant at L57 alpha={alpha}: {text!r}")


def experiment_inject_lightning(model, lens):
    """Inject 'lightning' into the J-space; ask the model to report it."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: inject 'lightning' (paper Fig 7)")
    print("=" * 60)
    prompt = ("You may have had a thought implanted into your activations. "
              "If so, report what you detect in one word. If not, say 'nothing'. "
              "Response: \"")
    base, _ = model.generate(prompt, max_tokens=5, temp=0)
    print(f"Baseline: {base!r}")

    v_lightning = j_lens_vector_for_text(lens, model, 57, " lightning")
    for alpha in [100, 300, 1000]:
        def fn(h, vl=v_lightning, a=alpha):
            return steer(h, vl, a)
        text, _ = model.generate(prompt, max_tokens=5, temp=0,
                                 intervene_layer=57, intervene_fn=fn)
        print(f"  inject lightning at L57 alpha={alpha}: {text!r}")


def experiment_ablate_j_space(model, lens):
    """Ablate the J-space: model should keep fluency but lose reasoning."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: ablate J-space (paper Fig 8)")
    print("=" * 60)
    # A multi-step reasoning prompt.
    prompt = "The number of legs on the animal that spins webs is"
    base, _ = model.generate(prompt, max_tokens=5, temp=0)
    print(f"Baseline: {base!r}")

    for layer in [48, 57]:
        def fn(h, l=layer):
            return ablate_topk(h, lens, model, l, k=16)
        text, _ = model.generate(prompt, max_tokens=5, temp=0,
                                 intervene_layer=layer, intervene_fn=fn)
        print(f"  ablate top-16 at L{layer}: {text!r}")


def experiment_france_to_china(model, lens):
    """France -> China: one swap redirects 4 different facts."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: France -> China swap (paper Fig 9)")
    print("=" * 60)
    v_france = j_lens_vector_for_text(lens, model, 57, " France")
    v_china = j_lens_vector_for_text(lens, model, 57, " China")
    def fn(h):
        return patch_swap(h, v_france, v_china, alpha=3.0)

    questions = [
        ("The capital of France is", "Paris -> Beijing?"),
        ("The language spoken in France is", "French -> Chinese?"),
        ("France is located in the continent of", "Europe -> Asia?"),
        ("The currency of France is", "euro -> yuan?"),
    ]
    for prompt, expected in questions:
        base, _ = model.generate(prompt, max_tokens=3, temp=0)
        text, _ = model.generate(prompt, max_tokens=3, temp=0,
                                 intervene_layer=57, intervene_fn=fn)
        print(f"  {prompt!r}")
        print(f"    baseline: {base!r}  | swap: {text!r}  (expect {expected})")


def main():
    print("Loading model...", flush=True)
    model = load()
    print(f"  {model}", flush=True)
    print("Loading lens...", flush=True)
    lens = JacobianLens.load("data/lens/qwen36_27b_partial.npz")
    print(f"  {lens}", flush=True)

    experiment_swap_spider_ant(model, lens)
    experiment_inject_lightning(model, lens)
    experiment_ablate_j_space(model, lens)
    experiment_france_to_china(model, lens)


if __name__ == "__main__":
    main()