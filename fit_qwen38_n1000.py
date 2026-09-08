"""Detached n=1000 full-depth Jacobian lens fit for Qwen3.8-27B-4bit.

Uses fit_analytic (the optimized analytic path, ~437s/prompt on an M4 Pro),
NOT scripts/run_fit.py, which calls the older fit() at ~34.5 min/prompt.

Checkpoints every prompt, so a crash or reboot resumes at the last completed
prompt. Writes a provenance sidecar recording the model the lens was fitted
against: 3.6 and 3.8 share identical shapes (64 layers, d_model 5120,
vocab 248320), so nothing in JacobianLens.load() would catch a mismatch.
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jlens_qwen.model import load
from jlens_qwen.fit_analytic import fit_analytic
from jlens_qwen.lens import JacobianLens
from jlens_qwen.prompts import load_prompts

MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
N_PROMPTS = 1000
MIN_CHARS = 150
SOURCE_LAYERS = list(range(63))  # L0-L62, matching the shipped v0.2-fulldepth lens
OUT = "data/lens/qwen38_27b_n1000.npz"
CKPT = "data/lens/qwen38_27b_n1000.ckpt.npy"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        stream=sys.stdout,
    )

    os.makedirs("data/lens", exist_ok=True)

    print(f"loading model {MODEL_ID!r}...", flush=True)
    model = load(MODEL_ID)
    print(f"  {model}", flush=True)

    prompts = load_prompts(n=N_PROMPTS, min_chars=MIN_CHARS)
    if len(prompts) != N_PROMPTS or len(set(prompts)) != N_PROMPTS:
        raise SystemExit(
            f"corpus is not {N_PROMPTS} unique prompts "
            f"(got {len(prompts)} total, {len(set(prompts))} unique) — "
            "refusing to fit on a degenerate corpus"
        )
    print(f"  {len(prompts)} unique prompts", flush=True)

    print(f"fitting {len(prompts)} prompts x {len(SOURCE_LAYERS)} layers...", flush=True)
    t0 = time.perf_counter()
    J = fit_analytic(
        model,
        prompts,
        source_layers=SOURCE_LAYERS,
        checkpoint_path=CKPT,
        checkpoint_every=1,
        resume=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"fit done in {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)

    lens = JacobianLens(J, n_prompts=N_PROMPTS, d_model=model.d_model)
    lens.save(OUT)
    print(f"saved {OUT}: {lens}", flush=True)

    with open(OUT.replace(".npz", ".provenance.json"), "w") as f:
        json.dump({
            "model_id": MODEL_ID,
            "n_prompts": N_PROMPTS,
            "source_layers": SOURCE_LAYERS,
            "d_model": model.d_model,
            "corpus": f"prompts_{N_PROMPTS}_{MIN_CHARS}.jsonl (c4)",
            "include_gbeta": True,
            "fit_seconds": round(elapsed),
            "host": os.uname().nodename,
        }, f, indent=2)


if __name__ == "__main__":
    main()
