"""Decode benchmark: the ledger's baseline number comes from here.

Drives the REAL production functions in-process — StreamSession.extend →
serve._readout_at_positions → greedy sample → incremental detok → JSON
snapshot assembly — i.e. everything the SSE chat loop does per token
except the socket write. Reports the median per-token wall time and a
call-level split (each call is synchronized at exit, so call-level
timing is exact attribution without instrumentation).

    uv run python scripts/bench_decode.py                # baseline
    JLENS_PERF=1 uv run python scripts/bench_decode.py   # + readout stages

Knobs: BENCH_TOKENS (default 32), BENCH_WARMUP (default 4),
JLENS_MODEL, JLENS_PATH (default data/lens/full_depth_analytic.npz).
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = (
    "The J-lens visual debugger shows which words a language model is "
    "pushing toward at every layer and token position. In this benchmark "
    "we generate a continuation and read out the top tokens per layer, "
    "exactly as the chat viewer does:"
)
N_TOKENS = int(os.environ.get("BENCH_TOKENS", "32"))
N_WARMUP = int(os.environ.get("BENCH_WARMUP", "4"))
TOP_N = 10
LENS_PATH = os.environ.get(
    "JLENS_PATH", os.path.join(REPO, "data", "lens", "full_depth_analytic.npz")
)


def main() -> None:
    from jlens_qwen import perf, serve
    from jlens_qwen.lens import JacobianLens
    from jlens_qwen.model import load
    from jlens_qwen.patch_gdn import set_inference_mode

    model_id = os.environ.get("JLENS_MODEL", "mlx-community/Qwen3.6-27B-4bit")
    print(f"loading {model_id} ...", flush=True)
    model = load(model_id)
    set_inference_mode(True)
    lens = JacobianLens.load(LENS_PATH) if os.path.exists(LENS_PATH) else None
    if lens is not None:
        t0 = time.perf_counter()
        lens.warm()
        print(f"lens warmed in {time.perf_counter() - t0:.1f}s "
              "(one-time; excluded from prefill numbers)", flush=True)
    serve._model = model
    serve._lens = lens

    layers = (
        sorted(set(lens.source_layers) | {model.n_layers - 1})
        if lens is not None
        else [model.n_layers - 1]
    )
    tok = model.tokenizer

    session = model.make_stream(capture_layers=layers)
    ids = model.encode(PROMPT)[0].tolist()

    t0 = time.perf_counter()
    logits, acts = session.extend(ids)
    prefill_forward_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    serve._readout_at_positions(acts, list(range(len(ids))), layers, TOP_N)
    prefill_readout_s = time.perf_counter() - t0
    del acts

    next_tok = int(mx.argmax(logits.astype(mx.float32)).tolist())

    per_token, t_extend, t_readout, t_sample, t_emit = [], [], [], [], []
    gen_ids: list[int] = []
    gen_prev = ""
    pos = len(ids)

    for step in range(N_WARMUP + N_TOKENS):
        s0 = time.perf_counter()
        logits, acts = session.extend([next_tok])
        s1 = time.perf_counter()
        row = serve._readout_at_positions(acts, [0], layers, TOP_N)
        s2 = time.perf_counter()
        new_next = int(mx.argmax(logits.astype(mx.float32)).tolist())
        s3 = time.perf_counter()
        # Emit phase: cumulative detok + JSON snapshot, as the SSE loop does.
        gen_ids.append(next_tok)
        full = tok.decode(gen_ids)
        stable = full.rstrip("�")
        tok_str = stable[len(gen_prev):] if len(stable) >= len(gen_prev) else ""
        gen_prev = stable if len(stable) >= len(gen_prev) else gen_prev
        _ = json.dumps({
            "token_idx": step, "token": tok_str, "token_id": next_tok,
            "pos": pos, "grid": {"layers": layers, "positions": [pos], "cells": row},
        })
        s4 = time.perf_counter()
        del acts
        next_tok = new_next
        pos += 1
        if step >= N_WARMUP:
            per_token.append(s4 - s0)
            t_extend.append(s1 - s0)
            t_readout.append(s2 - s1)
            t_sample.append(s3 - s2)
            t_emit.append(s4 - s3)

    def med(xs):
        return 1e3 * statistics.median(xs)

    total = med(per_token)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
        capture_output=True, text=True,
    ).stdout.strip()
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True, text=True,
    ).stdout.strip()

    print(f"\n== decode benchmark @ {commit} ==")
    print(f"machine: {chip} / {platform.platform()} / mlx {mx.__version__}")
    print(f"workload: {len(ids)}-tok prompt, {N_TOKENS} greedy tokens "
          f"(+{N_WARMUP} warmup), {len(layers)} layers, top_n={TOP_N}, "
          f"lens={'yes' if lens else 'no'}, JLENS_PERF={'1' if perf.ENABLED else '0'}")
    print(f"prefill: forward {1e3 * prefill_forward_s:.0f} ms, "
          f"readout {1e3 * prefill_readout_s:.0f} ms "
          f"({1e3 * prefill_readout_s / len(ids):.1f} ms/prompt-token)")
    print(f"\nper generated token (median of {N_TOKENS}):")
    print(f"  forward (extend) : {med(t_extend):8.1f} ms")
    print(f"  readout          : {med(t_readout):8.1f} ms")
    print(f"  sample           : {med(t_sample):8.1f} ms")
    print(f"  emit (detok+json): {med(t_emit):8.1f} ms")
    print(f"  TOTAL            : {total:8.1f} ms/token = {1e3 / total:.2f} tok/s")

    if perf.ENABLED:
        print("\nreadout stages (JLENS_PERF=1; sync points inflate totals):")
        for name, st in perf.report().items():
            print(f"  {name:24s} median {st['median_ms']:7.2f} ms  "
                  f"mean {st['mean_ms']:7.2f} ms  n={st['n']:.0f}")


if __name__ == "__main__":
    main()
