"""FastAPI backend for the J-space visualizer.

Endpoints:
- POST /api/slice  { prompt, max_seq_len } -> slice data (top-k per cell, ranks)
- POST /api/generate { prompt, ... } -> model's actual next-token logits
- GET  /api/lens  -> lens metadata (source layers, n_prompts, d_model)
- GET  /  -> serves the web UI
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens_qwen.model import load as load_model
from jlens_qwen.lens import JacobianLens

# Globals (loaded once at startup).
_model = None
_lens = None
_model_id = os.environ.get("JLENS_MODEL", "mlx-community/Qwen3.6-27B-4bit")
_lens_path = os.environ.get("JLENS_PATH", "data/lens/qwen36_27b.npz")

app = FastAPI(title="J-Space Visualizer")


class SliceRequest(BaseModel):
    prompt: str
    max_seq_len: int = 128
    top_n: int = 10


@app.on_event("startup")
async def load():
    global _model, _lens
    print(f"Loading model {_model_id!r}...", flush=True)
    _model = load_model(_model_id)
    print(f"  {_model}", flush=True)
    if os.path.exists(_lens_path):
        print(f"Loading lens from {_lens_path}...", flush=True)
        _lens = JacobianLens.load(_lens_path)
        print(f"  {_lens}", flush=True)
    else:
        print(f"  no lens at {_lens_path}; using logit lens (J=I)", flush=True)
        print(f"  (fit one with: uv run python scripts/run_fit.py)", flush=True)


@app.get("/api/lens")
async def lens_info():
    if _lens is None:
        return {"lens_loaded": False, "n_layers": _model.n_layers, "d_model": _model.d_model}
    return {
        "lens_loaded": True,
        "source_layers": _lens.source_layers,
        "n_prompts": _lens.n_prompts,
        "d_model": _lens.d_model,
        "n_layers": _model.n_layers,
    }


@app.post("/api/slice")
async def slice_endpoint(req: SliceRequest):
    if _model is None:
        raise HTTPException(503, "model not loaded")
    prompt = req.prompt
    max_seq_len = req.max_seq_len
    top_n = req.top_n

    # Run forward, get lens logits at all source layers (+ final).
    layers = list(_lens.source_layers) if _lens is not None else [_model.n_layers - 1]
    final_layer = _model.n_layers - 1
    record = sorted(set(layers) | {final_layer})

    input_ids = _model.encode(prompt, max_length=max_seq_len)
    final, acts = _model.forward(input_ids, capture_layers=record)
    for l in record:
        mx.eval(acts[l])

    # Token strings for the prompt.
    token_strs = [_model.tokenizer.decode([int(t)]) for t in input_ids[0].tolist()]
    seq_len = len(token_strs)

    # For each layer, compute top-n tokens at each position.
    # lens_logits[layer] = [seq_len, vocab]
    slice_data = {"layers": record, "seq_len": seq_len, "token_strs": token_strs,
                  "top_n": top_n, "cells": {}}

    for layer in record:
        h = acts[layer][0].astype(mx.float32)  # [seq_len, D]
        if _lens is not None and layer in _lens.jacobians:
            h = _lens.transport(h, layer)
        logits = _model.unembed(_model.final_norm(h))  # [seq_len, vocab]
        lf = logits.astype(mx.float32)
        # Top-n per position. mx.topk is buggy on large vocab; use argsort.
        # For memory, process per position.
        top_ids = []
        top_scores = []
        for pos in range(seq_len):
            v = lf[pos]
            sorted_idx = mx.argsort(v)
            top = [int(t) for t in sorted_idx[-top_n:][::-1].tolist()]
            scores = [float(mx.take(v, t).tolist()) for t in top]
            top_ids.append(top)
            top_scores.append(scores)
            top_tokens = [_model.tokenizer.decode([t]) for t in top]
        # Actually store as top_tokens for the response.
        top_tokens = [[_model.tokenizer.decode([t]) for t in pos_ids] for pos_ids in top_ids]
        slice_data["cells"][layer] = {
            "top_ids": top_ids,
            "top_tokens": top_tokens,
            "top_scores": top_scores,
        }
        del logits, lf

    return JSONResponse(slice_data)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent.parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>J-Space Visualizer</h1><p>web/index.html not found</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)