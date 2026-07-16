"""FastAPI backend for the J-space visualizer.

Endpoints:
- POST /api/slice  { prompt, max_seq_len } -> slice data (top-k per cell, ranks)
- POST /api/generate { prompt, ... } -> model's actual next-token logits
- POST /api/chat_stream { messages, ... } -> SSE stream of J-lens snapshots
- GET  /api/lens  -> lens metadata (source layers, n_prompts, d_model)
- GET  /  -> serves the web UI
"""

from __future__ import annotations

import asyncio
from collections import Counter
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens_qwen.model import load as load_model
from jlens_qwen.lens import JacobianLens
from jlens_qwen import perf

# Globals (loaded once at startup).
_model = None
_lens = None
# Measured functional bands for the loaded lens (sensory/workspace/motor),
# from scripts/measure_bands.py -> data/bands/<lens_stem>.json. None => the
# UI falls back to its percentage-split guess (bands_are_fallback).
_bands = None
_model_id = os.environ.get("JLENS_MODEL", "mlx-community/Qwen3.6-27B-4bit")
# Lens selection is ENFORCED at startup — there is no silent fallback.
# JLENS_PATH=<file> loads that file or refuses to start; JLENS_PATH=none
# runs lens-less (logit lens, final layer only) as an explicit choice;
# unset uses data/lens/lens.npz if present, auto-selects a lone candidate
# loudly, and refuses to start when the choice is ambiguous or empty.
# See resolve_lens_path().
# Deployment mode, decided at deployment time via JLENS_MODE:
# - "active" (default): full app — chat with the model, mark up blocks,
#   save/delete sessions, configure generation.
# - "presentation": strictly read-only display of historical data — view
#   saved sessions and jump between markups; every mutating endpoint is
#   rejected and the UI hides all editing affordances.
_app_mode = os.environ.get("JLENS_MODE", "active")
if _app_mode not in ("active", "presentation"):
    print(f"unknown JLENS_MODE {_app_mode!r}; falling back to 'active'", flush=True)
    _app_mode = "active"
_repo_root = Path(__file__).resolve().parent.parent
_sessions_dir = _repo_root / "data" / "sessions"

# Pause/resume control for the active generation stream.
# The generation loop checks this event before each token.
_pause_events: dict[str, asyncio.Event] = {}

# GPU work runs on worker threads (MLX releases the GIL during compute,
# measured: event-loop max tick 1.1 ms under to_thread vs 1063 ms inline)
# so the event loop stays responsive during generation. The lock
# serializes GPU access across concurrent streams, preserving the
# pre-thread semantics.
_gpu_lock = asyncio.Lock()

class LensResolutionError(RuntimeError):
    """Lens selection could not be resolved explicitly; refuse to start."""


def _lens_choices(lens_dir: str) -> str:
    from glob import glob

    lines = []
    for p in sorted(glob(os.path.join(lens_dir, "*.npz"))):
        size_gb = os.path.getsize(p) / 1e9
        lines.append(f"  JLENS_PATH={p}   ({size_gb:.1f} GB)")
    if not lines:
        lines.append(f"  (no .npz files in {lens_dir})")
    lines.append("  JLENS_PATH=none   (run lens-less: logit lens, final layer only)")
    return "Choose a lens explicitly:\n" + "\n".join(lines)


def resolve_lens_path(explicit: str | None, lens_dir: str) -> str | None:
    """Decide which lens to load — explicitly, never by silent fallback.

    Returns a path to load, or None for explicitly-chosen lens-less mode.
    Raises LensResolutionError (refusing startup) when the choice would
    otherwise be silent and wrong: an explicit path that doesn't exist,
    or no default with an ambiguous/empty candidate set. A lone candidate
    is auto-selected with a loud log line — an unambiguous choice can't
    be silently wrong.
    """
    from glob import glob

    if explicit:
        if explicit.strip().lower() in ("none", "logit"):
            return None
        if os.path.exists(explicit):
            return explicit
        raise LensResolutionError(
            f"JLENS_PATH points to a missing file: {explicit!r}\n"
            + _lens_choices(lens_dir)
        )
    default = os.path.join(lens_dir, "lens.npz")
    if os.path.exists(default):
        return default
    candidates = sorted(glob(os.path.join(lens_dir, "*.npz")))
    if len(candidates) == 1:
        print(f"LENS: auto-selected the only available lens: {candidates[0]}",
              flush=True)
        return candidates[0]
    if candidates:
        raise LensResolutionError(
            f"No lens at the default path {default} and JLENS_PATH is not "
            "set, but multiple lens files exist — refusing to guess.\n"
            + _lens_choices(lens_dir)
        )
    raise LensResolutionError(
        f"No lens files found in {lens_dir}.\n"
        "Fit one (uv run python scripts/run_fit.py), download the release "
        "lens (see README), or choose lens-less mode explicitly: "
        "JLENS_PATH=none"
    )


app = FastAPI(title="J-Space Visualizer")


class SliceRequest(BaseModel):
    prompt: str
    max_seq_len: int = 128
    top_n: int = 10


def _load_bands(lens_path: str | None) -> list[dict] | None:
    """Measured functional bands for this lens, if scripts/measure_bands.py
    has been run. Looked up by lens-file stem at data/bands/<stem>.json."""
    if not lens_path:
        return None
    bands_file = _repo_root / "data" / "bands" / (Path(lens_path).stem + ".json")
    if not bands_file.exists():
        return None
    try:
        data = json.loads(bands_file.read_text())
        bands = data.get("bands")
        if isinstance(bands, list) and bands:
            print(f"BANDS: measured bands from {bands_file.name}: "
                  + ", ".join(f"{b['name']} L{b['start_layer']}-L{b['end_layer']}"
                              for b in bands), flush=True)
            return bands
    except Exception as e:
        print(f"BANDS: failed to read {bands_file} ({e}); using UI fallback", flush=True)
    return None


@app.on_event("startup")
async def load():
    global _model, _lens, _bands
    # Resolve the lens FIRST so a wrong/ambiguous choice fails in
    # milliseconds instead of after a 15 GB model load.
    try:
        lens_path = resolve_lens_path(
            os.environ.get("JLENS_PATH"), str(_repo_root / "data" / "lens"))
    except LensResolutionError as e:
        print(f"\nLENS SELECTION FAILED\n{e}\n", flush=True)
        raise
    if lens_path is None:
        print("LENS: none — explicit lens-less mode (logit lens, final layer only)",
              flush=True)
    print(f"Loading model {_model_id!r}...", flush=True)
    _model = load_model(_model_id)
    print(f"  {_model}", flush=True)
    if lens_path is not None:
        print(f"Loading lens from {lens_path}...", flush=True)
        _lens = JacobianLens.load(lens_path)
        print(f"  {_lens}", flush=True)
        _t0 = time.perf_counter()
        _lens.warm()
        print(f"  lens warmed (fp16 GPU upload) in {time.perf_counter() - _t0:.1f}s", flush=True)
    _bands = _load_bands(lens_path)


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
        # Measured bands when available; else the UI uses its percentage guess.
        "bands": _bands or [],
        "bands_are_fallback": _bands is None,
    }


@app.get("/api/model")
async def model_info():
    return {"model_id": _model_id, "n_layers": _model.n_layers, "d_model": _model.d_model,
            "mode": _app_mode}


def _require_active_mode():
    """Reject state-changing requests in a presentation deployment."""
    if _app_mode != "active":
        raise HTTPException(403, "read-only presentation deployment")


def _record_layers() -> list[int]:
    layers = list(_lens.source_layers) if _lens is not None else [_model.n_layers - 1]
    return sorted(set(layers) | {_model.n_layers - 1})


def _blocked_topk(logits: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Exact top-k (ids, values) along the last axis, descending.

    Two-stage: per-block maxima -> top-k blocks by max -> exact sort of
    the k*C surviving candidates. Exact, not approximate: any top-k
    element has at most k-1 elements above it, which occupy at most k-1
    other blocks, so its own block ranks within the top-k blocks by max.

    Replaces a full argsort over the ~248k vocab: MLX's argsort,
    argpartition, and topk all run a full sort on GPU (measured ~14.7 ms
    per generated token at [64,1,248k] vs ~0.8 ms for this scheme; see
    docs/perf/LEDGER.md H1). Values match argsort bit-for-bit; ordering
    at exact score ties may differ, which the decode gate's golden
    readout pins.
    """
    *lead, V = logits.shape
    B = 1024                      # blocks -> C = ceil(V/B) candidates each
    C = (V + B - 1) // B
    pad = B * C - V
    x = logits
    if pad:
        fill = mx.full((*lead, pad), -float("inf"), dtype=x.dtype)
        x = mx.concatenate([x, fill], axis=-1)
    xb = x.reshape(*lead, B, C)
    bmax = xb.max(axis=-1)                                     # [..., B]
    top_blocks = mx.argsort(bmax, axis=-1)[..., -k:]           # [..., k]
    cand = mx.take_along_axis(
        xb, mx.broadcast_to(top_blocks[..., None], (*lead, k, C)), axis=-2
    )                                                          # [..., k, C]
    flat = cand.reshape(*lead, k * C)
    sub = mx.argsort(flat, axis=-1)[..., -k:][..., ::-1]       # [..., k]
    vals = mx.take_along_axis(flat, sub, axis=-1)
    blk = mx.take_along_axis(top_blocks, sub // C, axis=-1)
    ids = blk * C + (sub % C)
    return ids, vals


# Position-chunk size for the batched readout (see the loop below).
READOUT_CHUNK = int(os.environ.get("JLENS_READOUT_CHUNK", "8"))

_tok_str_cache: dict[int, str] = {}


def _sample_tok(logits: mx.array, temp: float) -> int:
    lf = logits.astype(mx.float32)
    if temp == 0:
        return int(mx.argmax(lf).tolist())
    # categorical() takes unnormalized logits, NOT probabilities.
    return int(mx.random.categorical(lf / temp).tolist())


_WS_GLYPHS = {" ": "␣", "\n": "⏎", "\t": "⇥"}


_u2b_cache: dict[str, int] = {}


def _u2b() -> dict[str, int]:
    """Inverse of GPT-2's byte<->unicode table (inlined: transformers no
    longer exports bytes_to_unicode at a stable path)."""
    if not _u2b_cache:
        bs = (list(range(ord("!"), ord("~") + 1))
              + list(range(ord("¡"), ord("¬") + 1))
              + list(range(ord("®"), ord("ÿ") + 1)))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _u2b_cache.update({chr(c): b for b, c in zip(bs, cs)})
    return _u2b_cache


def _token_bytes(tid: int) -> bytes | None:
    """Raw bytes of a byte-level BPE token. Returns None if the tokenizer
    isn't byte-level or the mapping fails."""
    try:
        t = _model.tokenizer.convert_ids_to_tokens([tid])[0]
        u2b = _u2b()
        return bytes(u2b[c] for c in t)
    except Exception:
        return None


def _tok_str(tid: int) -> str:
    """Display string for a single readout token (band cells only).

    A byte-level BPE token can be a fragment of a multi-byte UTF-8 char
    (CJK, emoji): decoded alone it yields U+FFFD, which the UI renders as
    a black diamond. Show the raw bytes instead. Pure-whitespace tokens
    otherwise render as blank cells; show visible glyphs.
    """
    s = _tok_str_cache.get(tid)
    if s is None:
        s = _model.tokenizer.decode([tid])
        if "�" in s:
            b = _token_bytes(tid)
            if b is not None:
                s = "⟨" + " ".join(f"{x:02X}" for x in b) + "⟩"
        elif s and all(c in _WS_GLYPHS for c in s):
            s = "".join(_WS_GLYPHS[c] for c in s)
        _tok_str_cache[tid] = s
    return s


def _readout_at_positions(
    acts: dict[int, mx.array],
    positions: list[int],
    layers: list[int],
    top_n: int,
) -> dict[int, dict]:
    """Compute top-n J-lens tokens at `positions` for each of `layers`.

    Batched across layers: ALL transported vectors go through ONE
    final_norm + unembed + argsort ([L, P, vocab]) instead of one per
    layer. This reads the ~636MB quantized lm_head once per call instead
    of |layers| times (~40GB of traffic per token at 64 layers) and does
    one GPU->CPU sync instead of two per layer. Identical values.

    Returns {layer: {"top_ids": [[int]*n]*len(positions),
                     "top_tokens": [[str]*n]*len(positions),
                     "top_scores": [[float]*n]*len(positions)}}.
    """
    out: dict[int, dict] = {l: {"top_ids": [], "top_tokens": [], "top_scores": []}
                            for l in layers}
    if not positions:
        return out

    # Chunk positions to bound the [L, P_chunk, vocab] logits tensor.
    # Each chunk pays one full read of the ~3.3GB J stack + ~636MB lm_head
    # REGARDLESS of how many positions it carries, so larger chunks
    # amortize weight traffic across positions; the transient fp32 logits
    # tensor is the ceiling (P=32 -> ~2GB).
    for c0 in range(0, len(positions), READOUT_CHUNK):
        chunk = positions[c0:c0 + READOUT_CHUNK]
        pos_idx = mx.array(chunk)
        t = perf.begin()
        hs = []
        for layer in layers:
            h = acts[layer][0][pos_idx].astype(mx.float32)  # [P, D]
            if _lens is not None and layer in _lens.jacobians:
                h = _lens.transport(h, layer)
            hs.append(h)
        hstack = mx.stack(hs)  # [L, P, D]
        t = perf.mark(t, "readout.transport", hstack)
        logits = _model.unembed(_model.final_norm(hstack)).astype(mx.float32)
        t = perf.mark(t, "readout.unembed", logits)
        order, vals = _blocked_topk(logits, top_n)  # [L, P, n]
        t = perf.mark(t, "readout.topk", order, vals)
        ids_l = order.tolist()
        vals_l = vals.tolist()
        perf.mark(t, "readout.tolist")
        for li, layer in enumerate(layers):
            o = out[layer]
            for pi in range(len(chunk)):
                ids = [int(t) for t in ids_l[li][pi]]
                o["top_ids"].append(ids)
                o["top_scores"].append([float(s) for s in vals_l[li][pi]])
                o["top_tokens"].append([_tok_str(t) for t in ids])
        del logits, order, vals
    return out


@app.post("/api/slice")
async def slice_endpoint(req: SliceRequest):
    _require_active_mode()
    if _model is None:
        raise HTTPException(503, "model not loaded")
    prompt = req.prompt
    max_seq_len = req.max_seq_len
    top_n = req.top_n

    # Same batched readout as the chat stream (one forward, one
    # transport/unembed/top-k pass over [L, P, vocab] chunks) — replaces
    # the original per-layer, per-position argsort loop that did
    # seq_len x n_layers full-vocab sorts with per-score GPU syncs.
    layers = list(_lens.source_layers) if _lens is not None else [_model.n_layers - 1]
    record = sorted(set(layers) | {_model.n_layers - 1})

    input_ids = _model.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])

    def _forward_acts():
        _, acts = _model.forward(input_ids, capture_layers=record)
        for l in record:
            mx.eval(acts[l])
        return acts

    async with _gpu_lock:
        acts = await asyncio.to_thread(_forward_acts)
        cells = await asyncio.to_thread(
            _readout_at_positions, acts, list(range(seq_len)), record, top_n)

    token_strs = [_model.tokenizer.decode([int(t)]) for t in input_ids[0].tolist()]
    return JSONResponse({"layers": record, "seq_len": seq_len,
                         "token_strs": token_strs, "top_n": top_n,
                         "cells": cells})


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 32
    temp: float = 0.0


@app.post("/api/generate")
async def generate_endpoint(req: GenerateRequest):
    """Generate a continuation (no intervention)."""
    _require_active_mode()
    if _model is None:
        raise HTTPException(503, "model not loaded")
    text, toks = _model.generate(req.prompt, max_tokens=req.max_tokens, temp=req.temp)
    return {"text": text, "token_ids": toks,
            "tokens": [_model.tokenizer.decode([t]) for t in toks]}


class InterveneRequest(BaseModel):
    prompt: str
    max_tokens: int = 32
    temp: float = 0.0
    layer: int
    mode: str  # "steer" | "swap" | "ablate"
    token: str | None = None       # for steer (the concept to inject)
    target: str | None = None      # for swap (replace `token` with `target`)
    alpha: float = 1.0
    positions: list[int] | None = None  # default: all positions
    each_step: bool = False


@app.post("/api/intervene")
async def intervene_endpoint(req: InterveneRequest):
    """Generate with a J-space intervention (single layer, uncached path).

    Legacy — superseded by the streaming `interventions` field on
    /api/chat_stream. Kept as a curl-able reference that matches
    scripts/intervention_sanity.py semantics.
    """
    _require_active_mode()
    if _model is None:
        raise HTTPException(503, "model not loaded")
    if _lens is None:
        raise HTTPException(400, "no lens loaded; fit one first")
    if req.layer not in _lens.jacobians:
        raise HTTPException(400, f"layer {req.layer} not fitted (source_layers={_lens.source_layers})")

    from .interventions import (
        j_lens_vector_for_text, steer as steer_op, patch_swap, ablate_topk
    )

    # Build the intervention function.
    if req.mode == "steer":
        if not req.token:
            raise HTTPException(400, "steer requires `token`")
        v = j_lens_vector_for_text(_lens, _model, req.layer, req.token)
        def fn(h):
            return steer_op(h, v, req.alpha)
    elif req.mode == "swap":
        if not req.token or not req.target:
            raise HTTPException(400, "swap requires `token` and `target`")
        v_s = j_lens_vector_for_text(_lens, _model, req.layer, req.token)
        v_t = j_lens_vector_for_text(_lens, _model, req.layer, req.target)
        def fn(h):
            return patch_swap(h, v_s, v_t, req.alpha)
    elif req.mode == "ablate":
        def fn(h):
            return ablate_topk(h, _lens, _model, req.layer, k=16)
    else:
        raise HTTPException(400, f"unknown mode {req.mode!r}")

    # Baseline
    base_text, base_toks = _model.generate(req.prompt, max_tokens=req.max_tokens, temp=req.temp)
    # With intervention
    int_text, int_toks = _model.generate(
        req.prompt, max_tokens=req.max_tokens, temp=req.temp,
        intervene_layer=req.layer, intervene_fn=fn,
        intervene_positions=req.positions, intervene_each_step=req.each_step,
    )
    return {
        "baseline": {"text": base_text, "tokens": [_model.tokenizer.decode([t]) for t in base_toks]},
        "intervened": {"text": int_text, "tokens": [_model.tokenizer.decode([t]) for t in int_toks]},
        "intervention": {"layer": req.layer, "mode": req.mode, "token": req.token,
                         "target": req.target, "alpha": req.alpha},
    }


class TokenizeRequest(BaseModel):
    text: str


@app.post("/api/tokenize")
async def tokenize_endpoint(req: TokenizeRequest):
    """Preview how a concept string tokenizes.

    Interventions use only the FIRST token of a concept string; the UI
    calls this to show which token that is before the user commits.
    """
    if _model is None:
        raise HTTPException(503, "model not loaded")
    ids = _model.tokenizer.encode(req.text, add_special_tokens=False)
    segs = _token_segments(_model.tokenizer, ids)
    return {
        "text": req.text,
        "tokens": [
            {"id": int(t), "text": segs[i], "display": _tok_str(int(t))}
            for i, t in enumerate(ids)
        ],
    }


class InterventionSpec(BaseModel):
    """One J-space edit applied during a chat stream.

    Position scopes are GLOBAL indices into the chat-templated token
    sequence — the same coordinates the readout grid uses — and exactly
    one of `positions`, `from_position`, `segment` must be set.
    `segment: "generation"` means every generated position of this
    request (resolved server-side to the generation-prefix length).
    The UI's "suppress" is steer with a negative alpha.
    """

    mode: str                        # "steer" | "swap" | "ablate"
    layers: list[int]                # lens source layers; n_layers-1 = plain unembed
    token: str | None = None         # steer concept / swap source (text...)
    token_id: int | None = None      # ...or exact token id (id wins over text)
    target: str | None = None        # swap target
    target_id: int | None = None
    alpha: float = 1.0               # steer: h += a*v (a<0 suppress); swap: coord scale
    positions: list[int] | None = None
    from_position: int | None = None
    segment: str | None = None       # "generation"
    ablate_token_ids: list[int] | None = None  # explicit directions to remove


def _validate_and_resolve_specs(
    specs: list[InterventionSpec],
    lens,
    tokenizer,
    n_layers: int,
) -> list[dict[str, Any]]:
    """Validate specs and resolve concept text -> token ids.

    Returns JSON-safe dicts that double as the `stream_start` echo and
    the compile_edits() inputs. Raises HTTPException(400) on any invalid
    spec — called in the endpoint body BEFORE the SSE stream starts, so
    failures surface as proper HTTP errors.
    """
    fitted = set(lens.source_layers) if lens is not None else set()
    allowed = fitted | {n_layers - 1}
    resolved: list[dict[str, Any]] = []
    for i, spec in enumerate(specs):
        where = f"interventions[{i}]"
        if spec.mode not in ("steer", "swap", "ablate"):
            raise HTTPException(400, f"{where}: unknown mode {spec.mode!r}")
        if spec.segment is not None and spec.segment != "generation":
            raise HTTPException(400, f"{where}: unknown segment {spec.segment!r}")
        layers = sorted(set(spec.layers))
        if not layers:
            raise HTTPException(400, f"{where}: layers is empty")
        bad = sorted(l for l in layers if l not in allowed)
        if bad:
            raise HTTPException(
                400,
                f"{where}: layers {bad} have no fitted Jacobian "
                f"(fitted={sorted(fitted)}, final={n_layers - 1})",
            )
        n_scopes = sum(
            s is not None for s in (spec.positions, spec.from_position, spec.segment)
        )
        if n_scopes != 1:
            raise HTTPException(
                400,
                f"{where}: exactly one of positions / from_position / segment "
                f"is required (got {n_scopes})",
            )
        positions = sorted(set(spec.positions)) if spec.positions is not None else None
        if positions is not None and not positions:
            raise HTTPException(400, f"{where}: positions is empty")
        if positions is not None and positions[0] < 0:
            raise HTTPException(400, f"{where}: positions must be >= 0")
        if spec.from_position is not None and spec.from_position < 0:
            raise HTTPException(400, f"{where}: from_position must be >= 0")

        def _resolve(text: str | None, tid: int | None, field: str) -> tuple[int, list[int]]:
            if tid is not None:
                return int(tid), [int(tid)]
            if not text:
                raise HTTPException(
                    400, f"{where}: {field} is required for mode {spec.mode!r}")
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise HTTPException(
                    400, f"{where}: {field} {text!r} encodes to no tokens")
            return int(ids[0]), [int(t) for t in ids]

        token_id = target_id = None
        token_ids_all = target_ids_all = None
        ablate_ids = None
        if spec.mode in ("steer", "swap"):
            token_id, token_ids_all = _resolve(spec.token, spec.token_id, "token")
        if spec.mode == "swap":
            target_id, target_ids_all = _resolve(spec.target, spec.target_id, "target")
        if spec.mode == "ablate":
            ablate_ids = sorted({int(t) for t in (spec.ablate_token_ids or [])})
            if not ablate_ids:
                raise HTTPException(400, f"{where}: ablate requires ablate_token_ids")

        token_str = _tok_str(token_id) if token_id is not None else None
        target_str = _tok_str(target_id) if target_id is not None else None
        if spec.mode == "steer":
            label = f"steer {token_str!r} a={spec.alpha:g}"
        elif spec.mode == "swap":
            label = f"swap {token_str!r}->{target_str!r} a={spec.alpha:g}"
        else:
            label = f"ablate k={len(ablate_ids)}"
        resolved.append({
            "index": i,
            "mode": spec.mode,
            "layers": layers,
            "token": spec.token, "token_id": token_id,
            "token_ids_all": token_ids_all, "token_str": token_str,
            "target": spec.target, "target_id": target_id,
            "target_ids_all": target_ids_all, "target_str": target_str,
            "alpha": spec.alpha,
            "positions": positions,
            "from_position": spec.from_position,
            "segment": spec.segment,
            "ablate_token_ids": ablate_ids,
            "label": label,
        })
    return resolved


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatStreamRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 32
    temp: float = 0.0
    top_n: int = 10
    # Qwen3.x thinking: disabled by default for the demo — the <think>
    # block burns hundreds of tokens before the answer. The J-lens stays
    # exact either way (this only changes the prompt template).
    enable_thinking: bool = False
    # J-space edits applied during prefill + generation. Empty list =
    # byte-identical clean behavior. NOTE: for a meaningful baseline vs
    # intervened comparison the client must keep every other field
    # (messages, temp, enable_thinking) identical across the two runs —
    # the chat template determines the global position coordinates.
    interventions: list[InterventionSpec] = []


# A planner probe is deliberately narrower than /api/chat_stream: it reuses
# the resident language model but never captures activations or computes a
# J-space readout.  The explicit limits make long-context timing experiments
# fail loudly instead of inheriting MLXLensModel.encode()'s 512-token tail
# truncation.
_PLANNER_CONTEXT_LIMIT = int(
    os.environ.get("JLENS_PLANNER_CONTEXT_LIMIT", "262144")
)
_PLANNER_MAX_OUTPUT_TOKENS = 256


class PlannerProbeRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    enable_thinking: bool = False
    prefill_chunk_size: int = 2048
    time_budget_seconds: float = 180.0
    max_input_tokens: int = 80000


async def _planner_gpu_call(fn, *args):
    """Run one GPU-bearing call off-loop without unsafe cancellation.

    Cancelling ``asyncio.to_thread`` cannot stop the underlying worker.  If
    the awaiter simply unwinds, its surrounding ``async with _gpu_lock`` can
    release the lock while MLX is still executing.  Shield the worker and, on
    cancellation, wait until it has really finished before re-raising so the
    shared lock remains authoritative.
    """
    worker = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancelled:
        # A caller may cancel more than once.  Keep shielding until the worker
        # has actually stopped; an MLX kernel already in flight is not
        # preemptible from Python.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        # Retrieve a possible worker exception to avoid an unobserved-task
        # warning.  Cancellation remains the externally visible result.
        try:
            worker.result()
        except BaseException:
            pass
        raise cancelled


def _planner_eos_ids(tokenizer) -> set[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(token_id) for token_id in value}
    return {int(value)}


@app.post("/api/planner_probe")
async def planner_probe(req: PlannerProbeRequest):
    """Run a timed, lens-free planner generation on the resident model.

    This endpoint is intended for intervention-planner experiments.  It uses
    the real chat template and cached inference session, but ``make_stream()``
    is called without capture layers and no readout helper is involved.
    Prefill is chunked so the time budget can be checked between bounded GPU
    calls.  The budget is cooperative: a single MLX call already in flight is
    allowed to finish before the request stops.
    """
    _require_active_mode()
    if _model is None:
        raise HTTPException(503, "model not loaded")
    if not req.messages:
        raise HTTPException(400, "messages is empty")
    bad_roles = sorted({
        message.role for message in req.messages
        if message.role not in ("system", "user", "assistant")
    })
    if bad_roles:
        raise HTTPException(400, f"unsupported message roles: {bad_roles}")
    if not (1 <= req.max_tokens <= _PLANNER_MAX_OUTPUT_TOKENS):
        raise HTTPException(
            400,
            f"max_tokens must be in [1, {_PLANNER_MAX_OUTPUT_TOKENS}]",
        )
    if not (0.0 <= req.temperature <= 2.0):
        raise HTTPException(400, "temperature must be in [0, 2]")
    if not (1 <= req.prefill_chunk_size <= 8192):
        raise HTTPException(400, "prefill_chunk_size must be in [1, 8192]")
    if not (0.001 <= req.time_budget_seconds <= 300.0):
        raise HTTPException(400, "time_budget_seconds must be in [0.001, 300]")
    if not (1 <= req.max_input_tokens <= _PLANNER_CONTEXT_LIMIT):
        raise HTTPException(
            400,
            f"max_input_tokens must be in [1, {_PLANNER_CONTEXT_LIMIT}]",
        )

    request_started = time.perf_counter()
    tok = _model.tokenizer
    prompt_messages = [
        {"role": message.role, "content": message.content}
        for message in req.messages
    ]

    tokenize_started = time.perf_counter()

    def _apply_template():
        ids = tok.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            enable_thinking=req.enable_thinking,
        )
        # transformers v5 may return a BatchEncoding even for one text
        # sequence.  Normalize its input_ids explicitly rather than iterating
        # the mapping keys ("input_ids", "attention_mask").
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            try:
                ids = ids["input_ids"]
            except (KeyError, TypeError):
                raise ValueError(
                    "chat template result does not contain input_ids"
                )
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        # Some tokenizer configurations retain a batch dimension of one.
        if (isinstance(ids, (list, tuple)) and len(ids) == 1
                and isinstance(ids[0], (list, tuple))):
            ids = ids[0]
        if isinstance(ids, list):
            return [int(token_id) for token_id in ids]
        return [int(token_id) for token_id in ids]

    # Tokenization can itself be noticeable for paper-sized context, so keep
    # it off the event loop too.  It is CPU-only and therefore does not need
    # the GPU lock.
    input_ids = await asyncio.to_thread(_apply_template)
    tokenize_ms = 1000.0 * (time.perf_counter() - tokenize_started)
    input_tokens = len(input_ids)
    if input_tokens == 0:
        raise HTTPException(400, "chat template produced no input tokens")
    if input_tokens > req.max_input_tokens:
        raise HTTPException(
            413,
            f"templated input has {input_tokens} tokens; "
            f"max_input_tokens is {req.max_input_tokens} (not truncated)",
        )
    if input_tokens + req.max_tokens > _PLANNER_CONTEXT_LIMIT:
        raise HTTPException(
            413,
            f"input ({input_tokens}) + max_tokens ({req.max_tokens}) exceeds "
            f"model context {_PLANNER_CONTEXT_LIMIT} (not truncated)",
        )

    def _budget_expired() -> bool:
        return time.perf_counter() - request_started >= req.time_budget_seconds

    queue_wait_ms = 0.0
    session_setup_ms = 0.0
    prefill_ms = 0.0
    decode_ms = 0.0
    total_service_ms = 0.0
    ttft_ms: float | None = None
    input_tokens_processed = 0
    prefill_chunks = 0
    decode_steps = 0
    output_ids: list[int] = []
    stop_reason = "time_budget" if _budget_expired() else ""
    eos_ids = _planner_eos_ids(tok)

    if not stop_reason:
        queue_started = time.perf_counter()
        async with _gpu_lock:
            lock_acquired = time.perf_counter()
            queue_wait_ms = 1000.0 * (lock_acquired - queue_started)
            service_started = lock_acquired
            session = None
            logits = None
            try:
                if _budget_expired():
                    stop_reason = "time_budget"
                else:
                    setup_started = time.perf_counter()
                    # No capture_layers argument: the returned activation dict
                    # stays empty and no J-space readout is possible.
                    session = await _planner_gpu_call(_model.make_stream)
                    session_setup_ms = 1000.0 * (
                        time.perf_counter() - setup_started
                    )

                    for offset in range(
                        0, input_tokens, req.prefill_chunk_size
                    ):
                        if _budget_expired():
                            stop_reason = "time_budget"
                            break
                        chunk = input_ids[
                            offset:offset + req.prefill_chunk_size
                        ]
                        prefill_started = time.perf_counter()
                        logits, _ = await _planner_gpu_call(
                            session.extend, chunk
                        )
                        prefill_ms += 1000.0 * (
                            time.perf_counter() - prefill_started
                        )
                        input_tokens_processed += len(chunk)
                        prefill_chunks += 1

                    if (not stop_reason
                            and input_tokens_processed == input_tokens):
                        if _budget_expired():
                            stop_reason = "time_budget"
                        else:
                            next_token = await _planner_gpu_call(
                                _sample_tok, logits, req.temperature
                            )
                            ttft_ms = 1000.0 * (
                                time.perf_counter() - request_started
                            )
                            if next_token in eos_ids:
                                stop_reason = "eos"
                            else:
                                output_ids.append(next_token)

                    while not stop_reason:
                        if len(output_ids) >= req.max_tokens:
                            stop_reason = "max_tokens"
                            break
                        if _budget_expired():
                            stop_reason = "time_budget"
                            break

                        # The first output token is predicted by prefill.  Each
                        # subsequent decision is one cached decode step.
                        decode_started = time.perf_counter()
                        logits, _ = await _planner_gpu_call(
                            session.extend, [output_ids[-1]]
                        )
                        next_token = await _planner_gpu_call(
                            _sample_tok, logits, req.temperature
                        )
                        decode_ms += 1000.0 * (
                            time.perf_counter() - decode_started
                        )
                        decode_steps += 1
                        if next_token in eos_ids:
                            stop_reason = "eos"
                            break
                        output_ids.append(next_token)
            finally:
                # Drop references to this request's potentially large caches
                # before another waiter acquires the GPU lock.
                logits = None
                session = None
                total_service_ms = 1000.0 * (
                    time.perf_counter() - service_started
                )

    # Decoding text is CPU-only and need not hold up the next GPU request.
    if output_ids:
        output_text = await asyncio.to_thread(
            tok.decode, output_ids, skip_special_tokens=True
        )
    else:
        output_text = ""
    total_request_ms = 1000.0 * (time.perf_counter() - request_started)

    prefill_seconds = prefill_ms / 1000.0
    decode_seconds = decode_ms / 1000.0
    return {
        "text": output_text,
        "token_ids": output_ids,
        "input_tokens": input_tokens,
        "input_tokens_processed": input_tokens_processed,
        "output_tokens": len(output_ids),
        "prefill_chunks": prefill_chunks,
        "decode_steps": decode_steps,
        "tokenize_ms": tokenize_ms,
        "queue_wait_ms": queue_wait_ms,
        "session_setup_ms": session_setup_ms,
        "prefill_ms": prefill_ms,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "total_service_ms": total_service_ms,
        "total_request_ms": total_request_ms,
        "prefill_tokens_per_second": (
            input_tokens_processed / prefill_seconds
            if prefill_seconds > 0 else 0.0
        ),
        "decode_tokens_per_second": (
            decode_steps / decode_seconds if decode_seconds > 0 else 0.0
        ),
        "stop_reason": stop_reason or "error",
        "lens_readout": False,
        "model_id": _model_id,
        "lens_n_prompts": (
            int(_lens.n_prompts) if _lens is not None else None
        ),
    }



def _token_segments(tok, ids: list[int]) -> list[str]:
    """Per-token text segments that concatenate to the exact decoded text.

    Byte-level BPE can split one UTF-8 character (e.g. a rare CJK char or an
    emoji) across tokens; decoding tokens individually yields U+FFFD
    replacement chars that can never be re-joined. Decode cumulatively and
    attribute each newly-stabilized span to the token that completed it
    (partial sequences hold back until complete).
    """
    segments, prev = [], ""
    for i in range(len(ids)):
        full = tok.decode(ids[:i + 1])
        stable = full.rstrip("\ufffd")
        if len(stable) >= len(prev):
            segments.append(stable[len(prev):])
            prev = stable
        else:
            segments.append("")
    # Any trailing incomplete bytes: surface them on the last token.
    full = tok.decode(ids)
    if len(full) > len(prev) and segments:
        segments[-1] += full[len(prev):]
    return segments


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\n".encode() + b"data: " + json.dumps(data).encode() + b"\n\n"


class ChatControlRequest(BaseModel):
    stream_id: str
    action: str  # "pause" | "resume"


class SessionSaveRequest(BaseModel):
    preview: str | None = None
    messages: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    layers: list[int] = []
    settings: dict[str, Any] = {}
    # Token markups: [{pos, layer, token}] bookmarks into the J-Space grid.
    markups: list[dict[str, Any]] = []
    # Intervention specs authored in the UI, the specs applied to the
    # current run, and the baseline-vs-intervened compare envelope.
    # Client-defined shapes, persisted verbatim (pydantic drops unknown
    # keys, so these MUST be declared to round-trip).
    interventions: list[dict[str, Any]] = []
    appliedInterventions: list[dict[str, Any]] = []
    compare: dict[str, Any] | None = None
    # Fire-and-forget full-fidelity save at stream end. localStorage is
    # quota-trimmed (~4.2MB) and silently loses old snapshots on long
    # conversations; the autosave keeps the complete state server-side
    # under a fixed id (overwritten each stream, never git-staged).
    autosave: bool = False


def _session_preview(data: dict[str, Any]) -> str:
    preview = str(data.get("preview") or "").strip()
    if preview:
        return preview[:160]
    for msg in data.get("messages") or []:
        if msg.get("role") == "user":
            content = str(msg.get("content") or "").strip()
            if content:
                return re.sub(r"\s+", " ", content)[:160]
    return "Untitled session"


def _session_slug(preview: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", preview.lower()).strip("-")
    return (slug or "session")[:48]


def _session_path(session_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "", session_id)
    path = (_sessions_dir / safe_id).resolve()
    if not str(path).startswith(str(_sessions_dir.resolve())) or path.suffix != ".json":
        raise HTTPException(400, "invalid session id")
    return path


def _stage_session_file(path: Path) -> bool:
    try:
        rel = path.relative_to(_repo_root)
        proc = subprocess.run(
            ["git", "add", str(rel)],
            cwd=_repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _remove_session_file(path: Path) -> bool:
    try:
        rel = path.relative_to(_repo_root)
        proc = subprocess.run(
            ["git", "rm", "-f", "--", str(rel)],
            cwd=_repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            return True
        if path.exists():
            path.unlink()
        proc = subprocess.run(
            ["git", "add", "-u", "--", str(rel)],
            cwd=_repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


@app.get("/api/sessions")
async def list_sessions():
    _sessions_dir.mkdir(parents=True, exist_ok=True)
    sessions = []
    for path in sorted(_sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        # The rolling autosave is a recovery backup (overwritten by every
        # stream), not a saved session — it stays fetchable by id for the
        # client's trimmed-cache recovery but is never listed.
        if path.name == "autosave-latest.json":
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        sessions.append({
            "id": path.name,
            "preview": _session_preview(data),
            "created_at": data.get("created_at"),
            "saved_at": data.get("saved_at"),
            "message_count": len(data.get("messages") or []),
            "snapshot_count": len(data.get("snapshots") or []),
            "path": str(path.relative_to(_repo_root)),
        })
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(404, "session not found")
    return JSONResponse(json.loads(path.read_text()))


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    _require_active_mode()
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(404, "session not found")
    staged = _remove_session_file(path)
    return {
        "id": session_id,
        "deleted": True,
        "staged": staged,
    }


@app.post("/api/sessions")
async def save_session(req: SessionSaveRequest):
    _require_active_mode()
    _sessions_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data = req.dict()
    autosave = data.pop("autosave", False)
    preview = _session_preview(data)
    if autosave:
        session_id = "autosave-latest.json"
    else:
        session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{_session_slug(preview)}.json"
    path = _sessions_dir / session_id
    data.update({
        "schema": 1,
        "preview": preview,
        "created_at": now,
        "saved_at": now,
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    staged = False if autosave else _stage_session_file(path)
    return {
        "id": session_id,
        "preview": preview,
        "created_at": now,
        "saved_at": now,
        "path": str(path.relative_to(_repo_root)),
        "staged": staged,
    }


@app.post("/api/chat_control")
async def chat_control(req: ChatControlRequest):
    """Pause or resume a generation stream.

    When paused, the server stops generating new tokens (the generation loop
    blocks on an asyncio.Event). When resumed, generation continues from where
    it left off. The stream_id is returned in the initial `event: stream_start`
    frame of the chat_stream response.
    """
    _require_active_mode()
    if req.stream_id not in _pause_events:
        raise HTTPException(404, f"stream {req.stream_id} not found")
    ev = _pause_events[req.stream_id]
    if req.action == "pause":
        ev.clear()
    elif req.action == "resume":
        ev.set()
    else:
        raise HTTPException(400, f"unknown action {req.action!r}")
    return {"stream_id": req.stream_id, "action": req.action, "paused": not ev.is_set()}


@app.post("/api/chat_stream")
async def chat_stream(req: ChatStreamRequest):
    """Stream J-lens snapshots as the assistant generates tokens.

    Emits `event: snapshot` frames:
      - prefill (token_idx = -1): grid for all prompt positions so far.
      - per-token (token_idx = k): row for the new position only.
    Then `event: done` when generation finishes (or EOS).

    The server streams unconditionally; the client decides which snapshot
    to render. Snapshots carry a monotonic `snapshot_id` so an SSE
    reconnect with `Last-Event-ID` can resume.
    """
    _require_active_mode()
    if _model is None:
        raise HTTPException(503, "model not loaded")

    import asyncio
    import uuid

    # Validate + resolve interventions BEFORE streaming starts so bad
    # specs fail as proper HTTP 400s instead of mid-stream error frames.
    resolved_specs: list[dict[str, Any]] = []
    if req.interventions:
        resolved_specs = _validate_and_resolve_specs(
            req.interventions, _lens, _model.tokenizer, _model.n_layers)

    layers = _record_layers()
    top_n = req.top_n
    temp = req.temp
    max_tokens = req.max_tokens

    # Create a unique stream_id and a pause event for this stream.
    stream_id = str(uuid.uuid4())[:8]
    pause_event = asyncio.Event()
    pause_event.set()  # start unpaused (generating)
    _pause_events[stream_id] = pause_event

    async def gen():
        snapshot_id = 0
        token_strs_global: list[str] = []

        # Emit the stream_start event so the client knows the stream_id for
        # pause/resume control (plus the resolved intervention echo).
        yield _sse("stream_start", {
            "stream_id": stream_id,
            "interventions": resolved_specs,
        })

        try:
            # Build an effective message list: the client sends user messages
            # (+ optionally prior assistant turns for multi-turn context). We
            # auto-append an empty assistant message at the end so generation
            # always runs after the latest user turn.
            eff_messages = list(req.messages)
            if not eff_messages or eff_messages[-1].role != "assistant":
                eff_messages.append(ChatMessage(role="assistant", content=""))

            tok = _model.tokenizer

            # One cached decoding session for the whole stream: each message
            # (and each generated token) is fed as a delta chunk, so the
            # per-step cost is O(chunk) instead of O(total sequence).
            session = _model.make_stream(capture_layers=layers)

            if resolved_specs:
                # "generation" scopes resolve to the first generated
                # position = the generation-prefix length of THIS
                # conversation (template-dependent, so server-side).
                gen_from_pos = None
                if any(r["segment"] == "generation" for r in resolved_specs):
                    last = eff_messages[-1]
                    if last.role == "assistant" and not last.content:
                        pm = [{"role": m.role, "content": m.content}
                              for m in eff_messages[:-1]]
                        pids = tok.apply_chat_template(
                            pm, add_generation_prompt=True,
                            enable_thinking=req.enable_thinking)
                        gen_from_pos = len(
                            pids if isinstance(pids, list) else list(pids))

                def _compile_all():
                    from .interventions import compile_edits
                    edits = []
                    for r in resolved_specs:
                        from_pos = r["from_position"]
                        if r["segment"] == "generation":
                            if gen_from_pos is None:
                                # Nothing generated this request -> inert.
                                continue
                            from_pos = gen_from_pos
                        edits.extend(compile_edits(
                            _lens, _model,
                            mode=r["mode"], layers=r["layers"],
                            token_id=r["token_id"], target_id=r["target_id"],
                            alpha=r["alpha"], positions=r["positions"],
                            from_pos=from_pos,
                            ablate_token_ids=r["ablate_token_ids"],
                            label=r["label"],
                        ))
                    return edits

                # Vector compile is a handful of matvecs against the
                # resident fp16 J stack — never the [vocab, D] matrix.
                async with _gpu_lock:
                    session.set_edits(await asyncio.to_thread(_compile_all))

            # We build the formatted token sequence using the chat template.
            # For each message, the template wraps it as:
            #   <|im_start|>{role}\n{content}<|im_end|>\n
            # We need to track which token positions are the CONTENT of each
            # message (excluding the markers), so readouts align to the
            # actual content tokens.
            #
            # Strategy: for each prefix of eff_messages, apply the template
            # with add_generation_prompt=False to get the token IDs up to and
            # including that message's <|im_end|>\n. The content tokens are
            # between the "{role}\n" marker and the "<|im_end|>" marker.

            def content_positions_for(formatted_ids, role, content_ids):
                """Find the start and end positions of the content within formatted_ids.
                The template wraps as: <|im_start|>{role}\n{content}<|im_end|>\n
                We find the content by locating <|im_start|> ... {role}\n ... <|im_end|>."""
                # For the LAST message, the template might not add <|im_end|>\n
                # (depends on add_generation_prompt). We handle both cases.
                # Simple approach: find the LAST <|im_start|> in the sequence,
                # then skip past the role + \n, that's the content start.
                # The content end is the next <|im_end|> after that, or end of seq.
                im_start_id = 248045  # <|im_start|>
                im_end_id = 248046    # <|im_end|>
                # Find the last <|im_start|>
                last_im_start = -1
                for i in range(len(formatted_ids) - 1, -1, -1):
                    if formatted_ids[i] == im_start_id:
                        last_im_start = i
                        break
                if last_im_start < 0:
                    return 0, len(formatted_ids)
                # After <|im_start|> comes the role word + \n. The role is
                # encoded as one or more tokens. We find the first \n (198)
                # after last_im_start; content starts right after it.
                content_start = last_im_start + 1
                for i in range(last_im_start + 1, len(formatted_ids)):
                    if formatted_ids[i] == 198:  # \n
                        content_start = i + 1
                        break
                # Content end: next <|im_end|> after content_start, or end.
                content_end = len(formatted_ids)
                for i in range(content_start, len(formatted_ids)):
                    if formatted_ids[i] == im_end_id:
                        content_end = i
                        break
                return content_start, content_end

            global_pos = 0
            input_ids = None

            # Process all messages except the last (which is the empty assistant
            # to generate). Each prior message is context: we apply the template
            # for the prefix, find the content positions, forward, emit prefill.
            n_context = len(eff_messages) - 1  # all but the trailing empty asst
            # Actually, the trailing message might have content if the client
            # sent an assistant message (but we only auto-append empty). So:
            # - If the last message has content, it's historical (process as context).
            # - If the last message is empty, it's the generation target.
            last_msg = eff_messages[-1]
            is_generate = (last_msg.role == "assistant" and not last_msg.content)
            n_to_process = len(eff_messages) if not is_generate else len(eff_messages) - 1

            for msg_idx in range(n_to_process):
                msg = eff_messages[msg_idx]
                # Apply the template for messages[0..msg_idx] without generation prompt.
                prefix_msgs = [{"role": m.role, "content": m.content} for m in eff_messages[:msg_idx + 1]]
                formatted_ids = tok.apply_chat_template(prefix_msgs, add_generation_prompt=False, enable_thinking=req.enable_thinking)
                if isinstance(formatted_ids, list):
                    formatted_ids_list = formatted_ids
                else:
                    formatted_ids_list = formatted_ids.tolist() if hasattr(formatted_ids, 'tolist') else list(formatted_ids)

                # Find content positions for this message.
                content_start, content_end = content_positions_for(formatted_ids_list, msg.role, None)
                content_ids = formatted_ids_list[content_start:content_end]

                # Feed only the DELTA since what the session has consumed
                # (chat-template prefixes are extend-only for fixed priors).
                start_pos = content_start
                end_pos = content_end
                n_new = end_pos - start_pos
                if n_new <= 0:
                    continue
                chunk_start = session.n_consumed
                delta = formatted_ids_list[chunk_start:]
                if not delta:
                    continue
                positions = list(range(start_pos, end_pos))
                async with _gpu_lock:
                    _, acts = await asyncio.to_thread(session.extend, delta)
                    # acts cover only the delta chunk -> chunk-local indices.
                    local = [p - chunk_start for p in positions]
                    row = await asyncio.to_thread(
                        _readout_at_positions, acts, local, layers, top_n)
                new_token_strs = _token_segments(tok, content_ids)

                snapshot_id += 1
                yield _sse("snapshot", {
                    "snapshot_id": snapshot_id,
                    "msg_idx": msg_idx,
                    "token_idx": -1,
                    "role": msg.role,
                    "prefill_pos": start_pos,
                    "n_tokens": n_new,
                    "token_strs": new_token_strs,
                    "grid": {"layers": layers, "positions": positions, "cells": row},
                })
                del acts
                await asyncio.sleep(0)
                if msg.role == "assistant" and msg.content:
                    yield _sse("done", {"msg_idx": msg_idx, "n_tokens": n_new, "historical": True})

            # Now handle the generation target (if the last message is an empty assistant).
            if is_generate:
                msg_idx = len(eff_messages) - 1
                # Apply the template with add_generation_prompt=True to get
                # the generation prefix (ends with <|im_start|>assistant\n...).
                prefix_msgs = [{"role": m.role, "content": m.content} for m in eff_messages[:-1]]
                gen_prefix_ids = tok.apply_chat_template(prefix_msgs, add_generation_prompt=True, enable_thinking=req.enable_thinking)
                if isinstance(gen_prefix_ids, list):
                    gen_prefix_list = gen_prefix_ids
                else:
                    gen_prefix_list = gen_prefix_ids.tolist() if hasattr(gen_prefix_ids, 'tolist') else list(gen_prefix_ids)

                global_pos = len(gen_prefix_list)

                # Feed the generation-prefix delta; readout at the frontier.
                chunk_start = session.n_consumed
                delta = gen_prefix_list[chunk_start:]
                async with _gpu_lock:
                    logits, acts = (await asyncio.to_thread(session.extend, delta)) if delta else (None, {})
                    prefill_positions = [global_pos - 1] if global_pos > 0 else []
                    prefill_row = (await asyncio.to_thread(
                        _readout_at_positions, acts, [len(delta) - 1], layers, top_n
                    )) if (prefill_positions and delta) else {l: {"top_ids": [], "top_tokens": [], "top_scores": []} for l in layers}
                    next_tok = await asyncio.to_thread(_sample_tok, logits, temp)
                del logits

                snapshot_id += 1
                yield _sse("snapshot", {
                    "snapshot_id": snapshot_id,
                    "msg_idx": msg_idx,
                    "token_idx": -1,
                    "role": "assistant",
                    "prefill_pos": global_pos - 1,
                    "n_tokens": 0,
                    "token_strs": [],
                    "grid": {"layers": layers, "positions": prefill_positions, "cells": prefill_row},
                })
                del acts
                await asyncio.sleep(0)

                n_gen = 0
                gen_ids: list[int] = []
                gen_prev = ""
                eos_id = getattr(tok, "eos_token_id", None)
                eos_hit = eos_id is not None and next_tok == eos_id
                generation_stop_reason = "eos" if eos_hit else None
                while not eos_hit and n_gen < max_tokens:
                    # Block here if the client has paused generation.
                    await pause_event.wait()

                    # One-token cached step: acts cover just this token.
                    new_pos = global_pos
                    async with _gpu_lock:
                        logits, acts = await asyncio.to_thread(session.extend, [next_tok])
                        row = await asyncio.to_thread(
                            _readout_at_positions, acts, [0], layers, top_n)
                        new_next_tok = await asyncio.to_thread(_sample_tok, logits, temp)
                    del logits

                    # Cumulative decode: emit only the newly-stabilized text
                    # so UTF-8 chars split across BPE tokens survive (a lone
                    # tok.decode([t]) turns each half into U+FFFD forever).
                    gen_ids.append(next_tok)
                    _full = tok.decode(gen_ids)
                    _stable = _full.rstrip("\ufffd")
                    if len(_stable) >= len(gen_prev):
                        tok_str = _stable[len(gen_prev):]
                        gen_prev = _stable
                    else:
                        tok_str = ""
                    snapshot_id += 1
                    yield _sse("snapshot", {
                        "snapshot_id": snapshot_id,
                        "msg_idx": msg_idx,
                        "token_idx": n_gen,
                        "role": "assistant",
                        "token": tok_str,
                        "token_id": next_tok,
                        "pos": new_pos,
                        "grid": {"layers": layers, "positions": [new_pos], "cells": row},
                    })
                    del acts
                    global_pos += 1
                    n_gen += 1
                    # A malformed intervention can otherwise keep a very large
                    # max-token run in a short token loop indefinitely. Baseline
                    # generation is deliberately untouched by this breaker.
                    if resolved_specs and _intervention_repetition(gen_ids):
                        generation_stop_reason = "repetition"
                        break
                    next_tok = new_next_tok
                    if eos_id is not None and next_tok == eos_id:
                        eos_hit = True
                        generation_stop_reason = "eos"
                    await asyncio.sleep(0)

                if generation_stop_reason is None:
                    generation_stop_reason = "max_tokens"
                yield _sse("done", {
                    "msg_idx": msg_idx, "n_tokens": n_gen,
                    "stop_reason": generation_stop_reason,
                })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[chat_stream] error after snapshot {snapshot_id}: {e}\n{tb}", flush=True)
            yield _sse("error", {"snapshot_id": snapshot_id, "error": str(e), "traceback": tb})
        finally:
            # Clean up the pause event when the stream ends.
            _pause_events.pop(stream_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Intervention scan: goal-directed probe grid over layers × strength.
#
# For a concept edit (swap source→target, or steer ±source) the endpoint
# runs a small grid of SHORT greedy probes — one layer × one alpha each —
# classifies every outcome, and streams the results as SSE so the client
# can paint the map incrementally:
#   unchanged   output identical to the baseline        (too weak)
#   derived     changed, not the literal target          (the win)
#   parrot      target appears, but the result is not a
#               strict conclusion success                (partial/off-target)
#   degenerate  repetition / collapse                    (way too strong)
#   success     terminated conclusion output contains one bounded goal and
#               no bounded occurrence of the old source
# Probes intentionally skip activation capture (make_stream() bare), so a
# probe costs a prefill + max_tokens decode steps and nothing else.
# ---------------------------------------------------------------------------

class InterventionScanRequest(BaseModel):
    messages: list[ChatMessage]
    mode: str = "swap"          # "swap" | "steer" (validator rejects others)
    token: str | None = None
    token_id: int | None = None
    target: str | None = None
    target_id: int | None = None
    # Full selected source word/phrase for conclusion-goal validation. This
    # can differ from the single token direction used by the intervention.
    source_text: str | None = None
    # Optional behavioral goal used only to classify probe output. The
    # intervention itself may resolve a multi-piece concept to one token,
    # while success should still mean producing the complete requested text.
    goal_text: str | None = None
    layers: list[int] | None = None       # default: spread across the workspace band
    alphas: list[float] | None = None     # default: mode-appropriate ladder
    # Exactly one effective scope is used. Omitting both preserves the old
    # scan default (from position zero); supplying `positions` selects only
    # those global token coordinates.
    positions: list[int] | None = None
    from_position: int | None = None
    max_tokens: int = 4
    enable_thinking: bool = False


class InterventionSearchCell(BaseModel):
    """One exact write site in an explicit intervention recipe."""

    layer: int
    position: int
    alpha: float


class InterventionSearchCandidate(BaseModel):
    """A caller-supplied recipe; cells are applied together."""

    id: str
    cells: list[InterventionSearchCell]


class InterventionSearchRequest(BaseModel):
    """Evaluate explicit swap recipes under a cooperative wall-time budget.

    This endpoint deliberately does not generate candidate coordinates.  A
    planner or UI supplies exact workspace cells and the server verifies each
    recipe with a fresh, full greedy replay.
    """

    messages: list[ChatMessage]
    token: str | None = None
    token_id: int | None = None
    target: str | None = None
    target_id: int | None = None
    source_text: str
    goal_text: str
    candidates: list[InterventionSearchCandidate]
    max_tokens: int = 64
    time_budget_seconds: float = 60.0
    enable_thinking: bool = False
    stop_on_success: bool = True


def _normalized_probe_text(text: str) -> str:
    """Normalize superficial presentation differences, not content.

    Goal checks deliberately retain punctuation and words so bounded phrase
    matching can distinguish a selected word from malformed adjacency.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _probe_is_degenerate(ids: list[int]) -> bool:
    """Detect short repetitive collapses before considering goal success."""
    n = len(ids)
    if n < 4:
        return False

    counts = Counter(ids)
    # Low-diversity/token-loop collapse, including A A * A A * ... .
    if len(counts) <= 2:
        return True
    most_common = counts.most_common(1)[0][1]
    if most_common >= 3 and most_common * 2 >= n:
        return True

    # Three adjacent copies of any short token pattern. This also catches
    # multi-piece words whose BPE ids alternate and evade a token-count test.
    for width in range(1, min(4, n // 3) + 1):
        for start in range(0, n - 3 * width + 1):
            chunk = ids[start:start + width]
            if (ids[start + width:start + 2 * width] == chunk
                    and ids[start + 2 * width:start + 3 * width] == chunk):
                return True
    return False


def _bounded_phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    """Compile the same conservative lexical boundary used by goal checks."""
    needle = _normalized_probe_text(phrase)
    if not needle:
        return None
    # Apply lexical boundaries to ASCII word edges (the failure we need to
    # reject is e.g. BeijingParis). CJK and other scripts commonly have no
    # whitespace word separators, so a Python `\w` boundary would incorrectly
    # reject 北京 inside a normal Chinese sentence.
    def _ascii_word(ch: str) -> bool:
        return ch.isascii() and (ch.isalnum() or ch == "_")

    left = r"(?<![A-Za-z0-9_])" if _ascii_word(needle[0]) else ""
    right = r"(?![A-Za-z0-9_])" if _ascii_word(needle[-1]) else ""
    return re.compile(left + re.escape(needle) + right)


def _bounded_phrase_count(text: str, phrase: str) -> int:
    """Count a normalized phrase only at Unicode word boundaries."""
    pattern = _bounded_phrase_pattern(phrase)
    if pattern is None:
        return 0
    return len(pattern.findall(_normalized_probe_text(text)))


def _expected_single_source_replacement(
    baseline: str,
    source_text: str,
    goal_text: str,
) -> str | None:
    """Normalized literal replacement oracle, when it is unambiguous.

    Behavioral success does not require this stricter oracle: a sound edit can
    legitimately change surrounding grammar.  It is reported separately as a
    useful high-precision signal when the baseline contains the old source
    exactly once.
    """
    pattern = _bounded_phrase_pattern(source_text)
    normalized = _normalized_probe_text(baseline)
    if pattern is None or len(pattern.findall(normalized)) != 1:
        return None
    return pattern.sub(_normalized_probe_text(goal_text), normalized, count=1)


def _classify_probe(
    ids: list[int],
    text: str,
    baseline: str,
    target_str: str | None,
    *,
    eos: bool,
    goal_text: str | None = None,
    source_text: str | None = None,
) -> str:
    t = _normalized_probe_text(text)
    if t == _normalized_probe_text(baseline):
        return "unchanged"
    # A loop that happens to begin with the goal is still a failed probe.
    if _probe_is_degenerate(ids):
        return "degenerate"
    if goal_text is not None:
        # Conclusion scans have a behavioral goal that may be a phrase inside
        # a longer answer. A valid result must terminate, contain the goal
        # exactly once as a bounded phrase, and no longer contain the old word.
        # In particular, `BeijingParis`, repeated Beijing, Paris+Beijing, and
        # max-token truncations are not successes.
        goal_count = _bounded_phrase_count(text, goal_text)
        source_gone = (
            not source_text or _bounded_phrase_count(text, source_text) == 0
        )
        if eos and goal_count == 1 and source_gone:
            return "success"
        # Orange conclusion cells are useful partial candidates: the requested
        # output appeared and the source disappeared, but the probe did not
        # satisfy the stricter green contract (for example, it hit max_tokens
        # after "... Beijing. Wait,"). Degeneration remains authoritative
        # because it is checked above this branch.
        if goal_count >= 1 and source_gone:
            return "parrot"
        return "derived"
    tgt = _normalized_probe_text(target_str or "")
    # Preserve the generic intervention-panel taxonomy: a literal target is a
    # `parrot`, while another changed answer is `derived`. Degeneration is
    # checked first so a target-prefixed loop is never adopted as healthy.
    if tgt and t.startswith(tgt):
        return "parrot"
    return "derived"


def _intervention_repetition(ids: list[int]) -> bool:
    """Conservative circuit breaker for runaway intervened generation.

    Require at least twelve generated tokens, then detect a short cycle whose
    repeated suffix spans at least twelve tokens, or an extremely low-diversity
    sixteen-token tail. This is intentionally less eager than scan-probe
    classification because it terminates a live generation.
    """
    n = len(ids)
    if n < 12:
        return False
    for width in range(1, 5):
        repeats = max(3, (12 + width - 1) // width)
        span = width * repeats
        if n < span:
            continue
        pattern = ids[-width:]
        start = n - span
        if all(ids[start + i:start + i + width] == pattern
               for i in range(0, span, width)):
            return True
    if n >= 16:
        tail = ids[-16:]
        counts = Counter(tail)
        if len(counts) <= 3 or counts.most_common(1)[0][1] >= 12:
            return True
    return False


def _scan_default_layers(fitted: set[int]) -> list[int]:
    """~6 write sites spread across the measured workspace band (or the
    middle half of the fitted range when no bands were measured)."""
    ws = next((b for b in (_bands or []) if b.get("name") == "workspace"), None)
    if ws:
        lo, hi = int(ws["start_layer"]), int(ws["end_layer"])
    else:
        lo = min(fitted) + (max(fitted) - min(fitted)) // 4
        hi = min(fitted) + (max(fitted) - min(fitted)) * 3 // 4
    span = [l for l in range(lo, hi + 1) if l in fitted]
    if not span:
        return sorted(fitted)[:6]
    step = max(1, (len(span) - 1) // 5) if len(span) > 1 else 1
    picked = span[::step]
    if span[-1] not in picked:
        picked.append(span[-1])
    return picked[:7]


def _chat_template_token_ids(
    tokenizer,
    messages: list[ChatMessage],
    *,
    enable_thinking: bool,
) -> list[int]:
    """Apply the model's real chat template and normalize common containers."""
    prompt_messages = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]
    ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        try:
            ids = ids["input_ids"]
        except (KeyError, TypeError):
            raise ValueError("chat template result does not contain input_ids")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if (isinstance(ids, (list, tuple)) and len(ids) == 1
            and isinstance(ids[0], (list, tuple))):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _greedy_intervention_replay(
    prompt_ids: list[int],
    edits,
    max_tokens: int,
    eos_ids: set[int],
) -> dict[str, Any]:
    """Run one fresh bare stream; safe to execute inside the GPU worker."""
    replay_started = time.perf_counter()

    setup_started = time.perf_counter()
    # Intentionally bare: no capture layers and therefore no lens readout.
    session = _model.make_stream()
    if edits:
        session.set_edits(edits)
    setup_ms = 1000.0 * (time.perf_counter() - setup_started)

    prefill_started = time.perf_counter()
    logits, _ = session.extend(prompt_ids)
    prefill_ms = 1000.0 * (time.perf_counter() - prefill_started)

    output_ids: list[int] = []
    eos = False
    repetition = False
    stop_reason = "max_tokens"
    decode_started = time.perf_counter()
    for _ in range(max_tokens):
        token_id = int(mx.argmax(logits.astype(mx.float32)).tolist())
        if token_id in eos_ids:
            eos = True
            stop_reason = "eos"
            break
        output_ids.append(token_id)
        if _intervention_repetition(output_ids):
            repetition = True
            stop_reason = "repetition"
            break
        if len(output_ids) >= max_tokens:
            break
        logits, _ = session.extend([token_id])
    decode_ms = 1000.0 * (time.perf_counter() - decode_started)

    return {
        "ids": output_ids,
        "eos": eos,
        "repetition": repetition,
        "stop_reason": stop_reason,
        "timings_ms": {
            "setup": setup_ms,
            "prefill": prefill_ms,
            "decode": decode_ms,
            "total": 1000.0 * (time.perf_counter() - replay_started),
        },
    }


@app.post("/api/intervention_search")
async def intervention_search(req: InterventionSearchRequest):
    """Verify caller-supplied workspace recipes within a wall-time budget.

    The budget is cooperative between candidates: once a candidate starts,
    its compile and replay are allowed to finish.  Consequently the response
    reports any wall-time overshoot rather than pretending an in-flight MLX
    kernel was cancelled.  This is an evaluator, not a coordinate planner.
    """
    _require_active_mode()
    request_started = time.perf_counter()
    if _model is None:
        raise HTTPException(503, "model not loaded")
    if _lens is None:
        raise HTTPException(400, "intervention search requires a loaded lens")
    if not req.messages:
        raise HTTPException(400, "messages is empty")
    bad_roles = sorted({
        message.role for message in req.messages
        if message.role not in ("system", "user", "assistant")
    })
    if bad_roles:
        raise HTTPException(400, f"unsupported message roles: {bad_roles}")
    if not req.source_text.strip():
        raise HTTPException(400, "source_text is empty")
    if not req.goal_text.strip():
        raise HTTPException(400, "goal_text is empty")
    if not (1 <= req.max_tokens <= 128):
        raise HTTPException(400, "max_tokens must be in [1, 128]")
    if not (math.isfinite(req.time_budget_seconds)
            and 0.001 <= req.time_budget_seconds <= 300.0):
        raise HTTPException(
            400, "time_budget_seconds must be finite and in [0.001, 300]"
        )
    if not (1 <= len(req.candidates) <= 256):
        raise HTTPException(400, "candidates must contain 1 to 256 recipes")

    workspace = next(
        (band for band in (_bands or []) if band.get("name") == "workspace"),
        None,
    )
    if workspace is None:
        raise HTTPException(
            400, "intervention search requires a measured workspace band"
        )
    workspace_start = int(workspace["start_layer"])
    workspace_end = int(workspace["end_layer"])
    fitted = {int(layer) for layer in _lens.source_layers}
    workspace_layers = sorted(
        layer for layer in fitted
        if workspace_start <= layer <= workspace_end
    )
    if not workspace_layers:
        raise HTTPException(
            400, "measured workspace band contains no fitted lens layers"
        )
    allowed_layers = set(workspace_layers)

    candidate_ids: set[str] = set()
    for index, candidate in enumerate(req.candidates):
        where = f"candidates[{index}]"
        if not candidate.id.strip():
            raise HTTPException(400, f"{where}.id is empty")
        if candidate.id in candidate_ids:
            raise HTTPException(400, f"duplicate candidate id {candidate.id!r}")
        candidate_ids.add(candidate.id)
        if not (1 <= len(candidate.cells) <= 8):
            raise HTTPException(400, f"{where}.cells must contain 1 to 8 cells")
        for cell_index, cell in enumerate(candidate.cells):
            cell_where = f"{where}.cells[{cell_index}]"
            if cell.layer not in allowed_layers:
                raise HTTPException(
                    400,
                    f"{cell_where}.layer L{cell.layer} is outside the measured "
                    f"workspace fitted layers {workspace_layers}",
                )
            if cell.position < 0:
                raise HTTPException(400, f"{cell_where}.position must be >= 0")
            if not math.isfinite(cell.alpha):
                raise HTTPException(400, f"{cell_where}.alpha must be finite")

    tok = _model.tokenizer
    # Reuse the production intervention validator and its exact token-id-first
    # resolution. The stand-in scope/layer is discarded after resolution.
    resolution_spec = InterventionSpec(
        mode="swap",
        layers=[workspace_layers[0]],
        token=req.token,
        token_id=req.token_id,
        target=req.target,
        target_id=req.target_id,
        alpha=1.0,
        positions=[0],
    )
    resolved = _validate_and_resolve_specs(
        [resolution_spec], _lens, tok, _model.n_layers
    )[0]
    try:
        prompt_ids = await asyncio.to_thread(
            _chat_template_token_ids,
            tok,
            req.messages,
            enable_thinking=req.enable_thinking,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if not prompt_ids:
        raise HTTPException(400, "chat template produced no input tokens")
    for candidate_index, candidate in enumerate(req.candidates):
        for cell_index, cell in enumerate(candidate.cells):
            if cell.position >= len(prompt_ids):
                raise HTTPException(
                    400,
                    f"candidates[{candidate_index}].cells[{cell_index}].position "
                    f"{cell.position} is outside the templated input "
                    f"(length {len(prompt_ids)})",
                )
    eos_ids = _planner_eos_ids(tok)

    def _compile_and_replay(candidate: InterventionSearchCandidate):
        from .interventions import compile_edits

        candidate_started = time.perf_counter()
        compile_started = time.perf_counter()
        edits = []
        # Compile every exact cell independently. Passing all layers and all
        # positions to one compile_edits call would create the Cartesian
        # product rather than the requested (layer, position) pairs.
        for cell in candidate.cells:
            cell_edits = compile_edits(
                _lens,
                _model,
                mode="swap",
                layers=[cell.layer],
                token_id=resolved["token_id"],
                target_id=resolved["target_id"],
                alpha=cell.alpha,
                positions=[cell.position],
                from_pos=None,
                label=(
                    f"search {candidate.id} L{cell.layer} "
                    f"p{cell.position} a={cell.alpha:g}"
                ),
            )
            edits.extend(cell_edits)
        compile_ms = 1000.0 * (time.perf_counter() - compile_started)
        replay = _greedy_intervention_replay(
            prompt_ids, edits, req.max_tokens, eos_ids
        )
        replay["timings_ms"]["compile"] = compile_ms
        replay["timings_ms"]["total"] = 1000.0 * (
            time.perf_counter() - candidate_started
        )
        return replay

    async def _locked_worker(fn, *args):
        queue_started = time.perf_counter()
        async with _gpu_lock:
            queue_wait_ms = 1000.0 * (
                time.perf_counter() - queue_started
            )
            result = await _planner_gpu_call(fn, *args)
        return result, queue_wait_ms

    async def gen():
        tested = 0
        verified_ids: list[str] = []
        stopped_for_budget = False
        try:
            yield _sse("search_start", {
                "token_id": resolved["token_id"],
                "token_str": resolved["token_str"],
                "target_id": resolved["target_id"],
                "target_str": resolved["target_str"],
                "source_text": req.source_text,
                "goal_text": req.goal_text,
                "workspace": {
                    "start_layer": workspace_start,
                    "end_layer": workspace_end,
                    "layers": workspace_layers,
                },
                "n_candidates": len(req.candidates),
                "max_tokens": req.max_tokens,
                "time_budget_seconds": req.time_budget_seconds,
                "enable_thinking": req.enable_thinking,
                "stop_on_success": req.stop_on_success,
                "input_tokens": len(prompt_ids),
            })

            baseline, baseline_queue_ms = await _locked_worker(
                _greedy_intervention_replay,
                prompt_ids,
                None,
                req.max_tokens,
                eos_ids,
            )
            baseline_text = tok.decode(
                baseline["ids"], skip_special_tokens=True
            )
            expected_replacement = _expected_single_source_replacement(
                baseline_text, req.source_text, req.goal_text
            )
            baseline_timings = dict(baseline["timings_ms"])
            baseline_timings["queue_wait"] = baseline_queue_ms
            yield _sse("baseline", {
                "text": baseline_text,
                "eos": baseline["eos"],
                "repetition": baseline["repetition"],
                "stop_reason": baseline["stop_reason"],
                "source_count": _bounded_phrase_count(
                    baseline_text, req.source_text
                ),
                "goal_count": _bounded_phrase_count(
                    baseline_text, req.goal_text
                ),
                "expected_replacement_text": expected_replacement,
                "timings_ms": baseline_timings,
            })

            for index, candidate in enumerate(req.candidates, start=1):
                elapsed = time.perf_counter() - request_started
                if elapsed >= req.time_budget_seconds:
                    stopped_for_budget = True
                    break

                replay, queue_wait_ms = await _locked_worker(
                    _compile_and_replay, candidate
                )
                tested += 1
                text = tok.decode(replay["ids"], skip_special_tokens=True)
                goal_count = _bounded_phrase_count(text, req.goal_text)
                source_count = _bounded_phrase_count(text, req.source_text)
                verified = bool(
                    replay["eos"]
                    and not replay["repetition"]
                    and goal_count == 1
                    and source_count == 0
                )
                if verified:
                    classification = "verified"
                    verified_ids.append(candidate.id)
                elif replay["repetition"]:
                    classification = "repetition"
                elif (_normalized_probe_text(text)
                      == _normalized_probe_text(baseline_text)):
                    classification = "unchanged"
                elif goal_count >= 1 and source_count == 0:
                    classification = "partial"
                else:
                    classification = "off_target"

                exact_match = (
                    None if expected_replacement is None
                    else _normalized_probe_text(text) == expected_replacement
                )
                elapsed = time.perf_counter() - request_started
                timings = dict(replay["timings_ms"])
                timings["queue_wait"] = queue_wait_ms
                yield _sse("candidate", {
                    "id": candidate.id,
                    "index": index,
                    "total": len(req.candidates),
                    "cells": [cell.model_dump() for cell in candidate.cells],
                    "text": text,
                    "eos": replay["eos"],
                    "repetition": replay["repetition"],
                    "stop_reason": replay["stop_reason"],
                    "class": classification,
                    "verified": verified,
                    "goal_count": goal_count,
                    "source_count": source_count,
                    "exact_replacement_applicable": (
                        expected_replacement is not None
                    ),
                    "exact_replacement_match": exact_match,
                    "timings_ms": timings,
                    "elapsed_seconds": elapsed,
                    "budget_remaining_seconds": max(
                        0.0, req.time_budget_seconds - elapsed
                    ),
                })
                if verified and req.stop_on_success:
                    break
                await asyncio.sleep(0)

            elapsed = time.perf_counter() - request_started
            if (tested < len(req.candidates)
                    and not (verified_ids and req.stop_on_success)
                    and elapsed >= req.time_budget_seconds):
                stopped_for_budget = True
            if verified_ids:
                status = "success"
                stop_reason = (
                    "verified" if req.stop_on_success else "exhausted_candidates"
                )
            elif stopped_for_budget:
                status = "budget_exhausted"
                stop_reason = "time_budget"
            else:
                status = "no_verified_recipe"
                stop_reason = "exhausted_candidates"
            yield _sse("search_end", {
                "status": status,
                "stop_reason": stop_reason,
                "tested": tested,
                "total": len(req.candidates),
                "verified_candidate_id": (
                    verified_ids[0] if verified_ids else None
                ),
                "verified_candidate_ids": verified_ids,
                "elapsed_seconds": elapsed,
                "time_budget_seconds": req.time_budget_seconds,
                "budget_exhausted": stopped_for_budget,
                "budget_overshoot_seconds": max(
                    0.0, elapsed - req.time_budget_seconds
                ),
                "stopped_early": tested < len(req.candidates),
            })
        except Exception as error:
            print(
                f"[intervention_search] error: {error}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            elapsed = time.perf_counter() - request_started
            yield _sse("error", {"error": str(error)})
            yield _sse("search_end", {
                "status": "error",
                "stop_reason": "error",
                "tested": tested,
                "total": len(req.candidates),
                "verified_candidate_id": (
                    verified_ids[0] if verified_ids else None
                ),
                "verified_candidate_ids": verified_ids,
                "elapsed_seconds": elapsed,
                "time_budget_seconds": req.time_budget_seconds,
                "budget_exhausted": False,
                "budget_overshoot_seconds": max(
                    0.0, elapsed - req.time_budget_seconds
                ),
                "stopped_early": tested < len(req.candidates),
            })

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/intervention_scan")
async def intervention_scan(req: InterventionScanRequest):
    _require_active_mode()
    if _model is None:
        raise HTTPException(503, "model not loaded")
    if _lens is None:
        raise HTTPException(400, "the intervention scan requires a loaded lens")
    if not req.messages:
        raise HTTPException(400, "messages is empty")
    if req.mode not in ("swap", "steer"):
        raise HTTPException(400, f"scan supports swap/steer, not {req.mode!r}")
    tok = _model.tokenizer
    fitted = set(_lens.source_layers)

    if req.positions is not None and req.from_position is not None:
        raise HTTPException(
            400, "exactly one of positions / from_position may be supplied")
    # Backward compatibility: scans that omit scope retain the historic
    # from-position-zero behavior. Once positions are supplied they are the
    # only scope passed through to compile_edits().
    scan_positions = req.positions
    scan_from_position = req.from_position
    if scan_positions is None and scan_from_position is None:
        scan_from_position = 0

    # Reuse the spec validator for concept-token resolution (mode checks,
    # text→first-token, swap-needs-target); the probe layer is a stand-in.
    probe_spec = InterventionSpec(
        mode=req.mode, layers=[sorted(fitted)[0]],
        token=req.token, token_id=req.token_id,
        target=req.target, target_id=req.target_id,
        alpha=1.0, positions=scan_positions,
        from_position=scan_from_position,
    )
    resolved = _validate_and_resolve_specs(
        [probe_spec], _lens, tok, _model.n_layers)[0]

    layers = sorted(set(req.layers)) if req.layers else _scan_default_layers(fitted)
    bad = [l for l in layers if l not in fitted and l != _model.n_layers - 1]
    if bad:
        raise HTTPException(400, f"layers {bad} have no fitted Jacobian")
    alphas = req.alphas or ([0.5, 1, 2, 4, 8] if req.mode == "swap"
                            else [50, 100, 200, 400, 800])
    if not (1 <= req.max_tokens <= 16):
        raise HTTPException(400, "max_tokens must be in [1, 16]")
    if len(layers) * len(alphas) > 80:
        raise HTTPException(400, f"scan too large: {len(layers)}×{len(alphas)} probes (cap 80)")

    pm = [{"role": m.role, "content": m.content} for m in req.messages]
    pids = tok.apply_chat_template(pm, add_generation_prompt=True,
                                   enable_thinking=req.enable_thinking)
    pids = pids if isinstance(pids, list) else list(pids)

    eos_id = tok.eos_token_id

    def _probe(edits) -> tuple[list[int], bool, str]:
        session = _model.make_stream()
        if edits:
            session.set_edits(edits)
        logits, _ = session.extend(pids)
        out: list[int] = []
        for _ in range(req.max_tokens):
            t = int(mx.argmax(logits.astype(mx.float32)).tolist())
            if t == eos_id:
                return out, True, "eos"
            out.append(t)
            logits, _ = session.extend([t])
        return out, False, "max_tokens"

    async def gen():
        from .interventions import compile_edits
        try:
            yield _sse("scan_start", {
                "mode": req.mode, "layers": layers, "alphas": alphas,
                "token_id": resolved["token_id"],
                "token_str": resolved["token_str"],
                "target_id": resolved["target_id"],
                "target_str": resolved["target_str"],
                "goal_text": req.goal_text,
                "source_text": req.source_text,
                "positions": resolved["positions"],
                "from_position": resolved["from_position"],
                "n_probes": len(layers) * len(alphas),
            })
            async with _gpu_lock:
                base_ids, base_eos, base_stop = await asyncio.to_thread(_probe, None)
            base_text = tok.decode(base_ids, skip_special_tokens=True)
            yield _sse("probe", {
                "kind": "baseline", "text": base_text,
                "eos": base_eos, "stop_reason": base_stop,
            })
            k, total = 0, len(layers) * len(alphas)
            for a in alphas:
                for l in layers:
                    async with _gpu_lock:
                        edits = await asyncio.to_thread(
                            lambda: compile_edits(
                                _lens, _model, mode=req.mode, layers=[l],
                                token_id=resolved["token_id"],
                                target_id=resolved["target_id"],
                                alpha=a, positions=resolved["positions"],
                                from_pos=resolved["from_position"],
                                label=f"scan L{l} a={a:g}"))
                        out_ids, eos, stop_reason = await asyncio.to_thread(
                            _probe, edits)
                    text = tok.decode(out_ids, skip_special_tokens=True)
                    cls = _classify_probe(
                        out_ids, text, base_text,
                        resolved["target_str"] or resolved["token_str"],
                        eos=eos,
                        goal_text=req.goal_text,
                        source_text=req.source_text or req.token
                        or resolved["token_str"],
                    )
                    k += 1
                    yield _sse("probe", {
                        "kind": "probe", "layer": l, "alpha": a,
                        "text": text,
                        "cls": cls, "success": cls == "success",
                        "eos": eos, "stop_reason": stop_reason,
                        "index": k, "total": total,
                    })
                    await asyncio.sleep(0)
            yield _sse("scan_end", {})
        except Exception as e:
            import traceback
            print(f"[intervention_scan] error: {e}\n{traceback.format_exc()}", flush=True)
            yield _sse("error", {"error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent.parent / "web" / "index.html"
    if html_path.exists():
        html = html_path.read_text()
        # Deployment mode is injected before any style/script runs so the
        # presentation UI never flashes active-mode controls.
        if _app_mode != "active":
            html = html.replace(
                'window.JLENS_MODE = "active"',
                f'window.JLENS_MODE = "{_app_mode}"',
                1,
            )
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
    return HTMLResponse("<h1>J-Space Visualizer</h1><p>web/index.html not found</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
