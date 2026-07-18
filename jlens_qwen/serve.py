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
from collections import Counter, deque
from difflib import SequenceMatcher
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

# Cooperative control for adaptive intervention searches.  These entries live
# only for the lifetime of their SSE response: pausing deliberately keeps that
# response open, so the controller can resume the exact in-memory scheduler
# without serialising queues or replaying candidates.
_adaptive_search_controls: dict[str, "_AdaptiveSearchControlState"] = {}

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
    interventionRecipes: list[dict[str, Any]] = []
    selectedInterventionRecipeId: str | None = None
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


class AdaptiveSearchHit(BaseModel):
    """A source-token readout occurrence in one workspace cell."""

    layer: int
    rank: int


class AdaptivePositionEvidence(BaseModel):
    """Compact baseline-grid evidence for one exact prefill position.

    The client already owns these readouts.  Sending only source-token hits
    lets the search rank positions without repeating the expensive full-grid
    lens readout on the server.
    """

    position: int
    msg_idx: int | None = None
    role: str = "user"
    token_text: str = ""
    source_hits: list[AdaptiveSearchHit] = []


class AdaptivePromisingSingle(BaseModel):
    """One improving single-cell result carried into a deeper search.

    The endpoint validates that ``cells`` contains exactly one measured
    workspace cell and that its recipe key is also excluded from replay.
    ``similarity_to_desired`` is the score from the standard search's full
    deterministic replay; it is used only to rank/seed combinations.
    """

    cells: list[InterventionSearchCell]
    similarity_to_desired: float


class AdaptiveInterventionSearchRequest(BaseModel):
    """Ask the server to discover and verify an exact workspace recipe."""

    messages: list[ChatMessage]
    token: str | None = None
    token_id: int | None = None
    target: str | None = None
    target_id: int | None = None
    # User-visible replacement. This is intentionally distinct from `target`,
    # whose leading whitespace may encode the source token's BPE boundary.
    replacement_text: str
    baseline_response: str
    # Unicode code-point offsets into baseline_response (Python string
    # indexing), not UTF-8 bytes or JavaScript UTF-16 code units.
    selected_start: int
    selected_end: int
    position_evidence: list[AdaptivePositionEvidence] = []
    # Recipe keys returned by an earlier standard search. Thorough/deeper
    # search passes these back so its extra two minutes do not repeat work.
    exclude_recipe_keys: list[str] = []
    # At most sixteen improving single-cell results from the standard search.
    # They seed the deeper combination beam but are never accepted as verified
    # without a new full replay of the resulting pair/triple recipe.
    prior_promising: list[AdaptivePromisingSingle] = []
    profile: str = "standard"       # standard (<=60s) | thorough (<=180s)
    max_tokens: int = 64
    time_budget_seconds: float = 60.0
    enable_thinking: bool = False
    # Opt in to one logical, cooperatively pausable search.  A standard search
    # pauses when its initial budget/queue ends; the control endpoint may then
    # add exactly 120 active seconds and resume the same scheduler in thorough
    # mode.  The SSE connection must remain open while paused.
    allow_continuation: bool = False
    # Opt in to the latent-premise stage: when the literal-direction singles
    # run dry (or the search is extended/thorough), the resident model
    # proposes a premise swap (e.g. France→China) and band-clamped premise
    # recipes are replay-tested. Redirects are reported as premise results
    # with their larger causal footprint, never as exact verified recipes.
    enable_premise_search: bool = False
    # End the search at the first verified recipe (evaluation harnesses).
    # By default the search exhausts its time budget and reports every
    # verified recipe it finds; a verified recipe also suppresses the
    # premise proposal, which exists for when the literal direction fails.
    stop_on_verified: bool = False


class AdaptiveInterventionSearchControlRequest(BaseModel):
    search_id: str
    action: str  # pause | resume | extend
    additional_time_seconds: float | None = None


class _AdaptiveSearchControlState:
    """Live cooperative clock and control plane for one SSE search.

    Paused wall time never consumes the active search budget.  The generator,
    rather than the control request, acknowledges a pause at a safe boundary;
    this guarantees an already-started MLX replay remains atomic.
    """

    def __init__(
        self,
        search_id: str,
        *,
        started: float,
        initial_budget_seconds: float,
        can_extend: bool,
    ):
        self.search_id = search_id
        self.started = started
        self.initial_budget_seconds = initial_budget_seconds
        self.time_budget_seconds = initial_budget_seconds
        self.can_extend = can_extend
        self.extended = False
        self.awaiting_extension = False
        self.closed = False
        self.pause_reason: str | None = None
        self.resume_reason: str | None = None
        self.pause_requested = asyncio.Event()
        self.run_event = asyncio.Event()
        self.run_event.set()
        self.paused_at: float | None = None
        self.paused_total_seconds = 0.0
        self.charged_active_seconds = 0.0

    def elapsed_active(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        effective_now = self.paused_at if self.paused_at is not None else now
        return max(
            0.0,
            effective_now
            - self.started
            - self.paused_total_seconds
            + self.charged_active_seconds,
        )

    def remaining_active(self) -> float:
        return max(0.0, self.time_budget_seconds - self.elapsed_active())

    def request_pause(self) -> None:
        if self.paused_at is None:
            self.pause_requested.set()

    def cancel_or_resume(self) -> None:
        # A resume arriving before the worker acknowledges the request simply
        # cancels that pending pause.  Otherwise it wakes the paused stream.
        self.pause_requested.clear()
        self.resume_reason = "user"
        self.run_event.set()

    def acknowledge_pause(
        self,
        reason: str,
        *,
        awaiting_extension: bool = False,
        charge_to_budget: bool = False,
    ) -> None:
        now = time.perf_counter()
        if self.paused_at is None:
            if charge_to_budget:
                self.charged_active_seconds += max(
                    0.0, self.time_budget_seconds - self.elapsed_active(now)
                )
            self.paused_at = now
        self.pause_reason = reason
        self.awaiting_extension = awaiting_extension
        self.pause_requested.clear()
        self.run_event.clear()

    def extend(self, seconds: float) -> None:
        self.time_budget_seconds += seconds
        self.extended = True
        self.awaiting_extension = False
        self.resume_reason = "extended"
        self.run_event.set()

    def finish_resume(self) -> None:
        if self.paused_at is not None:
            self.paused_total_seconds += time.perf_counter() - self.paused_at
            self.paused_at = None
        self.pause_reason = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "paused": self.paused_at is not None,
            "pause_requested": self.pause_requested.is_set(),
            "pause_reason": self.pause_reason,
            "awaiting_extension": self.awaiting_extension,
            "can_extend": self.can_extend and not self.extended,
            "extended": self.extended,
            "elapsed_active_seconds": self.elapsed_active(),
            "paused_total_seconds": self.paused_total_seconds,
            "time_budget_seconds": self.time_budget_seconds,
            "budget_remaining_seconds": self.remaining_active(),
            "closed": self.closed,
        }


class _AdaptiveSearchDeadlineExpired(Exception):
    """The search deadline elapsed before queued GPU work could begin."""

    def __init__(self, queue_wait_ms: float):
        super().__init__("adaptive search deadline expired while waiting for GPU")
        self.queue_wait_ms = queue_wait_ms


class _AdaptiveSearchPauseRequested(Exception):
    """A queued GPU call was cooperatively paused before it began."""

    def __init__(self, queue_wait_ms: float):
        super().__init__("adaptive search paused while waiting for GPU")
        self.queue_wait_ms = queue_wait_ms


def _adaptive_longest_layer_run(
    hit_layers: set[int], workspace_layers: list[int]
) -> int:
    """Longest consecutive run in fitted-layer order, not raw layer IDs."""
    longest = current = 0
    for layer in workspace_layers:
        if layer in hit_layers:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _adaptive_rank_positions(
    evidence: list[AdaptivePositionEvidence],
    *,
    frontier: int,
    workspace_layers: list[int],
    selected_source: str,
) -> list[dict[str, Any]]:
    """Rank prefill positions from observed J-space coherence.

    A coherent source readout is the strongest signal.  A weaker source hit
    or literal source token in the latest user context follows, then the
    causal frontier, remaining user rows, and older context.  All tie breaks
    are explicit so the same grid always produces the same search order.
    """
    allowed = set(workspace_layers)
    by_position: dict[int, dict[str, Any]] = {}
    for row in evidence:
        merged = by_position.setdefault(row.position, {
            "position": row.position,
            "msg_idx": row.msg_idx,
            "role": row.role,
            "token_text": row.token_text,
            "hit_ranks": {},
        })
        # Prefer the newest/most specific metadata if duplicate rows arrive,
        # while merging duplicate cell evidence by its best observed rank.
        if row.msg_idx is not None:
            merged["msg_idx"] = row.msg_idx
        if row.role:
            merged["role"] = row.role
        if row.token_text:
            merged["token_text"] = row.token_text
        for hit in row.source_hits:
            if hit.layer not in allowed:
                continue
            previous = merged["hit_ranks"].get(hit.layer)
            if previous is None or hit.rank < previous:
                merged["hit_ranks"][hit.layer] = hit.rank

    by_position.setdefault(frontier, {
        "position": frontier,
        "msg_idx": None,
        "role": "frontier",
        "token_text": "",
        "hit_ranks": {},
    })
    source_key = _normalized_probe_text(selected_source)
    ranked: list[dict[str, Any]] = []
    for row in by_position.values():
        hit_ranks = row["hit_ranks"]
        hit_layers = set(hit_ranks)
        longest_run = _adaptive_longest_layer_run(
            hit_layers, workspace_layers
        )
        top1_count = sum(rank == 1 for rank in hit_ranks.values())
        reciprocal_rank = sum(1.0 / rank for rank in hit_ranks.values())
        literal_source = bool(
            source_key
            and _normalized_probe_text(row["token_text"]) == source_key
        )
        coherent = longest_run >= 2 or top1_count >= 2
        role = row["role"]
        if coherent:
            tier, reason = 0, "coherent_source_readout"
        elif hit_layers or literal_source:
            tier, reason = 1, (
                "source_readout" if hit_layers else "literal_source_token"
            )
        elif row["position"] == frontier:
            tier, reason = 2, "causal_frontier"
        elif (role == "template"
              and row["position"] >= max(0, frontier - 2)):
            # The final assistant-prefix/control tokens are real prefill
            # coordinates even though they do not belong to a chat message.
            # They often form the last causal bridge into the first reply
            # token, so keep them ahead of the broad user-context sweep.
            tier, reason = 2, "causal_template_prefix"
        elif role == "user":
            tier, reason = 3, "recent_user_context"
        else:
            tier, reason = 4, "older_context"
        ranked.append({
            "position": row["position"],
            "msg_idx": row["msg_idx"],
            "role": role,
            "token_text": row["token_text"],
            "reason": reason,
            "tier": tier,
            "hit_layers": sorted(
                hit_layers,
                key=lambda layer: (hit_ranks[layer], -layer),
            ),
            "hit_ranks": {
                str(layer): hit_ranks[layer] for layer in sorted(hit_ranks)
            },
            "longest_layer_run": longest_run,
            "top1_count": top1_count,
            "hit_count": len(hit_layers),
            "reciprocal_rank_score": reciprocal_rank,
            "literal_source": literal_source,
        })
    ranked.sort(key=lambda row: (
        row["tier"],
        -row["longest_layer_run"],
        -row["top1_count"],
        -row["hit_count"],
        -row["reciprocal_rank_score"],
        -(row["msg_idx"] if row["msg_idx"] is not None else -1),
        -row["position"],
    ))
    return ranked


def _adaptive_coarse_layers(workspace_layers: list[int], count: int = 7) -> list[int]:
    """Evenly spread fitted workspace layers, searched late-first."""
    if len(workspace_layers) <= count:
        return list(reversed(workspace_layers))
    picked: list[int] = []
    for index in range(count):
        at = round(index * (len(workspace_layers) - 1) / (count - 1))
        layer = workspace_layers[at]
        if layer not in picked:
            picked.append(layer)
    return list(reversed(picked))


def _adaptive_recipe_key(cells: list[dict[str, Any]]) -> tuple:
    return tuple(sorted(
        (int(cell["position"]), int(cell["layer"]), float(cell["alpha"]))
        for cell in cells
    ))


def _adaptive_recipe_string(cells: list[dict[str, Any]]) -> str:
    """Stable wire key used to resume a deeper search without repetition."""
    return "|".join(
        f"p{position}:l{layer}:a{alpha:g}"
        for position, layer, alpha in _adaptive_recipe_key(cells)
    )


def _adaptive_candidate(
    stage: str,
    cells: list[dict[str, Any]],
    *,
    parents: list[str] | None = None,
) -> dict[str, Any]:
    bits = "+".join(
        f"p{cell['position']}l{cell['layer']}a{cell['alpha']:g}"
        for cell in sorted(
            cells, key=lambda value: (value["position"], value["layer"])
        )
    )
    return {
        "id": f"{stage}-{bits}",
        "stage": stage,
        "cells": cells,
        "parents": list(parents or []),
    }


def _adaptive_initial_candidates(
    ranked_positions: list[dict[str, Any]],
    workspace_layers: list[int],
) -> list[dict[str, Any]]:
    """Round-robin source-evidence and coarse singles across positions."""
    coarse = _adaptive_coarse_layers(workspace_layers)
    layer_orders: list[tuple[int, list[int]]] = []
    for row in ranked_positions:
        layers: list[int] = []
        for layer in row["hit_layers"][:4] + coarse:
            if layer not in layers:
                layers.append(layer)
        layer_orders.append((row["position"], layers))
    candidates: list[dict[str, Any]] = []
    max_lanes = max((len(layers) for _, layers in layer_orders), default=0)
    for lane in range(max_lanes):
        for position, layers in layer_orders:
            if lane >= len(layers):
                continue
            layer = layers[lane]
            stage = "evidence_single" if lane < min(4, len(
                next(row for row in ranked_positions
                     if row["position"] == position)["hit_layers"]
            )) else "coarse_single"
            candidates.append(_adaptive_candidate(stage, [{
                "position": position, "layer": layer, "alpha": 1.0,
            }]))
    # Evidence/coarse overlap can produce the same recipe at different lanes.
    unique: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for candidate in candidates:
        key = _adaptive_recipe_key(candidate["cells"])
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _adaptive_deep_single_candidates(
    ranked_positions: list[dict[str, Any]],
    workspace_layers: list[int],
) -> list[dict[str, Any]]:
    """Broaden an extended search beyond the coarse one-minute sweep.

    The standard phase samples seven layers and refines only cells that move
    the answer.  If every coarse probe is unchanged, there would otherwise be
    nothing for the added two minutes to do.  The extension therefore retains
    the same scheduler and adds a deterministic full workspace sweep at both
    normal and half strength; submitted recipes are still removed by `_enqueue`.
    """
    candidates: list[dict[str, Any]] = []
    positions = [row["position"] for row in ranked_positions]
    for alpha in (1.0, 0.5):
        for layer in reversed(workspace_layers):
            for position in positions:
                candidates.append(_adaptive_candidate("deep_single", [{
                    "position": position,
                    "layer": layer,
                    "alpha": alpha,
                }]))
    return candidates


def _adaptive_refinement_candidates(
    parent: dict[str, Any], workspace_layers: list[int]
) -> list[dict[str, Any]]:
    """Refine a safe changed single around its fitted layer and strength."""
    if len(parent["cells"]) != 1:
        return []
    cell = parent["cells"][0]
    try:
        center = workspace_layers.index(cell["layer"])
    except ValueError:
        return []
    cells = [{
        "position": cell["position"],
        "layer": cell["layer"],
        "alpha": 0.5,
    }]
    for delta in (-1, 1, -2, 2):
        index = center + delta
        if 0 <= index < len(workspace_layers):
            cells.append({
                "position": cell["position"],
                "layer": workspace_layers[index],
                "alpha": 1.0,
            })
    return [
        _adaptive_candidate("refinement_single", [value], parents=[parent["id"]])
        for value in cells
    ]


def _adaptive_pair_candidates(
    newest: dict[str, Any], promising: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pair adjacent-layer or same-layer improving singles only."""
    if len(newest["cells"]) != 1:
        return []
    left = newest["cells"][0]
    out: list[dict[str, Any]] = []
    for other in promising:
        if other["id"] == newest["id"] or len(other["cells"]) != 1:
            continue
        right = other["cells"][0]
        adjacent_layers = (
            left["position"] == right["position"]
            and 0 < abs(left["layer"] - right["layer"]) <= 2
        )
        same_layer_positions = (
            left["layer"] == right["layer"]
            and left["position"] != right["position"]
        )
        if not (adjacent_layers or same_layer_positions):
            continue
        cells = [
            {"position": value["position"], "layer": value["layer"], "alpha": 0.5}
            for value in (left, right)
        ]
        out.append(_adaptive_candidate(
            "pair", cells, parents=[newest["id"], other["id"]]
        ))
    return out


def _adaptive_triple_candidates(
    pair: dict[str, Any], promising: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extend an improving pair by one related improving single."""
    if len(pair["cells"]) != 2:
        return []
    occupied = {(cell["position"], cell["layer"]) for cell in pair["cells"]}
    out: list[dict[str, Any]] = []
    for single in promising:
        if len(single["cells"]) != 1:
            continue
        cell = single["cells"][0]
        coordinate = (cell["position"], cell["layer"])
        if coordinate in occupied:
            continue
        related = any(
            (cell["position"] == member["position"]
             and abs(cell["layer"] - member["layer"]) <= 2)
            or (cell["layer"] == member["layer"]
                and cell["position"] != member["position"])
            for member in pair["cells"]
        )
        if not related:
            continue
        cells = [dict(member) for member in pair["cells"]] + [{
            "position": cell["position"],
            "layer": cell["layer"],
            "alpha": 0.5,
        }]
        out.append(_adaptive_candidate(
            "triple", cells, parents=[pair["id"], single["id"]]
        ))
    return out


# ----- Latent-premise stage -------------------------------------------------
#
# When every literal-direction single leaves the reply unchanged, the reply
# token is usually the CONCLUSION of a computation whose workspace variable
# is an earlier PREMISE (paper example: edit France→China, and capital,
# language, and continent circuits recompute for China). The premise stage
# asks the resident model for that latent direction, then replay-tests it as
# a band-clamped swap. Its successes are reported as premise redirects with
# their larger causal footprint — never as exact verified recipes, because a
# coherent model that now believes the new premise also verbalizes it.

_PREMISE_PROPOSAL_BRIEF = """\
You analyze one finished chat exchange for J-Space intervention planning.
The user selected a span of the assistant reply and requested a replacement.
Do not restate that reply edit. Name the earlier semantic premise whose
counterfactual change would make the requested reply the natural answer:
when the requested span belongs to a different parent entity than the
current span (a capital to its country, an author to their book), the
premise is that parent entity.

Return JSON only, no Markdown. The top-level object has a "premises" array
with at most two objects, each with short string fields "source_concept"
(the premise as currently stated, at most three words), "target_concept"
(the counterfactual premise, at most three words), "reason" (at most eight
words), and "confidence" of "low", "medium", or "high". Concepts are words
from or implied by the conversation, never layer or position numbers. If no
responsible premise exists, return {"premises": []}. Keep the entire output
under 90 tokens."""


def _premise_proposal_messages(
    messages: list["ChatMessage"],
    selected_source: str,
    replacement_text: str,
) -> list["ChatMessage"]:
    transcript = "\n".join(
        f"{message.role}: {message.content}" for message in messages
    )
    task = (
        f"{_PREMISE_PROPOSAL_BRIEF}\n\n"
        f"Conversation:\n{transcript}\n\n"
        f"Selected assistant span: {selected_source!r}\n"
        f"Requested replacement: {replacement_text!r}\n"
        "Propose the premises now."
    )
    return [ChatMessage(role="user", content=task)]


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object in free-form text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find("{", start + 1)
    return None


def _parse_premise_proposals(text: str) -> list[dict[str, str]]:
    """Extract at most two premise directions from planner output.

    Tolerates fenced or chatty output around the JSON, the recipe brief's
    "candidates" key, and non-enum confidence values; drops rows with
    missing, identical, or over-long concepts.
    """
    blob = _extract_json_object(text or "")
    if blob is None:
        return []
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    rows = parsed.get("premises")
    if not isinstance(rows, list):
        rows = parsed.get("candidates")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source_concept") or "").strip()
        target = str(row.get("target_concept") or "").strip()
        if not source or not target:
            continue
        if len(source.split()) > 3 or len(target.split()) > 3:
            continue
        key = (source.casefold(), target.casefold())
        if key[0] == key[1] or key in seen:
            continue
        seen.add(key)
        confidence = str(row.get("confidence") or "").strip().lower()
        if confidence not in ("low", "medium", "high"):
            confidence = "unstated"
        out.append({
            "source_concept": source,
            "target_concept": target,
            "reason": str(row.get("reason") or "").strip()[:80],
            "confidence": confidence,
        })
        if len(out) == 2:
            break
    return out


def _premise_band_halves(layers: list[int]) -> list[list[int]]:
    """Bisect a passing clamp band to shrink its footprint; floor is four."""
    if len(layers) < 8:
        return []
    mid = len(layers) // 2
    return [layers[:mid], layers[mid:]]


def _premise_recipe_string(
    source_token_id: int, target_token_id: int, layers: list[int]
) -> str:
    return (
        f"premise:{source_token_id}->{target_token_id}:"
        f"L{layers[0]}-L{layers[-1]}:n{len(layers)}"
    )


def _premise_search_candidate(
    direction: dict[str, Any],
    layers: list[int],
    *,
    parents: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": (
            f"premise-{direction['source_token_id']}-"
            f"{direction['target_token_id']}-L{layers[0]}-{layers[-1]}"
        ),
        "stage": "premise_band",
        "kind": "premise",
        "cells": [],
        "premise": {
            "source_concept": direction["source_concept"],
            "target_concept": direction["target_concept"],
            "source_str": direction["source_str"],
            "target_str": direction["target_str"],
            "source_token_id": direction["source_token_id"],
            "target_token_id": direction["target_token_id"],
            "layers": [int(layer) for layer in layers],
            "from_position": 0,
        },
        "recipe_key": _premise_recipe_string(
            direction["source_token_id"],
            direction["target_token_id"],
            layers,
        ),
        "parents": list(parents or []),
    }


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


@app.post("/api/intervention_search_adaptive_control")
async def intervention_search_adaptive_control(
    req: AdaptiveInterventionSearchControlRequest,
):
    """Cooperatively pause/resume/extend a live adaptive-search stream.

    This is a control plane for the still-open SSE response.  It never
    cancels in-flight GPU work: ``pause`` is acknowledged by the search at
    the next safe boundary.  ``extend`` is accepted only after the initial
    standard phase has emitted ``search_paused`` and adds exactly two active
    minutes to that same scheduler.
    """
    _require_active_mode()
    control = _adaptive_search_controls.get(req.search_id)
    if control is None or control.closed:
        raise HTTPException(404, f"adaptive search {req.search_id!r} not found")
    if req.action == "pause":
        if control.awaiting_extension:
            raise HTTPException(
                409, "search is already paused awaiting a two-minute extension"
            )
        control.request_pause()
    elif req.action == "resume":
        if control.awaiting_extension:
            raise HTTPException(
                409, "initial search ended; use extend instead of resume"
            )
        control.cancel_or_resume()
    elif req.action == "extend":
        seconds = (
            120.0
            if req.additional_time_seconds is None
            else float(req.additional_time_seconds)
        )
        if not math.isfinite(seconds) or abs(seconds - 120.0) > 1e-9:
            raise HTTPException(
                400, "additional_time_seconds must be exactly 120"
            )
        if not control.can_extend or control.extended:
            raise HTTPException(409, "this search cannot be extended again")
        if not control.awaiting_extension or control.paused_at is None:
            raise HTTPException(
                409, "deeper search is available only after the initial phase ends"
            )
        control.extend(seconds)
    else:
        raise HTTPException(400, f"unknown action {req.action!r}")
    return {"action": req.action, **control.snapshot()}


@app.post("/api/intervention_search_adaptive")
async def intervention_search_adaptive(req: AdaptiveInterventionSearchRequest):
    """Discover and verify an exact workspace recipe under one deadline.

    Unlike :func:`intervention_search`, this production controller generates
    candidates lazily.  Baseline J-space evidence supplied by the UI ranks
    prefill positions; every proposed recipe is still judged by a fresh,
    full, deterministic, lens-free replay.
    """
    _require_active_mode()
    request_started = time.perf_counter()
    if _model is None:
        raise HTTPException(503, "model not loaded")
    if _lens is None:
        raise HTTPException(400, "adaptive search requires a loaded lens")
    if not req.messages:
        raise HTTPException(400, "messages is empty")
    bad_roles = sorted({
        message.role for message in req.messages
        if message.role not in ("system", "user", "assistant")
    })
    if bad_roles:
        raise HTTPException(400, f"unsupported message roles: {bad_roles}")
    if req.profile not in ("standard", "thorough"):
        raise HTTPException(400, "profile must be 'standard' or 'thorough'")
    profile_ceiling = 60.0 if req.profile == "standard" else 180.0
    if (not math.isfinite(req.time_budget_seconds)
            or req.time_budget_seconds < 0.001
            or req.time_budget_seconds > profile_ceiling):
        raise HTTPException(
            400,
            f"time_budget_seconds must be finite and in [0.001, "
            f"{profile_ceiling:g}] for the {req.profile} profile",
        )
    deadline = request_started + req.time_budget_seconds
    if not (1 <= req.max_tokens <= 128):
        raise HTTPException(400, "max_tokens must be in [1, 128]")
    if req.enable_thinking:
        raise HTTPException(400, "adaptive search requires enable_thinking=false")
    if not (0 <= req.selected_start < req.selected_end
            <= len(req.baseline_response)):
        raise HTTPException(
            400, "selected_start/selected_end are outside baseline_response"
        )
    if len(req.position_evidence) > 4096:
        raise HTTPException(400, "position_evidence is limited to 4096 rows")
    if not req.replacement_text:
        raise HTTPException(400, "replacement_text is empty")
    if len(req.exclude_recipe_keys) > 768:
        raise HTTPException(400, "exclude_recipe_keys is limited to 768 recipes")
    if any(not key.strip() for key in req.exclude_recipe_keys):
        raise HTTPException(400, "exclude_recipe_keys contains an empty key")
    if len(req.prior_promising) > 16:
        raise HTTPException(400, "prior_promising is limited to 16 singles")
    if req.prior_promising and req.profile != "thorough":
        raise HTTPException(
            400, "prior_promising is accepted only by the thorough profile"
        )

    workspace = next(
        (band for band in (_bands or []) if band.get("name") == "workspace"),
        None,
    )
    if workspace is None:
        raise HTTPException(
            400, "adaptive search requires a measured workspace band"
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

    tok = _model.tokenizer
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
            enable_thinking=False,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if not prompt_ids:
        raise HTTPException(400, "chat template produced no input tokens")
    frontier = len(prompt_ids) - 1

    excluded_recipe_keys = set(req.exclude_recipe_keys)
    prior_promising: list[dict[str, Any]] = []
    prior_recipe_keys: set[str] = set()
    for prior_index, prior in enumerate(req.prior_promising):
        where = f"prior_promising[{prior_index}]"
        if len(prior.cells) != 1:
            raise HTTPException(
                400, f"{where}.cells must contain exactly one cell"
            )
        score = float(prior.similarity_to_desired)
        if not math.isfinite(score) or not (0.0 <= score <= 1.0):
            raise HTTPException(
                400,
                f"{where}.similarity_to_desired must be finite and in [0, 1]",
            )
        supplied = prior.cells[0]
        if not (0 <= supplied.position <= frontier):
            raise HTTPException(
                400,
                f"{where}.cells[0].position {supplied.position} is outside "
                f"the prefill frontier {frontier}",
            )
        if supplied.layer not in allowed_layers:
            raise HTTPException(
                400, f"{where}.cells[0].layer is outside the measured workspace"
            )
        alpha = float(supplied.alpha)
        if not math.isfinite(alpha) or not (0.0 < alpha <= 1.0):
            raise HTTPException(
                400, f"{where}.cells[0].alpha must be finite and in (0, 1]"
            )
        cells = [{
            "position": int(supplied.position),
            "layer": int(supplied.layer),
            "alpha": alpha,
        }]
        recipe_key = _adaptive_recipe_string(cells)
        if recipe_key not in excluded_recipe_keys:
            raise HTTPException(
                400,
                f"{where} recipe must also appear in exclude_recipe_keys",
            )
        if recipe_key in prior_recipe_keys:
            raise HTTPException(
                400, f"{where} duplicates an earlier prior_promising recipe"
            )
        prior_recipe_keys.add(recipe_key)
        promising = _adaptive_candidate("prior_single", cells)
        promising["score"] = score
        prior_promising.append(promising)

    allowed_evidence_roles = {
        "system", "user", "assistant", "template", "frontier",
    }
    for row_index, row in enumerate(req.position_evidence):
        where = f"position_evidence[{row_index}]"
        if not (0 <= row.position <= frontier):
            raise HTTPException(
                400,
                f"{where}.position {row.position} is outside the prefill "
                f"frontier {frontier}",
            )
        if row.role not in allowed_evidence_roles:
            raise HTTPException(400, f"{where}.role is unsupported")
        if row.msg_idx is not None:
            if not (0 <= row.msg_idx < len(req.messages)):
                raise HTTPException(400, f"{where}.msg_idx is outside messages")
            expected_role = req.messages[row.msg_idx].role
            if row.role != expected_role:
                raise HTTPException(
                    400,
                    f"{where}.role {row.role!r} does not match "
                    f"messages[{row.msg_idx}].role {expected_role!r}",
                )
        if len(row.source_hits) > len(workspace_layers):
            raise HTTPException(
                400, f"{where}.source_hits exceeds the workspace layer count"
            )
        seen_hit_layers: set[int] = set()
        for hit_index, hit in enumerate(row.source_hits):
            hit_where = f"{where}.source_hits[{hit_index}]"
            if hit.layer not in allowed_layers:
                raise HTTPException(
                    400, f"{hit_where}.layer is outside the measured workspace"
                )
            if hit.layer in seen_hit_layers:
                raise HTTPException(400, f"{where} has duplicate source-hit layers")
            seen_hit_layers.add(hit.layer)
            if not (1 <= hit.rank <= 1000):
                raise HTTPException(400, f"{hit_where}.rank must be in [1, 1000]")

    selected_source = req.baseline_response[
        req.selected_start:req.selected_end
    ]
    if not selected_source:
        raise HTTPException(400, "selected response span is empty")
    if (_normalized_probe_text(req.replacement_text)
            == _normalized_probe_text(selected_source)):
        raise HTTPException(
            400,
            "replacement_text is equivalent to the selected response span",
        )
    desired_response = (
        req.baseline_response[:req.selected_start]
        + req.replacement_text
        + req.baseline_response[req.selected_end:]
    )
    all_ranked_positions = _adaptive_rank_positions(
        req.position_evidence,
        frontier=frontier,
        workspace_layers=workspace_layers,
        selected_source=selected_source,
    )
    position_limit = 8 if req.profile == "standard" else 16
    ranked_positions = all_ranked_positions[:position_limit]
    initial = deque(_adaptive_initial_candidates(
        ranked_positions, workspace_layers
    ))
    max_candidates = 384 if req.profile == "standard" else 768
    eos_ids = _planner_eos_ids(tok)

    control: _AdaptiveSearchControlState | None = None
    if req.allow_continuation:
        if req.profile != "standard":
            raise HTTPException(
                400, "allow_continuation requires the standard profile"
            )
        import uuid

        search_id = str(uuid.uuid4())
        control = _AdaptiveSearchControlState(
            search_id,
            started=request_started,
            initial_budget_seconds=req.time_budget_seconds,
            can_extend=True,
        )
        _adaptive_search_controls[search_id] = control

    def _elapsed_seconds() -> float:
        if control is not None:
            return control.elapsed_active()
        return time.perf_counter() - request_started

    def _time_budget_seconds() -> float:
        if control is not None:
            return control.time_budget_seconds
        return req.time_budget_seconds

    def _budget_remaining_seconds() -> float:
        return max(0.0, _time_budget_seconds() - _elapsed_seconds())

    def _compile_and_replay(candidate: dict[str, Any]):
        from .interventions import compile_edits

        candidate_started = time.perf_counter()
        compile_started = time.perf_counter()
        edits = []
        if candidate.get("kind") == "premise":
            premise = candidate["premise"]
            edits.extend(compile_edits(
                _lens,
                _model,
                mode="swap",
                layers=list(premise["layers"]),
                token_id=premise["source_token_id"],
                target_id=premise["target_token_id"],
                alpha=1.0,
                positions=None,
                from_pos=premise["from_position"],
                label=(
                    f"adaptive {candidate['id']} "
                    f"L{premise['layers'][0]}-{premise['layers'][-1]} clamp"
                ),
            ))
        else:
            for cell in candidate["cells"]:
                edits.extend(compile_edits(
                    _lens,
                    _model,
                    mode="swap",
                    layers=[cell["layer"]],
                    token_id=resolved["token_id"],
                    target_id=resolved["target_id"],
                    alpha=cell["alpha"],
                    positions=[cell["position"]],
                    from_pos=None,
                    label=(
                        f"adaptive {candidate['id']} L{cell['layer']} "
                        f"p{cell['position']} a={cell['alpha']:g}"
                    ),
                ))
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
        """Start GPU work only if this request acquires the lock in time.

        Once the worker has begun it is deliberately allowed to finish: MLX
        work cannot be cancelled safely. Queueing, however, is cancellable and
        belongs to the same global search budget as compilation and replay.
        """
        queue_started = time.perf_counter()
        remaining = (
            control.remaining_active()
            if control is not None
            else deadline - queue_started
        )
        if remaining <= 0:
            raise _AdaptiveSearchDeadlineExpired(0.0)
        acquired = False
        acquire_task: asyncio.Task | None = None
        pause_task: asyncio.Task | None = None
        try:
            if control is None:
                try:
                    await asyncio.wait_for(
                        _gpu_lock.acquire(), timeout=remaining
                    )
                    acquired = True
                except asyncio.TimeoutError as error:
                    queue_wait_ms = 1000.0 * (
                        time.perf_counter() - queue_started
                    )
                    raise _AdaptiveSearchDeadlineExpired(
                        queue_wait_ms
                    ) from error
            else:
                # A pause request can interrupt lock queueing, but never a
                # replay that has already acquired the GPU lock.
                if control.pause_requested.is_set():
                    raise _AdaptiveSearchPauseRequested(0.0)
                acquire_task = asyncio.create_task(_gpu_lock.acquire())
                pause_task = asyncio.create_task(
                    control.pause_requested.wait()
                )
                done, _pending = await asyncio.wait(
                    {acquire_task, pause_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                queue_wait_ms = 1000.0 * (
                    time.perf_counter() - queue_started
                )
                if not done:
                    raise _AdaptiveSearchDeadlineExpired(queue_wait_ms)
                if (pause_task in done
                        and control.pause_requested.is_set()):
                    # If both won the race, release the newly acquired lock;
                    # the requested pause takes precedence over new GPU work.
                    if acquire_task.done() and not acquire_task.cancelled():
                        acquired = bool(acquire_task.result())
                    raise _AdaptiveSearchPauseRequested(queue_wait_ms)
                acquired = bool(acquire_task.result())
            queue_wait_ms = 1000.0 * (time.perf_counter() - queue_started)
            if _budget_remaining_seconds() <= 0:
                raise _AdaptiveSearchDeadlineExpired(queue_wait_ms)
            result = await _planner_gpu_call(fn, *args)
            return result, queue_wait_ms
        finally:
            for task in (acquire_task, pause_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (acquire_task, pause_task):
                if task is not None and task.cancelled():
                    continue
                if task is not None and not task.done():
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            # Close the tiny race where acquire completed after pause won the
            # FIRST_COMPLETED set but before its task could be cancelled.
            if (not acquired
                    and acquire_task is not None
                    and acquire_task.done()
                    and not acquire_task.cancelled()):
                acquired = bool(acquire_task.result())
            if acquired:
                _gpu_lock.release()

    async def gen():
        tested = 0
        verified_candidates: list[dict[str, Any]] = []
        budget_exhausted = False
        deadline_queue_wait_ms = 0.0
        current_stage: str | None = None
        refinement_queue: deque[dict[str, Any]] = deque()
        pair_queue: deque[dict[str, Any]] = deque()
        triple_queue: deque[dict[str, Any]] = deque()
        promising_singles = [dict(candidate) for candidate in prior_promising]
        promising_singles.sort(
            key=lambda value: (-value["score"], value["id"])
        )
        scores_by_id = {
            candidate["id"]: candidate["score"]
            for candidate in promising_singles
        }
        promising_pairs: list[dict[str, Any]] = []
        submitted_keys: set[tuple] = set()
        excluded_keys = set(req.exclude_recipe_keys)
        candidate_durations: list[float] = []
        pair_count = triple_count = 0
        extension_seeded = False
        premise_queue: deque[dict[str, Any]] = deque()
        submitted_premise_keys: set[str] = set()
        # The proposal generation needs prefill plus ~90 output tokens
        # (~10-15 s measured in docs/perf/planner-01.md), so it only starts
        # with enough budget to also replay at least one band clamp after it.
        premise_min_budget_seconds = 25.0
        premise_max_replays = 12
        # Fire the proposal after the first evidence batch (~8 replays,
        # ~10 s) instead of only when the literal queue dries out: on hard
        # cases the premise hypothesis is then verified inside the first
        # minute, while an easy literal win still lands first.
        premise_early_after_tested = 8
        premise_state: dict[str, Any] = {
            "enabled": bool(req.enable_premise_search),
            "attempted": False,
            "proposal_status": None,
            "directions": [],
            "replays": 0,
            "redirects": 0,
            "best": None,
        }

        def _enqueue(
            queue: deque[dict[str, Any]], candidate: dict[str, Any]
        ) -> bool:
            key = _adaptive_recipe_key(candidate["cells"])
            wire_key = _adaptive_recipe_string(candidate["cells"])
            if key in submitted_keys or wire_key in excluded_keys:
                return False
            submitted_keys.add(key)
            queue.append(candidate)
            return True

        def _enqueue_premise(candidate: dict[str, Any]) -> bool:
            wire_key = candidate["recipe_key"]
            if (wire_key in submitted_premise_keys
                    or wire_key in excluded_keys):
                return False
            submitted_premise_keys.add(wire_key)
            premise_queue.append(candidate)
            return True

        def _queue_premise_proposal(origin: str) -> None:
            """One proposal per search, and only with budget to use it.

            A verified literal recipe also suppresses the proposal: the
            premise stage exists for when the literal direction fails.
            """
            if (not premise_state["enabled"]
                    or premise_state["attempted"]
                    or verified_candidates
                    or _budget_remaining_seconds()
                    < premise_min_budget_seconds):
                return
            premise_state["attempted"] = True
            premise_queue.append({
                "id": f"premise-proposal-{origin}",
                "stage": "premise_proposal",
                "kind": "premise_proposal",
                "origin": origin,
                "cells": [],
                "parents": [],
            })

        async def _pause_frames(
            reason: str,
            *,
            awaiting_extension: bool = False,
            charge_to_budget: bool = False,
        ):
            """Yield the acknowledged pause, then block until control resumes."""
            assert control is not None
            control.acknowledge_pause(
                reason,
                awaiting_extension=awaiting_extension,
                charge_to_budget=charge_to_budget,
            )
            yield _sse("search_paused", {
                **control.snapshot(),
                "reason": reason,
                "tested": tested,
                "max_candidates": max_candidates,
            })
            await control.run_event.wait()
            resume_reason = control.resume_reason or "user"
            added_seconds = (
                120.0 if resume_reason == "extended" else 0.0
            )
            control.finish_resume()
            yield _sse("search_resumed", {
                **control.snapshot(),
                "reason": resume_reason,
                "added_time_seconds": added_seconds,
                "tested": tested,
                "max_candidates": max_candidates,
            })

        def _seed_extension() -> None:
            """Broaden the retained standard scheduler to thorough mode."""
            nonlocal extension_seeded, max_candidates, position_limit
            nonlocal pair_count, triple_count
            if control is None or not control.extended or extension_seeded:
                return
            extension_seeded = True
            max_candidates = 768
            position_limit = 16
            # After a full minute of unchanged literal singles, one premise
            # proposal carries more expected information than the deep-single
            # flood below; premise_queue outranks `initial` in the scheduler.
            _queue_premise_proposal("extension")
            expanded = _adaptive_initial_candidates(
                all_ranked_positions[:position_limit], workspace_layers
            )
            for candidate in expanded:
                _enqueue(initial, candidate)
            for candidate in _adaptive_deep_single_candidates(
                all_ranked_positions[:position_limit], workspace_layers
            ):
                _enqueue(initial, candidate)
            for newest in promising_singles:
                for pair in _adaptive_pair_candidates(
                    newest, promising_singles
                ):
                    if pair_count >= 128:
                        break
                    if _enqueue(pair_queue, pair):
                        pair_count += 1
                if pair_count >= 128:
                    break
            for pair in promising_pairs:
                for triple in _adaptive_triple_candidates(
                    pair, promising_singles
                ):
                    if triple_count >= 64:
                        break
                    if _enqueue(triple_queue, triple):
                        triple_count += 1
                if triple_count >= 64:
                    break

        # Seed the dedupe set without changing the initial deterministic order.
        deduped_initial: deque[dict[str, Any]] = deque()
        while initial:
            candidate = initial.popleft()
            if _enqueue(deduped_initial, candidate):
                continue
        initial.extend(deduped_initial)

        # Reuse the standard search's improving singles as evidence, not as
        # claimed solutions: their exact recipes remain excluded above, while
        # newly formed combinations still require a fresh full replay.  An
        # improving pair may then seed the thorough-only triple beam below.
        if req.profile == "thorough":
            for newest in promising_singles:
                for pair in _adaptive_pair_candidates(
                    newest, promising_singles
                ):
                    if pair_count >= 128:
                        break
                    if _enqueue(pair_queue, pair):
                        pair_count += 1
                if pair_count >= 128:
                    break

        # Do not let a large coarse/refinement sweep consume the entire
        # deadline after the search has enough evidence to try a combination.
        # A pair gets one deterministic slot after at most two single-cell
        # probes; an improving pair's thorough-only triple gets the next
        # combination slot. Seeded prior singles therefore yield an early
        # pair in a deeper search instead of waiting behind fresh singles.
        singles_since_combination = 2 if pair_queue else 0

        try:
            yield _sse("search_start", {
                "profile": req.profile,
                "search_id": control.search_id if control else None,
                "allow_continuation": control is not None,
                "time_budget_seconds": _time_budget_seconds(),
                "extension_seconds": 120.0 if control else 0.0,
                "max_candidates": max_candidates,
                "max_tokens": req.max_tokens,
                "input_tokens": len(prompt_ids),
                "frontier_position": frontier,
                "token_id": resolved["token_id"],
                "token_str": resolved["token_str"],
                "target_id": resolved["target_id"],
                "target_str": resolved["target_str"],
                "replacement_text": req.replacement_text,
                "selected_source": selected_source,
                "desired_response": desired_response,
                "prior_promising_count": len(prior_promising),
                "workspace": {
                    "start_layer": workspace_start,
                    "end_layer": workspace_end,
                    "layers": workspace_layers,
                },
            })
            yield _sse("position_ranking", {
                "frontier_position": frontier,
                "limit": position_limit,
                "positions": ranked_positions,
            })

            while True:
                if control is not None and control.pause_requested.is_set():
                    async for frame in _pause_frames("user"):
                        yield frame
                try:
                    baseline, baseline_queue_ms = await _locked_worker(
                        _greedy_intervention_replay,
                        prompt_ids,
                        None,
                        req.max_tokens,
                        eos_ids,
                    )
                    break
                except _AdaptiveSearchPauseRequested:
                    assert control is not None
                    async for frame in _pause_frames("user"):
                        yield frame
                    continue
                except _AdaptiveSearchDeadlineExpired as expired:
                    if control is not None and not control.extended:
                        async for frame in _pause_frames(
                            "initial_budget",
                            awaiting_extension=True,
                            charge_to_budget=True,
                        ):
                            yield frame
                        _seed_extension()
                        yield _sse("position_ranking", {
                            "frontier_position": frontier,
                            "limit": position_limit,
                            "positions": all_ranked_positions[:position_limit],
                            "extended": True,
                        })
                        continue
                    deadline_queue_wait_ms = expired.queue_wait_ms
                    elapsed = _elapsed_seconds()
                    yield _sse("search_end", {
                        "status": "budget_exhausted",
                        "stop_reason": "time_budget",
                        "tested": 0,
                        "max_candidates": max_candidates,
                        "verified_candidate_id": None,
                        "verified_cells": None,
                        "desired_response": desired_response,
                        "elapsed_seconds": elapsed,
                        "time_budget_seconds": _time_budget_seconds(),
                        "budget_exhausted": True,
                        "budget_overshoot_seconds": max(
                            0.0, elapsed - _time_budget_seconds()
                        ),
                        "deadline_queue_wait_ms": deadline_queue_wait_ms,
                    })
                    return
            # The chat UI cumulatively decodes generated ids with the
            # tokenizer's default behavior. Use the same representation here
            # so equality means byte-for-byte equality with what was shown.
            baseline_text = tok.decode(baseline["ids"])
            baseline_matches = bool(
                baseline["eos"]
                and not baseline["repetition"]
                and baseline_text == req.baseline_response
            )
            baseline_timings = dict(baseline["timings_ms"])
            baseline_timings["queue_wait"] = baseline_queue_ms
            baseline_timings["total_with_queue"] = (
                baseline_timings["total"] + baseline_queue_ms
            )
            candidate_durations.append(
                baseline_timings["total_with_queue"] / 1000.0
            )
            yield _sse("baseline", {
                "text": baseline_text,
                "displayed_text": req.baseline_response,
                "matches_displayed": baseline_matches,
                "eos": baseline["eos"],
                "repetition": baseline["repetition"],
                "stop_reason": baseline["stop_reason"],
                "timings_ms": baseline_timings,
            })
            if not baseline_matches:
                elapsed = _elapsed_seconds()
                yield _sse("search_end", {
                    "status": "baseline_mismatch",
                    "stop_reason": "baseline_mismatch",
                    "tested": 0,
                    "verified_candidate_id": None,
                    "verified_cells": None,
                    "elapsed_seconds": elapsed,
                    "time_budget_seconds": _time_budget_seconds(),
                    "budget_exhausted": False,
                    "budget_overshoot_seconds": max(
                        0.0, elapsed - _time_budget_seconds()
                    ),
                })
                return

            desired_norm = _normalized_probe_text(desired_response)
            baseline_norm = _normalized_probe_text(baseline_text)
            baseline_similarity = SequenceMatcher(
                None, baseline_norm, desired_norm
            ).ratio()

            # A direct three-minute search has room for the premise stage up
            # front; the standard minute earns it only when its literal
            # singles run dry (see the scheduler's exhaustion branch).
            if req.profile == "thorough":
                _queue_premise_proposal("thorough")

            # The inner loop retains every queue and score across pauses.  The
            # outer loop exists solely to broaden this same scheduler after
            # the standard phase receives its one allowed extension.
            while True:
                phase_exhausted = False
                while tested < max_candidates:
                    if (control is not None
                            and control.pause_requested.is_set()):
                        async for frame in _pause_frames("user"):
                            yield frame

                    thorough_mode = bool(
                        req.profile == "thorough"
                        or (control is not None and control.extended)
                    )
                    singles_available = bool(refinement_queue or initial)
                    source_queue: deque[dict[str, Any]] | None = None
                    if thorough_mode and triple_queue:
                        source_queue = triple_queue
                    elif (pair_queue
                          and (singles_since_combination >= 2
                               or not singles_available)):
                        source_queue = pair_queue
                    elif refinement_queue:
                        source_queue = refinement_queue
                    elif premise_queue:
                        # Above `initial` so the premise stage preempts the
                        # extension's deep-single flood; below refinements
                        # and pairs because those grow from live leads.
                        source_queue = premise_queue
                    elif initial:
                        source_queue = initial
                    elif pair_queue:
                        source_queue = pair_queue
                    elif thorough_mode and triple_queue:
                        source_queue = triple_queue
                    else:
                        _queue_premise_proposal("exhausted")
                        if premise_queue:
                            source_queue = premise_queue
                        else:
                            phase_exhausted = True
                            break
                    candidate = source_queue.popleft()

                    if candidate.get("kind") == "premise_proposal":
                        proposal_origin = candidate.get("origin", "unknown")
                        if (_budget_remaining_seconds()
                                < premise_min_budget_seconds):
                            premise_state["proposal_status"] = "skipped_budget"
                            yield _sse("premise_proposal", {
                                "status": "skipped_budget",
                                "origin": proposal_origin,
                                "directions": [],
                                "elapsed_seconds": _elapsed_seconds(),
                            })
                            continue
                        current_stage = "premise_proposal"
                        yield _sse("stage", {
                            "stage": current_stage,
                            "tested": tested,
                            "elapsed_seconds": _elapsed_seconds(),
                            "budget_remaining_seconds": (
                                _budget_remaining_seconds()
                            ),
                        })
                        planner_messages = _premise_proposal_messages(
                            req.messages, selected_source,
                            req.replacement_text,
                        )
                        try:
                            planner_ids = await asyncio.to_thread(
                                _chat_template_token_ids,
                                tok,
                                planner_messages,
                                enable_thinking=False,
                            )
                        except ValueError:
                            planner_ids = []
                        if not planner_ids or len(planner_ids) > 4096:
                            premise_state["proposal_status"] = (
                                "context_too_large" if planner_ids
                                else "template_failed"
                            )
                            yield _sse("premise_proposal", {
                                "status": premise_state["proposal_status"],
                                "origin": proposal_origin,
                                "directions": [],
                                "elapsed_seconds": _elapsed_seconds(),
                            })
                            continue
                        proposal = None
                        while True:
                            try:
                                proposal, queue_wait_ms = (
                                    await _locked_worker(
                                        _greedy_intervention_replay,
                                        planner_ids,
                                        None,
                                        128,
                                        eos_ids,
                                    )
                                )
                                break
                            except _AdaptiveSearchPauseRequested:
                                assert control is not None
                                async for frame in _pause_frames("user"):
                                    yield frame
                                continue
                            except _AdaptiveSearchDeadlineExpired as expired:
                                deadline_queue_wait_ms = (
                                    expired.queue_wait_ms
                                )
                                source_queue.appendleft(candidate)
                                budget_exhausted = True
                                break
                        if proposal is None:
                            break
                        candidate_durations.append(
                            (proposal["timings_ms"]["total"] + queue_wait_ms)
                            / 1000.0
                        )
                        proposal_text = tok.decode(proposal["ids"])
                        directions: list[dict[str, Any]] = []
                        for row in _parse_premise_proposals(proposal_text):
                            premise_spec = InterventionSpec(
                                mode="swap",
                                layers=[workspace_layers[0]],
                                token=" " + row["source_concept"],
                                target=" " + row["target_concept"],
                                alpha=1.0,
                                positions=[0],
                            )
                            try:
                                resolved_premise = (
                                    _validate_and_resolve_specs(
                                        [premise_spec], _lens, tok,
                                        _model.n_layers,
                                    )[0]
                                )
                            except HTTPException:
                                continue
                            source_token_id = resolved_premise["token_id"]
                            target_token_id = resolved_premise["target_id"]
                            if source_token_id == target_token_id:
                                continue
                            if (source_token_id == resolved["token_id"]
                                    and target_token_id
                                    == resolved["target_id"]):
                                # The literal reply direction is already the
                                # subject of every cell stage.
                                continue
                            if any(
                                direction["source_token_id"]
                                == source_token_id
                                and direction["target_token_id"]
                                == target_token_id
                                for direction in directions
                            ):
                                continue
                            directions.append({
                                **row,
                                "source_str": resolved_premise["token_str"],
                                "target_str": resolved_premise["target_str"],
                                "source_token_id": source_token_id,
                                "target_token_id": target_token_id,
                            })
                        premise_state["directions"] = directions
                        premise_state["proposal_status"] = (
                            "ok" if directions else "no_usable_direction"
                        )
                        proposal_timings = dict(proposal["timings_ms"])
                        proposal_timings["queue_wait"] = queue_wait_ms
                        yield _sse("premise_proposal", {
                            "status": premise_state["proposal_status"],
                            "origin": proposal_origin,
                            "directions": [
                                {
                                    "source_concept": d["source_concept"],
                                    "target_concept": d["target_concept"],
                                    "source_str": d["source_str"],
                                    "target_str": d["target_str"],
                                    "source_token_id": d["source_token_id"],
                                    "target_token_id": d["target_token_id"],
                                    "reason": d["reason"],
                                    "confidence": d["confidence"],
                                }
                                for d in directions
                            ],
                            "proposal_text": proposal_text[:400],
                            "timings_ms": proposal_timings,
                            "elapsed_seconds": _elapsed_seconds(),
                            "budget_remaining_seconds": (
                                _budget_remaining_seconds()
                            ),
                        })
                        for direction in directions:
                            _enqueue_premise(_premise_search_candidate(
                                direction, list(workspace_layers)
                            ))
                        continue

                    elapsed = _elapsed_seconds()
                    ordered_durations = sorted(candidate_durations)
                    p95_index = round(0.95 * (len(ordered_durations) - 1))
                    estimated_atomic = ordered_durations[p95_index]
                    reserve = max(0.05, min(5.0, 2.0 * estimated_atomic))
                    if elapsed + reserve >= _time_budget_seconds():
                        source_queue.appendleft(candidate)
                        budget_exhausted = True
                        break

                    if candidate.get("kind") == "premise":
                        # Premise clamps are neither singles nor combinations
                        # for the pair-pacing counter.
                        pass
                    elif len(candidate["cells"]) == 1:
                        singles_since_combination += 1
                    else:
                        singles_since_combination = 0

                    if candidate["stage"] != current_stage:
                        current_stage = candidate["stage"]
                        yield _sse("stage", {
                            "stage": current_stage,
                            "tested": tested,
                            "elapsed_seconds": elapsed,
                            "budget_remaining_seconds": (
                                _budget_remaining_seconds()
                            ),
                        })

                    candidate_elapsed = _elapsed_seconds()
                    yield _sse("candidate_start", {
                        "id": candidate["id"],
                        "recipe_key": (
                            candidate.get("recipe_key")
                            or _adaptive_recipe_string(candidate["cells"])
                        ),
                        "stage": candidate["stage"],
                        "parents": candidate["parents"],
                        "index": tested + 1,
                        "max_candidates": max_candidates,
                        "cells": candidate["cells"],
                        "premise": candidate.get("premise"),
                        "elapsed_seconds": candidate_elapsed,
                        "budget_remaining_seconds": (
                            _budget_remaining_seconds()
                        ),
                    })

                    replay = None
                    while True:
                        try:
                            replay, queue_wait_ms = await _locked_worker(
                                _compile_and_replay, candidate
                            )
                            break
                        except _AdaptiveSearchPauseRequested:
                            assert control is not None
                            async for frame in _pause_frames("user"):
                                yield frame
                            continue
                        except _AdaptiveSearchDeadlineExpired as expired:
                            deadline_queue_wait_ms = expired.queue_wait_ms
                            source_queue.appendleft(candidate)
                            budget_exhausted = True
                            replay = None
                            break
                    if replay is None:
                        break

                    tested += 1
                    candidate_durations.append(
                        (replay["timings_ms"]["total"] + queue_wait_ms)
                        / 1000.0
                    )
                    text = tok.decode(replay["ids"])
                    text_norm = _normalized_probe_text(text)
                    exact = text == desired_response
                    changed = text != baseline_text
                    verified = bool(
                        replay["eos"]
                        and not replay["repetition"]
                        and exact
                        and changed
                    )
                    safe_changed = bool(
                        replay["eos"]
                        and not replay["repetition"]
                        and changed
                    )
                    similarity = SequenceMatcher(
                        None, text_norm, desired_norm
                    ).ratio()
                    improvement = similarity - baseline_similarity
                    target_count = _bounded_phrase_count(
                        text, req.replacement_text
                    )
                    source_count = _bounded_phrase_count(text, selected_source)
                    improving = bool(
                        safe_changed
                        and (improvement > 1e-9
                             or (target_count >= 1 and source_count == 0))
                    )
                    is_premise = candidate.get("kind") == "premise"
                    if is_premise:
                        premise_state["replays"] += 1
                    # A coherent model that now believes the swapped premise
                    # also verbalizes it, so a premise clamp is judged by
                    # redirect (goal present, source absent, clean stop) —
                    # byte-exact equality stays reserved for cell recipes.
                    premise_redirect = bool(
                        is_premise
                        and not verified
                        and safe_changed
                        and target_count >= 1
                        and source_count == 0
                    )
                    if verified:
                        classification = "verified"
                    elif replay["repetition"]:
                        classification = "repetition"
                    elif not changed:
                        classification = "unchanged"
                    elif premise_redirect:
                        classification = "premise_redirect"
                    elif target_count >= 1 and source_count == 0:
                        classification = "partial"
                    else:
                        classification = "off_target"

                    scores_by_id[candidate["id"]] = similarity
                    elapsed = _elapsed_seconds()
                    timings = dict(replay["timings_ms"])
                    timings["queue_wait"] = queue_wait_ms
                    timings["total_with_queue"] = (
                        timings["total"] + queue_wait_ms
                    )
                    yield _sse("candidate", {
                        "id": candidate["id"],
                        "recipe_key": (
                            candidate.get("recipe_key")
                            or _adaptive_recipe_string(candidate["cells"])
                        ),
                        "stage": candidate["stage"],
                        "parents": candidate["parents"],
                        "index": tested,
                        "max_candidates": max_candidates,
                        "cells": candidate["cells"],
                        "premise": candidate.get("premise"),
                        "premise_verified": (
                            classification == "premise_redirect"
                        ),
                        "text": text,
                        "eos": replay["eos"],
                        "repetition": replay["repetition"],
                        "stop_reason": replay["stop_reason"],
                        "class": classification,
                        "verified": verified,
                        "exact_response_match": exact,
                        "safe_changed": safe_changed,
                        "improving": improving,
                        "similarity_to_desired": similarity,
                        "improvement_over_baseline": improvement,
                        "timings_ms": timings,
                        "elapsed_seconds": elapsed,
                        "budget_remaining_seconds": (
                            _budget_remaining_seconds()
                        ),
                        "progress": {
                            "tested": tested,
                            "limit": max_candidates,
                            "initial_queued": len(initial),
                            "refinements_queued": len(refinement_queue),
                            "pairs_queued": len(pair_queue),
                            "triples_queued": len(triple_queue),
                        },
                    })
                    if verified:
                        # The search keeps going: a verified recipe joins the
                        # results (and may seed refinements and pairs below)
                        # while the budget hunts for more, unless the caller
                        # asked for first-hit termination.
                        verified_candidates.append(candidate)
                        if req.stop_on_verified:
                            break

                    if is_premise and classification == "premise_redirect":
                        premise_state["redirects"] += 1
                        premise = candidate["premise"]
                        record = {
                            "source_concept": premise["source_concept"],
                            "target_concept": premise["target_concept"],
                            "source_str": premise["source_str"],
                            "target_str": premise["target_str"],
                            "source_token_id": premise["source_token_id"],
                            "target_token_id": premise["target_token_id"],
                            "layers": list(premise["layers"]),
                            "recipe_key": candidate["recipe_key"],
                            "text": text,
                            "similarity_to_desired": similarity,
                        }
                        best = premise_state["best"]
                        if (best is None
                                or len(record["layers"])
                                < len(best["layers"])
                                or (len(record["layers"])
                                    == len(best["layers"])
                                    and similarity
                                    > best["similarity_to_desired"])):
                            premise_state["best"] = record
                        # Bisect the passing band to shrink the reported
                        # footprint while the replay budget lasts.
                        if premise_state["replays"] < premise_max_replays:
                            direction = {
                                key: premise[key]
                                for key in (
                                    "source_concept", "target_concept",
                                    "source_str", "target_str",
                                    "source_token_id", "target_token_id",
                                )
                            }
                            for half in _premise_band_halves(
                                premise["layers"]
                            ):
                                _enqueue_premise(_premise_search_candidate(
                                    direction, half,
                                    parents=[candidate["id"]],
                                ))

                    if len(candidate["cells"]) == 1:
                        if (safe_changed
                                and candidate["stage"]
                                != "refinement_single"):
                            for refined in _adaptive_refinement_candidates(
                                candidate, workspace_layers
                            ):
                                _enqueue(refinement_queue, refined)
                        if improving:
                            promising = dict(candidate)
                            promising["score"] = similarity
                            promising_singles.append(promising)
                            promising_singles.sort(
                                key=lambda value: (
                                    -value["score"], value["id"]
                                )
                            )
                            promising_singles[:] = promising_singles[:16]
                            pair_limit = 128 if thorough_mode else 48
                            if pair_count < pair_limit:
                                for pair in _adaptive_pair_candidates(
                                    promising, promising_singles
                                ):
                                    if pair_count >= pair_limit:
                                        break
                                    if _enqueue(pair_queue, pair):
                                        pair_count += 1
                    elif len(candidate["cells"]) == 2:
                        parent_score = max(
                            (scores_by_id.get(parent, baseline_similarity)
                             for parent in candidate["parents"]),
                            default=baseline_similarity,
                        )
                        if safe_changed and similarity > parent_score + 1e-9:
                            improving_pair = dict(candidate)
                            improving_pair["score"] = similarity
                            promising_pairs.append(improving_pair)
                            promising_pairs.sort(
                                key=lambda value: (
                                    -value["score"], value["id"]
                                )
                            )
                            promising_pairs[:] = promising_pairs[:16]
                            if thorough_mode:
                                for triple in _adaptive_triple_candidates(
                                    candidate, promising_singles
                                ):
                                    if triple_count >= 64:
                                        break
                                    if _enqueue(triple_queue, triple):
                                        triple_count += 1

                    if tested >= premise_early_after_tested:
                        _queue_premise_proposal("early")
                    await asyncio.sleep(0)

                if verified_candidates and req.stop_on_verified:
                    break
                if (control is not None
                        and not control.extended
                        and (budget_exhausted or phase_exhausted)):
                    async for frame in _pause_frames(
                        "initial_budget" if budget_exhausted
                        else "initial_candidates_exhausted",
                        awaiting_extension=True,
                        charge_to_budget=True,
                    ):
                        yield frame
                    budget_exhausted = False
                    _seed_extension()
                    yield _sse("position_ranking", {
                        "frontier_position": frontier,
                        "limit": position_limit,
                        "positions": all_ranked_positions[:position_limit],
                        "extended": True,
                    })
                    continue
                break

            elapsed = _elapsed_seconds()
            if (not verified_candidates
                    and elapsed >= _time_budget_seconds()):
                budget_exhausted = True
            if verified_candidates:
                status = "success"
                stop_reason = (
                    "verified" if req.stop_on_verified
                    else "time_budget" if budget_exhausted
                    else "exhausted_candidates"
                )
            elif budget_exhausted:
                status, stop_reason = "budget_exhausted", "time_budget"
            else:
                status, stop_reason = "no_verified_recipe", "exhausted_candidates"
            yield _sse("search_end", {
                "status": status,
                "stop_reason": stop_reason,
                "tested": tested,
                "max_candidates": max_candidates,
                "verified_candidate_id": (
                    verified_candidates[0]["id"]
                    if verified_candidates else None
                ),
                "verified_cells": (
                    verified_candidates[0]["cells"]
                    if verified_candidates else None
                ),
                "verified_count": len(verified_candidates),
                "desired_response": desired_response,
                "elapsed_seconds": elapsed,
                "time_budget_seconds": _time_budget_seconds(),
                "budget_exhausted": budget_exhausted,
                "budget_overshoot_seconds": max(
                    0.0, elapsed - _time_budget_seconds()
                ),
                "deadline_queue_wait_ms": deadline_queue_wait_ms,
                "stage_counts": {
                    "prior_promising_singles": len(prior_promising),
                    "promising_singles": len(promising_singles),
                    "pairs_enqueued": pair_count,
                    "triples_enqueued": triple_count,
                },
                "premise": {
                    "enabled": premise_state["enabled"],
                    "attempted": premise_state["attempted"],
                    "proposal_status": premise_state["proposal_status"],
                    "directions": [
                        {
                            "source_str": d["source_str"],
                            "target_str": d["target_str"],
                            "confidence": d["confidence"],
                        }
                        for d in premise_state["directions"]
                    ],
                    "replays": premise_state["replays"],
                    "redirects": premise_state["redirects"],
                    "best": premise_state["best"],
                },
            })
        except Exception as error:
            print(
                f"[intervention_search_adaptive] error: {error}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            elapsed = _elapsed_seconds()
            yield _sse("error", {"error": str(error)})
            yield _sse("search_end", {
                "status": "error",
                "stop_reason": "error",
                "tested": tested,
                "verified_candidate_id": None,
                "verified_cells": None,
                "elapsed_seconds": elapsed,
                "time_budget_seconds": _time_budget_seconds(),
                "budget_exhausted": False,
                "budget_overshoot_seconds": max(
                    0.0, elapsed - _time_budget_seconds()
                ),
            })
        finally:
            if control is not None:
                control.closed = True
                if _adaptive_search_controls.get(control.search_id) is control:
                    _adaptive_search_controls.pop(control.search_id, None)

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
