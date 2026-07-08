"""FastAPI backend for the J-space visualizer.

Endpoints:
- POST /api/slice  { prompt, max_seq_len } -> slice data (top-k per cell, ranks)
- POST /api/generate { prompt, ... } -> model's actual next-token logits
- POST /api/chat_stream { messages, ... } -> SSE stream of J-lens snapshots
- GET  /api/lens  -> lens metadata (source layers, n_prompts, d_model)
- GET  /  -> serves the web UI
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
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

# Globals (loaded once at startup).
_model = None
_lens = None
_model_id = os.environ.get("JLENS_MODEL", "mlx-community/Qwen3.6-27B-4bit")
_lens_path = os.environ.get("JLENS_PATH", "data/lens/qwen36_27b.npz")
_repo_root = Path(__file__).resolve().parent.parent
_sessions_dir = _repo_root / "data" / "sessions"

# Pause/resume control for the active generation stream.
# The generation loop checks this event before each token.
_pause_events: dict[str, asyncio.Event] = {}

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


@app.get("/api/model")
async def model_info():
    return {"model_id": _model_id, "n_layers": _model.n_layers, "d_model": _model.d_model}


def _record_layers() -> list[int]:
    layers = list(_lens.source_layers) if _lens is not None else [_model.n_layers - 1]
    return sorted(set(layers) | {_model.n_layers - 1})


_tok_str_cache: dict[int, str] = {}


def _tok_str(tid: int) -> str:
    s = _tok_str_cache.get(tid)
    if s is None:
        s = _model.tokenizer.decode([tid])
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

    # Chunk positions so the [L, P_chunk, vocab] logits tensor stays small
    # (64 layers x 8 positions x 248k vocab fp32 ~= 0.5GB).
    CHUNK = 8
    for c0 in range(0, len(positions), CHUNK):
        chunk = positions[c0:c0 + CHUNK]
        pos_idx = mx.array(chunk)
        hs = []
        for layer in layers:
            h = acts[layer][0][pos_idx].astype(mx.float32)  # [P, D]
            if _lens is not None and layer in _lens.jacobians:
                h = _lens.transport(h, layer)
            hs.append(h)
        hstack = mx.stack(hs)  # [L, P, D]
        logits = _model.unembed(_model.final_norm(hstack)).astype(mx.float32)
        order = mx.argsort(logits, axis=-1)[..., -top_n:][..., ::-1]  # [L, P, n]
        vals = mx.take_along_axis(logits, order, axis=-1)
        ids_l = order.tolist()
        vals_l = vals.tolist()
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


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 32
    temp: float = 0.0


@app.post("/api/generate")
async def generate_endpoint(req: GenerateRequest):
    """Generate a continuation (no intervention)."""
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
    """Generate with a J-space intervention."""
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


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatStreamRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 32
    temp: float = 0.0
    top_n: int = 10



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
    if _model is None:
        raise HTTPException(503, "model not loaded")

    import asyncio
    import uuid

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
        # pause/resume control.
        yield _sse("stream_start", {"stream_id": stream_id})

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
                formatted_ids = tok.apply_chat_template(prefix_msgs, add_generation_prompt=False)
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
                _, acts = session.extend(delta)
                positions = list(range(start_pos, end_pos))
                # acts cover only the delta chunk -> chunk-local indices.
                local = [p - chunk_start for p in positions]
                row = _readout_at_positions(acts, local, layers, top_n)
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
                gen_prefix_ids = tok.apply_chat_template(prefix_msgs, add_generation_prompt=True)
                if isinstance(gen_prefix_ids, list):
                    gen_prefix_list = gen_prefix_ids
                else:
                    gen_prefix_list = gen_prefix_ids.tolist() if hasattr(gen_prefix_ids, 'tolist') else list(gen_prefix_ids)

                global_pos = len(gen_prefix_list)

                # Feed the generation-prefix delta; readout at the frontier.
                chunk_start = session.n_consumed
                delta = gen_prefix_list[chunk_start:]
                logits, acts = session.extend(delta) if delta else (None, {})
                prefill_positions = [global_pos - 1] if global_pos > 0 else []
                prefill_row = _readout_at_positions(
                    acts, [len(delta) - 1], layers, top_n
                ) if (prefill_positions and delta) else {l: {"top_ids": [], "top_tokens": [], "top_scores": []} for l in layers}

                lf = logits.astype(mx.float32)
                if temp == 0:
                    next_tok = int(mx.argmax(lf).tolist())
                else:
                    # categorical() takes unnormalized logits, NOT probabilities.
                    next_tok = int(mx.random.categorical(lf / temp).tolist())
                del logits, lf

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
                while not eos_hit and n_gen < max_tokens:
                    # Block here if the client has paused generation.
                    await pause_event.wait()

                    # One-token cached step: acts cover just this token.
                    new_pos = global_pos
                    logits, acts = session.extend([next_tok])
                    row = _readout_at_positions(acts, [0], layers, top_n)
                    lf = logits.astype(mx.float32)
                    if temp == 0:
                        new_next_tok = int(mx.argmax(lf).tolist())
                    else:
                        # categorical() takes unnormalized logits, NOT probabilities.
                        new_next_tok = int(mx.random.categorical(lf / temp).tolist())
                    del logits, lf

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
                    next_tok = new_next_tok
                    if eos_id is not None and next_tok == eos_id:
                        eos_hit = True
                    await asyncio.sleep(0)

                yield _sse("done", {"msg_idx": msg_idx, "n_tokens": n_gen})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[chat_stream] error after snapshot {snapshot_id}: {e}\n{tb}", flush=True)
            yield _sse("error", {"snapshot_id": snapshot_id, "error": str(e), "traceback": tb})
        finally:
            # Clean up the pause event when the stream ends.
            _pause_events.pop(stream_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent.parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>J-Space Visualizer</h1><p>web/index.html not found</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
