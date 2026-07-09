"""Decode correctness gate.

Regression tests that pin the decode hot path's BEHAVIOR so performance
work cannot silently change what the viewer shows. Run with:

    uv run python -m pytest tests/test_decode_gate.py -v

Heavy: loads the 27B model once per session (~15 GB from HF cache) and a
full-depth lens (~3.3 GB). Golden data lives in
tests/golden/decode_gate_golden.json and freezes CURRENT behavior;
regenerate it (scripts/gen_decode_gate_golden.py) only as a deliberate
act — when a numerics-changing optimization has an approved tolerance
policy (docs/perf/LEDGER.md, Gate section) or the model/lens/MLX version
changes.

Why golden self-consistency instead of a cached-vs-uncached reference:
the cached (StreamSession) and uncached (full re-forward) paths are NOT
token-identical over long horizons in bf16 — measured divergence at step
21/64 on the gate prompt, a near-tie flip between '\\n' and '\\n\\n'.
Likewise the batched readout and a per-position reference differ in ulps
(different matmul batch shapes), which swaps near-tied ranks. Golden
data pins the shipped behavior exactly; the tie-aware equivalence test
(below) documents the batched-vs-naive relationship with a measured
epsilon.

Invariants:

1. Golden token identity: cached greedy decode reproduces the frozen
   64-token sequence exactly.
2. Golden readout identity: `_readout_at_positions` reproduces the
   frozen top-10 ids exactly (scores within 1e-3).
3. Batched-vs-naive readout equivalence, tie-aware: identical id sets
   and scores except where near-ties (< EPS) legitimately swap.
4. Streaming detokenization: per-token segments concatenate to exactly
   `tokenizer.decode(all_ids)`, including UTF-8 chars split across
   tokens.
"""

from __future__ import annotations

import json
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(REPO, "tests", "golden", "decode_gate_golden.json")

# Tie-aware epsilon for batched-vs-naive score comparisons. Calibrated by
# scripts/gen_decode_gate_golden.py's diagnostic: across 192 cells the
# worst common-id score diff was 1.089e-03 and the worst boundary-tie gap
# 2.8e-05 (2026-07-09, MLX 0.31.2). 5e-3 gives ~4.6x headroom over the
# measured shape-dependent ulp noise while staying far below any real
# regression signal.
EPS = 5e-3


@pytest.fixture(scope="session")
def golden():
    if not os.path.exists(GOLDEN_PATH):
        pytest.fail(
            f"golden data missing at {GOLDEN_PATH}; "
            "run: uv run python scripts/gen_decode_gate_golden.py"
        )
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def model(golden):
    from jlens_qwen.model import load
    from jlens_qwen.patch_gdn import set_inference_mode

    m = load(golden["meta"]["model"])
    set_inference_mode(True)
    return m


@pytest.fixture(scope="session")
def lens(golden):
    rel = golden["meta"]["lens"]
    if rel is None:
        return None
    path = os.path.join(REPO, rel)
    if not os.path.exists(path):
        pytest.fail(f"lens missing at {path} (golden data was generated with it)")
    from jlens_qwen.lens import JacobianLens

    return JacobianLens.load(path)


def _record_layers(model, lens):
    if lens is None:
        return [model.n_layers - 1]
    return sorted(set(lens.source_layers) | {model.n_layers - 1})


def test_golden_token_identity(model, golden):
    """Cached greedy decode reproduces the frozen token sequence exactly."""
    meta = golden["meta"]
    input_ids = model.encode(meta["prompt"])

    session = model.make_stream()
    logits, _ = session.extend(input_ids[0].tolist())
    got = []
    for _ in range(meta["horizon"]):
        tok = int(mx.argmax(logits.astype(mx.float32)).tolist())
        got.append(tok)
        logits, _ = session.extend([tok])

    want = golden["tokens"]
    if got != want:
        i = next(k for k in range(len(want)) if got[k] != want[k])
        raise AssertionError(
            f"token divergence from golden at step {i}/{meta['horizon']}: "
            f"got={got[i]} ({model.tokenizer.decode([got[i]])!r}) "
            f"golden={want[i]} ({model.tokenizer.decode([want[i]])!r}); "
            f"prefix={model.tokenizer.decode(got[:i])!r}"
        )


@pytest.fixture(scope="session")
def readout(model, lens, golden):
    """One prefill + batched readout, shared by the two readout tests."""
    from jlens_qwen import serve

    serve._model = model
    serve._lens = lens
    layers = _record_layers(model, lens)
    positions = golden["meta"]["positions"]

    session = model.make_stream(capture_layers=layers)
    ids = model.encode(golden["meta"]["prompt"])[0].tolist()
    _, acts = session.extend(ids)
    out = serve._readout_at_positions(acts, positions, layers, golden["meta"]["top_n"])
    return layers, positions, acts, out


def test_golden_readout_identity(readout, golden):
    """Batched readout reproduces the frozen top-10 ids exactly."""
    layers, positions, _, out = readout
    for layer in layers:
        g = golden["readout"][str(layer)]
        for pi in range(len(positions)):
            assert out[layer]["top_ids"][pi] == g["ids"][pi], (
                f"golden id mismatch at layer {layer} pos index {pi}: "
                f"got {out[layer]['top_ids'][pi]} golden {g['ids'][pi]}"
            )
            got_s = np.array(out[layer]["top_scores"][pi])
            want_s = np.array(g["scores"][pi])
            assert np.allclose(got_s, want_s, atol=1e-3), (
                f"golden score drift at layer {layer} pos index {pi}: "
                f"max diff {np.max(np.abs(got_s - want_s)):.3e}"
            )


def test_readout_batched_vs_naive_tie_aware(readout, model, lens):
    """Batched readout == per-position naive reference, modulo near-ties.

    The two paths run quantized matmuls at different batch shapes, so
    scores differ in ulps and near-tied ranks may swap. Criterion:
    - common ids: |score diff| < EPS
    - ids present in only one list: must be boundary ties (score within
      EPS of the k-th score of the other list)
    - internal ordering must be descending in both.
    """
    layers, positions, acts, out = readout
    top_n = 10

    worst_common = 0.0
    worst_boundary = 0.0
    for layer in layers:
        for pi, pos in enumerate(positions):
            h = acts[layer][0][pos].astype(mx.float32)[None]
            if lens is not None and layer in lens.jacobians:
                h = lens.transport(h, layer)
            logits = model.unembed(model.final_norm(h)).astype(mx.float32)[0]
            order = mx.argsort(logits)[-top_n:][::-1]
            ref_ids = [int(t) for t in order.tolist()]
            ref_scores = np.array(mx.take_along_axis(logits, order, axis=-1).tolist())

            got_ids = out[layer]["top_ids"][pi]
            got_scores = np.array(out[layer]["top_scores"][pi])

            assert all(np.diff(got_scores) <= 1e-6), (
                f"batched scores not descending at layer {layer} pos {pos}"
            )

            gm = dict(zip(got_ids, got_scores))
            rm = dict(zip(ref_ids, ref_scores))
            for cid in set(got_ids) & set(ref_ids):
                d = abs(gm[cid] - rm[cid])
                worst_common = max(worst_common, d)
                assert d < EPS, (
                    f"score diff {d:.3e} >= EPS for id {cid} "
                    f"at layer {layer} pos {pos}"
                )
            kth = min(ref_scores[-1], got_scores[-1])
            for oid in set(got_ids) ^ set(ref_ids):
                s = gm.get(oid, rm.get(oid))
                gap = abs(s - kth)
                worst_boundary = max(worst_boundary, gap)
                assert gap < EPS, (
                    f"non-tie id-set mismatch at layer {layer} pos {pos}: "
                    f"id {oid} is {gap:.3e} from the k-th score (>= EPS)"
                )
    print(
        f"\n[gate] tie-aware equivalence: worst common-id diff "
        f"{worst_common:.3e}, worst boundary gap {worst_boundary:.3e}, "
        f"EPS {EPS:.1e}"
    )


def test_readout_token_display(model):
    """Band-cell display strings (_tok_str) for pathological tokens.

    Reproducer for the '悖' bug: byte-level BPE splits multi-byte UTF-8
    chars across tokens; decoding a fragment token alone yields U+FFFD,
    which the band rendered as black diamonds. _tok_str must render
    fragments as their raw bytes ("⟨E6 82⟩"), pure-whitespace tokens as
    visible glyphs, and everything else as its plain decode.
    """
    import re

    from jlens_qwen import serve

    serve._model = model
    serve._tok_str_cache.clear()
    tok = model.tokenizer

    hexpat = re.compile(r"^⟨([0-9A-F]{2}( [0-9A-F]{2})*)⟩$")

    for text in ("悖论", "😄", "🧑‍🚀", "巍峨"):
        ids = tok.encode(text, add_special_tokens=False)
        recovered = b""
        for tid in ids:
            plain = tok.decode([tid])
            shown = serve._tok_str(tid)
            assert "�" not in shown, (
                f"replacement char leaked for token {tid} of {text!r}: {shown!r}"
            )
            if "�" in plain:
                m = hexpat.match(shown)
                assert m, f"fragment token {tid} not rendered as bytes: {shown!r}"
            # accumulate the true bytes for the round-trip property
            b = serve._token_bytes(tid)
            assert b is not None, f"byte recovery failed for token {tid}"
            recovered += b
        assert recovered == text.encode(), (
            f"byte round-trip mismatch for {text!r}: {recovered!r}"
        )

    # whitespace tokens render visibly
    for text, want in (("\n\n", "⏎⏎"), (" ", "␣"), ("\n", "⏎")):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1:  # only assert when it is a single token
            assert serve._tok_str(ids[0]) == want, (
                f"{text!r} rendered {serve._tok_str(ids[0])!r}, want {want!r}"
            )

    # normal tokens pass through unchanged
    for tid in tok.encode("hello world, this is plain text.", add_special_tokens=False):
        plain = tok.decode([tid])
        if "�" not in plain and plain.strip():
            assert serve._tok_str(tid) == plain


def test_streaming_detok_identity(model):
    """Per-token segments concatenate to the exact full decode."""
    from jlens_qwen.serve import _token_segments

    tok = model.tokenizer
    samples = [
        "héllo wörld — “smart quotes” … ellipsis",
        "emoji storm: 👋🌍🎉🇯🇵🇺🇸 🧑‍🚀🏳️‍🌈",
        "你好，世界！这是一个测试。日本語のテキストもある。",
        "mixed: café ☕ + naïve résumé + Здравствуйте + مرحبا",
        "code: `let x = \"π ≈ 3.14159\";` <|not_a_token|>",
    ]
    for text in samples:
        ids = tok.encode(text, add_special_tokens=False)
        segments = _token_segments(tok, ids)
        assert len(segments) == len(ids)
        joined = "".join(segments)
        full = tok.decode(ids)
        assert joined == full, (
            f"segment concat != decode for {text!r}: {joined!r} vs {full!r}"
        )
