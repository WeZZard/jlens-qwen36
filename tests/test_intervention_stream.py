"""Causal gate for STREAMING interventions (cached path).

Mirrors scripts/intervention_sanity.py's France->China swap, but through
the StreamSession edit hook the server uses — proving the cached engine
is causal end to end, plus a clean-path guard (empty edits == no edits).

Heavy: loads the 27B model + 3 lens layers. Gated so the default suite
(which already loads the model once for the decode gate) doesn't pay a
second model instance:

    JLENS_SLOW_TESTS=1 uv run pytest tests/test_intervention_stream.py -v
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LENS_PATH = os.path.join(REPO, "data", "lens", "full_depth_analytic.npz")
LAYERS = [30, 40, 48]
FRANCE_PROMPT = (
    "Question: What is the capital of France?\nAnswer: The capital of France is"
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("JLENS_SLOW_TESTS"),
    reason="model-loading gate; set JLENS_SLOW_TESTS=1 to run",
)


@pytest.fixture(scope="session")
def model():
    from jlens_qwen.model import load
    from jlens_qwen.patch_gdn import set_inference_mode

    m = load()
    set_inference_mode(True)
    return m


@pytest.fixture(scope="session")
def lens():
    if not os.path.exists(LENS_PATH):
        pytest.skip(f"lens missing at {LENS_PATH}")
    from jlens_qwen.lens import JacobianLens

    # Load only the intervention layers (npz members read lazily).
    data = np.load(LENS_PATH, allow_pickle=True)
    jac = {l: data[f"J_{l}"].astype(np.float32) for l in LAYERS}
    return JacobianLens(
        jac, n_prompts=int(data["n_prompts"]), d_model=int(data["d_model"])
    )


def _greedy(model, session, ids, n=6):
    logits, _ = session.extend(ids)
    out = []
    for _ in range(n):
        tok = int(mx.argmax(logits.astype(mx.float32)).tolist())
        out.append(tok)
        logits, _ = session.extend([tok])
    return out


def test_france_to_beijing_streaming(model, lens):
    from jlens_qwen.interventions import compile_edits

    ids = model.encode(FRANCE_PROMPT)[0].tolist()
    tok = model.tokenizer
    france_id = tok.encode(" France", add_special_tokens=False)[0]
    china_id = tok.encode(" China", add_special_tokens=False)[0]

    base = _greedy(model, model.make_stream(), ids)
    base_text = tok.decode(base)
    assert "paris" in base_text.lower(), f"unexpected baseline: {base_text!r}"

    edits = compile_edits(
        lens, model, mode="swap", layers=LAYERS,
        token_id=france_id, target_id=china_id, alpha=1.0, from_pos=0,
    )
    session = model.make_stream()
    session.set_edits(edits)
    swapped = _greedy(model, session, ids)
    text = tok.decode(swapped)
    assert any(w in text.lower() for w in ("beijing", "china", "peking")), (
        f"swap did not redirect: {text!r} (baseline {base_text!r})"
    )


def test_empty_edits_identical(model):
    ids = model.encode(FRANCE_PROMPT)[0].tolist()
    plain = _greedy(model, model.make_stream(), ids)
    with_empty = model.make_stream()
    with_empty.set_edits([])
    assert _greedy(model, with_empty, ids) == plain
