"""Validation + resolution of streaming intervention specs (no model).

_validate_and_resolve_specs must 400 on every malformed spec BEFORE the
SSE stream starts, resolve concept text to first-token ids (explicit ids
win), and emit the JSON-safe echo that stream_start carries.
"""
import pytest
from fastapi import HTTPException

import jlens_qwen.serve as serve
from jlens_qwen.serve import InterventionSpec, _validate_and_resolve_specs

N_LAYERS = 64


class StubTokenizer:
    """Encodes each character to its codepoint; decodes to 'T<id>'."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(f"T{t}" for t in ids)


class StubLens:
    source_layers = [30, 40, 48]


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    class _M:
        tokenizer = StubTokenizer()

    monkeypatch.setattr(serve, "_model", _M())
    # _tok_str memoizes per token id; keep runs independent.
    monkeypatch.setattr(serve, "_tok_str_cache", {})


TOK = StubTokenizer()
LENS = StubLens()


def resolve(**kw):
    spec = InterventionSpec(**kw)
    return _validate_and_resolve_specs([spec], LENS, TOK, N_LAYERS)


def err(**kw):
    with pytest.raises(HTTPException) as ei:
        resolve(**kw)
    assert ei.value.status_code == 400
    return ei.value.detail


def test_steer_text_resolves_first_token():
    out = resolve(mode="steer", layers=[40], token="ab", alpha=5.0, from_position=0)
    r = out[0]
    assert r["token_id"] == ord("a")
    assert r["token_ids_all"] == [ord("a"), ord("b")]
    assert r["token_str"] == f"T{ord('a')}"
    assert r["layers"] == [40]
    assert r["from_position"] == 0 and r["positions"] is None and r["segment"] is None
    assert r["mode"] == "steer" and r["alpha"] == 5.0
    assert "steer" in r["label"]


def test_token_id_wins_over_text():
    out = resolve(mode="steer", layers=[30], token="zz", token_id=7, from_position=0)
    assert out[0]["token_id"] == 7
    assert out[0]["token_ids_all"] == [7]


def test_swap_requires_both_concepts():
    assert "target" in err(mode="swap", layers=[30], token="a", from_position=0)
    out = resolve(mode="swap", layers=[30], token="a", target="b", from_position=0)
    assert out[0]["token_id"] == ord("a") and out[0]["target_id"] == ord("b")


def test_ablate_ids_required_and_deduped():
    assert "ablate_token_ids" in err(mode="ablate", layers=[30], from_position=0)
    out = resolve(
        mode="ablate", layers=[30], ablate_token_ids=[9, 3, 9], from_position=0)
    assert out[0]["ablate_token_ids"] == [3, 9]


def test_unknown_mode_and_segment():
    assert "mode" in err(mode="clamp", layers=[30], token="a", from_position=0)
    assert "segment" in err(mode="steer", layers=[30], token="a", segment="prefill")


def test_layers_validated():
    assert "empty" in err(mode="steer", layers=[], token="a", from_position=0)
    detail = err(mode="steer", layers=[31], token="a", from_position=0)
    assert "[31]" in detail and "fitted" in detail
    # duplicates collapse, order normalizes
    out = resolve(mode="steer", layers=[48, 30, 48], token="a", from_position=0)
    assert out[0]["layers"] == [30, 48]


def test_final_layer_allowed_even_without_lens():
    out = _validate_and_resolve_specs(
        [InterventionSpec(mode="steer", layers=[N_LAYERS - 1], token="a", from_position=0)],
        None, TOK, N_LAYERS)
    assert out[0]["layers"] == [N_LAYERS - 1]
    with pytest.raises(HTTPException):
        _validate_and_resolve_specs(
            [InterventionSpec(mode="steer", layers=[40], token="a", from_position=0)],
            None, TOK, N_LAYERS)


def test_exactly_one_scope():
    assert "exactly one" in err(mode="steer", layers=[30], token="a")
    assert "exactly one" in err(
        mode="steer", layers=[30], token="a", positions=[1], from_position=2)
    for kw in ({"positions": [3]}, {"from_position": 3}, {"segment": "generation"}):
        out = resolve(mode="steer", layers=[30], token="a", **kw)
        assert out[0]["mode"] == "steer"


def test_positions_normalized_and_bounded():
    out = resolve(mode="steer", layers=[30], token="a", positions=[5, 2, 5])
    assert out[0]["positions"] == [2, 5]
    assert "empty" in err(mode="steer", layers=[30], token="a", positions=[])
    assert ">= 0" in err(mode="steer", layers=[30], token="a", positions=[-1])
    assert ">= 0" in err(mode="steer", layers=[30], token="a", from_position=-2)


def test_text_encoding_to_nothing_fails():
    # Empty string is caught by the required-field check...
    assert "required" in err(mode="steer", layers=[30], token="", from_position=0)
    # ...and text that ENCODES to nothing by the tokenizer check.
    class EmptyTok(StubTokenizer):
        def encode(self, text, add_special_tokens=False):
            return []

    with pytest.raises(HTTPException) as ei:
        _validate_and_resolve_specs(
            [InterventionSpec(mode="steer", layers=[30], token="x", from_position=0)],
            LENS, EmptyTok(), N_LAYERS)
    assert "no tokens" in ei.value.detail


def test_multiple_specs_indexed():
    specs = [
        InterventionSpec(mode="steer", layers=[30], token="a", from_position=0),
        InterventionSpec(mode="swap", layers=[40, 48], token="a", target="b",
                         segment="generation"),
    ]
    out = _validate_and_resolve_specs(specs, LENS, TOK, N_LAYERS)
    assert [r["index"] for r in out] == [0, 1]
    assert out[1]["segment"] == "generation" and out[1]["from_position"] is None
