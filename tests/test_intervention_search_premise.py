"""Latent-premise stage tests for the adaptive search (no model or GPU)."""

from __future__ import annotations

import asyncio
import json

import jlens_qwen.interventions as interventions
import jlens_qwen.serve as serve
import pytest


# ----- proposal parsing ------------------------------------------------------

def test_parse_premise_proposals_accepts_plain_and_fenced_json():
    plain = json.dumps({"premises": [{
        "source_concept": "France",
        "target_concept": "China",
        "reason": "capital belongs to China",
        "confidence": "high",
    }]})
    fenced = f"Here you go:\n```json\n{plain}\n```"
    for text in (plain, fenced):
        rows = serve._parse_premise_proposals(text)
        assert [(row["source_concept"], row["target_concept"]) for row in rows] \
            == [("France", "China")]
        assert rows[0]["confidence"] == "high"


def test_parse_premise_proposals_tolerates_candidates_key_and_bad_confidence():
    text = json.dumps({"candidates": [
        {"source_concept": "France", "target_concept": "China",
         "confidence": 0.8},
        {"source_concept": "France", "target_concept": "china"},
        {"source_concept": "Paris", "target_concept": "Paris"},
        {"source_concept": "a very long concept name", "target_concept": "b"},
        {"source_concept": "spider", "target_concept": "ant",
         "confidence": "medium"},
        {"source_concept": "third", "target_concept": "extra"},
    ]})
    rows = serve._parse_premise_proposals(text)
    assert [(row["source_concept"], row["target_concept"]) for row in rows] \
        == [("France", "China"), ("spider", "ant")]
    assert rows[0]["confidence"] == "unstated"
    assert rows[1]["confidence"] == "medium"


def test_parse_premise_proposals_rejects_garbage():
    assert serve._parse_premise_proposals("") == []
    assert serve._parse_premise_proposals("no json here") == []
    assert serve._parse_premise_proposals("{broken json") == []
    assert serve._parse_premise_proposals('{"premises": "not a list"}') == []
    assert serve._parse_premise_proposals('[1, 2, 3]') == []


def test_premise_band_halves_floor():
    assert serve._premise_band_halves(list(range(26, 60))) == [
        list(range(26, 43)), list(range(43, 60)),
    ]
    assert serve._premise_band_halves(list(range(8))) == [
        [0, 1, 2, 3], [4, 5, 6, 7],
    ]
    assert serve._premise_band_halves(list(range(7))) == []


def test_premise_recipe_string_is_stable():
    assert serve._premise_recipe_string(8, 9, [26, 27, 28, 29]) \
        == "premise:8->9:L26-L29:n4"


# ----- endpoint flow ---------------------------------------------------------

_PROPOSAL_JSON = (
    '{"premises":[{"source_concept":"France","target_concept":"China",'
    '"reason":"capital belongs to China","confidence":"high"}]}'
)


class _Scalar:
    def __init__(self, value: int):
        self.value = value

    def tolist(self):
        return self.value


class _Logits:
    def __init__(self, value: int):
        self.value = value

    def astype(self, _dtype):
        return self


class _MX:
    float32 = object()

    @staticmethod
    def argmax(logits: _Logits):
        return _Scalar(logits.value)


class _Tokenizer:
    eos_token_id = 0
    _pieces = {
        1: "Paris",
        2: "Beijing",
        3: " is",
        8: " France",
        9: " China",
        20: _PROPOSAL_JSON,
    }

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {
            "Paris": [1],
            "Beijing": [2],
            "France": [8],
            "China": [9],
        }[text.strip()]

    def decode(self, ids, skip_special_tokens: bool = False):
        del skip_special_tokens
        return "".join(self._pieces.get(int(token_id), "") for token_id in ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if any("Propose the premises now." in m["content"] for m in messages):
            return list(range(30))
        return list(range(12))


class _Session:
    """Pick the scripted output by what this replay is actually doing."""

    def __init__(self, model):
        self.model = model
        self.edits = None
        self.step = 0
        self.sequence = None

    def set_edits(self, edits):
        self.edits = edits

    def extend(self, ids):
        if self.sequence is None:
            if self.edits and any("premise" in edit for edit in self.edits):
                self.sequence = self.model.premise_sequence
            elif self.edits:
                self.sequence = self.model.candidate_sequence
            elif len(ids) == 30:
                self.sequence = self.model.proposal_sequence
            else:
                self.sequence = self.model.baseline_sequence
        token_id = self.sequence[min(self.step, len(self.sequence) - 1)]
        self.step += 1
        return _Logits(token_id), None


class _Model:
    tokenizer = _Tokenizer()
    n_layers = 64

    def __init__(self):
        self.baseline_sequence = [1, 0]           # "Paris"
        self.candidate_sequence = [1, 0]          # unchanged literal singles
        self.proposal_sequence = [20, 0]          # premise JSON
        self.premise_sequence = [2, 3, 0]         # "Beijing is"
        self.sessions: list[_Session] = []

    def make_stream(self):
        session = _Session(self)
        self.sessions.append(session)
        return session


class _Lens:
    source_layers = list(range(15, 22))


def _parse_sse(raw: bytes):
    events = []
    for frame in raw.decode().strip().split("\n\n"):
        lines = frame.splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


async def _run(request):
    response = await serve.intervention_search_adaptive(request)
    raw = b""
    async for chunk in response.body_iterator:
        raw += chunk if isinstance(chunk, bytes) else chunk.encode()
    return _parse_sse(raw)


def _request(**overrides):
    values = {
        "messages": [serve.ChatMessage(role="user", content="Name a city")],
        "token": "Paris",
        "target": "Beijing",
        "replacement_text": "Beijing",
        "baseline_response": "Paris",
        "selected_start": 0,
        "selected_end": 5,
        "position_evidence": [{
            "position": 7,
            "msg_idx": 0,
            "role": "user",
            "token_text": "city",
            "source_hits": [
                {"layer": 17, "rank": 1},
                {"layer": 18, "rank": 2},
            ],
        }],
        "profile": "standard",
        "max_tokens": 16,
        "time_budget_seconds": 60.0,
        "enable_thinking": False,
    }
    values.update(overrides)
    return serve.AdaptiveInterventionSearchRequest(**values)


@pytest.fixture
def premise_env(monkeypatch):
    model = _Model()
    compiled = []

    def fake_compile(_lens, _model, **kwargs):
        if kwargs.get("from_pos") == 0:
            edit = {
                "premise": True,
                "layers": list(kwargs["layers"]),
                "token_id": kwargs["token_id"],
                "target_id": kwargs["target_id"],
            }
        else:
            edit = {
                "position": kwargs["positions"][0],
                "layer": kwargs["layers"][0],
                "alpha": kwargs["alpha"],
            }
        compiled.append(edit)
        return [edit]

    monkeypatch.setattr(serve, "_app_mode", "active")
    monkeypatch.setattr(serve, "_model", model)
    monkeypatch.setattr(serve, "_lens", _Lens())
    monkeypatch.setattr(serve, "_bands", [{
        "name": "workspace", "start_layer": 15, "end_layer": 21,
    }])
    monkeypatch.setattr(serve, "_tok_str_cache", {})
    monkeypatch.setattr(serve, "_gpu_lock", asyncio.Lock())
    monkeypatch.setattr(serve, "mx", _MX)
    monkeypatch.setattr(interventions, "compile_edits", fake_compile)
    return model, compiled


def test_premise_proposal_fires_early_after_first_evidence_batch(premise_env):
    model, compiled = premise_env
    events = asyncio.run(_run(_request(enable_premise_search=True)))
    by_name = {}
    for name, data in events:
        by_name.setdefault(name, []).append(data)

    proposals = by_name.get("premise_proposal") or []
    assert len(proposals) == 1
    assert proposals[0]["status"] == "ok"
    assert proposals[0]["origin"] == "early"

    # Eight literal singles run first, then the premise band replay
    # preempts the remaining literal queue.
    candidates = by_name.get("candidate", [])
    premise_flags = [bool(data.get("premise")) for data in candidates]
    assert premise_flags[:9] == [False] * 8 + [True]
    assert any(not flag for flag in premise_flags[9:])
    assert proposals[0]["directions"] == [{
        "source_concept": "France",
        "target_concept": "China",
        "source_str": " France",
        "target_str": " China",
        "source_token_id": 8,
        "target_token_id": 9,
        "reason": "capital belongs to China",
        "confidence": "high",
    }]

    premise_candidates = [
        data for data in by_name.get("candidate", [])
        if data.get("premise")
    ]
    assert len(premise_candidates) == 1
    redirect = premise_candidates[0]
    assert redirect["class"] == "premise_redirect"
    assert redirect["premise_verified"] is True
    assert redirect["verified"] is False
    assert redirect["text"] == "Beijing is"
    assert redirect["premise"]["layers"] == list(range(15, 22))
    assert redirect["premise"]["from_position"] == 0
    assert redirect["recipe_key"] == "premise:8->9:L15-L21:n7"
    # Every earlier literal single left the reply unchanged.
    literal = [
        data for data in by_name.get("candidate", [])
        if not data.get("premise")
    ]
    assert literal and all(data["class"] == "unchanged" for data in literal)

    end = by_name["search_end"][-1]
    assert end["status"] == "no_verified_recipe"
    assert end["premise"]["attempted"] is True
    assert end["premise"]["proposal_status"] == "ok"
    assert end["premise"]["redirects"] == 1
    assert end["premise"]["replays"] == 1
    assert end["premise"]["best"]["text"] == "Beijing is"
    assert end["premise"]["best"]["layers"] == list(range(15, 22))
    # A seven-layer band is below the bisection floor: no halves were queued.
    premise_edits = [edit for edit in compiled if edit.get("premise")]
    assert len(premise_edits) == 1


def test_premise_proposal_falls_back_to_exhaustion_on_a_small_queue(premise_env):
    events = asyncio.run(_run(_request(
        enable_premise_search=True, position_evidence=[],
    )))
    by_name = {}
    for name, data in events:
        by_name.setdefault(name, []).append(data)

    proposals = by_name.get("premise_proposal") or []
    assert len(proposals) == 1
    assert proposals[0]["origin"] == "exhausted"
    literal = [
        data for data in by_name.get("candidate", [])
        if not data.get("premise")
    ]
    assert 0 < len(literal) < 8
    end = by_name["search_end"][-1]
    assert end["premise"]["redirects"] == 1


def test_search_continues_after_a_verified_recipe(premise_env):
    model, _ = premise_env
    # Every intervened replay says the desired reply, so every literal
    # candidate verifies.
    model.candidate_sequence = [2, 0]
    events = asyncio.run(_run(_request(enable_premise_search=True)))
    by_name = {}
    for name, data in events:
        by_name.setdefault(name, []).append(data)

    candidates = by_name.get("candidate", [])
    verified = [data for data in candidates if data["verified"]]
    assert len(candidates) > 1
    assert len(verified) > 1

    end = by_name["search_end"][-1]
    assert end["status"] == "success"
    assert end["stop_reason"] == "exhausted_candidates"
    assert end["verified_count"] == len(verified)
    assert end["verified_candidate_id"] == verified[0]["id"]
    assert end["verified_cells"] == verified[0]["cells"]
    # A verified literal recipe suppresses the premise proposal.
    assert "premise_proposal" not in by_name
    assert end["premise"]["attempted"] is False


def test_stop_on_verified_ends_at_the_first_recipe(premise_env):
    model, _ = premise_env
    model.candidate_sequence = [2, 0]
    events = asyncio.run(_run(_request(stop_on_verified=True)))
    candidates = [data for name, data in events if name == "candidate"]
    assert len(candidates) == 1
    assert candidates[0]["verified"] is True
    end = [data for name, data in events if name == "search_end"][-1]
    assert end["status"] == "success"
    assert end["stop_reason"] == "verified"
    assert end["verified_count"] == 1


def test_premise_stage_is_off_by_default(premise_env):
    events = asyncio.run(_run(_request()))
    names = {name for name, _ in events}
    assert "premise_proposal" not in names
    end = [data for name, data in events if name == "search_end"][-1]
    assert end["premise"]["enabled"] is False
    assert end["premise"]["attempted"] is False
    assert end["premise"]["replays"] == 0
