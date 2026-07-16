"""Bounded intervention recipe evaluator tests (no model or GPU)."""

from __future__ import annotations

import asyncio
import json
import time

import jlens_qwen.interventions as interventions
import jlens_qwen.serve as serve
import pytest


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
        2: "New",
        3: " York",
        4: "London",
        5: "x",
    }

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {
            "Paris": [1],
            "New York": [2, 3],
        }[text.strip()]

    def decode(self, ids, skip_special_tokens: bool = False):
        del skip_special_tokens
        return "".join(self._pieces.get(int(token_id), "") for token_id in ids)

    def apply_chat_template(self, messages, **kwargs):
        del messages
        assert kwargs == {
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return list(range(12))


class _Session:
    def __init__(self, *, candidate_delay: float = 0.0):
        self.edits = None
        self.step = 0
        self.candidate_delay = candidate_delay

    def set_edits(self, edits):
        self.edits = edits

    def _sequence(self):
        if self.edits is None:
            return [1, 0]
        alphas = tuple(edit["alpha"] for edit in self.edits)
        if alphas == (0.5,):
            return [4, 0]
        if alphas == (1.0, 2.0):
            return [2, 3, 0]
        if alphas == (9.0,):
            return [5] * 128
        return [4, 0]

    def extend(self, _ids):
        if self.edits and self.candidate_delay:
            time.sleep(self.candidate_delay)
        sequence = self._sequence()
        token_id = sequence[min(self.step, len(sequence) - 1)]
        self.step += 1
        return _Logits(token_id), None


class _Model:
    tokenizer = _Tokenizer()
    n_layers = 64

    def __init__(self, *, candidate_delay: float = 0.0):
        self.sessions = []
        self.candidate_delay = candidate_delay

    def make_stream(self):
        session = _Session(candidate_delay=self.candidate_delay)
        self.sessions.append(session)
        return session


class _Lens:
    source_layers = [15, 21, 40]


def _parse_sse(raw: bytes):
    events = []
    for frame in raw.decode().strip().split("\n\n"):
        lines = frame.splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


def _request(**overrides):
    values = {
        "messages": [serve.ChatMessage(role="user", content="Name a city")],
        "token": "Paris",
        "target": "New York",
        "source_text": "Paris",
        "goal_text": "New York",
        "candidates": [
            {
                "id": "wrong",
                "cells": [{"layer": 15, "position": 7, "alpha": 0.5}],
            },
            {
                "id": "pair",
                "cells": [
                    {"layer": 15, "position": 7, "alpha": 1.0},
                    {"layer": 21, "position": 9, "alpha": 2.0},
                ],
            },
            {
                "id": "not-reached",
                "cells": [{"layer": 21, "position": 9, "alpha": 0.5}],
            },
        ],
        "max_tokens": 16,
        "time_budget_seconds": 60.0,
        "enable_thinking": False,
        "stop_on_success": True,
    }
    values.update(overrides)
    return serve.InterventionSearchRequest(**values)


@pytest.fixture
def search_env(monkeypatch):
    model = _Model()
    compiled = []

    def fake_compile(_lens, _model, **kwargs):
        cell = {
            "layer": kwargs["layers"][0],
            "position": kwargs["positions"][0],
            "alpha": kwargs["alpha"],
            "token_id": kwargs["token_id"],
            "target_id": kwargs["target_id"],
        }
        compiled.append(cell)
        return [cell]

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


async def _run(request):
    response = await serve.intervention_search(request)
    raw = b""
    async for chunk in response.body_iterator:
        raw += chunk if isinstance(chunk, bytes) else chunk.encode()
    return _parse_sse(raw)


def test_search_replays_explicit_recipes_and_stops_on_verified_pair(search_env):
    model, compiled = search_env

    events = asyncio.run(_run(_request()))

    assert [event for event, _ in events] == [
        "search_start", "baseline", "candidate", "candidate", "search_end",
    ]
    start = events[0][1]
    assert start["token_id"] == 1
    assert start["target_id"] == 2
    assert start["workspace"] == {
        "start_layer": 15,
        "end_layer": 21,
        "layers": [15, 21],
    }
    assert start["input_tokens"] == 12

    baseline = events[1][1]
    assert baseline["text"] == "Paris"
    assert baseline["eos"] is True
    assert baseline["stop_reason"] == "eos"
    assert baseline["expected_replacement_text"] == "new york"
    assert set(baseline["timings_ms"]) == {
        "queue_wait", "setup", "prefill", "decode", "total",
    }

    wrong, pair = events[2][1], events[3][1]
    assert wrong["id"] == "wrong"
    assert wrong["class"] == "off_target"
    assert wrong["verified"] is False
    assert pair["id"] == "pair"
    assert pair["text"] == "New York"
    assert pair["class"] == "verified"
    assert pair["verified"] is True
    assert pair["goal_count"] == 1
    assert pair["source_count"] == 0
    assert pair["exact_replacement_applicable"] is True
    assert pair["exact_replacement_match"] is True
    assert set(pair["timings_ms"]) == {
        "queue_wait", "compile", "setup", "prefill", "decode", "total",
    }

    end = events[-1][1]
    assert end["status"] == "success"
    assert end["stop_reason"] == "verified"
    assert end["tested"] == 2
    assert end["total"] == 3
    assert end["verified_candidate_id"] == "pair"
    assert end["stopped_early"] is True

    # One fresh bare session for the baseline and each tested candidate.
    assert len(model.sessions) == 3
    # Pair cells compile independently, preserving exact pair semantics. The
    # third candidate is never compiled because stop_on_success is enabled.
    assert [(cell["layer"], cell["position"], cell["alpha"])
            for cell in compiled] == [
        (15, 7, 0.5),
        (15, 7, 1.0),
        (21, 9, 2.0),
    ]
    assert {cell["token_id"] for cell in compiled} == {1}
    assert {cell["target_id"] for cell in compiled} == {2}


@pytest.mark.parametrize(
    ("override", "detail"),
    [
        ({"candidates": []}, "1 to 256"),
        ({"candidates": [{"id": "bad", "cells": []}]}, "1 to 8"),
        ({"candidates": [{"id": "bad", "cells": [
            {"layer": 15, "position": -1, "alpha": 1.0},
        ]}]}, "position must be >= 0"),
        ({"candidates": [{"id": "bad", "cells": [
            {"layer": 40, "position": 1, "alpha": 1.0},
        ]}]}, "outside the measured workspace"),
        ({"max_tokens": 129}, "[1, 128]"),
        ({"time_budget_seconds": 301}, "[0.001, 300]"),
    ],
)
def test_search_rejects_invalid_bounds_before_gpu(search_env, override, detail):
    model, _ = search_env

    with pytest.raises(serve.HTTPException, match=detail):
        asyncio.run(serve.intervention_search(_request(**override)))

    assert model.sessions == []


def test_search_requires_measured_workspace_metadata(search_env, monkeypatch):
    model, _ = search_env
    monkeypatch.setattr(serve, "_bands", None)

    with pytest.raises(serve.HTTPException, match="measured workspace band"):
        asyncio.run(serve.intervention_search(_request()))

    assert model.sessions == []


def test_search_stops_repetitive_candidate_before_max_tokens(search_env):
    model, _ = search_env
    request = _request(
        candidates=[{
            "id": "loop",
            "cells": [{"layer": 15, "position": 7, "alpha": 9.0}],
        }],
        max_tokens=128,
    )

    events = asyncio.run(_run(request))

    candidate = next(data for event, data in events if event == "candidate")
    assert candidate["class"] == "repetition"
    assert candidate["verified"] is False
    assert candidate["repetition"] is True
    assert candidate["stop_reason"] == "repetition"
    assert len(candidate["text"]) == 12
    assert len(model.sessions) == 2


def test_search_budget_is_checked_between_atomic_candidates(
    search_env, monkeypatch,
):
    del search_env
    model = _Model(candidate_delay=0.08)
    monkeypatch.setattr(serve, "_model", model)
    request = _request(
        candidates=[
            {"id": "slow", "cells": [
                {"layer": 15, "position": 7, "alpha": 0.5},
            ]},
            {"id": "not-started", "cells": [
                {"layer": 21, "position": 9, "alpha": 0.5},
            ]},
        ],
        max_tokens=2,
        time_budget_seconds=0.05,
        stop_on_success=False,
    )

    events = asyncio.run(_run(request))

    candidates = [data for event, data in events if event == "candidate"]
    assert [candidate["id"] for candidate in candidates] == ["slow"]
    end = events[-1][1]
    assert end["status"] == "budget_exhausted"
    assert end["stop_reason"] == "time_budget"
    assert end["tested"] == 1
    assert end["budget_exhausted"] is True
    assert end["budget_overshoot_seconds"] > 0
    assert end["stopped_early"] is True
    assert len(model.sessions) == 2
