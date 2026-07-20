"""Fast endpoint/SSE contract tests for intervention scans (no model)."""

from __future__ import annotations

import asyncio
import json

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
    _pieces = {1: "Paris", 2: "New", 3: " York"}

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
        del messages, kwargs
        return [99]


class _Session:
    def __init__(self):
        self.edits = None
        self.step = 0

    def set_edits(self, edits):
        self.edits = edits

    def extend(self, _ids):
        alpha = None if self.edits is None else self.edits["alpha"]
        # The weaker probe emits only the first token of the multi-token goal.
        # The stronger probe emits the complete goal.
        sequence = {
            None: [1, 0],
            0.5: [2, 0],
            1.0: [2, 3, 0],
        }[alpha]
        token_id = sequence[min(self.step, len(sequence) - 1)]
        self.step += 1
        return _Logits(token_id), None


class _Model:
    tokenizer = _Tokenizer()
    n_layers = 64

    def make_stream(self):
        return _Session()


class _Lens:
    source_layers = [15, 21]


def _parse_sse(raw: bytes):
    events = []
    for frame in raw.decode().strip().split("\n\n"):
        lines = frame.splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


def test_scan_sse_order_and_full_goal_classification(monkeypatch):
    compiled = []

    def fake_compile(_lens, _model, **kwargs):
        compiled.append({
            "layer": kwargs["layers"][0],
            "alpha": kwargs["alpha"],
            "positions": kwargs["positions"],
            "from_pos": kwargs["from_pos"],
            "token_id": kwargs["token_id"],
            "target_id": kwargs["target_id"],
        })
        return {"alpha": kwargs["alpha"]}

    monkeypatch.setattr(serve, "_app_mode", "active")
    monkeypatch.setattr(serve, "_model", _Model())
    monkeypatch.setattr(serve, "_lens", _Lens())
    monkeypatch.setattr(serve, "_bands", [])
    monkeypatch.setattr(serve, "_tok_str_cache", {})
    monkeypatch.setattr(serve, "mx", _MX)
    monkeypatch.setattr(interventions, "compile_edits", fake_compile)

    async def run_scan():
        request = serve.InterventionScanRequest(
            messages=[serve.ChatMessage(role="user", content="Name a city")],
            mode="swap",
            token="Paris",
            target="New York",
            goal_text="New York",
            layers=[15, 21],
            alphas=[0.5, 1.0],
            from_position=0,
            max_tokens=3,
        )
        response = await serve.intervention_scan(request)
        raw = b""
        async for chunk in response.body_iterator:
            raw += chunk if isinstance(chunk, bytes) else chunk.encode()
        return _parse_sse(raw)

    events = asyncio.run(run_scan())

    assert [event for event, _ in events] == [
        "scan_start",
        "probe",
        "probe",
        "probe",
        "probe",
        "probe",
        "scan_end",
    ]
    start = events[0][1]
    assert start["token_id"] == 1
    assert start["target_str"] == "New"
    assert start["target_id"] == 2
    assert start["goal_text"] == "New York"
    assert start["positions"] is None
    assert start["from_position"] == 0
    assert start["n_probes"] == 4
    assert events[1][1] == {
        "kind": "baseline", "text": "Paris",
        "eos": True, "stop_reason": "eos",
    }

    probes = [
        data
        for event, data in events
        if event == "probe" and data["kind"] == "probe"
    ]
    assert [(probe["layer"], probe["alpha"]) for probe in probes] == [
        (15, 0.5),
        (21, 0.5),
        (15, 1.0),
        (21, 1.0),
    ]
    assert [probe["index"] for probe in probes] == [1, 2, 3, 4]
    assert {probe["total"] for probe in probes} == {4}

    first_token_only = [probe for probe in probes if probe["text"] == "New"]
    full_goal = [probe for probe in probes if probe["text"] == "New York"]
    assert [probe["cls"] for probe in first_token_only] == ["derived", "derived"]
    assert [probe["success"] for probe in first_token_only] == [False, False]
    assert [probe["cls"] for probe in full_goal] == ["success", "success"]
    assert [probe["success"] for probe in full_goal] == [True, True]
    assert all(probe["eos"] for probe in probes)
    assert {probe["stop_reason"] for probe in probes} == {"eos"}
    assert compiled == [
        {"layer": 15, "alpha": 0.5, "positions": None, "from_pos": 0,
         "token_id": 1, "target_id": 2},
        {"layer": 21, "alpha": 0.5, "positions": None, "from_pos": 0,
         "token_id": 1, "target_id": 2},
        {"layer": 15, "alpha": 1.0, "positions": None, "from_pos": 0,
         "token_id": 1, "target_id": 2},
        {"layer": 21, "alpha": 1.0, "positions": None, "from_pos": 0,
         "token_id": 1, "target_id": 2},
    ]


def test_goal_success_is_bounded_unique_source_free_and_terminated():
    classify = serve._classify_probe

    assert classify(
        [9, 2, 3, 8], "The answer is New York.", "Paris",
        "New", eos=True, goal_text="New York", source_text="Paris",
    ) == "success"
    assert classify(
        [2, 3], "New York", "Paris",
        "New", eos=False, goal_text="New York", source_text="Paris",
    ) == "parrot"
    assert classify(
        [2, 3, 4], "New York today", "Paris",
        "New", eos=True, goal_text="New York", source_text="Paris",
    ) == "success"
    assert classify(
        [2, 3, 1], "Paris New York", "Paris",
        "New", eos=True, goal_text="New York", source_text="Paris",
    ) == "derived"
    assert classify(
        [2, 3, 1], "New YorkParis", "Paris",
        "New", eos=True, goal_text="New York", source_text="Paris",
    ) == "derived"
    assert classify(
        [9, 8, 7, 6, 5], "The answer is New York. Wait,", "Paris",
        "New", eos=False, goal_text="New York", source_text="Paris",
    ) == "parrot"
    assert classify(
        [9, 2, 3, 8, 2, 3, 7], "New York, then New York.", "Paris",
        "New", eos=True, goal_text="New York", source_text="Paris",
    ) == "parrot"
    assert classify(
        [9, 8, 7], "The answer changed to Boston.", "Paris",
        "New", eos=False, goal_text="New York", source_text="Paris",
    ) == "derived"
    assert classify(
        [1, 2, 3, 8], "Paris changed to New York.", "Paris",
        "New", eos=False, goal_text="New York", source_text="Paris",
    ) == "derived"
    assert classify(
        [2, 3, 2, 3, 2, 3], "New York New York New York", "Paris",
        "New", eos=True,
        goal_text="New York New York New York", source_text="Paris",
    ) == "degenerate"


def test_scan_exact_positions_are_authoritative(monkeypatch):
    compiled = []

    def fake_compile(_lens, _model, **kwargs):
        compiled.append(kwargs)
        return {"alpha": kwargs["alpha"]}

    monkeypatch.setattr(serve, "_app_mode", "active")
    monkeypatch.setattr(serve, "_model", _Model())
    monkeypatch.setattr(serve, "_lens", _Lens())
    monkeypatch.setattr(serve, "_bands", [])
    monkeypatch.setattr(serve, "_tok_str_cache", {})
    monkeypatch.setattr(serve, "mx", _MX)
    monkeypatch.setattr(interventions, "compile_edits", fake_compile)

    async def run_scan():
        request = serve.InterventionScanRequest(
            messages=[serve.ChatMessage(role="user", content="Name a city")],
            mode="swap", token="Paris", target="New York",
            goal_text="New York", source_text="Paris",
            layers=[15], alphas=[1.0], positions=[42, 9, 42], max_tokens=3,
        )
        response = await serve.intervention_scan(request)
        raw = b""
        async for chunk in response.body_iterator:
            raw += chunk if isinstance(chunk, bytes) else chunk.encode()
        return _parse_sse(raw)

    events = asyncio.run(run_scan())
    start = events[0][1]
    assert start["positions"] == [9, 42]
    assert start["from_position"] is None
    assert compiled[0]["positions"] == [9, 42]
    assert compiled[0]["from_pos"] is None


def test_scan_rejects_ambiguous_position_scope(monkeypatch):
    monkeypatch.setattr(serve, "_app_mode", "active")
    monkeypatch.setattr(serve, "_model", _Model())
    monkeypatch.setattr(serve, "_lens", _Lens())

    request = serve.InterventionScanRequest(
        messages=[serve.ChatMessage(role="user", content="Name a city")],
        mode="swap", token="Paris", target="New York",
        positions=[4], from_position=0,
    )
    with pytest.raises(serve.HTTPException, match="exactly one"):
        asyncio.run(serve.intervention_scan(request))


def test_intervention_repetition_breaker_is_conservative():
    repeats = serve._intervention_repetition

    assert not repeats([1, 2] * 5)                 # fewer than 12 tokens
    assert repeats([1, 2] * 6)                    # six two-token cycles
    assert repeats([1, 2, 3, 4] * 3)              # three four-token cycles
    assert repeats([1] * 12)                      # one-token collapse
    assert not repeats(list(range(16)))            # healthy diverse tail
    assert not repeats([1, 2, 3, 4, 5, 6] * 2)    # no <=4-token cycle
