"""Fast contract tests for the resident-model planner probe (no 27B load)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi import HTTPException

import jlens_qwen.serve as serve


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
    argmax_threads: list[int] = []

    @classmethod
    def argmax(cls, logits: _Logits):
        cls.argmax_threads.append(threading.get_ident())
        return _Scalar(logits.value)


class _BatchEncoding:
    """A transformers-v5-like mapping that is not a ``dict`` subclass."""

    def __init__(self, input_ids):
        self._data = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class _Tokenizer:
    eos_token_id = 0

    def __init__(self, input_ids=None):
        self.input_ids = list(input_ids or [10, 11, 12, 13])
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs, threading.get_ident()))
        return _BatchEncoding(self.input_ids)

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join({1: "A", 2: "B"}.get(int(token_id), "")
                       for token_id in ids)


class _Session:
    def __init__(self, values=None, delay=0.0):
        self.values = list(values or [99, 1, 2, 0])
        self.delay = delay
        self.calls: list[list[int]] = []
        self.threads: list[int] = []

    def extend(self, ids):
        self.calls.append(list(ids))
        self.threads.append(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)
        return _Logits(self.values[len(self.calls) - 1]), {}


class _Model:
    def __init__(self, tokenizer=None, session_factory=None):
        self.tokenizer = tokenizer or _Tokenizer()
        self.session_factory = session_factory or _Session
        self.sessions = []
        self.make_stream_threads = []

    def make_stream(self):
        self.make_stream_threads.append(threading.get_ident())
        session = self.session_factory()
        self.sessions.append(session)
        return session


class _Lens:
    n_prompts = 1000


def _request(**overrides):
    values = {
        "messages": [
            serve.ChatMessage(role="system", content="Return JSON."),
            serve.ChatMessage(role="user", content="Find causal cells."),
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "enable_thinking": False,
        "prefill_chunk_size": 2,
        "time_budget_seconds": 5.0,
        "max_input_tokens": 32,
    }
    values.update(overrides)
    return serve.PlannerProbeRequest(**values)


@pytest.fixture
def probe_env(monkeypatch):
    model = _Model()
    _MX.argmax_threads = []
    monkeypatch.setattr(serve, "_app_mode", "active")
    monkeypatch.setattr(serve, "_model", model)
    monkeypatch.setattr(serve, "_model_id", "fake/resident-model")
    monkeypatch.setattr(serve, "_lens", _Lens())
    monkeypatch.setattr(serve, "_gpu_lock", asyncio.Lock())
    monkeypatch.setattr(serve, "_PLANNER_CONTEXT_LIMIT", 128)
    monkeypatch.setattr(serve, "mx", _MX)

    def readout_must_not_run(*_args, **_kwargs):
        raise AssertionError("planner probe must not compute J-space readout")

    monkeypatch.setattr(serve, "_readout_at_positions", readout_must_not_run)
    return model


def test_probe_uses_full_chat_template_bare_stream_and_reports_timing(probe_env):
    main_thread = threading.get_ident()
    result = asyncio.run(serve.planner_probe(_request()))

    assert result["text"] == "AB"
    assert result["token_ids"] == [1, 2]
    assert result["input_tokens"] == 4
    assert result["input_tokens_processed"] == 4
    assert result["output_tokens"] == 2
    assert result["prefill_chunks"] == 2
    # One step predicts B and a second predicts EOS.
    assert result["decode_steps"] == 2
    assert result["stop_reason"] == "eos"
    assert result["lens_readout"] is False
    assert result["model_id"] == "fake/resident-model"
    assert result["lens_n_prompts"] == 1000
    assert result["ttft_ms"] is not None
    for field in (
        "tokenize_ms", "queue_wait_ms", "session_setup_ms", "prefill_ms",
        "decode_ms", "total_service_ms", "total_request_ms",
        "prefill_tokens_per_second", "decode_tokens_per_second",
    ):
        assert result[field] >= 0, field

    assert len(probe_env.sessions) == 1
    assert probe_env.sessions[0].calls == [[10, 11], [12, 13], [1], [2]]
    # make_stream() accepts no capture argument, and all MLX-bearing calls
    # execute on workers rather than the event-loop thread.
    assert probe_env.make_stream_threads == [probe_env.make_stream_threads[0]]
    assert probe_env.make_stream_threads[0] != main_thread
    assert all(tid != main_thread for tid in probe_env.sessions[0].threads)
    assert _MX.argmax_threads
    assert all(tid != main_thread for tid in _MX.argmax_threads)

    messages, kwargs, template_thread = probe_env.tokenizer.template_calls[0]
    assert messages == [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Find causal cells."},
    ]
    assert kwargs == {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert template_thread != main_thread


def test_probe_rejects_overlong_input_without_truncating_or_touching_gpu(
    probe_env,
):
    probe_env.tokenizer.input_ids = list(range(9))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(serve.planner_probe(_request(max_input_tokens=8)))

    assert exc_info.value.status_code == 413
    assert "9 tokens" in exc_info.value.detail
    assert "not truncated" in exc_info.value.detail
    assert probe_env.sessions == []


def test_probe_enforces_output_cap_before_touching_gpu(probe_env):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(serve.planner_probe(_request(max_tokens=257)))

    assert exc_info.value.status_code == 400
    assert "[1, 256]" in exc_info.value.detail
    assert probe_env.sessions == []


def test_probe_stops_chunked_prefill_after_cooperative_time_budget(
    probe_env,
):
    probe_env.session_factory = lambda: _Session(
        values=[99, 1, 2, 0], delay=0.15
    )

    result = asyncio.run(serve.planner_probe(_request(
        time_budget_seconds=0.1,
    )))

    assert result["stop_reason"] == "time_budget"
    assert result["input_tokens_processed"] == 2
    assert result["input_tokens_processed"] < result["input_tokens"]
    assert result["prefill_chunks"] == 1
    assert result["output_tokens"] == 0
    assert result["ttft_ms"] is None


def test_cancellation_keeps_gpu_lock_until_worker_finishes(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class _BlockingSession:
        def extend(self, _ids):
            started.set()
            assert release.wait(timeout=2.0)
            return _Logits(0), {}

    model = _Model(
        tokenizer=_Tokenizer([10]),
        session_factory=_BlockingSession,
    )
    _MX.argmax_threads = []
    monkeypatch.setattr(serve, "_app_mode", "active")
    monkeypatch.setattr(serve, "_model", model)
    monkeypatch.setattr(serve, "_lens", _Lens())
    monkeypatch.setattr(serve, "_gpu_lock", asyncio.Lock())
    monkeypatch.setattr(serve, "_PLANNER_CONTEXT_LIMIT", 128)
    monkeypatch.setattr(serve, "mx", _MX)

    async def exercise():
        first = asyncio.create_task(serve.planner_probe(_request(
            max_tokens=1,
            prefill_chunk_size=1,
        )))
        assert await asyncio.to_thread(started.wait, 1.0)

        first.cancel()
        await asyncio.sleep(0.02)
        assert serve._gpu_lock.locked()

        second = asyncio.create_task(serve.planner_probe(_request(
            max_tokens=1,
            prefill_chunk_size=1,
        )))
        await asyncio.sleep(0.02)
        # The second request has tokenized, but cannot create a model session
        # while the cancelled request's worker is still on the GPU.
        assert len(model.sessions) == 1

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        second_result = await second
        assert second_result["stop_reason"] == "eos"
        assert len(model.sessions) == 2

    asyncio.run(exercise())
