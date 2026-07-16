"""Adaptive intervention-search controller tests (no model or GPU)."""

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
    _pieces = {
        1: "Paris",
        2: "Beijing",
        3: " is",
        4: " wrong",
        5: "London",
        6: "🙂",
        7: "<display-special>",
    }

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {
            "Paris": [1],
            "Beijing": [2],
        }[text.strip()]

    def decode(self, ids, skip_special_tokens: bool = False):
        return "".join(
            self._pieces.get(int(token_id), "")
            for token_id in ids
            if not (skip_special_tokens and int(token_id) == 7)
        )

    def apply_chat_template(self, messages, **kwargs):
        del messages
        assert kwargs == {
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return list(range(12))


class _Session:
    def __init__(self, candidate_sequence: list[int], baseline_sequence: list[int]):
        self.edits = None
        self.step = 0
        self.candidate_sequence = candidate_sequence
        self.baseline_sequence = baseline_sequence

    def set_edits(self, edits):
        self.edits = edits

    def extend(self, _ids):
        sequence = (
            self.candidate_sequence if self.edits else self.baseline_sequence
        )
        token_id = sequence[min(self.step, len(sequence) - 1)]
        self.step += 1
        return _Logits(token_id), None


class _Model:
    tokenizer = _Tokenizer()
    n_layers = 64

    def __init__(
        self,
        *,
        baseline_sequence: list[int] | None = None,
        candidate_sequence: list[int] | None = None,
    ):
        self.baseline_sequence = baseline_sequence or [1, 0]
        self.candidate_sequence = candidate_sequence or [2, 0]
        self.sessions: list[_Session] = []

    def make_stream(self):
        session = _Session(self.candidate_sequence, self.baseline_sequence)
        self.sessions.append(session)
        return session


class _RecipeSession:
    """Choose output by exact recipe size for scheduler-focused tests."""

    def __init__(self, sequences_by_cell_count):
        self.sequences_by_cell_count = sequences_by_cell_count
        self.edits = None
        self.step = 0

    def set_edits(self, edits):
        self.edits = edits

    def extend(self, _ids):
        sequence = self.sequences_by_cell_count[len(self.edits or [])]
        token_id = sequence[min(self.step, len(sequence) - 1)]
        self.step += 1
        return _Logits(token_id), None


class _RecipeModel(_Model):
    def __init__(self, sequences_by_cell_count):
        super().__init__()
        self.sequences_by_cell_count = sequences_by_cell_count

    def make_stream(self):
        session = _RecipeSession(self.sequences_by_cell_count)
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


async def _next_event(iterator):
    chunk = await iterator.__anext__()
    encoded = chunk if isinstance(chunk, bytes) else chunk.encode()
    parsed = _parse_sse(encoded)
    assert len(parsed) == 1
    return parsed[0]


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
def adaptive_env(monkeypatch):
    model = _Model()
    compiled = []

    def fake_compile(_lens, _model, **kwargs):
        cell = {
            "position": kwargs["positions"][0],
            "layer": kwargs["layers"][0],
            "alpha": kwargs["alpha"],
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


def test_position_ranking_prefers_coherent_source_evidence_to_frontier():
    rows = [serve.AdaptivePositionEvidence(
        position=7,
        msg_idx=0,
        role="user",
        token_text="France",
        source_hits=[
            serve.AdaptiveSearchHit(layer=17, rank=1),
            serve.AdaptiveSearchHit(layer=18, rank=2),
        ],
    )]

    ranked = serve._adaptive_rank_positions(
        rows,
        frontier=11,
        workspace_layers=list(range(15, 22)),
        selected_source="Paris",
    )

    assert [row["position"] for row in ranked[:2]] == [7, 11]
    assert ranked[0]["reason"] == "coherent_source_readout"
    assert ranked[0]["longest_layer_run"] == 2
    assert ranked[0]["top1_count"] == 1
    assert ranked[1]["reason"] == "causal_frontier"


def test_position_ranking_keeps_recent_template_prefix_ahead_of_context():
    rows = [
        serve.AdaptivePositionEvidence(
            position=position,
            msg_idx=0,
            role="user",
            token_text=f"user-{position}",
        )
        for position in range(3, 10)
    ] + [
        serve.AdaptivePositionEvidence(
            position=10,
            msg_idx=None,
            role="template",
            token_text="<assistant-prefix>",
        )
    ]

    ranked = serve._adaptive_rank_positions(
        rows,
        frontier=11,
        workspace_layers=list(range(15, 22)),
        selected_source="Paris",
    )

    assert [row["position"] for row in ranked[:2]] == [11, 10]
    assert ranked[1]["reason"] == "causal_template_prefix"


def test_endpoint_accepts_message_less_template_evidence(adaptive_env):
    del adaptive_env

    events = asyncio.run(_run(_request(position_evidence=[{
        "position": 10,
        "msg_idx": None,
        "role": "template",
        "token_text": "<assistant-prefix>",
        "source_hits": [],
    }])))

    assert events[0][0] == "search_start"
    assert not any(name == "error" for name, _ in events)


def test_initial_singles_are_round_robin_across_ranked_positions():
    ranked = [
        {"position": 7, "hit_layers": [17]},
        {"position": 11, "hit_layers": []},
    ]

    candidates = serve._adaptive_initial_candidates(
        ranked, list(range(15, 22))
    )

    assert [candidate["cells"][0]["position"] for candidate in candidates[:2]] == [
        7, 11,
    ]
    assert candidates[0]["cells"] == [{
        "position": 7, "layer": 17, "alpha": 1.0,
    }]
    assert all(
        0 < cell["alpha"] <= 1
        for candidate in candidates
        for cell in candidate["cells"]
    )


def test_deep_singles_cover_every_workspace_layer_and_two_strengths():
    ranked = [{"position": 7}, {"position": 11}]

    candidates = serve._adaptive_deep_single_candidates(
        ranked, [15, 16, 17]
    )

    assert len(candidates) == 12
    assert all(candidate["stage"] == "deep_single" for candidate in candidates)
    assert [candidate["cells"][0] for candidate in candidates[:4]] == [
        {"position": 7, "layer": 17, "alpha": 1.0},
        {"position": 11, "layer": 17, "alpha": 1.0},
        {"position": 7, "layer": 16, "alpha": 1.0},
        {"position": 11, "layer": 16, "alpha": 1.0},
    ]
    assert {candidate["cells"][0]["alpha"] for candidate in candidates} == {
        0.5, 1.0,
    }


def test_refinement_pairs_and_triples_preserve_exact_cells():
    parent = serve._adaptive_candidate("coarse_single", [{
        "position": 7, "layer": 18, "alpha": 1.0,
    }])
    refinements = serve._adaptive_refinement_candidates(
        parent, list(range(15, 22))
    )
    assert refinements[0]["cells"] == [{
        "position": 7, "layer": 18, "alpha": 0.5,
    }]
    assert {candidate["cells"][0]["layer"] for candidate in refinements[1:]} == {
        16, 17, 19, 20,
    }

    other = serve._adaptive_candidate("coarse_single", [{
        "position": 7, "layer": 19, "alpha": 1.0,
    }])
    pairs = serve._adaptive_pair_candidates(parent, [parent, other])
    assert len(pairs) == 1
    assert len(pairs[0]["cells"]) == 2
    assert {cell["alpha"] for cell in pairs[0]["cells"]} == {0.5}

    third = serve._adaptive_candidate("coarse_single", [{
        "position": 7, "layer": 20, "alpha": 1.0,
    }])
    triples = serve._adaptive_triple_candidates(pairs[0], [parent, other, third])
    assert len(triples) == 1
    assert len(triples[0]["cells"]) == 3


def test_endpoint_ranks_then_verifies_exact_response(adaptive_env):
    model, compiled = adaptive_env

    events = asyncio.run(_run(_request()))

    assert [name for name, _ in events] == [
        "search_start",
        "position_ranking",
        "baseline",
        "stage",
        "candidate_start",
        "candidate",
        "search_end",
    ]
    ranking = events[1][1]
    assert ranking["positions"][0]["position"] == 7
    assert ranking["positions"][0]["reason"] == "coherent_source_readout"
    baseline = events[2][1]
    assert baseline["matches_displayed"] is True
    candidate_start = events[4][1]
    candidate = events[5][1]
    assert candidate_start["id"] == candidate["id"]
    assert candidate_start["stage"] == candidate["stage"]
    assert candidate_start["cells"] == candidate["cells"]
    assert candidate_start["index"] == 1
    assert candidate_start["budget_remaining_seconds"] > 0
    assert candidate["cells"] == [{
        "position": 7, "layer": 17, "alpha": 1.0,
    }]
    assert candidate["exact_response_match"] is True
    assert candidate["verified"] is True
    assert events[-1][1]["status"] == "success"
    assert events[-1][1]["verified_cells"] == candidate["cells"]
    assert compiled == candidate["cells"]
    assert len(model.sessions) == 2


def test_selected_offsets_are_unicode_code_points_before_emoji_span(
    adaptive_env, monkeypatch,
):
    del adaptive_env
    model = _Model(
        baseline_sequence=[6, 1, 0],
        candidate_sequence=[6, 2, 0],
    )
    monkeypatch.setattr(serve, "_model", model)
    request = _request(
        baseline_response="🙂Paris",
        selected_start=1,
        selected_end=6,
    )

    events = asyncio.run(_run(request))

    start = events[0][1]
    assert start["selected_source"] == "Paris"
    assert start["desired_response"] == "🙂Beijing"
    assert events[-1][1]["status"] == "success"


def test_excluded_recipe_key_is_not_replayed(adaptive_env):
    _, compiled = adaptive_env
    excluded = serve._adaptive_recipe_string([{
        "position": 7, "layer": 17, "alpha": 1.0,
    }])

    events = asyncio.run(_run(_request(exclude_recipe_keys=[excluded])))

    candidate = next(data for name, data in events if name == "candidate")
    assert candidate["recipe_key"] != excluded
    assert compiled[0] != {"position": 7, "layer": 17, "alpha": 1.0}


def test_thorough_search_seeds_pairs_from_excluded_promising_singles(
    adaptive_env,
):
    model, compiled = adaptive_env
    request = _request(profile="thorough", time_budget_seconds=180.0)
    ranked = serve._adaptive_rank_positions(
        request.position_evidence,
        frontier=11,
        workspace_layers=list(range(15, 22)),
        selected_source="Paris",
    )
    excluded = [
        serve._adaptive_recipe_string(candidate["cells"])
        for candidate in serve._adaptive_initial_candidates(
            ranked, list(range(15, 22))
        )
    ]
    prior = [
        {
            "cells": [{"position": 7, "layer": layer, "alpha": 1.0}],
            "similarity_to_desired": score,
        }
        for layer, score in ((17, 0.72), (18, 0.68))
    ]

    events = asyncio.run(_run(_request(
        profile="thorough",
        time_budget_seconds=180.0,
        exclude_recipe_keys=excluded,
        prior_promising=prior,
    )))

    candidates = [data for name, data in events if name == "candidate"]
    assert len(candidates) == 1
    assert candidates[0]["stage"] == "pair"
    assert candidates[0]["verified"] is True
    assert len(candidates[0]["cells"]) == 2
    assert {cell["alpha"] for cell in candidates[0]["cells"]} == {0.5}
    assert compiled == candidates[0]["cells"]
    assert all(len(session.edits or []) != 1 for session in model.sessions)
    assert events[0][1]["prior_promising_count"] == 2
    assert events[-1][1]["stage_counts"]["prior_promising_singles"] == 2


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({
            "profile": "thorough",
            "time_budget_seconds": 180.0,
            "exclude_recipe_keys": ["p7:l17:a1"],
            "prior_promising": [{
                "cells": [{"position": 7, "layer": 17, "alpha": 1.0}],
                "similarity_to_desired": 0.7,
            }] * 17,
        }, "limited to 16 singles"),
        ({
            "prior_promising": [{
                "cells": [{"position": 7, "layer": 17, "alpha": 1.0}],
                "similarity_to_desired": 0.7,
            }],
        }, "accepted only by the thorough profile"),
        ({
            "profile": "thorough",
            "time_budget_seconds": 180.0,
            "exclude_recipe_keys": ["p7:l17:a1|p7:l18:a1"],
            "prior_promising": [{
                "cells": [
                    {"position": 7, "layer": 17, "alpha": 1.0},
                    {"position": 7, "layer": 18, "alpha": 1.0},
                ],
                "similarity_to_desired": 0.7,
            }],
        }, "exactly one cell"),
        ({
            "profile": "thorough",
            "time_budget_seconds": 180.0,
            "exclude_recipe_keys": ["p7:l14:a1"],
            "prior_promising": [{
                "cells": [{"position": 7, "layer": 14, "alpha": 1.0}],
                "similarity_to_desired": 0.7,
            }],
        }, "outside the measured workspace"),
        ({
            "profile": "thorough",
            "time_budget_seconds": 180.0,
            "exclude_recipe_keys": ["p7:l17:a1"],
            "prior_promising": [{
                "cells": [{"position": 7, "layer": 17, "alpha": 1.0}],
                "similarity_to_desired": 1.1,
            }],
        }, "finite and in \\[0, 1\\]"),
        ({
            "profile": "thorough",
            "time_budget_seconds": 180.0,
            "prior_promising": [{
                "cells": [{"position": 7, "layer": 17, "alpha": 1.0}],
                "similarity_to_desired": 0.7,
            }],
        }, "also appear in exclude_recipe_keys"),
    ],
)
def test_endpoint_rejects_invalid_prior_promising(
    adaptive_env, overrides, detail,
):
    model, _ = adaptive_env

    with pytest.raises(serve.HTTPException, match=detail):
        asyncio.run(serve.intervention_search_adaptive(_request(**overrides)))

    assert model.sessions == []


def test_baseline_mismatch_ends_without_testing_candidates(
    adaptive_env, monkeypatch,
):
    del adaptive_env
    model = _Model(baseline_sequence=[5, 0])
    monkeypatch.setattr(serve, "_model", model)

    events = asyncio.run(_run(_request()))

    assert [name for name, _ in events] == [
        "search_start", "position_ranking", "baseline", "search_end",
    ]
    assert events[2][1]["matches_displayed"] is False
    assert events[-1][1]["status"] == "baseline_mismatch"
    assert events[-1][1]["tested"] == 0
    assert len(model.sessions) == 1


def test_baseline_acceptance_is_raw_display_equality(
    adaptive_env,
):
    model, _ = adaptive_env

    events = asyncio.run(_run(_request(
        baseline_response="PARIS",
        selected_start=0,
        selected_end=5,
    )))

    baseline = next(data for name, data in events if name == "baseline")
    assert baseline["text"] == "Paris"
    assert baseline["matches_displayed"] is False
    assert events[-1][1]["status"] == "baseline_mismatch"
    assert len(model.sessions) == 1


def test_candidate_acceptance_is_raw_desired_response_equality(
    adaptive_env, monkeypatch,
):
    model, _ = adaptive_env
    only = serve._adaptive_candidate("refinement_single", [{
        "position": 7, "layer": 17, "alpha": 1.0,
    }])
    monkeypatch.setattr(
        serve, "_adaptive_initial_candidates", lambda *_args: [only]
    )

    events = asyncio.run(_run(_request(replacement_text="BEIJING")))

    candidates = [data for name, data in events if name == "candidate"]
    assert len(candidates) == 1
    assert candidates[0]["text"] == "Beijing"
    assert candidates[0]["exact_response_match"] is False
    assert candidates[0]["verified"] is False
    assert events[-1][1]["status"] == "no_verified_recipe"
    assert len(model.sessions) == 2


def test_replay_decode_matches_display_special_token_behavior(
    adaptive_env, monkeypatch,
):
    del adaptive_env
    model = _Model(
        baseline_sequence=[7, 1, 0],
        candidate_sequence=[7, 2, 0],
    )
    monkeypatch.setattr(serve, "_model", model)
    prefix = "<display-special>"

    events = asyncio.run(_run(_request(
        baseline_response=prefix + "Paris",
        selected_start=len(prefix),
        selected_end=len(prefix) + len("Paris"),
    )))

    baseline = next(data for name, data in events if name == "baseline")
    assert baseline["text"] == prefix + "Paris"
    assert baseline["matches_displayed"] is True
    candidate = next(data for name, data in events if name == "candidate")
    assert candidate["text"] == prefix + "Beijing"
    assert candidate["verified"] is True


@pytest.mark.parametrize("replacement", [
    "Paris",
    " PARIS ",
    "Ｐａｒｉｓ",
])
def test_equivalent_replacement_is_rejected_as_noop(
    adaptive_env, replacement,
):
    model, _ = adaptive_env

    with pytest.raises(
        serve.HTTPException,
        match="replacement_text is equivalent to the selected response span",
    ):
        asyncio.run(serve.intervention_search_adaptive(_request(
            replacement_text=replacement,
        )))

    assert model.sessions == []


def test_target_phrase_inside_wrong_sentence_is_never_verified(
    adaptive_env, monkeypatch,
):
    del adaptive_env
    model = _Model(candidate_sequence=[2, 3, 4, 0])
    monkeypatch.setattr(serve, "_model", model)

    events = asyncio.run(_run(_request()))

    candidates = [data for name, data in events if name == "candidate"]
    assert candidates
    assert candidates[0]["text"] == "Beijing is wrong"
    assert candidates[0]["class"] == "partial"
    assert candidates[0]["exact_response_match"] is False
    assert all(candidate["verified"] is False for candidate in candidates)
    assert events[-1][1]["status"] == "no_verified_recipe"
    assert events[-1][1]["verified_cells"] is None


def test_deadline_expires_while_baseline_waits_for_gpu(
    adaptive_env, monkeypatch,
):
    model, _ = adaptive_env

    async def scenario():
        lock = asyncio.Lock()
        await lock.acquire()
        monkeypatch.setattr(serve, "_gpu_lock", lock)
        task = asyncio.create_task(_run(_request(time_budget_seconds=0.02)))
        await asyncio.sleep(0.04)
        lock.release()
        return await task

    events = asyncio.run(scenario())

    assert [name for name, _ in events] == [
        "search_start", "position_ranking", "search_end",
    ]
    ending = events[-1][1]
    assert ending["status"] == "budget_exhausted"
    assert ending["tested"] == 0
    assert ending["deadline_queue_wait_ms"] >= 10
    assert ending["elapsed_seconds"] >= 0.02
    assert model.sessions == []


def test_deadline_expires_while_candidate_waits_without_starting_replay(
    adaptive_env, monkeypatch,
):
    model, _ = adaptive_env

    async def scenario():
        lock = asyncio.Lock()
        monkeypatch.setattr(serve, "_gpu_lock", lock)
        response = await serve.intervention_search_adaptive(
            _request(time_budget_seconds=0.1)
        )
        iterator = response.body_iterator
        events = []

        # Search start, ranking, and the completed baseline replay.
        for _ in range(3):
            chunk = await iterator.__anext__()
            encoded = chunk if isinstance(chunk, bytes) else chunk.encode()
            events.extend(_parse_sse(encoded))

        await lock.acquire()
        try:
            async for chunk in iterator:
                encoded = chunk if isinstance(chunk, bytes) else chunk.encode()
                events.extend(_parse_sse(encoded))
        finally:
            lock.release()
        return events

    events = asyncio.run(scenario())

    assert [name for name, _ in events][-3:] == [
        "stage", "candidate_start", "search_end",
    ]
    ending = events[-1][1]
    assert ending["status"] == "budget_exhausted"
    assert ending["tested"] == 0
    assert ending["deadline_queue_wait_ms"] > 0
    # Only the baseline session was created; queued candidate work never ran.
    assert len(model.sessions) == 1


def test_live_search_pause_resume_keeps_one_stream_and_active_clock(
    adaptive_env,
):
    model, _ = adaptive_env

    async def scenario():
        response = await serve.intervention_search_adaptive(_request(
            allow_continuation=True,
        ))
        iterator = response.body_iterator
        events = [await _next_event(iterator)]
        search_id = events[0][1]["search_id"]
        assert search_id in serve._adaptive_search_controls

        pause = await serve.intervention_search_adaptive_control(
            serve.AdaptiveInterventionSearchControlRequest(
                search_id=search_id,
                action="pause",
            )
        )
        assert pause["pause_requested"] is True
        events.append(await _next_event(iterator))  # position ranking
        events.append(await _next_event(iterator))  # acknowledged pause
        paused = events[-1][1]
        assert paused["reason"] == "user"
        assert paused["paused"] is True
        before = paused["elapsed_active_seconds"]

        await asyncio.sleep(0.02)
        still_paused = await serve.intervention_search_adaptive_control(
            serve.AdaptiveInterventionSearchControlRequest(
                search_id=search_id,
                action="pause",
            )
        )
        assert still_paused["elapsed_active_seconds"] == pytest.approx(
            before, abs=0.003
        )
        await serve.intervention_search_adaptive_control(
            serve.AdaptiveInterventionSearchControlRequest(
                search_id=search_id,
                action="resume",
            )
        )
        async for chunk in iterator:
            encoded = chunk if isinstance(chunk, bytes) else chunk.encode()
            events.extend(_parse_sse(encoded))
        return search_id, events

    search_id, events = asyncio.run(scenario())

    names = [name for name, _ in events]
    assert names[:4] == [
        "search_start", "position_ranking", "search_paused", "search_resumed",
    ]
    assert names.count("baseline") == 1
    assert names.count("candidate") == 1
    assert events[-1][1]["status"] == "success"
    assert len(model.sessions) == 2
    assert search_id not in serve._adaptive_search_controls


def test_live_pause_interrupts_gpu_queue_without_starting_or_losing_work(
    adaptive_env, monkeypatch,
):
    model, _ = adaptive_env

    async def scenario():
        lock = asyncio.Lock()
        await lock.acquire()
        monkeypatch.setattr(serve, "_gpu_lock", lock)
        response = await serve.intervention_search_adaptive(_request(
            allow_continuation=True,
        ))
        iterator = response.body_iterator
        start = await _next_event(iterator)
        await _next_event(iterator)  # position ranking

        waiting = asyncio.create_task(_next_event(iterator))
        await asyncio.sleep(0)
        await serve.intervention_search_adaptive_control(
            serve.AdaptiveInterventionSearchControlRequest(
                search_id=start[1]["search_id"],
                action="pause",
            )
        )
        paused = await asyncio.wait_for(waiting, timeout=0.1)
        assert paused[0] == "search_paused"
        assert model.sessions == []
        # This is still the lock held by the simulated competing request; the
        # interrupted acquire did not leak a second ownership.
        assert lock.locked()
        lock.release()

        await serve.intervention_search_adaptive_control(
            serve.AdaptiveInterventionSearchControlRequest(
                search_id=start[1]["search_id"],
                action="resume",
            )
        )
        events = []
        async for chunk in iterator:
            encoded = chunk if isinstance(chunk, bytes) else chunk.encode()
            events.extend(_parse_sse(encoded))
        return events, lock

    events, lock = asyncio.run(scenario())
    assert events[0][0] == "search_resumed"
    assert sum(name == "baseline" for name, _ in events) == 1
    assert events[-1][1]["status"] == "success"
    assert len(model.sessions) == 2
    assert not lock.locked()


def test_initial_phase_extends_same_scheduler_without_recipe_replay(
    adaptive_env, monkeypatch,
):
    del adaptive_env
    model = _RecipeModel({
        0: [1, 0],
        1: [2, 4, 0],       # improving single: "Beijing wrong"
        2: [2, 3, 0],       # improving pair: "Beijing is"
        3: [2, 0],          # exact triple
    })
    monkeypatch.setattr(serve, "_model", model)

    async def scenario():
        response = await serve.intervention_search_adaptive(_request(
            allow_continuation=True,
        ))
        iterator = response.body_iterator
        events = []
        while True:
            event = await _next_event(iterator)
            events.append(event)
            if event[0] == "search_paused":
                break
        paused = events[-1][1]
        assert paused["reason"] == "initial_candidates_exhausted"
        assert paused["elapsed_active_seconds"] == pytest.approx(60.0)
        assert paused["time_budget_seconds"] == 60.0
        assert paused["awaiting_extension"] is True

        control = await serve.intervention_search_adaptive_control(
            serve.AdaptiveInterventionSearchControlRequest(
                search_id=paused["search_id"],
                action="extend",
                additional_time_seconds=120.0,
            )
        )
        assert control["time_budget_seconds"] == 180.0
        assert control["budget_remaining_seconds"] == pytest.approx(120.0)
        async for chunk in iterator:
            encoded = chunk if isinstance(chunk, bytes) else chunk.encode()
            events.extend(_parse_sse(encoded))
        return events

    events = asyncio.run(scenario())

    paused_index = next(
        index for index, (name, _) in enumerate(events)
        if name == "search_paused"
    )
    before = {
        data["recipe_key"]
        for name, data in events[:paused_index]
        if name == "candidate"
    }
    after = {
        data["recipe_key"]
        for name, data in events[paused_index + 1:]
        if name == "candidate"
    }
    assert before
    assert after
    assert before.isdisjoint(after)
    assert sum(name == "baseline" for name, _ in events) == 1
    assert any(
        name == "search_resumed"
        and data["reason"] == "extended"
        and data["added_time_seconds"] == 120.0
        for name, data in events
    )
    assert events[-1][1]["status"] == "success"
    verified = next(
        data for name, data in events
        if name == "candidate" and data["verified"]
    )
    assert verified["stage"] == "triple"
    assert len(model.sessions) == 1 + len(before) + len(after)


def test_live_search_disconnect_discards_continuation(adaptive_env):
    del adaptive_env

    async def scenario():
        response = await serve.intervention_search_adaptive(_request(
            allow_continuation=True,
        ))
        iterator = response.body_iterator
        start = await _next_event(iterator)
        search_id = start[1]["search_id"]
        assert search_id in serve._adaptive_search_controls
        await iterator.aclose()
        return search_id

    search_id = asyncio.run(scenario())
    assert search_id not in serve._adaptive_search_controls


def test_pairs_are_interleaved_before_single_queues_drain(
    adaptive_env, monkeypatch,
):
    del adaptive_env
    model = _RecipeModel({
        0: [1, 0],
        1: [2, 3, 4, 0],
        2: [2, 0],
    })
    monkeypatch.setattr(serve, "_model", model)

    events = asyncio.run(_run(_request()))

    candidates = [data for name, data in events if name == "candidate"]
    assert candidates[0]["stage"] == "evidence_single"
    assert all(
        len(candidate["cells"]) == 1 for candidate in candidates[:-1]
    )
    pair = candidates[-1]
    assert pair["stage"] == "pair"
    assert pair["verified"] is True
    # Once a compatible second single is found, the pair gets the next
    # reserved combination slot rather than waiting for the sweep to finish.
    assert pair["index"] <= 6
    assert pair["progress"]["initial_queued"] > 0
    assert events[-1][1]["tested"] == pair["index"]


def test_thorough_triple_runs_before_remaining_seeded_pairs(
    adaptive_env, monkeypatch,
):
    del adaptive_env
    model = _RecipeModel({
        0: [1, 0],
        1: [2, 3, 4, 0],
        2: [2, 3, 4, 0],
        3: [2, 0],
    })
    monkeypatch.setattr(serve, "_model", model)
    cells = [
        {"position": 7, "layer": layer, "alpha": 1.0}
        for layer in (17, 18, 19)
    ]
    excluded = [
        serve._adaptive_recipe_string([cell]) for cell in cells
    ]
    prior = [
        {"cells": [cell], "similarity_to_desired": 0.1}
        for cell in cells
    ]

    events = asyncio.run(_run(_request(
        profile="thorough",
        time_budget_seconds=180.0,
        exclude_recipe_keys=excluded,
        prior_promising=prior,
    )))

    candidates = [data for name, data in events if name == "candidate"]
    assert [candidate["stage"] for candidate in candidates[:2]] == [
        "pair", "triple",
    ]
    assert candidates[0]["progress"]["pairs_queued"] > 0
    assert candidates[1]["verified"] is True
    assert len(candidates[1]["cells"]) == 3


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"profile": "standard", "time_budget_seconds": 60.1}, r"\[0.001, 60\]"),
        ({"profile": "thorough", "time_budget_seconds": 180.1}, r"\[0.001, 180\]"),
        ({"enable_thinking": True}, "enable_thinking=false"),
        ({"selected_start": 4, "selected_end": 8}, "outside baseline_response"),
        ({"position_evidence": [{
            "position": 12, "role": "user", "token_text": "x", "source_hits": [],
        }]}, "outside the prefill frontier"),
        ({"position_evidence": [{
            "position": 7, "role": "user", "token_text": "x",
            "source_hits": [{"layer": 17, "rank": 0}],
        }]}, "rank must be in"),
    ],
)
def test_endpoint_rejects_invalid_adaptive_contract(
    adaptive_env, overrides, detail,
):
    model, _ = adaptive_env

    with pytest.raises(serve.HTTPException, match=detail):
        asyncio.run(serve.intervention_search_adaptive(_request(**overrides)))

    assert model.sessions == []
