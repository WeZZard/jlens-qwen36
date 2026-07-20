from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO / "data/benchmarks/jspace_contradiction_prompts.json"
SCRIPT_PATH = REPO / "scripts/benchmark_planner_latency.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_planner_latency", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contradiction_corpus_has_unique_valid_cases() -> None:
    corpus = json.loads(CORPUS_PATH.read_text())
    cases = corpus["cases"]
    ids = [case["id"] for case in cases]

    assert corpus["schema_version"] == 1
    assert len(cases) >= 6
    assert len(ids) == len(set(ids))
    assert set(corpus["timing_suite"]).issubset(ids)

    for case in cases:
        conversation = case["conversation"]
        requested = case["requested_edit"]
        assert requested["source_span"] in conversation["assistant"]
        assert requested["source_span"] != requested["replacement_span"]
        assert case["provenance"]["url"].startswith("https://transformer-circuits.pub/")
        assert case["oracle"]


def test_planner_payload_never_leaks_evaluation_oracle() -> None:
    benchmark = _load_benchmark_module()
    case = json.loads(CORPUS_PATH.read_text())["cases"][0]
    payload = benchmark._planner_payload(case, {
        "tokens": [{
            "position": 0,
            "token_id": 123,
            "region": "assistant_reply",
            "token": "Paris",
        }],
        "selected_reply_positions": [12],
        "eligible_position_max": 11,
    })

    assert "oracle" not in payload
    assert case["oracle"]["workspace_concept_to"] not in payload
    assert case["requested_edit"]["replacement_span"] in payload


def test_article_extractor_ignores_non_visible_code() -> None:
    benchmark = _load_benchmark_module()
    parser = benchmark._ArticleText()
    parser.feed("<main>Hello <b>world</b><script>secret()</script></main>")

    assert parser.text() == "Hello\nworld"


def test_summary_accepts_a_budget_stop_before_first_token() -> None:
    benchmark = _load_benchmark_module()
    summary = benchmark._summarize([
        {
            "case_id": "country_capital",
            "variant": "paper",
            "outer_http_ms": 60_100.0,
            "json_parseable": False,
            "contract_valid": False,
            "server": {
                "total_service_ms": 60_000.0,
                "prefill_ms": 59_900.0,
                "ttft_ms": None,
                "input_tokens": 69_000,
                "input_tokens_processed": 6_000,
                "prefill_tokens_per_second": 100.0,
                "output_tokens": 0,
                "stop_reason": "time_budget",
            },
        }
    ])

    assert summary["paper"]["median_ttft_ms"] is None
    assert summary["paper"]["responses"] == 1
    assert summary["paper"]["budget_stops"] == 1
    assert summary["paper"]["median_input_tokens_processed"] == 6_000
    assert summary["paper"]["stop_reasons"] == ["time_budget"]


def test_candidate_validator_enforces_workspace_and_causal_cutoff() -> None:
    benchmark = _load_benchmark_module()
    valid = json.dumps({
        "candidates": [{
            "cells": [{"layer": 40, "position": 11}],
            "source_concept": "France",
            "target_concept": "China",
            "reason": "Earlier country concept may control the answer.",
            "confidence": "low",
        }]
    })
    invalid = json.dumps({
        "candidates": [{
            "cells": [{"layer": 60, "position": 12}],
            "source_concept": "France",
            "target_concept": "China",
            "reason": "Targets the emitted reply.",
            "confidence": "certain",
        }]
    })

    assert benchmark._validate_candidate_output(
        valid,
        workspace_start=26,
        workspace_end=59,
        eligible_position_max=11,
    )["contract_valid"]
    result = benchmark._validate_candidate_output(
        invalid,
        workspace_start=26,
        workspace_end=59,
        eligible_position_max=11,
    )
    assert result["json_parseable"]
    assert not result["contract_valid"]
    assert len(result["contract_errors"]) == 3
