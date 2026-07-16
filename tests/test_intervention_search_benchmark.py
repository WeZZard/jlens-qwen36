from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/benchmark_intervention_search.py"
CORPUS = REPO / "data/benchmarks/jspace_contradiction_prompts.json"


def _module():
    spec = importlib.util.spec_from_file_location("benchmark_intervention_search", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tokenizer():
    return AutoTokenizer.from_pretrained(
        "mlx-community/Qwen3.6-27B-4bit",
        local_files_only=True,
    )


def test_core_cases_have_unique_oracle_search_anchors() -> None:
    corpus = json.loads(CORPUS.read_text())
    core = [case for case in corpus["cases"] if case["tier"] == "core"]

    for case in core:
        anchor = case["oracle"]["workspace_anchor"]
        assert case["conversation"]["user"].count(anchor) == 1


def test_literal_position_order_uses_frontier_without_oracle_anchor() -> None:
    benchmark = _module()
    case = json.loads(CORPUS.read_text())["cases"][0]
    context = benchmark._position_context(_tokenizer(), case, "literal")

    assert context["position_priority"][0] == context["frontier_position"]
    assert context["anchor_positions"] == []
    assert all(
        position <= context["frontier_position"]
        for position in context["position_priority"]
    )


def test_oracle_position_order_starts_at_declared_anchor() -> None:
    benchmark = _module()
    case = json.loads(CORPUS.read_text())["cases"][0]
    context = benchmark._position_context(_tokenizer(), case, "oracle")

    assert context["anchor_positions"] == [12]
    assert context["position_priority"][0] == 12


def test_recipe_candidates_use_exact_cells_and_measured_workspace() -> None:
    benchmark = _module()
    candidates = benchmark._candidate_recipes(
        positions=[25, 12],
        workspace_start=26,
        workspace_end=59,
        recipe_sizes=[1, 2, 3],
        max_candidates=256,
    )

    assert candidates
    assert len(candidates) <= 256
    assert {len(candidate["cells"]) for candidate in candidates} == {1, 2, 3}
    assert len({candidate["id"] for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        assert 1 <= len(candidate["cells"]) <= 3
        for cell in candidate["cells"]:
            assert cell["position"] in {12, 25}
            assert 26 <= cell["layer"] <= 59
            assert cell["alpha"] in {0.5, 1.0}
