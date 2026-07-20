#!/usr/bin/env python3
"""Benchmark wall-clock-bounded, lens-free intervention recipe replay.

The live server evaluates explicit recipes through ``/api/intervention_search``.
Each recipe is one or more exact ``(position, layer, alpha)`` cells and receives
a fresh deterministic model session.  This script supplies an anytime ordering;
the endpoint owns the deadline and never cancels an in-flight MLX call.

Examples:

    uv run python scripts/benchmark_intervention_search.py \
      --cases country_capital,spider_ant,bandit_repeat_switch \
      --direction literal --budget-seconds 60 --stop-on-success

    uv run python scripts/benchmark_intervention_search.py \
      --cases spider_ant --direction literal --budget-seconds 180 \
      --recipe-sizes 1,2,3
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "data/benchmarks/jspace_contradiction_prompts.json"


def _http_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach {url}: {error.reason}") from error


def _post_sse(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    on_event=None,
) -> tuple[list[dict[str, Any]], float]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "jlens-qwen36-search-benchmark/1",
        },
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    try:
        with urlopen(request, timeout=timeout) as response:
            event_name = "message"
            data_lines: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if data_lines:
                        data = json.loads("\n".join(data_lines))
                        item = {
                            "event": event_name,
                            "data": data,
                            "outer_elapsed_seconds": time.perf_counter() - started,
                        }
                        events.append(item)
                        if on_event is not None:
                            on_event(item)
                    event_name = "message"
                    data_lines = []
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach {url}: {error.reason}") from error
    return events, time.perf_counter() - started


def _template_ids(tokenizer: Any, messages: list[dict[str, str]], **kwargs) -> list[int]:
    result = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(result, "keys") and "input_ids" in result:
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()
    if (isinstance(result, (list, tuple)) and len(result) == 1
            and isinstance(result[0], (list, tuple))):
        result = result[0]
    return [int(token_id) for token_id in result]


def _token_char_ranges(tokenizer: Any, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    decoded_prefixes = [""]
    for index in range(len(ids)):
        decoded_prefixes.append(tokenizer.decode(ids[: index + 1]))
    return decoded_prefixes[-1], [
        (len(decoded_prefixes[index]), len(decoded_prefixes[index + 1]))
        for index in range(len(ids))
    ]


def _position_context(
    tokenizer: Any,
    case: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    user_text = case["conversation"]["user"]
    messages = [{"role": "user", "content": user_text}]
    input_ids = _template_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text, ranges = _token_char_ranges(tokenizer, input_ids)
    user_start = full_text.find(user_text)
    if user_start < 0:
        raise ValueError(f"{case['id']}: user content not found in chat template")
    user_end = user_start + len(user_text)
    content_positions = [
        position
        for position, (start, end) in enumerate(ranges)
        if start < user_end and end > user_start
    ]
    if not content_positions:
        raise ValueError(f"{case['id']}: user content has no token positions")

    frontier = len(input_ids) - 1
    anchor_positions: list[int] = []
    if direction == "oracle":
        anchor = str(case["oracle"].get("workspace_anchor") or "")
        relative = user_text.find(anchor)
        if not anchor or relative < 0 or user_text.find(anchor, relative + 1) >= 0:
            raise ValueError(
                f"{case['id']}: oracle anchor must occur exactly once in user message"
            )
        anchor_start = user_start + relative
        anchor_end = anchor_start + len(anchor)
        anchor_positions = [
            position
            for position, (start, end) in enumerate(ranges)
            if start < anchor_end and end > anchor_start
        ]

    if direction == "oracle":
        raw_priority = (
            list(reversed(anchor_positions))
            + [position - 1 for position in anchor_positions]
            + [position + 1 for position in anchor_positions]
            + [frontier]
            + list(reversed(content_positions))
        )
    else:
        # No oracle concepts are used in production-visible literal mode.
        # Begin at the causal frontier, then walk semantic prompt tokens back.
        raw_priority = [frontier] + list(reversed(content_positions))
    priorities: list[int] = []
    for position in raw_priority:
        if 0 <= position <= frontier and position not in priorities:
            priorities.append(position)
    return {
        "messages": messages,
        "input_ids": input_ids,
        "frontier_position": frontier,
        "content_positions": content_positions,
        "anchor_positions": anchor_positions,
        "position_priority": priorities,
    }


def _direction(case: dict[str, Any], direction: str) -> tuple[str, str]:
    if direction == "literal":
        return (
            case["requested_edit"]["source_span"],
            case["requested_edit"]["replacement_span"],
        )
    if direction != "oracle":
        raise ValueError(f"unknown direction: {direction}")
    source = str(case["oracle"]["workspace_concept_from"])
    target = str(case["oracle"]["workspace_concept_to"])
    if "/" in source or "/" in target:
        raise ValueError(f"{case['id']}: oracle concept is not one token-like direction")
    # English concepts occur at internal word boundaries in the paper-style
    # swaps. Preserve that tokenizer boundary without changing CJK concepts.
    if case.get("language") == "en":
        source = " " + source.lstrip()
        target = " " + target.lstrip()
    return source, target


def _coarse_layers(workspace_start: int, workspace_end: int) -> list[int]:
    span = workspace_end - workspace_start
    ascending: list[int] = []
    for index in range(7):
        layer = round(workspace_start + index * span / 6)
        if layer not in ascending:
            ascending.append(layer)
    # Late-first is conservative for a literal reply edit and also reaches
    # the full measured band early. Fine layers follow after the coarse pass.
    return list(reversed(ascending))


def _candidate_recipes(
    *,
    positions: list[int],
    workspace_start: int,
    workspace_end: int,
    recipe_sizes: list[int],
    max_candidates: int,
) -> list[dict[str, Any]]:
    coarse = _coarse_layers(workspace_start, workspace_end)
    fine = [
        layer for layer in range(workspace_end, workspace_start - 1, -1)
        if layer not in coarse
    ]
    layer_order = coarse + fine
    candidates: list[dict[str, Any]] = []

    def append(candidate_id: str, cells: list[dict[str, Any]]) -> None:
        if len(candidates) >= max_candidates:
            return
        candidates.append({"id": candidate_id, "cells": cells})

    if 1 in recipe_sizes:
        for position in positions:
            for alpha in (1.0, 0.5):
                for layer in layer_order:
                    append(
                        f"s1-p{position}-l{layer}-a{alpha:g}",
                        [{"position": position, "layer": layer, "alpha": alpha}],
                    )

    # Static pair/triple batches measure recipe-size cost. A future adaptive
    # controller should build these only from non-degenerate promising singles.
    if 2 in recipe_sizes:
        adjacent_pairs: list[tuple[int, int]] = []
        for pair in list(zip(coarse, coarse[1:], strict=False)) + [
            (layer, layer - 1)
            for layer in range(workspace_end, workspace_start, -1)
        ]:
            if pair not in adjacent_pairs:
                adjacent_pairs.append(pair)
        for position in positions:
            for left, right in adjacent_pairs:
                append(
                    f"s2-p{position}-l{left}+l{right}-a0.5",
                    [
                        {"position": position, "layer": left, "alpha": 0.5},
                        {"position": position, "layer": right, "alpha": 0.5},
                    ],
                )
        if len(positions) >= 2:
            first, second = positions[:2]
            for layer in coarse:
                append(
                    f"s2-p{first}+p{second}-l{layer}-a0.5",
                    [
                        {"position": first, "layer": layer, "alpha": 0.5},
                        {"position": second, "layer": layer, "alpha": 0.5},
                    ],
                )

    if 3 in recipe_sizes:
        layer_triples: list[list[int]] = []
        for index in range(len(coarse) - 2):
            layer_triples.append(coarse[index:index + 3])
        for high in range(workspace_end, workspace_start + 1, -1):
            triple = [high, high - 1, high - 2]
            if triple not in layer_triples:
                layer_triples.append(triple)
        for position in positions:
            for layers in layer_triples:
                append(
                    f"s3-p{position}-l{'+l'.join(map(str, layers))}-a0.5",
                    [
                        {"position": position, "layer": layer, "alpha": 0.5}
                        for layer in layers
                    ],
                )
    return candidates


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summarize_events(events: list[dict[str, Any]], outer_seconds: float) -> dict[str, Any]:
    candidates = [item["data"] for item in events if item["event"] == "candidate"]
    end = next((item["data"] for item in reversed(events) if item["event"] == "search_end"), {})
    total_ms = [float(item["timings_ms"]["total"]) for item in candidates]
    recipe_sizes = Counter(len(item["cells"]) for item in candidates)
    return {
        "outer_seconds": outer_seconds,
        "tested": len(candidates),
        "candidates_per_minute": (
            60.0 * len(candidates) / outer_seconds if outer_seconds > 0 else 0.0
        ),
        "median_candidate_ms": statistics.median(total_ms) if total_ms else None,
        "p95_candidate_ms": _percentile(total_ms, 0.95),
        "classes": dict(Counter(item["class"] for item in candidates)),
        "stop_reasons": dict(Counter(item["stop_reason"] for item in candidates)),
        "recipe_sizes_tested": {str(key): value for key, value in sorted(recipe_sizes.items())},
        "verified_ids": [item["id"] for item in candidates if item["verified"]],
        "search_end": end,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cases", default="country_capital,spider_ant,bandit_repeat_switch")
    parser.add_argument("--direction", choices=("literal", "oracle"), default="literal")
    parser.add_argument("--budget-seconds", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--recipe-sizes", default="1")
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--position-limit", type=int, default=8)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--require-lens-prompts", type=int, default=1000)
    parser.add_argument("--request-timeout-seconds", type=float, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not 0.001 <= args.budget_seconds <= 300:
        raise SystemExit("--budget-seconds must be in [0.001, 300]")
    if not 1 <= args.max_candidates <= 256:
        raise SystemExit("--max-candidates must be in [1, 256]")
    if args.position_limit < 1:
        raise SystemExit("--position-limit must be at least 1")
    recipe_sizes = [int(value) for value in args.recipe_sizes.split(",") if value.strip()]
    if not recipe_sizes or any(size not in (1, 2, 3) for size in recipe_sizes):
        raise SystemExit("--recipe-sizes must be a comma-separated subset of 1,2,3")

    corpus = json.loads(args.corpus.read_text())
    cases_by_id = {case["id"]: case for case in corpus["cases"]}
    case_ids = [value.strip() for value in args.cases.split(",") if value.strip()]
    missing = sorted(set(case_ids) - set(cases_by_id))
    if missing:
        raise SystemExit(f"unknown case IDs: {', '.join(missing)}")

    base_url = args.base_url.rstrip("/")
    model_info = _http_json(f"{base_url}/api/model")
    lens_info = _http_json(f"{base_url}/api/lens")
    if model_info.get("mode") != "active":
        raise SystemExit(f"server is not active: {model_info}")
    if args.require_lens_prompts and lens_info.get("n_prompts") != args.require_lens_prompts:
        raise SystemExit(
            f"expected n{args.require_lens_prompts} lens configuration; "
            f"server reports n{lens_info.get('n_prompts')}"
        )
    workspace = next(
        (band for band in lens_info.get("bands", []) if band.get("name") == "workspace"),
        None,
    )
    if workspace is None:
        raise SystemExit("server did not report a measured workspace band")
    workspace_start = int(workspace["start_layer"])
    workspace_end = int(workspace["end_layer"])

    setup_started = time.perf_counter()
    model_id = args.model_id or model_info["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    setup_seconds = time.perf_counter() - setup_started
    timeout = args.request_timeout_seconds or (args.budget_seconds + 90.0)

    print(
        f"model={model_id} lens-config=n{lens_info.get('n_prompts')} "
        f"workspace=L{workspace_start}-L{workspace_end} direction={args.direction} "
        f"budget={args.budget_seconds:g}s cases={len(case_ids)}",
        flush=True,
    )
    runs: list[dict[str, Any]] = []
    for case_id in case_ids:
        case = cases_by_id[case_id]
        position_context = _position_context(tokenizer, case, args.direction)
        positions = position_context["position_priority"][: args.position_limit]
        candidates = _candidate_recipes(
            positions=positions,
            workspace_start=workspace_start,
            workspace_end=workspace_end,
            recipe_sizes=recipe_sizes,
            max_candidates=args.max_candidates,
        )
        token, target = _direction(case, args.direction)
        payload = {
            "messages": position_context["messages"],
            "token": token,
            "target": target,
            "source_text": case["requested_edit"]["source_span"],
            "goal_text": case["requested_edit"]["replacement_span"],
            "candidates": candidates,
            "max_tokens": args.max_tokens,
            "time_budget_seconds": args.budget_seconds,
            "enable_thinking": False,
            "stop_on_success": args.stop_on_success,
        }

        def show(item: dict[str, Any]) -> None:
            data = item["data"]
            if item["event"] == "baseline":
                print(
                    f"{case_id:24s} baseline={data['text']!r} "
                    f"stop={data['stop_reason']}",
                    flush=True,
                )
            elif item["event"] == "candidate" and (
                data["verified"] or data["index"] == 1 or data["index"] % 10 == 0
            ):
                print(
                    f"{case_id:24s} {data['index']:3d}/{data['total']} "
                    f"{data['id']} class={data['class']} "
                    f"elapsed={data['elapsed_seconds']:.1f}s",
                    flush=True,
                )

        events, outer_seconds = _post_sse(
            f"{base_url}/api/intervention_search",
            payload,
            timeout=timeout,
            on_event=show,
        )
        summary = _summarize_events(events, outer_seconds)
        baseline = next(
            (item["data"] for item in events if item["event"] == "baseline"),
            None,
        )
        run = {
            "case_id": case_id,
            "direction": args.direction,
            "direction_token": token,
            "direction_target": target,
            "positions": positions,
            "frontier_position": position_context["frontier_position"],
            "anchor_positions": position_context["anchor_positions"],
            "candidate_count_submitted": len(candidates),
            "recipe_sizes": recipe_sizes,
            "baseline": baseline,
            "events": events,
            "summary": summary,
        }
        runs.append(run)
        print(
            f"{case_id:24s} tested={summary['tested']} "
            f"rate={summary['candidates_per_minute']:.1f}/min "
            f"verified={summary['verified_ids']} "
            f"end={summary['search_end'].get('stop_reason')}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "server": {"model": model_info, "lens": lens_info},
        "benchmark": {
            "scope": "lens-free deterministic full recipe replay; visualization excluded",
            "corpus": str(args.corpus),
            "case_ids": case_ids,
            "direction": args.direction,
            "budget_seconds_per_case": args.budget_seconds,
            "max_tokens": args.max_tokens,
            "recipe_sizes": recipe_sizes,
            "max_candidates": args.max_candidates,
            "position_limit": args.position_limit,
            "stop_on_success": args.stop_on_success,
            "setup_seconds_excluded": setup_seconds,
            "workspace": {
                "start_layer": workspace_start,
                "end_layer": workspace_end,
            },
        },
        "runs": runs,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}", flush=True)
    else:
        print("\n" + rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
