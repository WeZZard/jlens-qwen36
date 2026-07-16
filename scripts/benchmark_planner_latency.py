#!/usr/bin/env python3
"""Profile a text-only backward J-Space planner against the live server.

This deliberately uses ``/api/planner_probe`` rather than chat generation.  The
probe reuses the resident model but performs no J-lens readout, so its timings
measure planner tokenization, prefill, and decode instead of visualization work.

Examples:

    uv run python scripts/benchmark_planner_latency.py
    uv run python scripts/benchmark_planner_latency.py --guidance minimal,distilled
    uv run python scripts/benchmark_planner_latency.py \
        --guidance paper --cases country_capital --paper-token-limit 0 \
        --time-budget-seconds 180

The paper variant downloads the official article at run time; the repository does
not vendor a copyrighted copy.  ``--paper-token-limit 0`` means the complete
extracted article.  A server-side time budget still bounds model execution.
"""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
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
DEFAULT_BRIEF = REPO / "data/benchmarks/jspace_planner_brief.md"
DEFAULT_PAPER_URL = "https://transformer-circuits.pub/2026/workspace/index.html"

MINIMAL_GUIDANCE = """You are suggesting hypotheses for a backward J-Space
intervention search. Given an existing user message, assistant reply, requested
reply edit, selected reply positions, causal cutoff, and indexed token rail, return
JSON only. Return exactly one compact candidate containing one or more layer/position
cells in WORKSPACE_BAND plus source_concept, target_concept, reason, and confidence.
Every position must be no later than eligible_position_max. Never claim a candidate
was verified; the application must replay it. If unsupported, return
{\"candidates\": []}. Keep the whole JSON under 120 tokens: at most three cells,
three words per concept, and eight words for the reason. Do not use Markdown."""


class _ArticleText(HTMLParser):
    """Conservative visible-text extraction without third-party dependencies."""

    _ignored = {"script", "style", "svg", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in self._ignored:
            self._ignore_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._ignored and self._ignore_depth:
            self._ignore_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignore_depth:
            stripped = " ".join(data.split())
            if stripped:
                self._parts.append(stripped)

    def text(self) -> str:
        return "\n".join(self._parts)


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 300.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "jlens-qwen36-planner-benchmark/1",
        },
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach {url}: {error.reason}") from error


def _fetch_paper(url: str) -> str:
    request = Request(url, headers={"User-Agent": "jlens-qwen36-planner-benchmark/1"})
    try:
        with urlopen(request, timeout=60.0) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Could not fetch official paper at {url}: {error}") from error
    parser = _ArticleText()
    parser.feed(raw)
    return html.unescape(parser.text())


def _token_segments(tokenizer: Any, ids: list[int]) -> list[str]:
    segments: list[str] = []
    previous = ""
    for index in range(len(ids)):
        decoded = tokenizer.decode(ids[: index + 1])
        stable = decoded.rstrip("\ufffd")
        if len(stable) >= len(previous):
            segment = stable[len(previous) :]
            previous = stable
        else:
            segment = ""
        if not segment:
            segment = tokenizer.convert_ids_to_tokens(ids[index]) or ""
        segments.append(segment)
    return segments


def _template_ids(tokenizer: Any, messages: list[dict[str, str]], **kwargs) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if (isinstance(ids, (list, tuple)) and len(ids) == 1
            and isinstance(ids[0], (list, tuple))):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _token_rail(tokenizer: Any, case: dict[str, Any]) -> dict[str, Any]:
    conversation = case["conversation"]
    user_message = {"role": "user", "content": conversation["user"]}
    assistant_message = {"role": "assistant", "content": conversation["assistant"]}
    messages = [
        user_message,
        assistant_message,
    ]
    ids = _template_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assistant_prefix_ids = _template_ids(
        tokenizer,
        [user_message],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if ids[: len(assistant_prefix_ids)] != assistant_prefix_ids:
        raise ValueError("assistant chat-template prefix does not match completed chat")

    full_text = tokenizer.decode(ids)
    assistant_start = len(tokenizer.decode(assistant_prefix_ids))
    assistant_text = conversation["assistant"]
    if not full_text.startswith(tokenizer.decode(assistant_prefix_ids)):
        raise ValueError("decoded assistant prefix does not match completed chat")
    if full_text[assistant_start: assistant_start + len(assistant_text)] != assistant_text:
        raise ValueError("assistant reply is not contiguous after generation prefix")
    assistant_end = assistant_start + len(assistant_text)

    source_span = case["requested_edit"]["source_span"]
    source_offset = assistant_text.find(source_span)
    if source_offset < 0 or assistant_text.find(source_span, source_offset + 1) >= 0:
        raise ValueError(
            f"{case['id']}: selected source span must occur exactly once in assistant reply"
        )
    selected_start = assistant_start + source_offset
    selected_end = selected_start + len(source_span)
    user_start = full_text.find(conversation["user"])
    user_end = user_start + len(conversation["user"]) if user_start >= 0 else -1

    segments = _token_segments(tokenizer, ids)
    tokens: list[dict[str, Any]] = []
    selected_positions: list[int] = []
    for position, (token_id, segment) in enumerate(zip(ids, segments, strict=True)):
        char_start = len(tokenizer.decode(ids[:position]))
        char_end = len(tokenizer.decode(ids[: position + 1]))
        selected = char_start < selected_end and char_end > selected_start
        if selected:
            selected_positions.append(position)
        if char_start < assistant_end and char_end > assistant_start:
            region = "assistant_reply"
        elif user_start >= 0 and char_start < user_end and char_end > user_start:
            region = "user"
        else:
            region = "template"
        tokens.append({
            "position": position,
            "token_id": token_id,
            "token": segment,
            "region": region,
            "selected": selected,
        })

    if not selected_positions:
        raise ValueError(f"{case['id']}: selected source span has no token positions")
    return {
        "tokens": tokens,
        "selected_reply_positions": selected_positions,
        # The hidden state at p predicts reply token p+1; do not target the
        # emitted token itself or the completed-chat suffix.
        "eligible_position_max": min(selected_positions) - 1,
    }


def _planner_payload(case: dict[str, Any], token_rail: dict[str, Any]) -> str:
    # Intentionally select fields one by one. In particular, never leak `oracle`.
    visible = {
        "conversation": case["conversation"],
        "requested_edit": case["requested_edit"],
        "token_rail_columns": ["position", "token_id", "region", "token"],
        "token_rail": [
            [token["position"], token["token_id"], token["region"], token["token"]]
            for token in token_rail["tokens"]
        ],
        "selected_reply_positions": token_rail["selected_reply_positions"],
        "eligible_position_max": token_rail["eligible_position_max"],
    }
    return (
        "Recommend candidate cell combinations for this requested reply edit. "
        "The indexed rail is the exact chat-template token sequence.\n"
        + json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
    )


def _guidance_text(
    variant: str,
    brief: str,
    paper_text: str | None,
    tokenizer: Any,
    paper_token_limit: int,
    workspace_start: int,
    workspace_end: int,
) -> tuple[str, dict[str, Any]]:
    minimal = MINIMAL_GUIDANCE.replace(
        "WORKSPACE_BAND", f"workspace layers L{workspace_start}-L{workspace_end}"
    )
    brief = brief.replace(
        "layers L26 through L59",
        f"layers L{workspace_start} through L{workspace_end}",
    )
    if variant == "minimal":
        return minimal, {
            "guidance_tokens": len(tokenizer.encode(minimal)),
            "quality_comparison_valid": True,
        }
    if variant == "distilled":
        return brief, {
            "guidance_tokens": len(tokenizer.encode(brief)),
            "quality_comparison_valid": True,
        }
    if variant != "paper":
        raise ValueError(f"unknown guidance variant: {variant}")
    if paper_text is None:
        raise ValueError("paper guidance requested without paper text")
    paper_ids = tokenizer.encode(paper_text, add_special_tokens=False)
    original_tokens = len(paper_ids)
    if paper_token_limit > 0 and original_tokens > paper_token_limit:
        paper_ids = paper_ids[:paper_token_limit]
        paper_text = tokenizer.decode(paper_ids)
    guidance = (
        "REFERENCE PAPER (background only):\n"
        + paper_text
        + "\n\nFINAL PLANNER CONTRACT (follow this after the reference):\n"
        + brief
    )
    return guidance, {
        "guidance_tokens": len(tokenizer.encode(guidance, add_special_tokens=False)),
        "paper_tokens_extracted": original_tokens,
        "paper_tokens_included": len(paper_ids),
        "paper_truncated": len(paper_ids) < original_tokens,
        "quality_comparison_valid": False,
        "quality_warning": "The paper contains oracle concepts for core cases; use this condition for latency only.",
    }


def _validate_candidate_output(
    text: str,
    *,
    workspace_start: int,
    workspace_end: int,
    eligible_position_max: int,
) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        return {
            "json_parseable": False,
            "candidate_count": None,
            "contract_valid": False,
            "contract_errors": [f"invalid JSON: {error.msg}"],
        }
    errors: list[str] = []
    values = parsed.get("candidates") if isinstance(parsed, dict) else None
    if not isinstance(parsed, dict) or set(parsed) != {"candidates"}:
        errors.append("top level must contain only candidates")
    if not isinstance(values, list):
        errors.append("candidates must be an array")
        values = []
    if len(values) > 1:
        errors.append("latency contract permits at most one candidate")

    for candidate_index, value in enumerate(values):
        prefix = f"candidates[{candidate_index}]"
        if not isinstance(value, dict):
            errors.append(f"{prefix} must be an object")
            continue
        required = {
            "cells", "source_concept", "target_concept", "reason", "confidence"
        }
        missing = required - set(value)
        if missing:
            errors.append(f"{prefix} missing {sorted(missing)}")
        cells = value.get("cells")
        if not isinstance(cells, list) or not cells:
            errors.append(f"{prefix}.cells must be a non-empty array")
        else:
            for cell_index, cell in enumerate(cells):
                cell_prefix = f"{prefix}.cells[{cell_index}]"
                if not isinstance(cell, dict):
                    errors.append(f"{cell_prefix} must be an object")
                    continue
                layer = cell.get("layer")
                position = cell.get("position")
                if (not isinstance(layer, int) or isinstance(layer, bool)
                        or not workspace_start <= layer <= workspace_end):
                    errors.append(
                        f"{cell_prefix}.layer must be in "
                        f"[{workspace_start}, {workspace_end}]"
                    )
                if (not isinstance(position, int) or isinstance(position, bool)
                        or not 0 <= position <= eligible_position_max):
                    errors.append(
                        f"{cell_prefix}.position must be in "
                        f"[0, {eligible_position_max}]"
                    )
        for field in ("source_concept", "target_concept", "reason"):
            if not isinstance(value.get(field), str) or not value[field].strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
        if value.get("confidence") not in {"low", "medium", "high"}:
            errors.append(f"{prefix}.confidence must be low, medium, or high")

    return {
        "json_parseable": True,
        "candidate_count": len(values),
        "contract_valid": not errors,
        "contract_errors": errors,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    variants = sorted({str(item["variant"]) for item in results})
    for variant in variants:
        rows = [item for item in results if item["variant"] == variant and "error" not in item]
        if not rows:
            summary[variant] = {"responses": 0}
            continue
        outer = [float(item["outer_http_ms"]) for item in rows]
        service = [float(item["server"]["total_service_ms"]) for item in rows]
        prefill = [float(item["server"]["prefill_ms"]) for item in rows]
        ttft = [
            float(item["server"]["ttft_ms"])
            for item in rows
            if item["server"].get("ttft_ms") is not None
        ]
        summary[variant] = {
            "responses": len(rows),
            "eos_completions": sum(
                item["server"]["stop_reason"] == "eos" for item in rows
            ),
            "max_token_stops": sum(
                item["server"]["stop_reason"] == "max_tokens" for item in rows
            ),
            "budget_stops": sum(
                item["server"]["stop_reason"] == "time_budget" for item in rows
            ),
            "median_outer_http_ms": statistics.median(outer),
            "p95_outer_http_ms": _percentile(outer, 0.95),
            "median_server_service_ms": statistics.median(service),
            "median_prefill_ms": statistics.median(prefill),
            "median_ttft_ms": statistics.median(ttft) if ttft else None,
            "median_input_tokens": statistics.median(
                float(item["server"]["input_tokens"]) for item in rows
            ),
            "median_input_tokens_processed": statistics.median(
                float(item["server"]["input_tokens_processed"]) for item in rows
            ),
            "median_prefill_tokens_per_second": statistics.median(
                float(item["server"]["prefill_tokens_per_second"]) for item in rows
            ),
            "median_output_tokens": statistics.median(
                float(item["server"]["output_tokens"]) for item in rows
            ),
            "json_parse_rate": sum(bool(item["json_parseable"]) for item in rows) / len(rows),
            "contract_valid_rate": sum(bool(item["contract_valid"]) for item in rows) / len(rows),
            "stop_reasons": sorted({str(item["server"]["stop_reason"]) for item in rows}),
        }
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF)
    parser.add_argument(
        "--guidance",
        default="minimal,distilled",
        help="comma-separated subset of minimal,distilled,paper",
    )
    parser.add_argument(
        "--cases",
        default="timing",
        help="timing, all, or comma-separated case IDs",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--time-budget-seconds", type=float, default=180.0)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=131072)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--require-lens-prompts", type=int, default=1000)
    parser.add_argument("--paper-url", default=DEFAULT_PAPER_URL)
    parser.add_argument(
        "--paper-token-limit",
        type=int,
        default=8192,
        help="0 includes the complete extracted paper",
    )
    parser.add_argument("--model-id", default=None, help="override server model ID for tokenizer")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    if args.paper_token_limit < 0:
        raise SystemExit("--paper-token-limit must be 0 or greater")
    variants = [value.strip() for value in args.guidance.split(",") if value.strip()]
    invalid = sorted(set(variants) - {"minimal", "distilled", "paper"})
    if invalid:
        raise SystemExit(f"unknown guidance variants: {', '.join(invalid)}")

    corpus = json.loads(args.corpus.read_text())
    brief = args.brief.read_text()
    cases_by_id = {case["id"]: case for case in corpus["cases"]}
    if args.cases == "timing":
        case_ids = corpus["timing_suite"]
    elif args.cases == "all":
        case_ids = list(cases_by_id)
    else:
        case_ids = [value.strip() for value in args.cases.split(",") if value.strip()]
    missing = sorted(set(case_ids) - set(cases_by_id))
    if missing:
        raise SystemExit(f"unknown case IDs: {', '.join(missing)}")

    base_url = args.base_url.rstrip("/")
    model_info = _http_json(f"{base_url}/api/model", timeout=30.0)
    lens_info = _http_json(f"{base_url}/api/lens", timeout=30.0)
    if model_info.get("mode") != "active":
        raise SystemExit(f"server is not in active mode: {model_info}")
    actual_prompts = lens_info.get("n_prompts")
    if args.require_lens_prompts and actual_prompts != args.require_lens_prompts:
        raise SystemExit(
            f"refusing benchmark: expected n{args.require_lens_prompts} lens, "
            f"server reports n{actual_prompts}"
        )
    workspace_bands = [
        band for band in lens_info.get("bands", [])
        if band.get("name") == "workspace"
    ]
    if len(workspace_bands) != 1:
        raise SystemExit(f"server did not report exactly one workspace band: {lens_info}")
    workspace_start = int(workspace_bands[0]["start_layer"])
    workspace_end = int(workspace_bands[0]["end_layer"])

    model_id = args.model_id or model_info["model_id"]
    setup_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    token_rails = {case_id: _token_rail(tokenizer, cases_by_id[case_id]) for case_id in case_ids}
    paper_text = _fetch_paper(args.paper_url) if "paper" in variants else None
    guidances = {
        variant: _guidance_text(
            variant,
            brief,
            paper_text,
            tokenizer,
            args.paper_token_limit,
            workspace_start,
            workspace_end,
        )
        for variant in variants
    }
    setup_seconds = time.perf_counter() - setup_started

    print(
        f"server={model_id} lens-config=n{actual_prompts} "
        f"workspace=L{workspace_start}-L{workspace_end} cases={len(case_ids)} "
        f"variants={','.join(variants)} setup={setup_seconds:.2f}s",
        flush=True,
    )
    for variant, (_, metadata) in guidances.items():
        print(f"guidance[{variant}]={metadata}", flush=True)

    warmup: dict[str, Any] | None = None
    if not args.skip_warmup:
        warmup_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": MINIMAL_GUIDANCE.replace(
                        "WORKSPACE_BAND",
                        f"workspace layers L{workspace_start}-L{workspace_end}",
                    ),
                },
                {"role": "user", "content": "Return {\"candidates\": []}."},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "enable_thinking": False,
            "prefill_chunk_size": min(args.prefill_chunk_size, 256),
            "time_budget_seconds": min(args.time_budget_seconds, 60.0),
            "max_input_tokens": args.max_input_tokens,
        }
        warmup = _http_json(
            f"{base_url}/api/planner_probe",
            warmup_payload,
            timeout=args.request_timeout_seconds,
        )
        print(
            f"warmup={warmup.get('total_service_ms', 0):.1f}ms "
            f"stop={warmup.get('stop_reason')}",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    run_order = 0
    for repetition in range(args.repetitions):
        for case_index, case_id in enumerate(case_ids):
            case = cases_by_id[case_id]
            user_payload = _planner_payload(case, token_rails[case_id])
            # Deterministic counterbalancing: each condition appears first on
            # alternating case/repetition pairs instead of always paying the
            # same cold/cache/thermal order effect.
            ordered_variants = (
                variants if (case_index + repetition) % 2 == 0
                else list(reversed(variants))
            )
            for variant in ordered_variants:
                guidance, guidance_metadata = guidances[variant]
                request_payload = {
                    "messages": [
                        {"role": "system", "content": guidance},
                        {"role": "user", "content": user_payload},
                    ],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "enable_thinking": False,
                    "prefill_chunk_size": args.prefill_chunk_size,
                    "time_budget_seconds": args.time_budget_seconds,
                    "max_input_tokens": args.max_input_tokens,
                }
                started = time.perf_counter()
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "variant": variant,
                    "repetition": repetition,
                    "run_order": run_order,
                    "guidance": guidance_metadata,
                }
                run_order += 1
                try:
                    server = _http_json(
                        f"{base_url}/api/planner_probe",
                        request_payload,
                        timeout=args.request_timeout_seconds,
                    )
                    row["outer_http_ms"] = (time.perf_counter() - started) * 1000.0
                    row["server"] = server
                    validation = _validate_candidate_output(
                        server.get("text", ""),
                        workspace_start=workspace_start,
                        workspace_end=workspace_end,
                        eligible_position_max=token_rails[case_id]["eligible_position_max"],
                    )
                    row.update(validation)
                    print(
                        f"{case_id:24s} {variant:9s} "
                        f"in={server.get('input_tokens', '?'):>6} "
                        f"prefill={server.get('prefill_ms', 0):8.1f}ms "
                        f"total={row['outer_http_ms']:8.1f}ms "
                        f"stop={server.get('stop_reason')} "
                        f"json={row['json_parseable']} valid={row['contract_valid']}",
                        flush=True,
                    )
                except Exception as error:  # preserve the rest of a long matrix
                    row["outer_http_ms"] = (time.perf_counter() - started) * 1000.0
                    row["error"] = str(error)
                    print(f"{case_id:24s} {variant:9s} ERROR {error}", flush=True)
                results.append(row)

    report = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "server": {"model": model_info, "lens": lens_info},
        "benchmark": {
            "corpus": str(args.corpus.relative_to(REPO)) if args.corpus.is_relative_to(REPO) else str(args.corpus),
            "case_ids": case_ids,
            "variants": variants,
            "repetitions": args.repetitions,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "time_budget_seconds": args.time_budget_seconds,
            "prefill_chunk_size": args.prefill_chunk_size,
            "max_input_tokens": args.max_input_tokens,
            "setup_seconds_excluded": setup_seconds,
            "scope": "planner request latency over canned conversations; baseline generation and intervention replay are excluded",
            "ordering": "conditions are deterministically counterbalanced by case and repetition",
            "workspace_band": {
                "start_layer": workspace_start,
                "end_layer": workspace_end,
            },
        },
        "warmup": warmup,
        "results": results,
        "summary": _summarize(results),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}", flush=True)
    else:
        print("\n" + rendered)
    return 0 if all("error" not in row for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
