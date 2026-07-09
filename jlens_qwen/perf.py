"""Flag-gated per-stage timers for the decode hot path.

Off by default; enable with JLENS_PERF=1. When enabled, `mark` forces the
given arrays to evaluate so lazily-built work is attributed to the stage
that built it — this adds sync points, so per-stage numbers slightly
inflate the total. End-to-end baselines must be measured with the flag
OFF; the stages are for attribution, not for the headline number.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import mlx.core as mx

ENABLED = os.environ.get("JLENS_PERF") == "1"
_acc: dict[str, list[float]] = defaultdict(list)


def begin() -> float | None:
    """Start a stage clock. Returns None (no-op) when disabled."""
    if not ENABLED:
        return None
    return time.perf_counter()


def mark(t0: float | None, name: str, *arrays: mx.array) -> float | None:
    """End the current stage: force `arrays` (or a full synchronize),
    record the elapsed time under `name`, and start the next stage clock.
    """
    if t0 is None:
        return None
    if arrays:
        mx.eval(*arrays)
    else:
        mx.synchronize()
    now = time.perf_counter()
    _acc[name].append(now - t0)
    return now


def report() -> dict[str, dict[str, float]]:
    """Per-stage stats in ms: median, mean, count. Resets the accumulator."""
    out = {}
    for name, xs in sorted(_acc.items()):
        s = sorted(xs)
        out[name] = {
            "median_ms": 1e3 * s[len(s) // 2],
            "mean_ms": 1e3 * sum(s) / len(s),
            "n": len(s),
        }
    _acc.clear()
    return out
