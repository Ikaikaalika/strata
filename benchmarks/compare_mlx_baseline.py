#!/usr/bin/env python3
"""Compare one standalone Lokahi model result with its exact MLX-LM control."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


class BaselineComparisonError(ValueError):
    """Raised when two reports are not safely comparable."""


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BaselineComparisonError(f"{name} must be an object")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineComparisonError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise BaselineComparisonError(f"{name} must be finite")
    return result


def _identity(report: Mapping[str, Any]) -> dict[str, Any]:
    model = _object(report.get("model"), "model")
    artifact = _object(model.get("artifact"), "model.artifact")
    weights = artifact.get("weights")
    if not isinstance(weights, list) or not weights:
        raise BaselineComparisonError("model.artifact.weights must be non-empty")
    return {
        "repo": model.get("repo"),
        "revision": model.get("revision"),
        "weights": [
            (item.get("filename"), item.get("bytes"), item.get("sha256"))
            for item in weights
            if isinstance(item, dict)
        ],
    }


def _reference_tokens(report: Mapping[str, Any]) -> Any:
    repetitions = report.get("repetitions")
    if not isinstance(repetitions, list) or not repetitions:
        raise BaselineComparisonError("repetitions must be non-empty")
    first = _object(repetitions[0], "repetitions[0]")
    tokens = first.get("token_ids")
    if not isinstance(tokens, list) or not tokens:
        raise BaselineComparisonError("repetitions[0].token_ids must be non-empty")
    return tokens


def _mean(report: Mapping[str, Any], metric: str) -> float:
    summary = _object(report.get("summary"), "summary")
    distribution = _object(summary.get(metric), f"summary.{metric}")
    return _number(distribution.get("mean"), f"summary.{metric}.mean")


def compare_mlx_baseline(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Return exact-identity correctness and performance deltas."""

    if baseline.get("runner") != "mlx-lm-baseline":
        raise BaselineComparisonError("baseline runner must be mlx-lm-baseline")
    if candidate.get("runner") == "mlx-lm-baseline":
        raise BaselineComparisonError("candidate must be a Lokahi runtime")
    for field in ("hardware", "workload"):
        if candidate.get(field) != baseline.get(field):
            raise BaselineComparisonError(f"candidate {field} does not match baseline")
    if _identity(candidate) != _identity(baseline):
        raise BaselineComparisonError("candidate artifact identity does not match baseline")

    candidate_correctness = _object(candidate.get("correctness"), "correctness")
    exact_tokens = _reference_tokens(candidate) == _reference_tokens(baseline)
    correctness_passed = (
        candidate_correctness.get("requested_token_count_passed") is True
        and candidate_correctness.get("repetition_token_invariance") is True
        and exact_tokens
    )

    directions = {
        "ttft_ms": "lower",
        "inter_token_latency_ms": "lower",
        "prompt_tokens_per_second": "higher",
        "decode_tokens_per_second": "higher",
        "aggregate_tokens_per_second_including_prefill": "higher",
        "peak_memory_gb": "lower",
    }
    comparisons = []
    improvements: dict[str, float] = {}
    for metric, direction in directions.items():
        control = _mean(baseline, metric)
        actual = _mean(candidate, metric)
        improvement = (
            100.0 * (actual - control) / control
            if direction == "higher"
            else 100.0 * (control - actual) / control
        )
        improvements[metric] = improvement
        comparisons.append(
            {
                "metric": metric,
                "direction": direction,
                "mlx_lm_mean": control,
                "lokahi_mean": actual,
                "improvement_percent": improvement,
            }
        )

    within_ttft = improvements["ttft_ms"] >= -5.0
    within_decode = improvements["decode_tokens_per_second"] >= -5.0
    ten_percent_win = any(value >= 10.0 for value in improvements.values())
    return {
        "schema_version": 1,
        "baseline_runner": "mlx-lm-baseline",
        "candidate_runner": candidate.get("runner"),
        "model": _identity(baseline),
        "correctness_passed": correctness_passed,
        "exact_greedy_token_parity": exact_tokens,
        "within_five_percent_ttft": within_ttft,
        "within_five_percent_decode": within_decode,
        "has_ten_percent_win": ten_percent_win,
        "standalone_mvp_gate_passed": (
            correctness_passed and within_ttft and within_decode and ten_percent_win
        ),
        "comparisons": comparisons,
    }


def _load(path: Path) -> Mapping[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineComparisonError(f"could not read {path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        report = compare_mlx_baseline(
            _load(arguments.baseline), _load(arguments.candidate)
        )
    except BaselineComparisonError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        if not output.parent.is_dir():
            raise BaselineComparisonError("output parent directory must exist")
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["standalone_mvp_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
