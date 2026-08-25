#!/usr/bin/env python3
"""Fail-closed comparison of a Strata result against a benchmark target.

This is offline evidence tooling, not part of the inference hot path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


class ComparisonError(ValueError):
    """Raised when a target or result cannot support a trustworthy comparison."""


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonError(f"{name} must be a JSON object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonError(f"{name} must be finite")
    return result


def compare_documents(
    target: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a deterministic comparison report or reject incomparable input."""

    target_id = _text(target.get("target_id"), "target.target_id")
    if _text(result.get("target_id"), "result.target_id") != target_id:
        raise ComparisonError("result target_id does not match the target manifest")

    expected_hardware = _object(target.get("required_hardware"), "required_hardware")
    actual_hardware = _object(result.get("hardware"), "result.hardware")
    for key, expected in expected_hardware.items():
        if actual_hardware.get(key) != expected:
            raise ComparisonError(
                f"result hardware {key!r} does not match required value {expected!r}"
            )

    expected_models = _object(target.get("required_models"), "required_models")
    actual_models = _object(result.get("models"), "result.models")
    for model_id, expected_revision in expected_models.items():
        if actual_models.get(model_id) != expected_revision:
            raise ComparisonError(
                f"result model {model_id!r} does not match required revision"
            )

    correctness = _object(result.get("correctness"), "result.correctness")
    required_checks = target.get("required_correctness")
    if not isinstance(required_checks, list) or not required_checks:
        raise ComparisonError("required_correctness must be a non-empty array")
    failed_correctness: list[str] = []
    for check in required_checks:
        check_name = _text(check, "required_correctness item")
        if correctness.get(check_name) is not True:
            failed_correctness.append(check_name)

    actual_metrics = _object(result.get("metrics"), "result.metrics")
    cells = target.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ComparisonError("target cells must be a non-empty array")

    comparisons: list[dict[str, Any]] = []
    for index, raw_cell in enumerate(cells):
        cell = _object(raw_cell, f"cells[{index}]")
        cell_id = _text(cell.get("id"), f"cells[{index}].id")
        direction = _text(cell.get("direction"), f"cells[{index}].direction")
        gate = _number(cell.get("strata_gate"), f"cells[{index}].strata_gate")
        actual = _number(actual_metrics.get(cell_id), f"metrics.{cell_id}")
        if direction == "higher":
            passed = actual >= gate
            margin_percent = 100.0 * (actual - gate) / gate
        elif direction == "lower":
            passed = actual <= gate
            margin_percent = 100.0 * (gate - actual) / gate
        else:
            raise ComparisonError(f"unsupported direction {direction!r} for {cell_id}")
        comparisons.append(
            {
                "id": cell_id,
                "unit": _text(cell.get("unit"), f"cells[{index}].unit"),
                "direction": direction,
                "darkbloom_best": _number(
                    cell.get("darkbloom_best"), f"cells[{index}].darkbloom_best"
                ),
                "strata_gate": gate,
                "actual": actual,
                "margin_percent_vs_gate": margin_percent,
                "passed": passed,
            }
        )

    metrics_passed = all(item["passed"] for item in comparisons)
    correctness_passed = not failed_correctness
    return {
        "schema_version": 1,
        "target_id": target_id,
        "passed": metrics_passed and correctness_passed,
        "metrics_passed": metrics_passed,
        "correctness_passed": correctness_passed,
        "failed_correctness": failed_correctness,
        "comparisons": comparisons,
    }


def _load(path: Path) -> Mapping[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"could not read valid JSON from {path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("result", type=Path)
    arguments = parser.parse_args()
    try:
        report = compare_documents(_load(arguments.target), _load(arguments.result))
    except ComparisonError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
