import json
import math
from pathlib import Path

import pytest

from benchmarks.compare_target import ComparisonError, compare_documents


def target_fixture():
    return {
        "target_id": "test-target",
        "required_hardware": {"chip": "Apple Test"},
        "required_models": {"model": "revision"},
        "required_correctness": ["oracle", "isolation"],
        "cells": [
            {
                "id": "throughput",
                "unit": "token/s",
                "direction": "higher",
                "darkbloom_best": 100.0,
                "strata_gate": 110.0,
            },
            {
                "id": "latency",
                "unit": "ms",
                "direction": "lower",
                "darkbloom_best": 10.0,
                "strata_gate": 9.0,
            },
        ],
    }


def result_fixture():
    return {
        "target_id": "test-target",
        "hardware": {"chip": "Apple Test"},
        "models": {"model": "revision"},
        "correctness": {"oracle": True, "isolation": True},
        "metrics": {"throughput": 111.0, "latency": 8.9},
    }


def test_comparison_passes_only_when_every_gate_and_correctness_check_passes():
    report = compare_documents(target_fixture(), result_fixture())

    assert report["passed"] is True
    assert report["metrics_passed"] is True
    assert report["correctness_passed"] is True
    assert {cell["id"] for cell in report["comparisons"]} == {
        "throughput",
        "latency",
    }


def test_comparison_reports_metric_and_correctness_failures():
    result = result_fixture()
    result["metrics"]["throughput"] = 109.9
    result["correctness"]["isolation"] = False

    report = compare_documents(target_fixture(), result)

    assert report["passed"] is False
    assert report["metrics_passed"] is False
    assert report["correctness_passed"] is False
    assert report["failed_correctness"] == ["isolation"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: result.update(target_id="wrong"), "target_id"),
        (
            lambda result: result["hardware"].update(chip="Different"),
            "hardware",
        ),
        (
            lambda result: result["models"].update(model="wrong"),
            "revision",
        ),
        (
            lambda result: result["metrics"].update(throughput=math.nan),
            "finite",
        ),
    ],
)
def test_comparison_rejects_incomparable_or_invalid_results(mutation, message):
    result = result_fixture()
    mutation(result)

    with pytest.raises(ComparisonError, match=message):
        compare_documents(target_fixture(), result)


def test_checked_in_darkbloom_target_has_ten_percent_performance_gates():
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "targets"
        / "darkbloom_m4_max_v080.json"
    )
    target = json.loads(path.read_text(encoding="utf-8"))

    assert len(target["cells"]) == 13
    for cell in target["cells"]:
        baseline = cell["darkbloom_best"]
        gate = cell["strata_gate"]
        if cell["metric"] == "peak_runtime_memory":
            assert gate == baseline
        elif cell["direction"] == "higher":
            assert gate == pytest.approx(baseline * 1.10)
        else:
            assert gate == pytest.approx(baseline * 0.90)

