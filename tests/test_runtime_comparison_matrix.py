from __future__ import annotations

import json
from pathlib import Path


def test_runtime_matrix_has_one_mlx_oracle_and_fail_closed_statuses() -> None:
    root = Path(__file__).parents[1]
    matrix = json.loads(
        (root / "benchmarks/targets/apple_m1_runtime_comparison_matrix_v1.json").read_text(
            encoding="utf-8"
        )
    )
    runtimes = {item["runtime_id"]: item for item in matrix["runtimes"]}

    assert matrix["oracle_runtime"] == "mlx-lm-baseline"
    assert set(runtimes["mlx-lm-baseline"]["model_ids"]) == set(matrix["models"])
    assert set(runtimes["strata-resident-mlx-route-selector"]["model_ids"]) == set(
        matrix["models"]
    )
    assert runtimes["strata-resident-mlx-route-selector"]["status"] == (
        "measured-route-promotions-not-native-backend"
    )
    assert runtimes["strata-paged-mlx-lab"]["status"] == "measured-capability-not-promoted"
    assert runtimes["strata-native-metal"]["status"] == "not-yet-full-model"
    assert runtimes["strata-ane-research"]["status"] == (
        "qualified-not-planner-eligible-not-yet-full-model"
    )
    assert runtimes["strata-ane-research"]["result_files"] == [
        "benchmarks/results/apple_m1_ane_projection_qualification_2026_08_25.json",
        "benchmarks/results/apple_m1_ane_resident_projection_benchmark_2026_08_25.json",
    ]

    for runtime in matrix["runtimes"]:
        for result_file in runtime["result_files"]:
            assert (root / result_file).is_file(), result_file
