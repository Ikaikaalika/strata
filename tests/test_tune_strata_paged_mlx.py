from __future__ import annotations

from pathlib import Path

from benchmarks.tune_strata_paged_mlx import report_to_candidate


def test_report_to_candidate_uses_runtime_parameters_and_medians() -> None:
    report = {
        "runner": "strata-paged-mlx-lab",
        "weight_residency": {
            "runtime": {
                "pinned_global_weight_bytes": 100,
                "layer_residency_budget_bytes": 200,
                "prefetch_distance": 2,
                "io_workers": 3,
                "pinned_layer_count": 4,
            }
        },
        "correctness": {
            "mlx_lm_exact_greedy_token_parity": True,
            "repetition_token_invariance": True,
            "requested_token_count_passed": True,
        },
        "summary": {
            "decode_tokens_per_second": {"median": 12.5},
            "ttft_ms": {"median": 800.0},
            "peak_memory_gb": {"median": 0.9},
        },
    }

    candidate = report_to_candidate(Path("candidate.json"), report)

    assert candidate.max_resident_weight_bytes == 300
    assert candidate.prefetch_distance == 2
    assert candidate.io_workers == 3
    assert candidate.pinned_layer_count == 4
    assert candidate.decode_tokens_per_second == 12.5
