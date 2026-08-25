from __future__ import annotations

from benchmarks.benchmark_resident_mlx_routes import compare_route_runs


def _run(ttft: float, itl: float, wall: float, tokens: list[int]) -> dict:
    return {
        "ttft_ms": ttft,
        "inter_token_latency_ms": itl,
        "wall_time_seconds": wall,
        "end_to_end_output_tokens_per_second": 32.0 / wall,
        "token_ids": tokens,
    }


def test_resident_route_promotes_exact_stable_speedup() -> None:
    control = [_run(500.0, 10.0, 0.84, [1, 2, 3]) for _ in range(3)]
    candidate = [_run(480.0, 9.9, 0.80, [1, 2, 3]) for _ in range(3)]

    result = compare_route_runs(control, candidate)

    assert result["exact_greedy_token_parity"] is True
    assert result["batch_one_route_promotion_passed"] is True
    assert result["median_improvement_percent"]["ttft_ms"] == 4.0


def test_resident_route_rejects_token_drift_even_when_faster() -> None:
    control = [_run(500.0, 10.0, 0.84, [1, 2, 3]) for _ in range(3)]
    candidate = [_run(450.0, 9.0, 0.70, [1, 2, 4]) for _ in range(3)]

    result = compare_route_runs(control, candidate)

    assert result["exact_greedy_token_parity"] is False
    assert result["batch_one_route_promotion_passed"] is False
