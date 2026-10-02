"""Synthetic rejection/ranking contracts; not measured model speed evidence."""
import copy

import numpy as np
import pytest

from benchmarks.benchmark_native_swiglu import (
    PROMPT_LENGTHS, ROUTES, affine_reference, compare_trace, error_gate, round_storage, summarize,
)


def records():
    rows = []
    for length in PROMPT_LENGTHS:
        for route in ROUTES:
            speed = 1 if route == "mlx_lm" else 2
            for round_ in range(3):
                rows.append({"route": route, "prompt_tokens": length, "round": round_, "warmup": False,
                             "output_token_ids": [1, 3, 7], "ttft_ms": 10 / speed,
                             "prefill_proxy_tps": length / (0.01 / speed), "decode_tps": 10 * speed,
                             "goodput_tps": 8 * speed, "peak_mlx_bytes": 1024})
    return rows


def test_frozen_numerical_and_state_gates():
    assert error_gate(np.zeros((1, 2)), np.zeros((1, 2)))["relative_l2"] == 0
    trace = ([1, 3], [np.ones((1, 2))], [[np.ones((1, 2)), np.ones((1, 2))]], [3])
    assert compare_trace(trace, trace)["exact_greedy_token_parity"] is True
    bad = copy.deepcopy(trace)
    bad[0][0] = 7
    with pytest.raises(ValueError, match="diverged"):
        compare_trace(bad, trace)
    for actual in (np.array([np.nan, 1]), np.array([2, 1]), np.array([])):
        with pytest.raises(ValueError):
            error_gate(actual, np.ones(2))


def test_independent_packed_reference_and_bfloat_rounding():
    # Eight codes 0..7 per word, scale 1/8, bias -1/4, all-one activation.
    x = np.ones((1, 64), dtype=np.float16)
    q = np.full((1, 8), 0x76543210, dtype=np.uint32)
    expected = affine_reference(x, q, np.array([[0.125]]), np.array([[-0.25]]))
    assert np.array_equal(expected, [[12]])
    # BF16 ties to even: 1 + 1/256 rounds down; 1 + 3/256 rounds up.
    assert np.array_equal(round_storage([1 + 1 / 256, 1 + 3 / 256], "bfloat16"), [1, 1 + 1 / 64])
    with pytest.raises(ValueError):
        round_storage([1], "float32")


def test_pareto_frontiers_and_shortlist_never_promote():
    result = summarize(records())
    assert result["promotion_eligible"] is result["full_model_serving_gate_passed"] is False
    assert result["development_speed_shortlist"] == list(ROUTES[1:])
    assert all("mlx_lm" not in f["routes"] for f in result["development_pareto_frontiers"])


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows[-1].update(output_token_ids=[8, 9]),
    lambda rows: rows[-1].update(decode_tps=float("nan")),
    lambda rows: rows[-1].update(ttft_ms=0),
    lambda rows: rows[-1].update(peak_mlx_bytes=True),
    lambda rows: rows[-1].update(route="unbounded"),
])
def test_missing_divergent_or_invalid_measurements_rejected(mutation):
    rows = records()
    mutation(rows)
    with pytest.raises(ValueError):
        summarize(rows)


def test_memory_regression_disqualifies_speed_shortlist():
    rows = records()
    for row in rows:
        if row["route"] == "fused_both":
            row["peak_mlx_bytes"] = 2048
    assert "fused_both" not in summarize(rows)["development_speed_shortlist"]
