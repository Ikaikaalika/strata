from __future__ import annotations

import pytest

from lokahi.planning import (
    PagingTuningCandidate,
    PagingTuningObjective,
    select_paging_candidate,
)


def _candidate(
    candidate_id: str,
    *,
    decode: float,
    ttft: float,
    memory: float,
    parity: bool = True,
) -> PagingTuningCandidate:
    return PagingTuningCandidate(
        candidate_id=candidate_id,
        max_resident_weight_bytes=640 * 1024 * 1024,
        prefetch_distance=1,
        io_workers=1,
        pinned_layer_count=12,
        exact_token_parity=parity,
        decode_tokens_per_second=decode,
        ttft_ms=ttft,
        peak_memory_gb=memory,
    )


def test_throughput_objective_selects_fastest_eligible_candidate() -> None:
    decision = select_paging_candidate(
        (
            _candidate("low-memory", decode=8.0, ttft=900, memory=0.7),
            _candidate("fast", decode=12.0, ttft=1000, memory=0.9),
            _candidate("wrong", decode=20.0, ttft=700, memory=0.8, parity=False),
        ),
        PagingTuningObjective(max_peak_memory_gb=1.0),
    )

    assert decision.selected.candidate_id == "fast"
    assert decision.evaluations[2].rejection_reasons == ("exact token parity failed",)


def test_weights_can_prefer_lower_memory_over_raw_throughput() -> None:
    decision = select_paging_candidate(
        (
            _candidate("small", decode=8.0, ttft=900, memory=0.5),
            _candidate("fast", decode=10.0, ttft=900, memory=1.0),
        ),
        PagingTuningObjective(decode_weight=1.0, memory_weight=1.0),
    )

    assert decision.selected.candidate_id == "small"


def test_hard_constraints_fail_closed_when_no_candidate_is_eligible() -> None:
    with pytest.raises(ValueError, match="no paging candidate"):
        select_paging_candidate(
            (_candidate("candidate", decode=8.0, ttft=900, memory=0.7),),
            PagingTuningObjective(max_peak_memory_gb=0.6),
        )
