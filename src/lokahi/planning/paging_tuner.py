"""Deterministic selection of measured SSD-paging configurations."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class PagingTuningObjective:
    """Hard gates and dimensionless weights for one tuning decision.

    Decode, TTFT, and memory use logarithms, so the weights express relative
    priorities rather than depending on whether latency is reported in ms or s.
    """

    max_peak_memory_gb: float | None = None
    max_ttft_ms: float | None = None
    min_decode_tokens_per_second: float | None = None
    decode_weight: float = 1.0
    ttft_weight: float = 0.0
    memory_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_peak_memory_gb",
            "max_ttft_ms",
            "min_decode_tokens_per_second",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if min(self.decode_weight, self.ttft_weight, self.memory_weight) < 0:
            raise ValueError("tuning weights must be non-negative")
        if self.decode_weight + self.ttft_weight + self.memory_weight <= 0:
            raise ValueError("at least one tuning weight must be positive")


@dataclass(frozen=True)
class PagingTuningCandidate:
    candidate_id: str
    max_resident_weight_bytes: int
    prefetch_distance: int
    io_workers: int
    pinned_layer_count: int
    exact_token_parity: bool
    decode_tokens_per_second: float
    ttft_ms: float
    peak_memory_gb: float

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.max_resident_weight_bytes <= 0:
            raise ValueError("max_resident_weight_bytes must be positive")
        if self.prefetch_distance < 0 or self.pinned_layer_count < 0:
            raise ValueError("prefetch distance and pinned layers must be non-negative")
        if self.io_workers <= 0:
            raise ValueError("io_workers must be positive")
        for name in (
            "decode_tokens_per_second",
            "ttft_ms",
            "peak_memory_gb",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class PagingCandidateEvaluation:
    candidate: PagingTuningCandidate
    eligible: bool
    score: float | None
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PagingTuningDecision:
    selected: PagingTuningCandidate
    evaluations: tuple[PagingCandidateEvaluation, ...]


def evaluate_paging_candidate(
    candidate: PagingTuningCandidate,
    objective: PagingTuningObjective,
) -> PagingCandidateEvaluation:
    reasons = []
    if not candidate.exact_token_parity:
        reasons.append("exact token parity failed")
    if (
        objective.max_peak_memory_gb is not None
        and candidate.peak_memory_gb > objective.max_peak_memory_gb
    ):
        reasons.append("peak memory exceeds objective")
    if objective.max_ttft_ms is not None and candidate.ttft_ms > objective.max_ttft_ms:
        reasons.append("TTFT exceeds objective")
    if (
        objective.min_decode_tokens_per_second is not None
        and candidate.decode_tokens_per_second
        < objective.min_decode_tokens_per_second
    ):
        reasons.append("decode throughput is below objective")
    if reasons:
        return PagingCandidateEvaluation(candidate, False, None, tuple(reasons))

    score = (
        objective.decode_weight * math.log(candidate.decode_tokens_per_second)
        - objective.ttft_weight * math.log(candidate.ttft_ms)
        - objective.memory_weight * math.log(candidate.peak_memory_gb)
    )
    return PagingCandidateEvaluation(candidate, True, score, ())


def select_paging_candidate(
    candidates: Sequence[PagingTuningCandidate],
    objective: PagingTuningObjective,
) -> PagingTuningDecision:
    """Select the highest-scoring eligible measurement with stable tie breaks."""

    if not candidates:
        raise ValueError("at least one tuning candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    evaluations = tuple(
        evaluate_paging_candidate(candidate, objective) for candidate in candidates
    )
    eligible = [evaluation for evaluation in evaluations if evaluation.eligible]
    if not eligible:
        reasons = sorted(
            {
                reason
                for evaluation in evaluations
                for reason in evaluation.rejection_reasons
            }
        )
        raise ValueError("no paging candidate satisfies objective: " + "; ".join(reasons))
    winner = min(
        eligible,
        key=lambda evaluation: (
            -float(evaluation.score),
            -evaluation.candidate.decode_tokens_per_second,
            evaluation.candidate.ttft_ms,
            evaluation.candidate.peak_memory_gb,
            evaluation.candidate.candidate_id,
        ),
    )
    return PagingTuningDecision(winner.candidate, evaluations)


__all__ = [
    "PagingCandidateEvaluation",
    "PagingTuningCandidate",
    "PagingTuningDecision",
    "PagingTuningObjective",
    "evaluate_paging_candidate",
    "select_paging_candidate",
]
