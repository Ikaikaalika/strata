"""Hardware-segmented prefill and decode plans."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

from .ir import InferencePhase, StrataIRGraph


class BackendTarget(str, Enum):
    CPU = "cpu"
    MLX = "mlx"
    METAL = "metal"
    ANE = "ane"
    COREML = "coreml"


@dataclass(frozen=True)
class ExecutionSegment:
    """A contiguous graph region assigned to one compute backend."""

    segment_id: str
    phase: InferencePhase
    target: BackendTarget
    operation_ids: Tuple[str, ...]
    input_tensors: Tuple[str, ...]
    output_tensors: Tuple[str, ...]
    estimated_working_set_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must not be empty")
        if not self.operation_ids:
            raise ValueError("execution segments must contain operations")
        if len(self.operation_ids) != len(set(self.operation_ids)):
            raise ValueError("segment operation IDs must be unique")
        if self.estimated_working_set_bytes < 0:
            raise ValueError("estimated working set must not be negative")


@dataclass(frozen=True)
class AdaptiveExecutionPlan:
    """Plan selected for one model, machine, quantization, and context bucket."""

    plan_id: str
    model_id: str
    model_hash: str
    hardware_fingerprint: str
    quantization: str
    batch_size: int
    context_min_tokens: int
    context_max_tokens: int
    segments: Tuple[ExecutionSegment, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "plan_id",
            "model_id",
            "model_hash",
            "hardware_fingerprint",
            "quantization",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.context_min_tokens < 0:
            raise ValueError("context_min_tokens must not be negative")
        if self.context_max_tokens < self.context_min_tokens:
            raise ValueError("context bucket maximum must be >= minimum")
        if not self.segments:
            raise ValueError("adaptive plan must contain at least one segment")
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment IDs must be unique")
        assignments = [
            (segment.phase, operation_id)
            for segment in self.segments
            for operation_id in segment.operation_ids
        ]
        if len(assignments) != len(set(assignments)):
            raise ValueError("an operation may be assigned only once per phase")

    def validate_graph(self, graph: StrataIRGraph) -> None:
        known = graph.operations_by_id().keys()
        unknown = {
            operation_id
            for segment in self.segments
            for operation_id in segment.operation_ids
            if operation_id not in known
        }
        if unknown:
            raise ValueError(f"plan references unknown operations: {sorted(unknown)}")

    def segments_by_phase(self) -> Dict[InferencePhase, Tuple[ExecutionSegment, ...]]:
        return {
            phase: tuple(segment for segment in self.segments if segment.phase is phase)
            for phase in InferencePhase
        }
