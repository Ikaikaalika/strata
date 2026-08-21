"""Capabilities declared by a compute runtime adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .hardware import ComputeUnit
from .ir import InferencePhase, OperationKind


@dataclass(frozen=True)
class RuntimeCapabilities:
    compute_units: Tuple[ComputeUnit, ...] = ()
    supported_phases: Tuple[InferencePhase, ...] = ()
    supported_operations: Tuple[OperationKind, ...] = ()
    supported_dtypes: Tuple[str, ...] = ()
    supports_weight_paging: bool = False
    supports_expert_paging: bool = False
    supports_async_prefetch: bool = False
    supports_external_kv_cache: bool = False
    supports_quantized_weights: bool = False
    supports_graph_capture: bool = False
    supports_memory_counters: bool = False
    supports_dynamic_shapes: bool = False
    supports_shared_iosurface: bool = False
    supports_concurrent_dispatch: bool = False
    requires_private_api: bool = False
    max_program_bytes: int | None = None

    def __post_init__(self) -> None:
        if len(self.compute_units) != len(set(self.compute_units)):
            raise ValueError("compute units must be unique")
        if len(self.supported_phases) != len(set(self.supported_phases)):
            raise ValueError("supported phases must be unique")
        if len(self.supported_operations) != len(set(self.supported_operations)):
            raise ValueError("supported operations must be unique")
        if len(self.supported_dtypes) != len(set(self.supported_dtypes)):
            raise ValueError("supported dtypes must be unique")
        if self.max_program_bytes is not None and self.max_program_bytes <= 0:
            raise ValueError("max_program_bytes must be positive when provided")
