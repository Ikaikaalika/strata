"""Evidence records that keep simulations separate from hardware results."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Tuple

from .ir import InferencePhase


class EvidenceKind(str, Enum):
    CORRECTNESS = "correctness"
    SYNTHETIC = "synthetic"
    HARDWARE = "hardware"


@dataclass(frozen=True)
class Measurement:
    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("measurement name and unit must not be empty")


@dataclass(frozen=True)
class EvidenceRecord:
    """One reproducible correctness or performance observation."""

    evidence_id: str
    kind: EvidenceKind
    model_id: str
    plan_id: str
    phase: InferencePhase
    hardware_fingerprint: str
    measurements: Tuple[Measurement, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "evidence_id",
            "model_id",
            "plan_id",
            "hardware_fingerprint",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if not self.measurements:
            raise ValueError("evidence records must contain measurements")
        names = [measurement.name for measurement in self.measurements]
        if len(names) != len(set(names)):
            raise ValueError("measurement names must be unique within one record")
