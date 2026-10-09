"""Runtime-neutral Apple Silicon hardware identity and measurements."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple


class ComputeUnit(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    ANE = "ane"


@dataclass(frozen=True)
class HardwareProfile:
    """Facts and measured ceilings used to key adaptive execution plans."""

    chip: str
    macos_version: str
    macos_build: str
    unified_memory_bytes: int
    cpu_logical_cores: int
    compute_units: Tuple[ComputeUnit, ...]
    gpu_recommended_working_set_bytes: Optional[int] = None
    ane_runtime_fingerprint: Optional[str] = None
    storage_read_bytes_per_second: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.chip or not self.macos_version or not self.macos_build:
            raise ValueError("chip and macOS identity fields must not be empty")
        if self.unified_memory_bytes <= 0:
            raise ValueError("unified_memory_bytes must be positive")
        if self.cpu_logical_cores <= 0:
            raise ValueError("cpu_logical_cores must be positive")
        if not self.compute_units:
            raise ValueError("at least one compute unit must be declared")
        if len(self.compute_units) != len(set(self.compute_units)):
            raise ValueError("compute units must be unique")
        if (
            self.gpu_recommended_working_set_bytes is not None
            and self.gpu_recommended_working_set_bytes <= 0
        ):
            raise ValueError("GPU working set must be positive when provided")
        if (
            self.storage_read_bytes_per_second is not None
            and self.storage_read_bytes_per_second <= 0
        ):
            raise ValueError("storage bandwidth must be positive when provided")

    @property
    def fingerprint(self) -> str:
        """Stable plan-cache key; measurements deliberately affect the key."""
        payload = {
            "ane": self.ane_runtime_fingerprint,
            "build": self.macos_build,
            "chip": self.chip,
            "compute_units": sorted(unit.value for unit in self.compute_units),
            "cpu_logical_cores": self.cpu_logical_cores,
            "gpu_working_set": self.gpu_recommended_working_set_bytes,
            "macos": self.macos_version,
            "storage_bps": self.storage_read_bytes_per_second,
            "unified_memory": self.unified_memory_bytes,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
