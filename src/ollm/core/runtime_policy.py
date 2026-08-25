"""Cross-runtime policy for full-resident, evidence-gated inference."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Tuple

from .adaptive_plan import BackendTarget
from .capabilities import RuntimeCapabilities
from .hardware import ComputeUnit
from .platform import ServiceObjective
from .ssd_offload import SSDOffloadMode, SSDOffloadPolicy


class OptimizationMode(str, Enum):
    MAXIMUM_SPEED = "maximum_speed"
    BALANCED = "balanced"
    CAPACITY = "capacity"


class ANEExecutionMode(str, Enum):
    """How ANE-backed runtime candidates may participate."""

    DISABLED = "disabled"
    PUBLIC_AUTO = "public_auto"
    PRIVATE_RESEARCH = "private_research"


_ALL_TARGETS = tuple(BackendTarget)


@dataclass(frozen=True)
class RuntimePerformancePolicy:
    """One policy consumed by MLX, Metal, CPU, Core ML, and direct ANE lanes.

    This policy makes compute participation conditional. It never promises that
    every engine runs during a request; a target must support the exact graph,
    pass correctness, and beat the comparable MLX phase measurement.
    """

    optimization_mode: OptimizationMode = OptimizationMode.MAXIMUM_SPEED
    ssd_offload: SSDOffloadPolicy = SSDOffloadPolicy()
    ane_execution: ANEExecutionMode = ANEExecutionMode.PUBLIC_AUTO
    allow_private_apis: bool = False
    allowed_targets: Tuple[BackendTarget, ...] = _ALL_TARGETS
    minimum_speedup_percent: float = 5.0
    require_mlx_fallback: bool = True

    def __post_init__(self) -> None:
        if len(self.allowed_targets) != len(set(self.allowed_targets)):
            raise ValueError("allowed_targets must be unique")
        if not self.allowed_targets:
            raise ValueError("allowed_targets must not be empty")
        if self.minimum_speedup_percent < 0:
            raise ValueError("minimum_speedup_percent must not be negative")
        if self.require_mlx_fallback and BackendTarget.MLX not in self.allowed_targets:
            raise ValueError("MLX must be allowed when an MLX fallback is required")
        if (
            self.optimization_mode is OptimizationMode.MAXIMUM_SPEED
            and self.ssd_offload.mode is not SSDOffloadMode.DISABLED
        ):
            raise ValueError("maximum_speed requires ssd_offload=disabled")
        if (
            self.ane_execution is ANEExecutionMode.PRIVATE_RESEARCH
            and not self.allow_private_apis
        ):
            raise ValueError("private_research ANE requires allow_private_apis=true")

    @classmethod
    def maximum_speed(
        cls,
        *,
        ane_execution: ANEExecutionMode = ANEExecutionMode.PUBLIC_AUTO,
        allow_private_apis: bool = False,
        minimum_speedup_percent: float = 5.0,
        allowed_targets: Tuple[BackendTarget, ...] = _ALL_TARGETS,
    ) -> "RuntimePerformancePolicy":
        return cls(
            optimization_mode=OptimizationMode.MAXIMUM_SPEED,
            ssd_offload=SSDOffloadPolicy(mode=SSDOffloadMode.DISABLED),
            ane_execution=ane_execution,
            allow_private_apis=allow_private_apis,
            allowed_targets=allowed_targets,
            minimum_speedup_percent=minimum_speedup_percent,
        )

    def apply(self, objective: ServiceObjective) -> ServiceObjective:
        """Apply the residency toggle before platform admission."""

        return self.ssd_offload.apply(objective)

    @classmethod
    def from_parameters(
        cls,
        parameters: Mapping[str, Any],
    ) -> "RuntimePerformancePolicy":
        """Parse the strict public runtime-policy parameter object."""

        if not isinstance(parameters, Mapping):
            raise TypeError("runtime policy parameters must be an object")
        expected = {
            "optimization_mode",
            "ssd_offload",
            "ane_execution",
            "allow_private_apis",
            "allowed_targets",
            "minimum_speedup_percent",
            "require_mlx_fallback",
        }
        unknown = set(parameters) - expected
        if unknown:
            raise ValueError(f"unknown runtime policy parameters: {sorted(unknown)}")
        ssd_parameters = parameters.get("ssd_offload", {"mode": "disabled"})
        if not isinstance(ssd_parameters, Mapping):
            raise TypeError("ssd_offload must be an object")
        ssd_expected = {
            "mode",
            "storage_target_id",
            "max_resident_weight_bytes",
            "prefetch_distance",
            "io_workers",
        }
        ssd_unknown = set(ssd_parameters) - ssd_expected
        if ssd_unknown:
            raise ValueError(f"unknown SSD offload parameters: {sorted(ssd_unknown)}")
        raw_targets = parameters.get(
            "allowed_targets", [target.value for target in _ALL_TARGETS]
        )
        if not isinstance(raw_targets, (list, tuple)):
            raise TypeError("allowed_targets must be an array")
        try:
            allowed_targets = tuple(BackendTarget(value) for value in raw_targets)
            ssd_policy = SSDOffloadPolicy(
                mode=SSDOffloadMode(ssd_parameters.get("mode", "disabled")),
                storage_target_id=ssd_parameters.get("storage_target_id"),
                max_resident_weight_bytes=ssd_parameters.get(
                    "max_resident_weight_bytes"
                ),
                prefetch_distance=ssd_parameters.get("prefetch_distance", 1),
                io_workers=ssd_parameters.get("io_workers", 1),
            )
            optimization_mode = OptimizationMode(
                parameters.get("optimization_mode", "maximum_speed")
            )
            ane_execution = ANEExecutionMode(
                parameters.get("ane_execution", "public_auto")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid runtime policy enum or SSD parameter: {exc}") from exc
        return cls(
            optimization_mode=optimization_mode,
            ssd_offload=ssd_policy,
            ane_execution=ane_execution,
            allow_private_apis=parameters.get("allow_private_apis", False),
            allowed_targets=allowed_targets,
            minimum_speedup_percent=parameters.get("minimum_speedup_percent", 5.0),
            require_mlx_fallback=parameters.get("require_mlx_fallback", True),
        )

    def to_parameters(self) -> dict[str, Any]:
        return {
            "optimization_mode": self.optimization_mode.value,
            "ssd_offload": {
                "mode": self.ssd_offload.mode.value,
                "storage_target_id": self.ssd_offload.storage_target_id,
                "max_resident_weight_bytes": self.ssd_offload.max_resident_weight_bytes,
                "prefetch_distance": self.ssd_offload.prefetch_distance,
                "io_workers": self.ssd_offload.io_workers,
            },
            "ane_execution": self.ane_execution.value,
            "allow_private_apis": self.allow_private_apis,
            "allowed_targets": [target.value for target in self.allowed_targets],
            "minimum_speedup_percent": self.minimum_speedup_percent,
            "require_mlx_fallback": self.require_mlx_fallback,
        }

    @property
    def coreml_compute_units(self) -> str:
        """Public Core ML configuration required by this ANE policy."""

        if self.ane_execution is ANEExecutionMode.DISABLED:
            return "cpuAndGPU"
        return "all"

    @property
    def identity_digest(self) -> str:
        payload = {
            "allow_private_apis": self.allow_private_apis,
            "allowed_targets": sorted(target.value for target in self.allowed_targets),
            "ane_execution": self.ane_execution.value,
            "minimum_speedup_percent": self.minimum_speedup_percent,
            "optimization_mode": self.optimization_mode.value,
            "require_mlx_fallback": self.require_mlx_fallback,
            "ssd_offload": {
                "io_workers": self.ssd_offload.io_workers,
                "max_resident_weight_bytes": self.ssd_offload.max_resident_weight_bytes,
                "mode": self.ssd_offload.mode.value,
                "prefetch_distance": self.ssd_offload.prefetch_distance,
                "storage_target_id": self.ssd_offload.storage_target_id,
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def target_rejection_reason(
        self,
        target: BackendTarget,
        capability: RuntimeCapabilities | None,
    ) -> str | None:
        if target not in self.allowed_targets:
            return f"target {target.value!r} is disabled by runtime policy"
        if capability is None:
            return None
        uses_ane = ComputeUnit.ANE in capability.compute_units
        if uses_ane and self.ane_execution is ANEExecutionMode.DISABLED:
            return "ANE is disabled by runtime policy"
        if capability.requires_private_api:
            if self.ane_execution is not ANEExecutionMode.PRIVATE_RESEARCH:
                return "private ANE requires private_research policy"
            if not self.allow_private_apis:
                return "private APIs are disabled by runtime policy"
        return None


__all__ = [
    "ANEExecutionMode",
    "OptimizationMode",
    "RuntimePerformancePolicy",
]
