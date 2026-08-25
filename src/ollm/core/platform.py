"""Live platform and workload contracts for adaptive admission.

Static hardware facts belong in :mod:`ollm.core.hardware` and key reusable
performance evidence.  The values here are deliberately short-lived: Common
Compute (or another host) refreshes them before admitting a request.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .hardware import HardwareProfile


class WorkloadClass(str, Enum):
    INTERACTIVE = "interactive"
    THROUGHPUT = "throughput"
    BACKGROUND = "background"


class DeploymentMode(str, Enum):
    SUPPORTED = "supported"
    RESEARCH = "research"


class ThermalState(str, Enum):
    NOMINAL = "nominal"
    FAIR = "fair"
    SERIOUS = "serious"
    CRITICAL = "critical"


class MemoryPressure(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class PowerSource(str, Enum):
    AC = "ac"
    BATTERY = "battery"
    UNKNOWN = "unknown"


class StorageMedium(str, Enum):
    INTERNAL_SSD = "internal_ssd"
    EXTERNAL_SSD = "external_ssd"
    HDD = "hdd"
    UNKNOWN = "unknown"


class ResidencyMode(str, Enum):
    FULL = "full"
    PAGED = "paged"


class ResidencyPreference(str, Enum):
    """Caller policy; the admission decision resolves it to a concrete mode."""

    AUTO = "auto"
    FULL = "full"
    PAGED = "paged"


class AdmissionReason(str, Enum):
    FULL_RESIDENCY = "full_residency"
    SSD_PAGING = "ssd_paging"
    HARDWARE_STATE_MISMATCH = "hardware_state_mismatch"
    THERMAL_PRESSURE = "thermal_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    POWER_POLICY = "power_policy"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    FULL_RESIDENCY_REQUIRED = "full_residency_required"
    SPILL_DISABLED = "spill_disabled"
    NO_QUALIFIED_SSD = "no_qualified_ssd"
    SSD_THROUGHPUT_BOUND = "ssd_throughput_bound"


@dataclass(frozen=True)
class ServiceObjective:
    """The result a host wants, independent of a particular backend."""

    workload_class: WorkloadClass
    context_tokens: int
    max_output_tokens: int
    batch_size: int = 1
    target_ttft_ms: Optional[float] = None
    min_decode_tokens_per_second: Optional[float] = None
    deadline_ms: Optional[float] = None
    max_runtime_memory_bytes: Optional[int] = None
    max_resident_weight_bytes: Optional[int] = None
    residency_preference: ResidencyPreference = ResidencyPreference.AUTO
    preferred_storage_target_id: Optional[str] = None
    allow_weight_spill: bool = False
    allow_battery: bool = False
    allow_low_power_mode: bool = False
    deployment_mode: DeploymentMode = DeploymentMode.SUPPORTED

    def __post_init__(self) -> None:
        if self.context_tokens < 0:
            raise ValueError("context_tokens must not be negative")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for field_name in (
            "target_ttft_ms",
            "min_decode_tokens_per_second",
            "deadline_ms",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when provided")
        if (
            self.max_runtime_memory_bytes is not None
            and self.max_runtime_memory_bytes <= 0
        ):
            raise ValueError(
                "max_runtime_memory_bytes must be positive when provided"
            )
        if (
            self.max_resident_weight_bytes is not None
            and self.max_resident_weight_bytes <= 0
        ):
            raise ValueError(
                "max_resident_weight_bytes must be positive when provided"
            )
        if self.preferred_storage_target_id == "":
            raise ValueError("preferred_storage_target_id must not be empty")


@dataclass(frozen=True)
class RuntimeDemand:
    """Measured or statically derived memory demand for one model request."""

    model_weight_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    temporary_bytes: int
    minimum_weight_window_bytes: int
    storage_bytes_required: int = 0
    paging_bytes_per_decode_token: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "model_weight_bytes",
            "kv_cache_bytes",
            "activation_bytes",
            "temporary_bytes",
            "minimum_weight_window_bytes",
            "storage_bytes_required",
            "paging_bytes_per_decode_token",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.model_weight_bytes <= 0:
            raise ValueError("model_weight_bytes must be positive")
        if self.minimum_weight_window_bytes <= 0:
            raise ValueError("minimum_weight_window_bytes must be positive")
        if self.minimum_weight_window_bytes > self.model_weight_bytes:
            raise ValueError(
                "minimum_weight_window_bytes must not exceed model_weight_bytes"
            )

    @property
    def full_residency_bytes(self) -> int:
        return (
            self.model_weight_bytes
            + self.kv_cache_bytes
            + self.activation_bytes
            + self.temporary_bytes
        )

    @property
    def minimum_working_set_bytes(self) -> int:
        return (
            self.minimum_weight_window_bytes
            + self.kv_cache_bytes
            + self.activation_bytes
            + self.temporary_bytes
        )


@dataclass(frozen=True)
class StorageTarget:
    """Opaque host-approved storage; paths never enter the planning contract."""

    target_id: str
    medium: StorageMedium
    user_approved: bool
    free_bytes: int
    measured_read_bytes_per_second: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if self.free_bytes < 0:
            raise ValueError("free_bytes must not be negative")
        if (
            self.measured_read_bytes_per_second is not None
            and self.measured_read_bytes_per_second <= 0
        ):
            raise ValueError(
                "measured_read_bytes_per_second must be positive when provided"
            )

    @property
    def qualified_for_adaptive_paging(self) -> bool:
        return (
            self.user_approved
            and self.medium
            in {StorageMedium.INTERNAL_SSD, StorageMedium.EXTERNAL_SSD}
            and self.measured_read_bytes_per_second is not None
        )


@dataclass(frozen=True)
class LivePlatformState:
    """A per-admission snapshot; it must not be part of the hardware fingerprint."""

    hardware_fingerprint: str
    available_memory_bytes: int
    gpu_allocated_bytes: int
    memory_pressure: MemoryPressure
    thermal_state: ThermalState
    power_source: PowerSource
    low_power_mode: bool
    storage_targets: Tuple[StorageTarget, ...] = ()

    def __post_init__(self) -> None:
        if not self.hardware_fingerprint:
            raise ValueError("hardware_fingerprint must not be empty")
        if self.available_memory_bytes < 0:
            raise ValueError("available_memory_bytes must not be negative")
        if self.gpu_allocated_bytes < 0:
            raise ValueError("gpu_allocated_bytes must not be negative")
        target_ids = [target.target_id for target in self.storage_targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("storage target IDs must be unique")


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    memory_budget_bytes: int
    weight_residency_budget_bytes: int
    residency_mode: Optional[ResidencyMode]
    storage_target_id: Optional[str]
    private_backends_allowed: bool
    reasons: Tuple[AdmissionReason, ...]
    estimated_storage_decode_ceiling_tps: Optional[float] = None

    def __post_init__(self) -> None:
        if self.memory_budget_bytes < 0:
            raise ValueError("memory_budget_bytes must not be negative")
        if self.weight_residency_budget_bytes < 0:
            raise ValueError("weight_residency_budget_bytes must not be negative")
        if self.weight_residency_budget_bytes > self.memory_budget_bytes:
            raise ValueError(
                "weight residency budget must not exceed total memory budget"
            )
        if self.admitted:
            if self.residency_mode is None:
                raise ValueError("admitted decisions require a residency mode")
            if not self.reasons:
                raise ValueError("admitted decisions require an admission reason")
        elif self.residency_mode is not None or self.storage_target_id is not None:
            raise ValueError("rejected decisions must not select residency or storage")
        if (
            self.residency_mode is ResidencyMode.PAGED
            and not self.storage_target_id
        ):
            raise ValueError("paged decisions require a storage target")
        if (
            self.residency_mode is not ResidencyMode.PAGED
            and self.storage_target_id is not None
        ):
            raise ValueError("only paged decisions may select storage")
        if (
            self.estimated_storage_decode_ceiling_tps is not None
            and self.estimated_storage_decode_ceiling_tps <= 0
        ):
            raise ValueError("storage decode ceiling must be positive when provided")
        if (
            self.residency_mode is not ResidencyMode.PAGED
            and self.estimated_storage_decode_ceiling_tps is not None
        ):
            raise ValueError("only paged decisions may have a storage decode ceiling")


@dataclass(frozen=True)
class AdaptiveAdmissionPolicy:
    """Fail-closed admission before the evidence-gated backend planner runs.

    The GPU headroom bound is retained even for heterogeneous plans because MLX
    is the mandatory recovery path.  Backend selection and performance scoring
    remain the responsibility of :class:`ollm.planning.AdaptivePlanner`.
    """

    system_reserve_bytes: int
    max_unified_memory_fraction: float = 0.85

    def __post_init__(self) -> None:
        if self.system_reserve_bytes < 0:
            raise ValueError("system_reserve_bytes must not be negative")
        if not 0 < self.max_unified_memory_fraction <= 1:
            raise ValueError("max_unified_memory_fraction must be in (0, 1]")

    def decide(
        self,
        *,
        hardware: HardwareProfile,
        platform: LivePlatformState,
        objective: ServiceObjective,
        demand: RuntimeDemand,
    ) -> AdmissionDecision:
        private_allowed = objective.deployment_mode is DeploymentMode.RESEARCH
        budget = self._memory_budget(hardware, platform, objective)
        weight_budget = self._weight_budget(budget, objective, demand)

        if platform.hardware_fingerprint != hardware.fingerprint:
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.HARDWARE_STATE_MISMATCH,
            )
        if platform.thermal_state in {ThermalState.SERIOUS, ThermalState.CRITICAL}:
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.THERMAL_PRESSURE,
            )
        if platform.memory_pressure is MemoryPressure.CRITICAL:
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.MEMORY_PRESSURE,
            )
        if platform.power_source is PowerSource.BATTERY and not objective.allow_battery:
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.POWER_POLICY,
            )
        if platform.low_power_mode and not objective.allow_low_power_mode:
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.POWER_POLICY,
            )
        if (
            demand.minimum_working_set_bytes > budget
            or demand.minimum_weight_window_bytes > weight_budget
        ):
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.INSUFFICIENT_MEMORY,
            )
        if (
            objective.residency_preference is not ResidencyPreference.PAGED
            and demand.full_residency_bytes <= budget
            and demand.model_weight_bytes <= weight_budget
        ):
            return AdmissionDecision(
                admitted=True,
                memory_budget_bytes=budget,
                weight_residency_budget_bytes=weight_budget,
                residency_mode=ResidencyMode.FULL,
                storage_target_id=None,
                private_backends_allowed=private_allowed,
                reasons=(AdmissionReason.FULL_RESIDENCY,),
            )
        if objective.residency_preference is ResidencyPreference.FULL:
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.FULL_RESIDENCY_REQUIRED,
            )
        if (
            objective.residency_preference is ResidencyPreference.AUTO
            and not objective.allow_weight_spill
        ):
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.SPILL_DISABLED,
            )

        target = self._select_storage_target(
            platform,
            demand.storage_bytes_required,
            objective.preferred_storage_target_id,
        )
        if target is None:
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.NO_QUALIFIED_SSD,
            )
        storage_ceiling = self._storage_decode_ceiling(target, demand)
        throughput_bound = (
            storage_ceiling is not None
            and objective.min_decode_tokens_per_second is not None
            and storage_ceiling < objective.min_decode_tokens_per_second
        )
        if (
            throughput_bound
            and objective.residency_preference is not ResidencyPreference.PAGED
        ):
            return self._reject(
                budget,
                weight_budget,
                private_allowed,
                AdmissionReason.SSD_THROUGHPUT_BOUND,
            )
        reasons = (
            (AdmissionReason.SSD_PAGING, AdmissionReason.SSD_THROUGHPUT_BOUND)
            if throughput_bound
            else (AdmissionReason.SSD_PAGING,)
        )
        return AdmissionDecision(
            admitted=True,
            memory_budget_bytes=budget,
            weight_residency_budget_bytes=weight_budget,
            residency_mode=ResidencyMode.PAGED,
            storage_target_id=target.target_id,
            private_backends_allowed=private_allowed,
            reasons=reasons,
            estimated_storage_decode_ceiling_tps=storage_ceiling,
        )

    @staticmethod
    def _storage_decode_ceiling(
        target: StorageTarget,
        demand: RuntimeDemand,
    ) -> Optional[float]:
        if demand.paging_bytes_per_decode_token <= 0:
            return None
        assert target.measured_read_bytes_per_second is not None
        return (
            target.measured_read_bytes_per_second
            / demand.paging_bytes_per_decode_token
        )

    def _memory_budget(
        self,
        hardware: HardwareProfile,
        platform: LivePlatformState,
        objective: ServiceObjective,
    ) -> int:
        limits = [
            int(hardware.unified_memory_bytes * self.max_unified_memory_fraction),
            max(0, platform.available_memory_bytes - self.system_reserve_bytes),
        ]
        if hardware.gpu_recommended_working_set_bytes is not None:
            limits.append(
                max(
                    0,
                    hardware.gpu_recommended_working_set_bytes
                    - platform.gpu_allocated_bytes,
                )
            )
        if objective.max_runtime_memory_bytes is not None:
            limits.append(objective.max_runtime_memory_bytes)
        return max(0, min(limits))

    @staticmethod
    def _weight_budget(
        memory_budget_bytes: int,
        objective: ServiceObjective,
        demand: RuntimeDemand,
    ) -> int:
        non_weight_bytes = (
            demand.kv_cache_bytes
            + demand.activation_bytes
            + demand.temporary_bytes
        )
        limits = [
            demand.model_weight_bytes,
            max(0, memory_budget_bytes - non_weight_bytes),
        ]
        if objective.max_resident_weight_bytes is not None:
            limits.append(objective.max_resident_weight_bytes)
        return max(0, min(limits))

    @staticmethod
    def _select_storage_target(
        platform: LivePlatformState,
        required_free_bytes: int,
        preferred_target_id: Optional[str] = None,
    ) -> Optional[StorageTarget]:
        candidates = [
            target
            for target in platform.storage_targets
            if target.qualified_for_adaptive_paging
            and target.free_bytes >= required_free_bytes
            and (
                preferred_target_id is None
                or target.target_id == preferred_target_id
            )
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (
                -float(target.measured_read_bytes_per_second or 0),
                target.target_id,
            ),
        )

    @staticmethod
    def _reject(
        memory_budget_bytes: int,
        weight_residency_budget_bytes: int,
        private_backends_allowed: bool,
        reason: AdmissionReason,
    ) -> AdmissionDecision:
        return AdmissionDecision(
            admitted=False,
            memory_budget_bytes=memory_budget_bytes,
            weight_residency_budget_bytes=weight_residency_budget_bytes,
            residency_mode=None,
            storage_target_id=None,
            private_backends_allowed=private_backends_allowed,
            reasons=(reason,),
        )


__all__ = [
    "AdaptiveAdmissionPolicy",
    "AdmissionDecision",
    "AdmissionReason",
    "DeploymentMode",
    "LivePlatformState",
    "MemoryPressure",
    "PowerSource",
    "ResidencyMode",
    "ResidencyPreference",
    "RuntimeDemand",
    "ServiceObjective",
    "StorageMedium",
    "StorageTarget",
    "ThermalState",
    "WorkloadClass",
]
