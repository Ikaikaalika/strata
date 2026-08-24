"""Runtime-neutral planning contracts for Strata."""

from .adaptive_plan import AdaptiveExecutionPlan, BackendTarget, ExecutionSegment
from .capabilities import OperationEnvelope, RuntimeCapabilities
from .evidence import EvidenceKind, EvidenceRecord, Measurement
from .execution_plan import ExecutionPlan, PlanStep
from .hardware import ComputeUnit, HardwareProfile
from .ir import (
    IROperation,
    InferencePhase,
    OperationKind,
    StrataIRGraph,
    TensorRole,
    TensorSpec,
)
from .model_spec import ModelSpec, WeightGroup
from .platform import (
    AdaptiveAdmissionPolicy,
    AdmissionDecision,
    AdmissionReason,
    DeploymentMode,
    LivePlatformState,
    MemoryPressure,
    PowerSource,
    ResidencyMode,
    ResidencyPreference,
    RuntimeDemand,
    ServiceObjective,
    StorageMedium,
    StorageTarget,
    ThermalState,
    WorkloadClass,
)
from .tensor_ref import TensorRef

__all__ = [
    "AdaptiveExecutionPlan",
    "AdaptiveAdmissionPolicy",
    "AdmissionDecision",
    "AdmissionReason",
    "BackendTarget",
    "ComputeUnit",
    "DeploymentMode",
    "EvidenceKind",
    "EvidenceRecord",
    "ExecutionPlan",
    "ExecutionSegment",
    "HardwareProfile",
    "IROperation",
    "InferencePhase",
    "LivePlatformState",
    "Measurement",
    "MemoryPressure",
    "ModelSpec",
    "OperationKind",
    "OperationEnvelope",
    "PlanStep",
    "PowerSource",
    "ResidencyMode",
    "ResidencyPreference",
    "RuntimeDemand",
    "RuntimeCapabilities",
    "ServiceObjective",
    "StorageMedium",
    "StorageTarget",
    "StrataIRGraph",
    "TensorRole",
    "TensorRef",
    "TensorSpec",
    "ThermalState",
    "WeightGroup",
    "WorkloadClass",
]
