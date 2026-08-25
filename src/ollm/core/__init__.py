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
from .model_manifest import (
    ArchitectureClass,
    ArchitectureSpec,
    ArtifactKind,
    AttentionPattern,
    AttentionSchedule,
    AttentionSpec,
    FeedForwardKind,
    FeedForwardSpec,
    PortableModelManifest,
    PromptProtocol,
    TokenizerSpec,
)
from .ssd_offload import SSDOffloadMode, SSDOffloadPolicy
from .runtime_policy import (
    ANEExecutionMode,
    OptimizationMode,
    RuntimePerformancePolicy,
)
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
    "ANEExecutionMode",
    "ArchitectureClass",
    "ArchitectureSpec",
    "ArtifactKind",
    "AttentionPattern",
    "AttentionSchedule",
    "AttentionSpec",
    "ComputeUnit",
    "DeploymentMode",
    "EvidenceKind",
    "EvidenceRecord",
    "ExecutionPlan",
    "ExecutionSegment",
    "FeedForwardKind",
    "FeedForwardSpec",
    "HardwareProfile",
    "IROperation",
    "InferencePhase",
    "LivePlatformState",
    "Measurement",
    "MemoryPressure",
    "ModelSpec",
    "OperationKind",
    "OperationEnvelope",
    "OptimizationMode",
    "PlanStep",
    "PowerSource",
    "PortableModelManifest",
    "SSDOffloadMode",
    "SSDOffloadPolicy",
    "PromptProtocol",
    "ResidencyMode",
    "ResidencyPreference",
    "RuntimeDemand",
    "RuntimeCapabilities",
    "RuntimePerformancePolicy",
    "ServiceObjective",
    "StorageMedium",
    "StorageTarget",
    "StrataIRGraph",
    "TensorRole",
    "TensorRef",
    "TensorSpec",
    "TokenizerSpec",
    "ThermalState",
    "WeightGroup",
    "WorkloadClass",
]
