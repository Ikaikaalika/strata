"""Runtime-neutral planning contracts for Strata."""

from .adaptive_plan import AdaptiveExecutionPlan, BackendTarget, ExecutionSegment
from .capabilities import RuntimeCapabilities
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
from .tensor_ref import TensorRef

__all__ = [
    "AdaptiveExecutionPlan",
    "BackendTarget",
    "ComputeUnit",
    "EvidenceKind",
    "EvidenceRecord",
    "ExecutionPlan",
    "ExecutionSegment",
    "HardwareProfile",
    "IROperation",
    "InferencePhase",
    "Measurement",
    "ModelSpec",
    "OperationKind",
    "PlanStep",
    "RuntimeCapabilities",
    "StrataIRGraph",
    "TensorRole",
    "TensorRef",
    "TensorSpec",
    "WeightGroup",
]
