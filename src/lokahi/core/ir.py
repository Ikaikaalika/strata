"""Canonical, runtime-neutral operation graph for Lokahi models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Dict, Mapping, Tuple, Union


Dimension = Union[int, str]


class InferencePhase(str, Enum):
    """The two performance regimes of autoregressive inference."""

    PREFILL = "prefill"
    DECODE = "decode"


class OperationKind(str, Enum):
    """Operations that model adapters may lower into backend segments."""

    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    LINEAR = "linear"
    ROPE = "rope"
    ATTENTION = "attention"
    RESIDUAL = "residual"
    ACTIVATION = "activation"
    MLP = "mlp"
    ROUTER = "router"
    EXPERT = "expert"
    CONVOLUTION = "convolution"
    STATE_SPACE = "state_space"
    LOGITS = "logits"


class TensorRole(str, Enum):
    """Lifetime and ownership role of a logical tensor."""

    INPUT = "input"
    WEIGHT = "weight"
    STATE = "state"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"


_DTYPE_BITS = {
    "bool": 8,
    "int4": 4,
    "uint4": 4,
    "int8": 8,
    "uint8": 8,
    "int16": 16,
    "uint16": 16,
    "float16": 16,
    "bfloat16": 16,
    "int32": 32,
    "uint32": 32,
    "float32": 32,
}


@dataclass(frozen=True)
class TensorSpec:
    """A logical tensor whose dimensions may be static or symbolic."""

    name: str
    shape: Tuple[Dimension, ...]
    dtype: str
    role: TensorRole
    layout: str = "logical"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if not self.shape:
            raise ValueError("tensor shape must not be empty")
        for dimension in self.shape:
            if isinstance(dimension, int) and dimension <= 0:
                raise ValueError("static tensor dimensions must be positive")
            if isinstance(dimension, str) and not dimension:
                raise ValueError("symbolic tensor dimensions must not be empty")
            if not isinstance(dimension, (int, str)):
                raise TypeError("tensor dimensions must be integers or symbols")
        if self.normalized_dtype not in _DTYPE_BITS:
            raise ValueError(f"unsupported LokahiIR dtype {self.dtype!r}")
        if not self.layout:
            raise ValueError("tensor layout must not be empty")

    @property
    def normalized_dtype(self) -> str:
        return self.dtype.lower().replace("mlx.core.", "").replace("torch.", "")

    @property
    def static_nbytes(self) -> int | None:
        """Return storage bytes when every dimension is statically known."""
        if any(isinstance(dimension, str) for dimension in self.shape):
            return None
        elements = reduce(mul, self.shape, 1)
        bits = elements * _DTYPE_BITS[self.normalized_dtype]
        return (bits + 7) // 8


@dataclass(frozen=True)
class IROperation:
    """One topologically ordered operation in a LokahiIR graph."""

    operation_id: str
    kind: OperationKind
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must not be empty")
        if not self.inputs:
            raise ValueError("operations must consume at least one tensor")
        if not self.outputs:
            raise ValueError("operations must produce at least one tensor")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError(f"operation {self.operation_id!r} has duplicate outputs")


@dataclass(frozen=True)
class LokahiIRGraph:
    """Validated graph shared by model adapters and compute backends."""

    graph_id: str
    tensors: Tuple[TensorSpec, ...]
    operations: Tuple[IROperation, ...]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise ValueError("graph_id must not be empty")
        tensors = self.tensors_by_name()
        if len(tensors) != len(self.tensors):
            raise ValueError("graph tensor names must be unique")
        operation_ids = [operation.operation_id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("graph operation IDs must be unique")
        for name in (*self.inputs, *self.outputs):
            if name not in tensors:
                raise ValueError(f"graph references unknown tensor {name!r}")

        available = {
            tensor.name
            for tensor in self.tensors
            if tensor.role in {TensorRole.INPUT, TensorRole.WEIGHT, TensorRole.STATE}
        }
        if not set(self.inputs).issubset(available):
            raise ValueError("graph inputs must have input, weight, or state roles")
        for operation in self.operations:
            unknown_inputs = set(operation.inputs) - available
            if unknown_inputs:
                raise ValueError(
                    f"operation {operation.operation_id!r} consumes tensors before "
                    f"they are available: {sorted(unknown_inputs)}"
                )
            unknown_outputs = set(operation.outputs) - tensors.keys()
            if unknown_outputs:
                raise ValueError(
                    f"operation {operation.operation_id!r} produces unknown tensors: "
                    f"{sorted(unknown_outputs)}"
                )
            available.update(operation.outputs)
        if not set(self.outputs).issubset(available):
            raise ValueError("graph outputs must be inputs or produced tensors")

    def tensors_by_name(self) -> Dict[str, TensorSpec]:
        return {tensor.name: tensor for tensor in self.tensors}

    def operations_by_id(self) -> Dict[str, IROperation]:
        return {operation.operation_id: operation for operation in self.operations}
