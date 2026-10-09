"""Deterministic NumPy reference executor for a bounded LokahiIR subset.

This module is a correctness oracle, not the optimized Lokahi CPU backend.  It
uses float32 intermediates for floating-point arithmetic and casts each result
to the dtype declared by the graph so accelerator backends can compare against
well-defined semantics.
"""
from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np

from ..core.capabilities import RuntimeCapabilities
from ..core.hardware import ComputeUnit
from ..core.ir import (
    IROperation,
    InferencePhase,
    OperationKind,
    LokahiIRGraph,
    TensorRole,
    TensorSpec,
)


_SUPPORTED_OPERATIONS = (
    OperationKind.LINEAR,
    OperationKind.NORMALIZATION,
    OperationKind.RESIDUAL,
    OperationKind.ACTIVATION,
    OperationKind.LOGITS,
)
_SUPPORTED_DTYPES = ("float16", "float32")
_SOURCE_ROLES = (TensorRole.INPUT, TensorRole.WEIGHT, TensorRole.STATE)


class CPUReferenceError(RuntimeError):
    """Raised when a graph cannot be executed with reference semantics."""


def cpu_reference_capabilities() -> RuntimeCapabilities:
    """Return conservative capabilities for the deterministic NumPy oracle."""

    return RuntimeCapabilities(
        compute_units=(ComputeUnit.CPU,),
        supported_phases=(InferencePhase.PREFILL, InferencePhase.DECODE),
        supported_operations=_SUPPORTED_OPERATIONS,
        supported_dtypes=_SUPPORTED_DTYPES,
        supports_dynamic_shapes=True,
    )


class CPUReferenceExecutor:
    """Execute a small, deterministic LokahiIR graph using NumPy.

    Inputs include graph inputs plus any weights or state consumed directly by
    operations.  Unknown feeds, feeds for intermediates, and missing sources
    fail closed so misspelled tensor names cannot silently change a reference
    result.
    """

    def capabilities(self) -> RuntimeCapabilities:
        return cpu_reference_capabilities()

    def execute(
        self,
        graph: LokahiIRGraph,
        tensors: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Execute ``graph`` and return its declared outputs by name."""

        specs = graph.tensors_by_name()
        self._validate_graph(graph)
        values, symbols = self._validate_feeds(graph, specs, tensors)

        for operation in graph.operations:
            if any(name not in values for name in operation.inputs):
                missing = sorted(name for name in operation.inputs if name not in values)
                raise CPUReferenceError(
                    f"operation {operation.operation_id!r} is missing runtime tensors: "
                    f"{missing}"
                )
            if any(name in values for name in operation.outputs):
                duplicate = sorted(name for name in operation.outputs if name in values)
                raise CPUReferenceError(
                    f"operation {operation.operation_id!r} overwrites materialized "
                    f"tensors: {duplicate}"
                )

            inputs = tuple(values[name] for name in operation.inputs)
            results = self._execute_operation(operation, inputs)
            if len(results) != len(operation.outputs):
                raise CPUReferenceError(
                    f"operation {operation.operation_id!r} produced {len(results)} "
                    f"results for {len(operation.outputs)} outputs"
                )

            for name, result in zip(operation.outputs, results):
                spec = specs[name]
                cast_result = self._cast_result(operation, spec, result)
                self._validate_shape(spec, cast_result.shape, symbols)
                values[name] = cast_result

        return {name: values[name] for name in graph.outputs}

    def _validate_graph(self, graph: LokahiIRGraph) -> None:
        unsupported = [
            operation
            for operation in graph.operations
            if operation.kind not in _SUPPORTED_OPERATIONS
        ]
        if unsupported:
            labels = ", ".join(
                f"{operation.operation_id}:{operation.kind.value}"
                for operation in unsupported
            )
            raise CPUReferenceError(f"unsupported CPU reference operations: {labels}")

        for tensor in graph.tensors:
            if tensor.normalized_dtype not in _SUPPORTED_DTYPES:
                raise CPUReferenceError(
                    f"tensor {tensor.name!r} uses unsupported CPU reference dtype "
                    f"{tensor.dtype!r}"
                )

        for operation in graph.operations:
            if len(operation.outputs) != 1:
                raise CPUReferenceError(
                    f"operation {operation.operation_id!r} must have exactly one output"
                )

    def _validate_feeds(
        self,
        graph: LokahiIRGraph,
        specs: Mapping[str, TensorSpec],
        tensors: Mapping[str, np.ndarray],
    ) -> tuple[MutableMapping[str, np.ndarray], MutableMapping[str, int]]:
        feed_names = set(tensors)
        unknown = sorted(feed_names - specs.keys())
        if unknown:
            raise CPUReferenceError(f"unknown runtime tensors: {unknown}")

        invalid_roles = sorted(
            name for name in feed_names if specs[name].role not in _SOURCE_ROLES
        )
        if invalid_roles:
            raise CPUReferenceError(
                f"runtime feeds may not materialize output/intermediate tensors: "
                f"{invalid_roles}"
            )

        required = set(graph.inputs)
        produced = {
            name for operation in graph.operations for name in operation.outputs
        }
        required.update(
            name
            for operation in graph.operations
            for name in operation.inputs
            if name not in produced
        )
        required.update(name for name in graph.outputs if name not in produced)
        missing = sorted(required - feed_names)
        if missing:
            raise CPUReferenceError(f"missing runtime tensors: {missing}")

        values: MutableMapping[str, np.ndarray] = {}
        symbols: MutableMapping[str, int] = {}
        for name, raw_value in tensors.items():
            value = np.asarray(raw_value)
            spec = specs[name]
            expected_dtype = np.dtype(spec.normalized_dtype)
            if value.dtype != expected_dtype:
                raise CPUReferenceError(
                    f"tensor {name!r} has dtype {value.dtype}, expected "
                    f"{expected_dtype}"
                )
            self._validate_shape(spec, value.shape, symbols)
            values[name] = value
        return values, symbols

    def _validate_shape(
        self,
        spec: TensorSpec,
        actual_shape: Sequence[int],
        symbols: MutableMapping[str, int],
    ) -> None:
        if len(actual_shape) != len(spec.shape):
            raise CPUReferenceError(
                f"tensor {spec.name!r} has rank {len(actual_shape)}, expected "
                f"{len(spec.shape)} for shape {spec.shape}"
            )
        for expected, actual in zip(spec.shape, actual_shape):
            actual = int(actual)
            if isinstance(expected, int):
                if actual != expected:
                    raise CPUReferenceError(
                        f"tensor {spec.name!r} has shape {tuple(actual_shape)}, "
                        f"expected {spec.shape}"
                    )
                continue
            bound = symbols.get(expected)
            if bound is None:
                symbols[expected] = actual
            elif bound != actual:
                raise CPUReferenceError(
                    f"tensor {spec.name!r} binds symbolic dimension {expected!r} "
                    f"to {actual}, but it is already bound to {bound}"
                )

    def _execute_operation(
        self,
        operation: IROperation,
        inputs: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        if operation.kind in (OperationKind.LINEAR, OperationKind.LOGITS):
            return (self._linear(operation, inputs),)
        if operation.kind is OperationKind.NORMALIZATION:
            return (self._rms_norm(operation, inputs),)
        if operation.kind is OperationKind.RESIDUAL:
            return (self._residual(operation, inputs),)
        if operation.kind is OperationKind.ACTIVATION:
            return (self._silu(operation, inputs),)
        # Graph validation makes this unreachable, but retain a fail-closed guard.
        raise CPUReferenceError(
            f"operation {operation.operation_id!r} has unsupported kind "
            f"{operation.kind.value!r}"
        )

    def _linear(
        self,
        operation: IROperation,
        inputs: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        self._expect_input_count(operation, inputs, (2, 3))
        activation, weight = inputs[:2]
        if activation.ndim < 1 or weight.ndim != 2:
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} requires an activation with "
                "rank >= 1 and a rank-2 weight"
            )

        layout = str(operation.attributes.get("weight_layout", "out_in")).lower()
        if layout == "out_in":
            input_features = weight.shape[1]
            output_features = weight.shape[0]
            weight_matrix = weight.astype(np.float32, copy=False).T
        elif layout == "in_out":
            input_features = weight.shape[0]
            output_features = weight.shape[1]
            weight_matrix = weight.astype(np.float32, copy=False)
        else:
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} has unsupported weight_layout "
                f"{layout!r}"
            )

        if activation.shape[-1] != input_features:
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} activation width "
                f"{activation.shape[-1]} does not match weight input width "
                f"{input_features}"
            )

        result = np.matmul(
            activation.astype(np.float32, copy=False),
            weight_matrix,
        )
        if len(inputs) == 3:
            bias = inputs[2]
            if bias.shape != (output_features,):
                raise CPUReferenceError(
                    f"operation {operation.operation_id!r} bias has shape "
                    f"{bias.shape}, expected {(output_features,)}"
                )
            result = result + bias.astype(np.float32, copy=False)
        return result

    def _rms_norm(
        self,
        operation: IROperation,
        inputs: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        self._expect_input_count(operation, inputs, (1, 2))
        norm_type = str(
            operation.attributes.get(
                "normalization",
                operation.attributes.get("type", "rmsnorm"),
            )
        ).lower()
        if norm_type not in {"rmsnorm", "rms_norm"}:
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} has unsupported normalization "
                f"{norm_type!r}"
            )

        epsilon = float(
            operation.attributes.get("epsilon", operation.attributes.get("eps", 1e-5))
        )
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} epsilon must be finite and "
                "positive"
            )

        activation = inputs[0]
        if activation.ndim < 1:
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} requires rank >= 1 input"
            )
        activation32 = activation.astype(np.float32, copy=False)
        mean_square = np.mean(
            activation32 * activation32,
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        result = activation32 / np.sqrt(mean_square + np.float32(epsilon))
        if len(inputs) == 2:
            scale = inputs[1]
            if scale.shape != (activation.shape[-1],):
                raise CPUReferenceError(
                    f"operation {operation.operation_id!r} RMSNorm scale has shape "
                    f"{scale.shape}, expected {(activation.shape[-1],)}"
                )
            result = result * scale.astype(np.float32, copy=False)
        return result

    def _residual(
        self,
        operation: IROperation,
        inputs: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        self._expect_input_count(operation, inputs, (2,))
        left, right = inputs
        if left.shape != right.shape:
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} requires equal residual "
                f"shapes, got {left.shape} and {right.shape}"
            )
        return left.astype(np.float32, copy=False) + right.astype(
            np.float32, copy=False
        )

    def _silu(
        self,
        operation: IROperation,
        inputs: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        self._expect_input_count(operation, inputs, (1,))
        activation_name = str(
            operation.attributes.get(
                "activation",
                operation.attributes.get("type", "silu"),
            )
        ).lower()
        if activation_name not in {"silu", "swish"}:
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} has unsupported activation "
                f"{activation_name!r}"
            )

        activation = inputs[0].astype(np.float32, copy=False)
        sigmoid = np.empty_like(activation, dtype=np.float32)
        nonnegative = activation >= 0
        sigmoid[nonnegative] = 1.0 / (1.0 + np.exp(-activation[nonnegative]))
        exponential = np.exp(activation[~nonnegative])
        sigmoid[~nonnegative] = exponential / (1.0 + exponential)
        return activation * sigmoid

    def _cast_result(
        self,
        operation: IROperation,
        spec: TensorSpec,
        result: np.ndarray,
    ) -> np.ndarray:
        try:
            target_dtype = np.dtype(spec.normalized_dtype)
        except TypeError as exc:  # pragma: no cover - guarded by graph validation
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} has invalid output dtype "
                f"{spec.dtype!r}"
            ) from exc
        return np.asarray(result, dtype=target_dtype)

    def _expect_input_count(
        self,
        operation: IROperation,
        inputs: tuple[np.ndarray, ...],
        allowed: tuple[int, ...],
    ) -> None:
        if len(inputs) not in allowed:
            rendered = " or ".join(str(count) for count in allowed)
            raise CPUReferenceError(
                f"operation {operation.operation_id!r} expects {rendered} inputs, "
                f"got {len(inputs)}"
            )


__all__ = [
    "CPUReferenceError",
    "CPUReferenceExecutor",
    "cpu_reference_capabilities",
]
