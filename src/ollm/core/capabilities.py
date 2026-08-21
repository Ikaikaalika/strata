"""Capabilities declared by a compute runtime adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .hardware import ComputeUnit
from .ir import InferencePhase, OperationKind


@dataclass(frozen=True)
class OperationEnvelope:
    """One exact logical operation signature implemented by a backend.

    Shapes are ordered to match ``IROperation.inputs`` and ``outputs``.  Every
    dimension is a concrete positive integer: an exact native kernel must not
    silently claim a symbolic graph shape.  Required attributes are stored as
    a canonical tuple so the frozen envelope is immutable and hashable.
    """

    kind: OperationKind
    input_shapes: Tuple[Tuple[int, ...], ...]
    output_shapes: Tuple[Tuple[int, ...], ...]
    required_attributes: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OperationKind):
            raise TypeError("operation envelope kind must be an OperationKind")
        object.__setattr__(
            self,
            "input_shapes",
            self._normalize_shapes(self.input_shapes, "input_shapes"),
        )
        object.__setattr__(
            self,
            "output_shapes",
            self._normalize_shapes(self.output_shapes, "output_shapes"),
        )
        object.__setattr__(
            self,
            "required_attributes",
            self._normalize_attributes(self.required_attributes),
        )

    @staticmethod
    def _normalize_shapes(
        raw_shapes: Tuple[Tuple[int, ...], ...],
        field_name: str,
    ) -> Tuple[Tuple[int, ...], ...]:
        try:
            shapes = tuple(tuple(shape) for shape in raw_shapes)
        except TypeError as exc:
            raise TypeError(f"operation envelope {field_name} must be sequences") from exc
        if not shapes:
            raise ValueError(f"operation envelope {field_name} must not be empty")
        for shape in shapes:
            if not shape:
                raise ValueError("operation envelope tensor shapes must not be empty")
            for dimension in shape:
                if type(dimension) is not int or dimension <= 0:
                    raise ValueError(
                        "operation envelope dimensions must be exact positive integers"
                    )
        return shapes

    @staticmethod
    def _normalize_attributes(
        raw_attributes: Tuple[Tuple[str, Any], ...],
    ) -> Tuple[Tuple[str, Any], ...]:
        if isinstance(raw_attributes, Mapping):
            raw_items = tuple(raw_attributes.items())
        else:
            try:
                raw_items = tuple(raw_attributes)
            except TypeError as exc:
                raise TypeError(
                    "operation envelope required_attributes must be key/value pairs"
                ) from exc

        attributes = []
        for item in raw_items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    "operation envelope required_attributes must be key/value pairs"
                )
            name, value = item
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "operation envelope required attribute names must not be empty"
                )
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError(
                    f"operation envelope attribute {name!r} must be immutable"
                ) from exc
            attributes.append((name, value))

        names = [name for name, _ in attributes]
        if len(names) != len(set(names)):
            raise ValueError("operation envelope required attribute names must be unique")
        return tuple(sorted(attributes, key=lambda item: item[0]))


@dataclass(frozen=True)
class RuntimeCapabilities:
    compute_units: Tuple[ComputeUnit, ...] = ()
    supported_phases: Tuple[InferencePhase, ...] = ()
    supported_operations: Tuple[OperationKind, ...] = ()
    supported_dtypes: Tuple[str, ...] = ()
    operation_envelopes: Tuple[OperationEnvelope, ...] = ()
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
        if any(
            not isinstance(envelope, OperationEnvelope)
            for envelope in self.operation_envelopes
        ):
            raise TypeError("operation_envelopes must contain OperationEnvelope values")
        if len(self.operation_envelopes) != len(set(self.operation_envelopes)):
            raise ValueError("operation envelopes must be unique")
        unsupported_envelopes = sorted(
            {
                envelope.kind.value
                for envelope in self.operation_envelopes
                if envelope.kind not in self.supported_operations
            }
        )
        if unsupported_envelopes:
            raise ValueError(
                "operation envelopes require matching supported operations: "
                f"{unsupported_envelopes}"
            )
        if self.max_program_bytes is not None and self.max_program_bytes <= 0:
            raise ValueError("max_program_bytes must be positive when provided")
