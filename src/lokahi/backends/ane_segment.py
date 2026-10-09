"""LokahiIR lowering for the one fixed ANE projection envelope."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from ..core.capabilities import OperationEnvelope, RuntimeCapabilities
from ..core.hardware import ComputeUnit
from ..core.ir import InferencePhase, OperationKind, LokahiIRGraph
from .ane_executor import ANEProjectionExecutor


ANE_LINEAR_ENVELOPE = OperationEnvelope(
    kind=OperationKind.LINEAR,
    input_shapes=((64, 256), (256, 256)),
    output_shapes=((64, 256),),
    required_attributes=(("weight_layout", "out_in"),),
)


class ANESegmentError(RuntimeError):
    """Raised when a graph is outside the fixed direct-ANE envelope."""


def ane_projection_segment_capabilities() -> RuntimeCapabilities:
    """Declare the exact logical graph region implemented by this adapter.

    These are implementation capabilities, not a hardware-availability claim.
    The hardware profile and correctness evidence independently gate scheduling.
    """
    return RuntimeCapabilities(
        compute_units=(ComputeUnit.ANE,),
        supported_phases=(InferencePhase.PREFILL,),
        supported_operations=(OperationKind.LINEAR,),
        supported_dtypes=("float16",),
        operation_envelopes=(ANE_LINEAR_ENVELOPE,),
        supports_dynamic_shapes=False,
        supports_shared_iosurface=True,
        requires_private_api=True,
    )


class ANEProjectionSegmentExecutor:
    """Lower one exact LokahiIR LINEAR graph into ``ANEProjectionExecutor``."""

    def __init__(self, executor: ANEProjectionExecutor | None = None) -> None:
        self.executor = executor or ANEProjectionExecutor()

    def capabilities(self) -> RuntimeCapabilities:
        return ane_projection_segment_capabilities()

    def execute(
        self,
        graph: LokahiIRGraph,
        tensors: Mapping[str, np.ndarray],
        *,
        phase: InferencePhase,
    ) -> Dict[str, np.ndarray]:
        operation = self._validate_graph(graph, phase)
        expected_feeds = set(operation.inputs)
        actual_feeds = set(tensors)
        if actual_feeds != expected_feeds:
            raise ANESegmentError(
                "ANE projection feeds do not match the exact operation inputs; "
                f"missing={sorted(expected_feeds - actual_feeds)}, "
                f"extra={sorted(actual_feeds - expected_feeds)}"
            )

        activation = tensors[operation.inputs[0]]
        weight = tensors[operation.inputs[1]]
        try:
            output = self.executor.execute(activation, weight)
        except (TypeError, ValueError) as exc:
            raise ANESegmentError(f"ANE projection tensor envelope mismatch: {exc}") from exc
        return {operation.outputs[0]: output}

    def _validate_graph(self, graph: LokahiIRGraph, phase: InferencePhase):
        if phase is not InferencePhase.PREFILL:
            raise ANESegmentError("fixed [64, 256] ANE projection supports prefill only")
        if len(graph.operations) != 1:
            raise ANESegmentError("ANE projection segment must contain exactly one operation")

        operation = graph.operations[0]
        if operation.kind is not OperationKind.LINEAR:
            raise ANESegmentError("ANE projection segment must be a LINEAR operation")
        if len(operation.inputs) != 2 or len(operation.outputs) != 1:
            raise ANESegmentError("ANE projection LINEAR requires two inputs and one output")

        specs = graph.tensors_by_name()
        input_shapes = tuple(tuple(specs[name].shape) for name in operation.inputs)
        output_shapes = tuple(tuple(specs[name].shape) for name in operation.outputs)
        if input_shapes != ANE_LINEAR_ENVELOPE.input_shapes:
            raise ANESegmentError(
                f"ANE projection input shapes {input_shapes} do not match "
                f"{ANE_LINEAR_ENVELOPE.input_shapes}"
            )
        if output_shapes != ANE_LINEAR_ENVELOPE.output_shapes:
            raise ANESegmentError(
                f"ANE projection output shapes {output_shapes} do not match "
                f"{ANE_LINEAR_ENVELOPE.output_shapes}"
            )
        names = (*operation.inputs, *operation.outputs)
        dtypes = tuple(specs[name].normalized_dtype for name in names)
        if dtypes != ("float16", "float16", "float16"):
            raise ANESegmentError("ANE projection tensors must all use float16")
        if operation.attributes.get("weight_layout") != "out_in":
            raise ANESegmentError("ANE projection requires explicit out_in weight layout")
        return operation


__all__ = [
    "ANE_LINEAR_ENVELOPE",
    "ANEProjectionSegmentExecutor",
    "ANESegmentError",
    "ane_projection_segment_capabilities",
]
