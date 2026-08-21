"""Deterministic, evidence-gated planning over Strata runtime contracts.

Capabilities answer whether a target *can* execute a graph.  Correctness
evidence unlocks non-baseline targets.  Comparable hardware measurements may
then prefer a verified target over MLX.  Synthetic measurements are never used
to select execution targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from statistics import median
from typing import Dict, Mapping, Sequence, Tuple

from ..core.adaptive_plan import (
    AdaptiveExecutionPlan,
    BackendTarget,
    ExecutionSegment,
)
from ..core.capabilities import RuntimeCapabilities
from ..core.evidence import EvidenceKind, EvidenceRecord
from ..core.hardware import HardwareProfile
from ..core.ir import InferencePhase, StrataIRGraph, TensorRole


_PHASES = (InferencePhase.PREFILL, InferencePhase.DECODE)
_TARGET_ORDER = (
    BackendTarget.MLX,
    BackendTarget.METAL,
    BackendTarget.ANE,
    BackendTarget.COREML,
    BackendTarget.CPU,
)
_QUANTIZED_DTYPES = {
    "int4",
    "uint4",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
}


class PlanningError(RuntimeError):
    """Raised when no evidence-safe execution plan can be built."""


@dataclass(frozen=True)
class CandidateEvaluation:
    """Auditable eligibility and evidence state for one target and phase."""

    target: BackendTarget
    phase: InferencePhase
    capability_eligible: bool
    correctness_verified: bool
    hardware_tokens_per_second: float | None
    rejection_reasons: Tuple[str, ...] = ()

    @property
    def execution_eligible(self) -> bool:
        return self.capability_eligible and self.correctness_verified


@dataclass(frozen=True)
class PlannerResult:
    """Selected plan plus a retained MLX recovery plan when MLX is capable."""

    selected_plan: AdaptiveExecutionPlan
    mlx_fallback_plan: AdaptiveExecutionPlan | None
    evaluations: Tuple[CandidateEvaluation, ...]

    @property
    def plan(self) -> AdaptiveExecutionPlan:
        """Convenience alias for callers that only consume the selected plan."""

        return self.selected_plan

    def evaluation(
        self,
        target: BackendTarget,
        phase: InferencePhase,
    ) -> CandidateEvaluation:
        for evaluation in self.evaluations:
            if evaluation.target is target and evaluation.phase is phase:
                return evaluation
        raise KeyError((target, phase))


class AdaptivePlanner:
    """Build whole-graph prefill/decode plans from facts and measurements."""

    def build(
        self,
        *,
        graph: StrataIRGraph,
        model_id: str,
        model_hash: str,
        quantization: str,
        batch_size: int,
        context_min_tokens: int,
        context_max_tokens: int,
        hardware: HardwareProfile,
        capabilities: Mapping[BackendTarget, RuntimeCapabilities],
        evidence: Sequence[EvidenceRecord] = (),
    ) -> PlannerResult:
        self._validate_request(
            model_id=model_id,
            model_hash=model_hash,
            quantization=quantization,
            batch_size=batch_size,
            context_min_tokens=context_min_tokens,
            context_max_tokens=context_max_tokens,
        )
        normalized_capabilities = self._normalize_capabilities(capabilities)

        evaluations = tuple(
            self._evaluate(
                graph=graph,
                model_id=model_id,
                model_hash=model_hash,
                quantization=quantization,
                batch_size=batch_size,
                context_min_tokens=context_min_tokens,
                context_max_tokens=context_max_tokens,
                hardware=hardware,
                target=target,
                phase=phase,
                capability=normalized_capabilities.get(target),
                evidence=evidence,
            )
            for phase in _PHASES
            for target in _TARGET_ORDER
        )

        selected_targets: Dict[InferencePhase, BackendTarget] = {}
        for phase in _PHASES:
            phase_evaluations = tuple(
                evaluation
                for evaluation in evaluations
                if evaluation.phase is phase
            )
            selected_targets[phase] = self._select_target(phase, phase_evaluations)

        selected_plan = self._make_plan(
            graph=graph,
            model_id=model_id,
            model_hash=model_hash,
            quantization=quantization,
            batch_size=batch_size,
            context_min_tokens=context_min_tokens,
            context_max_tokens=context_max_tokens,
            hardware=hardware,
            phase_targets=selected_targets,
        )

        mlx_fallback_plan = None
        mlx_by_phase = {
            phase: self._find_evaluation(evaluations, BackendTarget.MLX, phase)
            for phase in _PHASES
        }
        if all(
            evaluation.execution_eligible
            for evaluation in mlx_by_phase.values()
        ):
            fallback_targets = {phase: BackendTarget.MLX for phase in _PHASES}
            if selected_targets == fallback_targets:
                mlx_fallback_plan = selected_plan
            else:
                mlx_fallback_plan = self._make_plan(
                    graph=graph,
                    model_id=model_id,
                    model_hash=model_hash,
                    quantization=quantization,
                    batch_size=batch_size,
                    context_min_tokens=context_min_tokens,
                    context_max_tokens=context_max_tokens,
                    hardware=hardware,
                    phase_targets=fallback_targets,
                )

        return PlannerResult(selected_plan, mlx_fallback_plan, evaluations)

    def _validate_request(
        self,
        *,
        model_id: str,
        model_hash: str,
        quantization: str,
        batch_size: int,
        context_min_tokens: int,
        context_max_tokens: int,
    ) -> None:
        if not model_id or not model_hash or not quantization:
            raise PlanningError("model identity and quantization must not be empty")
        if batch_size <= 0:
            raise PlanningError("batch_size must be positive")
        if context_min_tokens < 0:
            raise PlanningError("context_min_tokens must not be negative")
        if context_max_tokens < context_min_tokens:
            raise PlanningError(
                "context_max_tokens must be greater than or equal to the minimum"
            )

    def _normalize_capabilities(
        self,
        capabilities: Mapping[BackendTarget, RuntimeCapabilities],
    ) -> Dict[BackendTarget, RuntimeCapabilities]:
        normalized: Dict[BackendTarget, RuntimeCapabilities] = {}
        for raw_target, capability in capabilities.items():
            try:
                target = (
                    raw_target
                    if isinstance(raw_target, BackendTarget)
                    else BackendTarget(raw_target)
                )
            except (TypeError, ValueError) as exc:
                raise PlanningError(f"unknown backend target {raw_target!r}") from exc
            if target in normalized:
                raise PlanningError(f"duplicate capabilities for target {target.value!r}")
            if not isinstance(capability, RuntimeCapabilities):
                raise PlanningError(
                    f"capabilities for target {target.value!r} have invalid type"
                )
            normalized[target] = capability
        return normalized

    def _evaluate(
        self,
        *,
        graph: StrataIRGraph,
        model_id: str,
        model_hash: str,
        quantization: str,
        batch_size: int,
        context_min_tokens: int,
        context_max_tokens: int,
        hardware: HardwareProfile,
        target: BackendTarget,
        phase: InferencePhase,
        capability: RuntimeCapabilities | None,
        evidence: Sequence[EvidenceRecord],
    ) -> CandidateEvaluation:
        reasons = self._capability_rejection_reasons(
            graph,
            hardware,
            phase,
            capability,
        )
        capability_eligible = not reasons

        if target is BackendTarget.MLX:
            # MLX is the established baseline, but capability checks still apply.
            correctness_verified = capability_eligible
        else:
            correctness_verified = capability_eligible and self._correctness_verified(
                evidence,
                graph=graph,
                model_id=model_id,
                model_hash=model_hash,
                quantization=quantization,
                batch_size=batch_size,
                context_min_tokens=context_min_tokens,
                context_max_tokens=context_max_tokens,
                hardware=hardware,
                target=target,
                phase=phase,
            )
            if capability_eligible and not correctness_verified:
                reasons.append("no passing correctness evidence")

        hardware_score = None
        if correctness_verified:
            hardware_score = self._hardware_tokens_per_second(
                evidence,
                graph=graph,
                model_id=model_id,
                model_hash=model_hash,
                quantization=quantization,
                batch_size=batch_size,
                context_min_tokens=context_min_tokens,
                context_max_tokens=context_max_tokens,
                hardware=hardware,
                target=target,
                phase=phase,
            )

        return CandidateEvaluation(
            target=target,
            phase=phase,
            capability_eligible=capability_eligible,
            correctness_verified=correctness_verified,
            hardware_tokens_per_second=hardware_score,
            rejection_reasons=tuple(reasons),
        )

    def _capability_rejection_reasons(
        self,
        graph: StrataIRGraph,
        hardware: HardwareProfile,
        phase: InferencePhase,
        capability: RuntimeCapabilities | None,
    ) -> list[str]:
        if capability is None:
            return ["capabilities not supplied"]

        reasons: list[str] = []
        if not capability.compute_units:
            reasons.append("no executable compute units")
        else:
            missing_units = sorted(
                unit.value
                for unit in capability.compute_units
                if unit not in hardware.compute_units
            )
            if missing_units:
                reasons.append(f"hardware lacks compute units {missing_units}")

        if phase not in capability.supported_phases:
            reasons.append(f"phase {phase.value!r} is unsupported")

        supported_operations = set(capability.supported_operations)
        unsupported_operations = sorted(
            {
                operation.kind.value
                for operation in graph.operations
                if operation.kind not in supported_operations
            }
        )
        if unsupported_operations:
            reasons.append(f"unsupported operations {unsupported_operations}")

        supported_dtypes = {
            dtype.lower().replace("mlx.core.", "").replace("torch.", "")
            for dtype in capability.supported_dtypes
        }
        unsupported_dtypes = sorted(
            {
                tensor.normalized_dtype
                for tensor in graph.tensors
                if tensor.normalized_dtype not in supported_dtypes
            }
        )
        if unsupported_dtypes:
            reasons.append(f"unsupported dtypes {unsupported_dtypes}")

        if not capability.supports_dynamic_shapes and any(
            isinstance(dimension, str)
            for tensor in graph.tensors
            for dimension in tensor.shape
        ):
            reasons.append("graph has symbolic shapes but target is fixed-shape")

        has_quantized_weight = any(
            tensor.role is TensorRole.WEIGHT
            and tensor.normalized_dtype in _QUANTIZED_DTYPES
            for tensor in graph.tensors
        )
        if has_quantized_weight and not capability.supports_quantized_weights:
            reasons.append("graph has quantized weights but target does not support them")

        return reasons

    def _correctness_verified(
        self,
        evidence: Sequence[EvidenceRecord],
        *,
        graph: StrataIRGraph,
        model_id: str,
        model_hash: str,
        quantization: str,
        batch_size: int,
        context_min_tokens: int,
        context_max_tokens: int,
        hardware: HardwareProfile,
        target: BackendTarget,
        phase: InferencePhase,
    ) -> bool:
        outcomes: list[float] = []
        for record in self._matching_evidence(
            evidence,
            kind=EvidenceKind.CORRECTNESS,
            graph=graph,
            model_id=model_id,
            model_hash=model_hash,
            quantization=quantization,
            batch_size=batch_size,
            context_min_tokens=context_min_tokens,
            context_max_tokens=context_max_tokens,
            hardware=hardware,
            target=target,
            phase=phase,
        ):
            for measurement in record.measurements:
                if (
                    measurement.name == "correctness_passed"
                    and measurement.unit == "bool"
                ):
                    outcomes.append(measurement.value)
        return bool(outcomes) and all(value == 1.0 for value in outcomes)

    def _hardware_tokens_per_second(
        self,
        evidence: Sequence[EvidenceRecord],
        *,
        graph: StrataIRGraph,
        model_id: str,
        model_hash: str,
        quantization: str,
        batch_size: int,
        context_min_tokens: int,
        context_max_tokens: int,
        hardware: HardwareProfile,
        target: BackendTarget,
        phase: InferencePhase,
    ) -> float | None:
        values: list[float] = []
        for record in self._matching_evidence(
            evidence,
            kind=EvidenceKind.HARDWARE,
            graph=graph,
            model_id=model_id,
            model_hash=model_hash,
            quantization=quantization,
            batch_size=batch_size,
            context_min_tokens=context_min_tokens,
            context_max_tokens=context_max_tokens,
            hardware=hardware,
            target=target,
            phase=phase,
        ):
            for measurement in record.measurements:
                if (
                    measurement.name == "tokens_per_second"
                    and measurement.unit == "token/s"
                    and math.isfinite(measurement.value)
                    and measurement.value > 0
                ):
                    values.append(float(measurement.value))
        return float(median(values)) if values else None

    def _matching_evidence(
        self,
        evidence: Sequence[EvidenceRecord],
        *,
        kind: EvidenceKind,
        graph: StrataIRGraph,
        model_id: str,
        model_hash: str,
        quantization: str,
        batch_size: int,
        context_min_tokens: int,
        context_max_tokens: int,
        hardware: HardwareProfile,
        target: BackendTarget,
        phase: InferencePhase,
    ) -> Tuple[EvidenceRecord, ...]:
        return tuple(
            record
            for record in evidence
            if record.kind is kind
            and record.model_id == model_id
            and record.hardware_fingerprint == hardware.fingerprint
            and record.phase is phase
            and record.metadata.get("target") == target.value
            and record.metadata.get("graph_id") == graph.graph_id
            and record.metadata.get("model_hash") == model_hash
            and record.metadata.get("quantization") == quantization
            and record.metadata.get("batch_size") == batch_size
            and record.metadata.get("context_min_tokens") == context_min_tokens
            and record.metadata.get("context_max_tokens") == context_max_tokens
        )

    def _select_target(
        self,
        phase: InferencePhase,
        evaluations: Sequence[CandidateEvaluation],
    ) -> BackendTarget:
        by_target = {evaluation.target: evaluation for evaluation in evaluations}
        executable = [
            evaluation for evaluation in evaluations if evaluation.execution_eligible
        ]
        if not executable:
            details = "; ".join(
                f"{evaluation.target.value}: "
                f"{', '.join(evaluation.rejection_reasons) or 'not verified'}"
                for evaluation in evaluations
            )
            raise PlanningError(
                f"no evidence-safe execution target for {phase.value}: {details}"
            )

        baseline = by_target[BackendTarget.MLX]
        if baseline.execution_eligible:
            baseline_score = baseline.hardware_tokens_per_second
            if baseline_score is None:
                return BackendTarget.MLX
            measured_alternatives = [
                evaluation
                for evaluation in executable
                if evaluation.target is not BackendTarget.MLX
                and evaluation.hardware_tokens_per_second is not None
                and evaluation.hardware_tokens_per_second > baseline_score
            ]
            if not measured_alternatives:
                return BackendTarget.MLX
            return min(
                measured_alternatives,
                key=lambda evaluation: (
                    -float(evaluation.hardware_tokens_per_second),
                    _TARGET_ORDER.index(evaluation.target),
                ),
            ).target

        measured = [
            evaluation
            for evaluation in executable
            if evaluation.hardware_tokens_per_second is not None
        ]
        if measured:
            return min(
                measured,
                key=lambda evaluation: (
                    -float(evaluation.hardware_tokens_per_second),
                    _TARGET_ORDER.index(evaluation.target),
                ),
            ).target
        if len(executable) == 1:
            return executable[0].target
        raise PlanningError(
            f"multiple verified targets exist for {phase.value}, but no comparable "
            "hardware evidence is available"
        )

    def _make_plan(
        self,
        *,
        graph: StrataIRGraph,
        model_id: str,
        model_hash: str,
        quantization: str,
        batch_size: int,
        context_min_tokens: int,
        context_max_tokens: int,
        hardware: HardwareProfile,
        phase_targets: Mapping[InferencePhase, BackendTarget],
    ) -> AdaptiveExecutionPlan:
        plan_id = self._plan_id(
            graph=graph,
            model_id=model_id,
            model_hash=model_hash,
            quantization=quantization,
            batch_size=batch_size,
            context_min_tokens=context_min_tokens,
            context_max_tokens=context_max_tokens,
            hardware=hardware,
            phase_targets=phase_targets,
        )
        operation_ids = tuple(operation.operation_id for operation in graph.operations)
        segments = tuple(
            ExecutionSegment(
                segment_id=f"{plan_id}:{phase.value}:{phase_targets[phase].value}",
                phase=phase,
                target=phase_targets[phase],
                operation_ids=operation_ids,
                input_tensors=graph.inputs,
                output_tensors=graph.outputs,
            )
            for phase in _PHASES
        )
        plan = AdaptiveExecutionPlan(
            plan_id=plan_id,
            model_id=model_id,
            model_hash=model_hash,
            hardware_fingerprint=hardware.fingerprint,
            quantization=quantization,
            batch_size=batch_size,
            context_min_tokens=context_min_tokens,
            context_max_tokens=context_max_tokens,
            segments=segments,
        )
        plan.validate_graph(graph)
        return plan

    def _plan_id(
        self,
        *,
        graph: StrataIRGraph,
        model_id: str,
        model_hash: str,
        quantization: str,
        batch_size: int,
        context_min_tokens: int,
        context_max_tokens: int,
        hardware: HardwareProfile,
        phase_targets: Mapping[InferencePhase, BackendTarget],
    ) -> str:
        targets = "-".join(
            f"{phase.value}-{phase_targets[phase].value}" for phase in _PHASES
        )
        payload = {
            "batch_size": batch_size,
            "context_max_tokens": context_max_tokens,
            "context_min_tokens": context_min_tokens,
            "graph_id": graph.graph_id,
            "hardware_fingerprint": hardware.fingerprint,
            "model_hash": model_hash,
            "model_id": model_id,
            "operation_ids": [operation.operation_id for operation in graph.operations],
            "quantization": quantization,
            "targets": targets,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return (
            f"{model_id}:{graph.graph_id}:{quantization}:b{batch_size}:"
            f"c{context_min_tokens}-{context_max_tokens}:{targets}:{digest}"
        )

    def _find_evaluation(
        self,
        evaluations: Sequence[CandidateEvaluation],
        target: BackendTarget,
        phase: InferencePhase,
    ) -> CandidateEvaluation:
        for evaluation in evaluations:
            if evaluation.target is target and evaluation.phase is phase:
                return evaluation
        raise AssertionError((target, phase))


__all__ = [
    "AdaptivePlanner",
    "CandidateEvaluation",
    "PlannerResult",
    "PlanningError",
]
