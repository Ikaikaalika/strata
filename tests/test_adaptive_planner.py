import unittest

from ollm.core import (
    BackendTarget,
    ComputeUnit,
    EvidenceKind,
    EvidenceRecord,
    HardwareProfile,
    IROperation,
    InferencePhase,
    Measurement,
    OperationEnvelope,
    OperationKind,
    RuntimeCapabilities,
    StrataIRGraph,
    TensorRole,
    TensorSpec,
)
from ollm.planning import AdaptivePlanner, PlanningError


def _graph(kind=OperationKind.LINEAR, *, symbolic=True, dtype="float16"):
    token_dimension = "tokens" if symbolic else 2
    tensors = (
        TensorSpec("x", (token_dimension, 4), dtype, TensorRole.INPUT),
        TensorSpec("w", (4, 4), dtype, TensorRole.WEIGHT),
        TensorSpec("y", (token_dimension, 4), dtype, TensorRole.OUTPUT),
    )
    operation = IROperation("op", kind, ("x", "w"), ("y",))
    return StrataIRGraph("tiny", tensors, (operation,), ("x",), ("y",))


def _hardware():
    return HardwareProfile(
        chip="Apple M1",
        macos_version="26.5",
        macos_build="25F90",
        unified_memory_bytes=16 * 1024**3,
        cpu_logical_cores=8,
        compute_units=(ComputeUnit.CPU, ComputeUnit.GPU, ComputeUnit.ANE),
        ane_runtime_fingerprint="probe-v1",
    )


def _capability(
    target,
    *,
    operations=(OperationKind.LINEAR,),
    dynamic=True,
    dtypes=("float16",),
    phases=(InferencePhase.PREFILL, InferencePhase.DECODE),
    envelopes=(),
):
    unit = {
        BackendTarget.MLX: ComputeUnit.GPU,
        BackendTarget.METAL: ComputeUnit.GPU,
        BackendTarget.ANE: ComputeUnit.ANE,
        BackendTarget.CPU: ComputeUnit.CPU,
        BackendTarget.COREML: ComputeUnit.ANE,
    }[target]
    return RuntimeCapabilities(
        compute_units=(unit,),
        supported_phases=phases,
        supported_operations=operations,
        supported_dtypes=dtypes,
        operation_envelopes=envelopes,
        supports_dynamic_shapes=dynamic,
        requires_private_api=target is BackendTarget.ANE,
    )


def _evidence(kind, target, phase, value, hardware, *, sequence=0, **scope_overrides):
    if kind is EvidenceKind.CORRECTNESS:
        measurement = Measurement("correctness_passed", value, "bool")
    else:
        measurement = Measurement("tokens_per_second", value, "token/s")
    scope = {
        "target": target.value,
        "graph_id": "tiny",
        "model_hash": "model-hash",
        "quantization": "fp16",
        "batch_size": 1,
        "context_min_tokens": 0,
        "context_max_tokens": 2048,
    }
    scope.update(scope_overrides)
    return EvidenceRecord(
        evidence_id=f"{kind.value}-{target.value}-{phase.value}-{sequence}",
        kind=kind,
        model_id="model",
        plan_id=f"candidate-{target.value}-{phase.value}",
        phase=phase,
        hardware_fingerprint=hardware.fingerprint,
        measurements=(measurement,),
        metadata=scope,
    )


def _build(planner, capabilities, evidence=(), **overrides):
    arguments = dict(
        graph=_graph(),
        model_id="model",
        model_hash="model-hash",
        quantization="fp16",
        batch_size=1,
        context_min_tokens=0,
        context_max_tokens=2048,
        hardware=_hardware(),
        capabilities=capabilities,
        evidence=evidence,
    )
    arguments.update(overrides)
    return planner.build(**arguments)


def _exact_linear_graph(*, tokens=64, width=256, weight_layout="out_in"):
    tensors = (
        TensorSpec("x", (tokens, width), "float16", TensorRole.INPUT),
        TensorSpec("w", (width, width), "float16", TensorRole.WEIGHT),
        TensorSpec("y", (tokens, width), "float16", TensorRole.OUTPUT),
    )
    operation = IROperation(
        "op",
        OperationKind.LINEAR,
        ("x", "w"),
        ("y",),
        {"weight_layout": weight_layout},
    )
    return StrataIRGraph("tiny", tensors, (operation,), ("x",), ("y",))


def _linear_envelope():
    return OperationEnvelope(
        OperationKind.LINEAR,
        ((64, 256), (256, 256)),
        ((64, 256),),
        (("weight_layout", "out_in"),),
    )


class AdaptivePlannerTest(unittest.TestCase):
    def test_capable_mlx_is_selected_and_retained_as_fallback(self):
        result = _build(
            AdaptivePlanner(),
            {BackendTarget.MLX: _capability(BackendTarget.MLX)},
        )
        self.assertIs(result.selected_plan, result.mlx_fallback_plan)
        self.assertEqual(
            tuple(segment.phase for segment in result.plan.segments),
            (InferencePhase.PREFILL, InferencePhase.DECODE),
        )
        self.assertTrue(
            all(segment.target is BackendTarget.MLX for segment in result.plan.segments)
        )

    def test_fixed_shape_metal_is_ineligible_for_symbolic_graph(self):
        hardware = _hardware()
        evidence = tuple(
            _evidence(kind, BackendTarget.METAL, phase, value, hardware)
            for phase in (InferencePhase.PREFILL, InferencePhase.DECODE)
            for kind, value in (
                (EvidenceKind.CORRECTNESS, 1.0),
                (EvidenceKind.HARDWARE, 100.0),
            )
        )
        result = _build(
            AdaptivePlanner(),
            {
                BackendTarget.MLX: _capability(BackendTarget.MLX),
                BackendTarget.METAL: _capability(
                    BackendTarget.METAL, dynamic=False
                ),
            },
            evidence,
        )
        self.assertTrue(
            all(segment.target is BackendTarget.MLX for segment in result.plan.segments)
        )
        metal = result.evaluation(BackendTarget.METAL, InferencePhase.PREFILL)
        self.assertFalse(metal.capability_eligible)
        self.assertIn(
            "graph has symbolic shapes but target is fixed-shape",
            metal.rejection_reasons,
        )

    def test_private_ane_discovery_with_zero_operations_never_executes(self):
        ane_discovery = RuntimeCapabilities(
            compute_units=(ComputeUnit.ANE,),
            supported_phases=(InferencePhase.PREFILL, InferencePhase.DECODE),
            supported_operations=(),
            supported_dtypes=("float16",),
            supports_dynamic_shapes=True,
            requires_private_api=True,
        )
        result = _build(
            AdaptivePlanner(),
            {
                BackendTarget.MLX: _capability(BackendTarget.MLX),
                BackendTarget.ANE: ane_discovery,
            },
        )
        self.assertTrue(
            all(segment.target is BackendTarget.MLX for segment in result.plan.segments)
        )
        ane = result.evaluation(BackendTarget.ANE, InferencePhase.DECODE)
        self.assertFalse(ane.capability_eligible)
        self.assertIn("unsupported operations ['linear']", ane.rejection_reasons)

    def test_verified_measured_alternative_can_win_one_phase(self):
        hardware = _hardware()
        evidence = (
            _evidence(
                EvidenceKind.HARDWARE,
                BackendTarget.MLX,
                InferencePhase.PREFILL,
                20.0,
                hardware,
            ),
            _evidence(
                EvidenceKind.HARDWARE,
                BackendTarget.MLX,
                InferencePhase.DECODE,
                10.0,
                hardware,
            ),
            _evidence(
                EvidenceKind.CORRECTNESS,
                BackendTarget.ANE,
                InferencePhase.PREFILL,
                1.0,
                hardware,
            ),
            _evidence(
                EvidenceKind.HARDWARE,
                BackendTarget.ANE,
                InferencePhase.PREFILL,
                30.0,
                hardware,
            ),
            _evidence(
                EvidenceKind.CORRECTNESS,
                BackendTarget.ANE,
                InferencePhase.DECODE,
                1.0,
                hardware,
            ),
            _evidence(
                EvidenceKind.HARDWARE,
                BackendTarget.ANE,
                InferencePhase.DECODE,
                8.0,
                hardware,
            ),
        )
        result = _build(
            AdaptivePlanner(),
            {
                BackendTarget.MLX: _capability(BackendTarget.MLX),
                BackendTarget.ANE: _capability(BackendTarget.ANE),
            },
            evidence,
        )
        selected = result.plan.segments_by_phase()
        self.assertIs(
            selected[InferencePhase.PREFILL][0].target,
            BackendTarget.ANE,
        )
        self.assertIs(
            selected[InferencePhase.DECODE][0].target,
            BackendTarget.MLX,
        )
        self.assertIsNotNone(result.mlx_fallback_plan)
        self.assertTrue(
            all(
                segment.target is BackendTarget.MLX
                for segment in result.mlx_fallback_plan.segments
            )
        )

    def test_correctness_without_hardware_cannot_displace_mlx(self):
        hardware = _hardware()
        correctness = tuple(
            _evidence(
                EvidenceKind.CORRECTNESS,
                BackendTarget.ANE,
                phase,
                1.0,
                hardware,
            )
            for phase in (InferencePhase.PREFILL, InferencePhase.DECODE)
        )
        synthetic = tuple(
            _evidence(
                EvidenceKind.SYNTHETIC,
                BackendTarget.ANE,
                phase,
                10_000.0,
                hardware,
            )
            for phase in (InferencePhase.PREFILL, InferencePhase.DECODE)
        )
        result = _build(
            AdaptivePlanner(),
            {
                BackendTarget.MLX: _capability(BackendTarget.MLX),
                BackendTarget.ANE: _capability(BackendTarget.ANE),
            },
            correctness + synthetic,
        )
        self.assertTrue(
            all(segment.target is BackendTarget.MLX for segment in result.plan.segments)
        )

    def test_evidence_from_a_different_context_bucket_is_ignored(self):
        hardware = _hardware()
        evidence = tuple(
            _evidence(
                kind,
                BackendTarget.ANE,
                phase,
                value,
                hardware,
                context_max_tokens=4096,
            )
            for phase in (InferencePhase.PREFILL, InferencePhase.DECODE)
            for kind, value in (
                (EvidenceKind.CORRECTNESS, 1.0),
                (EvidenceKind.HARDWARE, 10_000.0),
            )
        )
        result = _build(
            AdaptivePlanner(),
            {
                BackendTarget.MLX: _capability(BackendTarget.MLX),
                BackendTarget.ANE: _capability(BackendTarget.ANE),
            },
            evidence,
        )
        self.assertTrue(
            all(segment.target is BackendTarget.MLX for segment in result.plan.segments)
        )
        self.assertFalse(
            result.evaluation(
                BackendTarget.ANE, InferencePhase.PREFILL
            ).correctness_verified
        )

    def test_phase_support_is_checked_independently(self):
        hardware = _hardware()
        evidence = (
            _evidence(
                EvidenceKind.HARDWARE,
                BackendTarget.MLX,
                InferencePhase.PREFILL,
                20.0,
                hardware,
            ),
            _evidence(
                EvidenceKind.HARDWARE,
                BackendTarget.MLX,
                InferencePhase.DECODE,
                10.0,
                hardware,
            ),
            _evidence(
                EvidenceKind.CORRECTNESS,
                BackendTarget.METAL,
                InferencePhase.PREFILL,
                1.0,
                hardware,
            ),
            _evidence(
                EvidenceKind.HARDWARE,
                BackendTarget.METAL,
                InferencePhase.PREFILL,
                30.0,
                hardware,
            ),
        )
        result = _build(
            AdaptivePlanner(),
            {
                BackendTarget.MLX: _capability(BackendTarget.MLX),
                BackendTarget.METAL: _capability(
                    BackendTarget.METAL,
                    phases=(InferencePhase.PREFILL,),
                ),
            },
            evidence,
        )
        by_phase = result.plan.segments_by_phase()
        self.assertIs(
            by_phase[InferencePhase.PREFILL][0].target,
            BackendTarget.METAL,
        )
        self.assertIs(
            by_phase[InferencePhase.DECODE][0].target,
            BackendTarget.MLX,
        )
        decode = result.evaluation(BackendTarget.METAL, InferencePhase.DECODE)
        self.assertIn("phase 'decode' is unsupported", decode.rejection_reasons)

    def test_operation_unknown_to_every_target_fails_closed(self):
        with self.assertRaisesRegex(
            PlanningError,
            "no evidence-safe execution target for prefill",
        ):
            _build(
                AdaptivePlanner(),
                {BackendTarget.MLX: _capability(BackendTarget.MLX)},
                graph=_graph(OperationKind.STATE_SPACE),
            )

    def test_output_and_plan_id_are_deterministic_and_context_aware(self):
        planner = AdaptivePlanner()
        capabilities = {BackendTarget.MLX: _capability(BackendTarget.MLX)}
        first = _build(planner, capabilities)
        second = _build(planner, dict(reversed(tuple(capabilities.items()))))
        self.assertEqual(first, second)

        other_bucket = _build(planner, capabilities, context_max_tokens=4096)
        self.assertNotEqual(first.plan.plan_id, other_bucket.plan.plan_id)
        self.assertIn("c0-2048", first.plan.plan_id)
        self.assertIn("c0-4096", other_bucket.plan.plan_id)

    def test_dtype_support_is_required_even_with_positive_evidence(self):
        result = _build(
            AdaptivePlanner(),
            {
                BackendTarget.MLX: _capability(BackendTarget.MLX),
                BackendTarget.METAL: _capability(
                    BackendTarget.METAL, dtypes=("float32",)
                ),
            },
        )
        metal = result.evaluation(BackendTarget.METAL, InferencePhase.PREFILL)
        self.assertFalse(metal.capability_eligible)
        self.assertIn("unsupported dtypes ['float16']", metal.rejection_reasons)

    def test_exact_linear_envelope_still_requires_correctness_and_hardware(self):
        hardware = _hardware()
        capabilities = {
            BackendTarget.MLX: _capability(BackendTarget.MLX),
            BackendTarget.METAL: _capability(
                BackendTarget.METAL,
                dynamic=False,
                envelopes=(_linear_envelope(),),
            ),
        }
        graph = _exact_linear_graph()

        without_evidence = _build(
            AdaptivePlanner(),
            capabilities,
            graph=graph,
        )
        self.assertTrue(
            without_evidence.evaluation(
                BackendTarget.METAL, InferencePhase.PREFILL
            ).capability_eligible
        )
        self.assertTrue(
            all(
                segment.target is BackendTarget.MLX
                for segment in without_evidence.plan.segments
            )
        )

        evidence = tuple(
            _evidence(kind, target, phase, value, hardware)
            for phase in (InferencePhase.PREFILL, InferencePhase.DECODE)
            for target, kind, value in (
                (BackendTarget.MLX, EvidenceKind.HARDWARE, 10.0),
                (BackendTarget.METAL, EvidenceKind.CORRECTNESS, 1.0),
                (BackendTarget.METAL, EvidenceKind.HARDWARE, 20.0),
            )
        )
        verified = _build(
            AdaptivePlanner(),
            capabilities,
            evidence,
            graph=graph,
        )
        self.assertTrue(
            all(
                segment.target is BackendTarget.METAL
                for segment in verified.plan.segments
            )
        )

    def test_exact_envelope_rejects_shape_layout_and_symbolic_mismatches(self):
        metal = _capability(
            BackendTarget.METAL,
            dynamic=True,
            envelopes=(_linear_envelope(),),
        )
        capabilities = {
            BackendTarget.MLX: _capability(BackendTarget.MLX),
            BackendTarget.METAL: metal,
        }
        cases = (
            ("shape", _exact_linear_graph(tokens=32)),
            ("layout", _exact_linear_graph(weight_layout="in_out")),
            ("symbolic", _exact_linear_graph(tokens="tokens")),
        )
        for label, graph in cases:
            with self.subTest(label=label):
                result = _build(
                    AdaptivePlanner(),
                    capabilities,
                    graph=graph,
                )
                evaluation = result.evaluation(
                    BackendTarget.METAL,
                    InferencePhase.PREFILL,
                )
                self.assertFalse(evaluation.capability_eligible)
                self.assertIn(
                    "operations outside exact envelopes ['op']",
                    evaluation.rejection_reasons,
                )
                self.assertTrue(
                    all(
                        segment.target is BackendTarget.MLX
                        for segment in result.plan.segments
                    )
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
