import unittest

from lokahi.core import (
    AdaptiveExecutionPlan,
    BackendTarget,
    ComputeUnit,
    EvidenceKind,
    EvidenceRecord,
    ExecutionSegment,
    HardwareProfile,
    IROperation,
    InferencePhase,
    Measurement,
    OperationEnvelope,
    OperationKind,
    RuntimeCapabilities,
    LokahiIRGraph,
    TensorRole,
    TensorSpec,
)


def tiny_graph():
    tensors = (
        TensorSpec("hidden", ("tokens", 4), "float16", TensorRole.INPUT),
        TensorSpec("weight", (4, 4), "float16", TensorRole.WEIGHT),
        TensorSpec("projected", ("tokens", 4), "float16", TensorRole.INTERMEDIATE),
        TensorSpec("logits", ("tokens", 4), "float16", TensorRole.OUTPUT),
    )
    operations = (
        IROperation(
            "projection",
            OperationKind.LINEAR,
            ("hidden", "weight"),
            ("projected",),
        ),
        IROperation(
            "lm_head",
            OperationKind.LOGITS,
            ("projected", "weight"),
            ("logits",),
        ),
    )
    return LokahiIRGraph("tiny", tensors, operations, ("hidden",), ("logits",))


class LokahiIRContractTest(unittest.TestCase):
    def test_symbolic_tensor_has_no_static_byte_count(self):
        tensor = TensorSpec("x", ("tokens", 64), "float16", TensorRole.INPUT)
        self.assertIsNone(tensor.static_nbytes)
        weight = TensorSpec("w", (64, 64), "int4", TensorRole.WEIGHT)
        self.assertEqual(weight.static_nbytes, 2048)

    def test_graph_requires_topological_tensor_availability(self):
        graph = tiny_graph()
        self.assertEqual(tuple(graph.operations_by_id()), ("projection", "lm_head"))
        bad = IROperation("bad", OperationKind.LINEAR, ("future",), ("projected",))
        with self.assertRaisesRegex(ValueError, "before they are available"):
            LokahiIRGraph(graph.graph_id, graph.tensors, (bad,), graph.inputs, graph.outputs)


class AdaptivePlanContractTest(unittest.TestCase):
    def test_plan_can_assign_same_graph_differently_by_phase(self):
        graph = tiny_graph()
        segments = (
            ExecutionSegment(
                "prefill-metal",
                InferencePhase.PREFILL,
                BackendTarget.METAL,
                ("projection", "lm_head"),
                ("hidden",),
                ("logits",),
            ),
            ExecutionSegment(
                "decode-ane",
                InferencePhase.DECODE,
                BackendTarget.ANE,
                ("projection", "lm_head"),
                ("hidden",),
                ("logits",),
            ),
        )
        plan = AdaptiveExecutionPlan(
            "tiny-plan",
            "tiny",
            "model-hash",
            "hardware-hash",
            "fp16",
            1,
            0,
            4096,
            segments,
        )
        plan.validate_graph(graph)
        by_phase = plan.segments_by_phase()
        self.assertEqual(by_phase[InferencePhase.PREFILL][0].target, BackendTarget.METAL)
        self.assertEqual(by_phase[InferencePhase.DECODE][0].target, BackendTarget.ANE)

    def test_duplicate_phase_assignment_is_rejected(self):
        segment = ExecutionSegment(
            "a",
            InferencePhase.DECODE,
            BackendTarget.CPU,
            ("projection",),
            ("hidden",),
            ("projected",),
        )
        duplicate = ExecutionSegment(
            "b",
            InferencePhase.DECODE,
            BackendTarget.METAL,
            ("projection",),
            ("hidden",),
            ("projected",),
        )
        with self.assertRaisesRegex(ValueError, "only once per phase"):
            AdaptiveExecutionPlan(
                "bad",
                "tiny",
                "model-hash",
                "hardware-hash",
                "fp16",
                1,
                0,
                1,
                (segment, duplicate),
            )


class HardwareAndEvidenceContractTest(unittest.TestCase):
    def test_hardware_fingerprint_changes_with_ane_runtime(self):
        common = dict(
            chip="Apple M1",
            macos_version="26.5",
            macos_build="25F90",
            unified_memory_bytes=16 * 1024**3,
            cpu_logical_cores=8,
            compute_units=(ComputeUnit.CPU, ComputeUnit.GPU, ComputeUnit.ANE),
        )
        first = HardwareProfile(**common, ane_runtime_fingerprint="ane-a")
        second = HardwareProfile(**common, ane_runtime_fingerprint="ane-b")
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_evidence_kind_keeps_synthetic_results_explicit(self):
        record = EvidenceRecord(
            "scheduler-sim",
            EvidenceKind.SYNTHETIC,
            "tiny",
            "tiny-plan",
            InferencePhase.DECODE,
            "hardware-hash",
            (Measurement("modeled_tokens_per_second", 10.0, "token/s"),),
        )
        self.assertIs(record.kind, EvidenceKind.SYNTHETIC)

    def test_backend_capability_declares_device_and_private_api(self):
        capability = RuntimeCapabilities(
            compute_units=(ComputeUnit.ANE,),
            supported_phases=(InferencePhase.PREFILL, InferencePhase.DECODE),
            supported_operations=(OperationKind.LINEAR, OperationKind.MLP),
            supported_dtypes=("float16", "int8"),
            supports_shared_iosurface=True,
            requires_private_api=True,
        )
        self.assertEqual(capability.compute_units, (ComputeUnit.ANE,))
        self.assertTrue(capability.requires_private_api)


class OperationEnvelopeContractTest(unittest.TestCase):
    def test_exact_envelope_is_immutable_and_canonical(self):
        envelope = OperationEnvelope(
            OperationKind.LINEAR,
            ([64, 256], [256, 256]),
            ([64, 256],),
            {"weight_layout": "out_in", "fused_bias": False},
        )
        self.assertEqual(
            envelope.input_shapes,
            ((64, 256), (256, 256)),
        )
        self.assertEqual(
            envelope.required_attributes,
            (("fused_bias", False), ("weight_layout", "out_in")),
        )
        self.assertIsInstance(hash(envelope), int)

    def test_envelope_rejects_symbolic_or_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "exact positive integers"):
            OperationEnvelope(
                OperationKind.LINEAR,
                (("tokens", 256), (256, 256)),
                ((64, 256),),
            )
        with self.assertRaisesRegex(ValueError, "exact positive integers"):
            OperationEnvelope(
                OperationKind.LINEAR,
                ((64, 256), (256, 256)),
                ((0, 256),),
            )

    def test_capabilities_reject_duplicate_or_unadvertised_envelopes(self):
        envelope = OperationEnvelope(
            OperationKind.LINEAR,
            ((64, 256), (256, 256)),
            ((64, 256),),
            (("weight_layout", "out_in"),),
        )
        with self.assertRaisesRegex(ValueError, "operation envelopes must be unique"):
            RuntimeCapabilities(
                supported_operations=(OperationKind.LINEAR,),
                operation_envelopes=(envelope, envelope),
            )
        with self.assertRaisesRegex(ValueError, "matching supported operations"):
            RuntimeCapabilities(operation_envelopes=(envelope,))


if __name__ == "__main__":
    unittest.main()
