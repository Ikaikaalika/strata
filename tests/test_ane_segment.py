import unittest

import numpy as np

from lokahi.backends.ane_segment import (
    ANE_LINEAR_ENVELOPE,
    ANEProjectionSegmentExecutor,
    ANESegmentError,
    ane_projection_segment_capabilities,
)
from lokahi.backends.cpu_reference import CPUReferenceExecutor
from lokahi.core import (
    ComputeUnit,
    IROperation,
    InferencePhase,
    OperationKind,
    LokahiIRGraph,
    TensorRole,
    TensorSpec,
)


def _graph(*, tokens=64, weight_layout="out_in"):
    return LokahiIRGraph(
        "ane-linear-segment",
        (
            TensorSpec("x", (tokens, 256), "float16", TensorRole.INPUT),
            TensorSpec("w", (256, 256), "float16", TensorRole.WEIGHT),
            TensorSpec("y", (tokens, 256), "float16", TensorRole.OUTPUT),
        ),
        (
            IROperation(
                "projection",
                OperationKind.LINEAR,
                ("x", "w"),
                ("y",),
                {"weight_layout": weight_layout},
            ),
        ),
        ("x",),
        ("y",),
    )


class _FakeProjectionExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, activation, weight):
        self.calls.append((activation, weight))
        return (activation.astype(np.float32) @ weight.astype(np.float32).T).astype(
            np.float16
        )


class ANEProjectionSegmentTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.activation = rng.normal(0, 0.1, (64, 256)).astype(np.float16)
        self.weight = rng.normal(0, 0.03, (256, 256)).astype(np.float16)

    def test_capability_is_exact_private_ane_prefill_only(self):
        capability = ane_projection_segment_capabilities()

        self.assertEqual(capability.compute_units, (ComputeUnit.ANE,))
        self.assertEqual(capability.supported_phases, (InferencePhase.PREFILL,))
        self.assertEqual(capability.operation_envelopes, (ANE_LINEAR_ENVELOPE,))
        self.assertFalse(capability.supports_dynamic_shapes)
        self.assertTrue(capability.supports_shared_iosurface)
        self.assertTrue(capability.requires_private_api)

    def test_exact_graph_matches_cpu_reference(self):
        graph = _graph()
        fake = _FakeProjectionExecutor()
        actual = ANEProjectionSegmentExecutor(fake).execute(
            graph,
            {"x": self.activation, "w": self.weight},
            phase=InferencePhase.PREFILL,
        )["y"]
        expected = CPUReferenceExecutor().execute(
            graph,
            {"x": self.activation, "w": self.weight},
        )["y"]

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(fake.calls), 1)

    def test_decode_shape_layout_and_feed_mismatches_fail_before_dispatch(self):
        cases = (
            (_graph(), InferencePhase.DECODE, {"x": self.activation, "w": self.weight}, "prefill"),
            (_graph(tokens=32), InferencePhase.PREFILL, {"x": self.activation[:32], "w": self.weight}, "input shapes"),
            (_graph(weight_layout="in_out"), InferencePhase.PREFILL, {"x": self.activation, "w": self.weight}, "out_in"),
            (_graph(), InferencePhase.PREFILL, {"x": self.activation}, "missing"),
        )
        for graph, phase, feeds, message in cases:
            with self.subTest(message=message):
                fake = _FakeProjectionExecutor()
                with self.assertRaisesRegex(ANESegmentError, message):
                    ANEProjectionSegmentExecutor(fake).execute(
                        graph, feeds, phase=phase
                    )
                self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
