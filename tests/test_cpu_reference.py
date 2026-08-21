import unittest

import numpy as np

from ollm.backends.cpu_reference import (
    CPUReferenceError,
    CPUReferenceExecutor,
    cpu_reference_capabilities,
)
from ollm.core import (
    ComputeUnit,
    IROperation,
    InferencePhase,
    OperationKind,
    StrataIRGraph,
    TensorRole,
    TensorSpec,
)


def _reference_graph():
    tensors = (
        TensorSpec("hidden", ("tokens", 3), "float16", TensorRole.INPUT),
        TensorSpec("norm_weight", (3,), "float16", TensorRole.WEIGHT),
        TensorSpec("projection_weight", (2, 3), "float16", TensorRole.WEIGHT),
        TensorSpec("bias", (2,), "float16", TensorRole.WEIGHT),
        TensorSpec("residual", ("tokens", 2), "float16", TensorRole.INPUT),
        TensorSpec("normalized", ("tokens", 3), "float16", TensorRole.INTERMEDIATE),
        TensorSpec("projected", ("tokens", 2), "float16", TensorRole.INTERMEDIATE),
        TensorSpec("activated", ("tokens", 2), "float16", TensorRole.INTERMEDIATE),
        TensorSpec("combined", ("tokens", 2), "float16", TensorRole.INTERMEDIATE),
        TensorSpec("lm_weight", (4, 2), "float16", TensorRole.WEIGHT),
        TensorSpec("logits", ("tokens", 4), "float32", TensorRole.OUTPUT),
    )
    operations = (
        IROperation(
            "rms_norm",
            OperationKind.NORMALIZATION,
            ("hidden", "norm_weight"),
            ("normalized",),
            {"epsilon": 1e-5},
        ),
        IROperation(
            "projection",
            OperationKind.LINEAR,
            ("normalized", "projection_weight", "bias"),
            ("projected",),
        ),
        IROperation(
            "silu",
            OperationKind.ACTIVATION,
            ("projected",),
            ("activated",),
        ),
        IROperation(
            "add_residual",
            OperationKind.RESIDUAL,
            ("activated", "residual"),
            ("combined",),
        ),
        IROperation(
            "lm_head",
            OperationKind.LOGITS,
            ("combined", "lm_weight"),
            ("logits",),
        ),
    )
    return StrataIRGraph(
        "cpu-reference",
        tensors,
        operations,
        ("hidden", "residual"),
        ("logits",),
    )


def _feeds():
    return {
        "hidden": np.array([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]], dtype=np.float16),
        "norm_weight": np.array([0.5, 1.5, -1.0], dtype=np.float16),
        "projection_weight": np.array(
            [[1.0, -0.5, 0.25], [-1.0, 0.75, 0.5]], dtype=np.float16
        ),
        "bias": np.array([0.125, -0.25], dtype=np.float16),
        "residual": np.array([[0.1, 0.2], [-0.3, 0.4]], dtype=np.float16),
        "lm_weight": np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-1.0, 2.0]],
            dtype=np.float16,
        ),
    }


class CPUReferenceExecutorTest(unittest.TestCase):
    def test_executes_reference_subset_with_float32_accumulation(self):
        graph = _reference_graph()
        feeds = _feeds()

        actual = CPUReferenceExecutor().execute(graph, feeds)["logits"]

        hidden = feeds["hidden"].astype(np.float32)
        normalized = hidden / np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + 1e-5)
        normalized *= feeds["norm_weight"].astype(np.float32)
        # The graph declares float16 boundaries between operations.
        normalized = normalized.astype(np.float16).astype(np.float32)
        projected = normalized @ feeds["projection_weight"].astype(np.float32).T
        projected += feeds["bias"].astype(np.float32)
        projected = projected.astype(np.float16).astype(np.float32)
        activated = projected / (1.0 + np.exp(-projected))
        activated = activated.astype(np.float16).astype(np.float32)
        combined = activated + feeds["residual"].astype(np.float32)
        combined = combined.astype(np.float16).astype(np.float32)
        expected = combined @ feeds["lm_weight"].astype(np.float32).T

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_is_deterministic(self):
        executor = CPUReferenceExecutor()
        first = executor.execute(_reference_graph(), _feeds())["logits"]
        second = executor.execute(_reference_graph(), _feeds())["logits"]
        np.testing.assert_array_equal(first, second)

    def test_capabilities_are_conservative_and_cpu_only(self):
        capability = cpu_reference_capabilities()
        self.assertEqual(capability.compute_units, (ComputeUnit.CPU,))
        self.assertEqual(
            capability.supported_phases,
            (InferencePhase.PREFILL, InferencePhase.DECODE),
        )
        self.assertEqual(
            capability.supported_operations,
            (
                OperationKind.LINEAR,
                OperationKind.NORMALIZATION,
                OperationKind.RESIDUAL,
                OperationKind.ACTIVATION,
                OperationKind.LOGITS,
            ),
        )
        self.assertEqual(capability.supported_dtypes, ("float16", "float32"))
        self.assertTrue(capability.supports_dynamic_shapes)
        self.assertFalse(capability.supports_graph_capture)
        self.assertFalse(capability.supports_quantized_weights)

    def test_rejects_unsupported_operation_before_execution(self):
        tensors = (
            TensorSpec("x", (1, 2), "float32", TensorRole.INPUT),
            TensorSpec("y", (1, 2), "float32", TensorRole.OUTPUT),
        )
        graph = StrataIRGraph(
            "unsupported",
            tensors,
            (IROperation("attention", OperationKind.ATTENTION, ("x",), ("y",)),),
            ("x",),
            ("y",),
        )
        with self.assertRaisesRegex(
            CPUReferenceError,
            "unsupported CPU reference operations: attention:attention",
        ):
            CPUReferenceExecutor().execute(
                graph, {"x": np.ones((1, 2), dtype=np.float32)}
            )

    def test_rejects_symbolic_shape_mismatch(self):
        feeds = _feeds()
        feeds["residual"] = np.zeros((3, 2), dtype=np.float16)
        with self.assertRaisesRegex(CPUReferenceError, "symbolic dimension 'tokens'"):
            CPUReferenceExecutor().execute(_reference_graph(), feeds)

    def test_rejects_missing_and_unknown_tensor_names(self):
        feeds = _feeds()
        del feeds["lm_weight"]
        with self.assertRaisesRegex(CPUReferenceError, "missing runtime tensors"):
            CPUReferenceExecutor().execute(_reference_graph(), feeds)

        feeds = _feeds()
        feeds["typo"] = np.ones((1,), dtype=np.float16)
        with self.assertRaisesRegex(CPUReferenceError, "unknown runtime tensors"):
            CPUReferenceExecutor().execute(_reference_graph(), feeds)

    def test_rejects_feed_dtype_mismatch(self):
        feeds = _feeds()
        feeds["hidden"] = feeds["hidden"].astype(np.float32)
        with self.assertRaisesRegex(CPUReferenceError, "has dtype float32"):
            CPUReferenceExecutor().execute(_reference_graph(), feeds)

    def test_supports_declared_in_out_linear_layout(self):
        tensors = (
            TensorSpec("x", (1, 2), "float32", TensorRole.INPUT),
            TensorSpec("w", (2, 3), "float32", TensorRole.WEIGHT),
            TensorSpec("y", (1, 3), "float32", TensorRole.OUTPUT),
        )
        graph = StrataIRGraph(
            "in-out",
            tensors,
            (
                IROperation(
                    "linear",
                    OperationKind.LINEAR,
                    ("x", "w"),
                    ("y",),
                    {"weight_layout": "in_out"},
                ),
            ),
            ("x",),
            ("y",),
        )
        output = CPUReferenceExecutor().execute(
            graph,
            {
                "x": np.array([[2.0, 3.0]], dtype=np.float32),
                "w": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            },
        )["y"]
        np.testing.assert_array_equal(output, [[14.0, 19.0, 24.0]])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
