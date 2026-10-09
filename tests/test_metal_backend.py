import json
from pathlib import Path
import tempfile
import unittest

from lokahi.backends.base import BackendError
from lokahi.backends.metal_backend import MetalBackend
from lokahi.core.hardware import ComputeUnit
from lokahi.core.ir import InferencePhase, OperationKind


def _fake_probe(directory: str, payload, *, exit_status: int = 0) -> Path:
    path = Path(directory) / "fake-metal-probe"
    encoded = json.dumps(payload)
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{encoded}'\n"
        f"exit {exit_status}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _passing_payload(**overrides):
    payload = {
        "schema_version": 1,
        "backend": "metal",
        "operation": "rmsnorm_residual_f32",
        "device_name": "Fake Apple GPU",
        "dispatch_success": True,
        "failure_reason": None,
        "numerical_max_error": 2.5e-6,
        "dtype": "float32",
        "timing": {"average_dispatch_wall_time_ms": 0.125},
    }
    payload.update(overrides)
    return payload


class MetalBackendTest(unittest.TestCase):
    def test_missing_probe_fails_closed(self):
        backend = MetalBackend("/definitely/not/a/metal/probe")
        self.assertFalse(backend.is_available())
        self.assertEqual(backend.capabilities().compute_units, ())
        with self.assertRaises(BackendError):
            backend.resolve_device(None)

    def test_valid_probe_declares_only_proven_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _fake_probe(directory, _passing_payload())
            backend = MetalBackend(path)

            self.assertTrue(backend.is_available())
            self.assertEqual(backend.resolve_device("gpu"), "Fake Apple GPU")
            result = backend.probe()
            self.assertEqual(result.operation, "rmsnorm_residual_f32")
            self.assertEqual(result.average_dispatch_wall_time_ms, 0.125)

            capabilities = backend.capabilities()
            self.assertEqual(capabilities.compute_units, (ComputeUnit.GPU,))
            self.assertEqual(
                capabilities.supported_phases,
                (InferencePhase.PREFILL, InferencePhase.DECODE),
            )
            self.assertEqual(
                capabilities.supported_operations,
                (OperationKind.NORMALIZATION, OperationKind.RESIDUAL),
            )
            self.assertEqual(capabilities.supported_dtypes, ("float32",))
            self.assertFalse(capabilities.supports_dynamic_shapes)
            self.assertFalse(capabilities.supports_graph_capture)

    def test_numerical_error_over_threshold_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _fake_probe(
                directory,
                _passing_payload(numerical_max_error=0.25),
            )
            backend = MetalBackend(path, maximum_error=1.0e-4)
            self.assertFalse(backend.is_available())
            self.assertIn("capability contract", backend.probe().failure_reason)

    def test_invalid_evidence_envelope_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _fake_probe(
                directory,
                _passing_payload(schema_version=2, numerical_max_error=-1.0),
            )
            backend = MetalBackend(path)
            self.assertFalse(backend.is_available())
            self.assertIsNone(backend.probe().numerical_max_error)
            self.assertIn("evidence envelope", backend.probe().failure_reason)

    def test_nonzero_probe_exit_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _fake_probe(
                directory,
                _passing_payload(failure_reason="simulated GPU failure"),
                exit_status=7,
            )
            backend = MetalBackend(path)
            self.assertFalse(backend.is_available())
            self.assertEqual(backend.probe().failure_reason, "simulated GPU failure")

    def test_invalid_json_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-metal-probe"
            path.write_text("#!/bin/sh\nprintf 'not-json\\n'\n", encoding="utf-8")
            path.chmod(0o755)
            backend = MetalBackend(path)
            self.assertFalse(backend.is_available())
            self.assertIn("invalid JSON", backend.probe().failure_reason)


if __name__ == "__main__":
    unittest.main()
