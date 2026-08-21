from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import stat
import tempfile
import unittest

from ollm.backends.ane_backend import ANEBackend, ANEProbeError, ANEProbeReport
from ollm.core.hardware import ComputeUnit


def _valid_report(*, executed: bool = False) -> dict:
    classes = {
        "_ANEClient": {
            "present": True,
            "instance_methods": [
                "loadModel:options:qos:error:",
                "evaluateWithModel:options:request:qos:error:",
            ],
            "class_methods": ["sharedConnection"],
        },
        "_ANECompiler": {
            "present": False,
            "instance_methods": [],
            "class_methods": [],
        },
        "_ANEInMemoryModelDescriptor": {
            "present": True,
            "instance_methods": [],
            "class_methods": ["modelWithMILText:weights:optionsPlist:"],
        },
        "_ANEModel": {
            "present": True,
            "instance_methods": [],
            "class_methods": [],
        },
        "_ANEInMemoryModel": {
            "present": True,
            "instance_methods": [
                "compileWithQoS:options:error:",
                "loadWithQoS:options:error:",
                "evaluateWithQoS:options:request:error:",
                "unloadWithQoS:error:",
            ],
            "class_methods": ["inMemoryModelWithDescriptor:"],
        },
        "_ANERequest": {
            "present": True,
            "instance_methods": [],
            "class_methods": [
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"
            ],
        },
        "_ANEIOSurfaceObject": {
            "present": True,
            "instance_methods": [],
            "class_methods": ["objectWithIOSurface:"],
        },
    }
    execution = {
        "requested": executed,
        "operation": "fp16_projection",
        "shape": [1, 256, 1, 64],
        "input_dtype": "float16",
        "compute_dtype": "float16",
        "output_dtype": "float16",
        "tolerance": 0.002,
        "last_successful_stage": "numeric_verified" if executed else "discovery",
        "failure_stage": "",
        "error": "",
        "cleanup_error": "",
        "mil_generated": executed,
        "descriptor_created": executed,
        "model_created": executed,
        "artifacts_written": executed,
        "compile_attempted": executed,
        "compile_succeeded": executed,
        "load_attempted": executed,
        "load_succeeded": executed,
        "iosurfaces_created": executed,
        "request_created": executed,
        "dispatch_attempted": executed,
        "dispatch_succeeded": executed,
        "output_finite": executed,
        "numeric_verified": executed,
        "execution_verified": executed,
        "max_abs_error": 0.0 if executed else None,
        "dispatch_ms": 0.291 if executed else None,
    }
    return {
        "schema_version": 2,
        "probe": "strata-ane-capability",
        "platform": {
            "os": "macOS",
            "version": "26.5.2",
            "build": "25F84",
            "architecture": "arm64",
            "chip": "Apple M1",
        },
        "frameworks": {
            "AppleNeuralEngine": {
                "path": "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                "loaded": True,
                "error": "",
            },
            "ANECompiler": {
                "path": "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
                "loaded": True,
                "error": "",
            },
        },
        "objective_c": {
            "classes": classes,
            "descriptor_candidates": [
                "_ANEInMemoryModelDescriptor",
                "_ANEInMemoryModel",
                "_ANEModel",
            ],
            "descriptor_class": "_ANEInMemoryModelDescriptor",
        },
        "required_surface": {
            "private_frameworks_loaded": True,
            "ane_client_present": True,
            "ane_client_entrypoints_present": True,
            "ane_compiler_present": False,
            "compiler_equivalent_present": True,
            "compiler_class": "_ANEInMemoryModel",
            "descriptor_present": True,
            "descriptor_factory_present": True,
            "descriptor_class": "_ANEInMemoryModelDescriptor",
            "model_lifecycle_entrypoints_present": True,
            "satisfied": True,
            "request_class_present": True,
            "iosurface_object_class_present": True,
            "request_factory_present": True,
            "iosurface_object_factory_present": True,
            "execution_surface_satisfied": True,
        },
        "execution": execution,
        "errors": [],
    }


@contextmanager
def _fake_worker(stdout: str, *, exit_code: int = 0, delay_seconds: float = 0.0):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fake-ane-worker"
        script = "#!/bin/sh\n"
        if delay_seconds:
            script += f"sleep {delay_seconds}\n"
        script += "printf '%s\\n' " + repr(stdout) + "\n"
        script += f"exit {exit_code}\n"
        path.write_text(script, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        yield path


class ANEProbeReportTest(unittest.TestCase):
    def test_valid_discovery_report_does_not_claim_execution(self):
        report = ANEProbeReport.from_json(json.dumps(_valid_report()))

        self.assertTrue(report.required_surface_satisfied)
        self.assertEqual(report.chip, "Apple M1")
        self.assertEqual(report.compiler_class, "_ANEInMemoryModel")
        self.assertFalse(report.compile_attempted)
        self.assertFalse(report.dispatch_attempted)
        self.assertFalse(report.execution_verified)
        self.assertFalse(report.numeric_verified)

    def test_malformed_json_fails_closed(self):
        with self.assertRaisesRegex(ANEProbeError, "valid JSON"):
            ANEProbeReport.from_json("not-json")

    def test_extra_schema_field_is_rejected(self):
        document = _valid_report()
        document["untrusted_claim"] = True
        with self.assertRaisesRegex(ANEProbeError, "invalid keys"):
            ANEProbeReport.from_json(json.dumps(document))

    def test_inconsistent_surface_claim_is_rejected(self):
        document = _valid_report()
        document["required_surface"]["ane_client_entrypoints_present"] = False
        with self.assertRaisesRegex(ANEProbeError, "contradicts method observations"):
            ANEProbeReport.from_json(json.dumps(document))

    def test_unproven_execution_claim_is_rejected(self):
        document = _valid_report()
        document["execution"]["execution_verified"] = True
        with self.assertRaisesRegex(ANEProbeError, "execution-verification"):
            ANEProbeReport.from_json(json.dumps(document))

    def test_complete_projection_execution_report_is_verified(self):
        report = ANEProbeReport.from_json(json.dumps(_valid_report(executed=True)))
        repeat = ANEProbeReport.from_json(json.dumps(_valid_report(executed=True)))

        self.assertTrue(report.execution_requested)
        self.assertTrue(report.compile_succeeded)
        self.assertTrue(report.load_succeeded)
        self.assertTrue(report.dispatch_succeeded)
        self.assertTrue(report.numeric_verified)
        self.assertTrue(report.execution_verified)
        self.assertEqual(report.max_abs_error, 0.0)
        self.assertEqual(report.last_successful_stage, "numeric_verified")
        self.assertEqual(report.runtime_fingerprint, repeat.runtime_fingerprint)
        self.assertEqual(len(report.runtime_fingerprint), 64)

    def test_out_of_order_execution_progress_is_rejected(self):
        document = _valid_report(executed=True)
        document["execution"]["compile_succeeded"] = False
        with self.assertRaisesRegex(ANEProbeError, "load_attempted.*compile_succeeded"):
            ANEProbeReport.from_json(json.dumps(document))


class ANEBackendTest(unittest.TestCase):
    def test_valid_fake_worker_exposes_private_ane_discovery_only(self):
        with _fake_worker(json.dumps(_valid_report())) as worker:
            backend = ANEBackend(worker, timeout_seconds=1.0)

            self.assertTrue(backend.is_discovered())
            self.assertFalse(backend.is_available())
            capabilities = backend.capabilities()
            self.assertEqual(capabilities.compute_units, ())
            self.assertTrue(capabilities.requires_private_api)
            self.assertEqual(capabilities.supported_phases, ())
            self.assertEqual(capabilities.supported_operations, ())
            with self.assertRaisesRegex(Exception, "qualification is incomplete"):
                backend.resolve_device("ane")
            with self.assertRaisesRegex(Exception, "does not implement attention"):
                backend.attention_kernel()

    def test_explicit_execution_qualification_exposes_no_general_operations(self):
        with _fake_worker(json.dumps(_valid_report(executed=True))) as worker:
            backend = ANEBackend(
                worker,
                timeout_seconds=1.0,
                attempt_execution=True,
            )

            self.assertTrue(backend.is_discovered())
            self.assertTrue(backend.is_available())
            capabilities = backend.capabilities()
            self.assertEqual(capabilities.compute_units, (ComputeUnit.ANE,))
            self.assertEqual(capabilities.supported_operations, ())
            self.assertEqual(capabilities.supported_phases, ())
            self.assertEqual(capabilities.supported_dtypes, ("float16",))
            self.assertTrue(capabilities.supports_shared_iosurface)
            self.assertTrue(capabilities.requires_private_api)
            self.assertEqual(backend.resolve_device("ane").report.max_abs_error, 0.0)
            with self.assertRaisesRegex(Exception, "does not implement general"):
                backend.move_model_to_device(object(), object())

    def test_cleanup_failure_invalidates_execution_qualification(self):
        document = _valid_report(executed=True)
        document["execution"]["cleanup_error"] = "unload failed"
        document["errors"] = ["projection cleanup: unload failed"]
        with _fake_worker(json.dumps(document)) as worker:
            backend = ANEBackend(worker, attempt_execution=True)

            report = backend.probe()
            self.assertTrue(report.execution_verified)
            self.assertFalse(report.qualification_verified)
            self.assertFalse(backend.is_available())
            self.assertEqual(backend.capabilities().compute_units, ())

    def test_surface_unavailable_returns_empty_capabilities(self):
        document = _valid_report()
        document["objective_c"]["classes"]["_ANEClient"]["present"] = False
        document["required_surface"]["ane_client_present"] = False
        document["required_surface"]["satisfied"] = False
        document["required_surface"]["execution_surface_satisfied"] = False
        with _fake_worker(json.dumps(document)) as worker:
            backend = ANEBackend(worker)

            self.assertFalse(backend.is_available())
            capabilities = backend.capabilities()
            self.assertEqual(capabilities.compute_units, ())
            self.assertFalse(capabilities.requires_private_api)

    def test_non_apple_silicon_report_is_not_available(self):
        document = _valid_report()
        document["platform"]["architecture"] = "x86_64"
        document["platform"]["chip"] = "Intel Core i9"
        with _fake_worker(json.dumps(document)) as worker:
            backend = ANEBackend(worker)

            self.assertFalse(backend.is_available())
            self.assertEqual(backend.capabilities().compute_units, ())

    def test_missing_worker_is_unavailable(self):
        backend = ANEBackend("/definitely/not/a/strata-ane-worker")

        self.assertFalse(backend.is_available())
        self.assertEqual(backend.capabilities().compute_units, ())
        with self.assertRaisesRegex(ANEProbeError, "does not exist"):
            backend.probe()

    def test_worker_timeout_fails_closed(self):
        with _fake_worker("{}", delay_seconds=1.0) as worker:
            backend = ANEBackend(worker, timeout_seconds=0.01)

            self.assertFalse(backend.is_available())
            self.assertEqual(backend.capabilities().compute_units, ())
            with self.assertRaisesRegex(ANEProbeError, "timed out"):
                backend.probe()

    def test_worker_malformed_output_fails_closed(self):
        with _fake_worker("not-json") as worker:
            backend = ANEBackend(worker)

            self.assertFalse(backend.is_available())
            with self.assertRaisesRegex(ANEProbeError, "valid JSON"):
                backend.probe()

    def test_worker_nonzero_exit_fails_closed(self):
        with _fake_worker("ignored", exit_code=7) as worker:
            backend = ANEBackend(worker)

            self.assertFalse(backend.is_available())
            with self.assertRaisesRegex(ANEProbeError, "status 7"):
                backend.probe()


if __name__ == "__main__":
    unittest.main()
