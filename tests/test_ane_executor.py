from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import stat
import sys
import tempfile
import textwrap
import unittest

import numpy as np

from lokahi.backends.ane_executor import (
    ANEProjectionError,
    ANEProjectionExecutor,
    ANEProjectionReport,
)


_FAKE_WORKER = r"""
import json
from pathlib import Path
import sys
import time

import numpy as np

mode = MODE
if mode == "timeout":
    time.sleep(2)
if mode == "exit":
    print("synthetic worker failure", file=sys.stderr)
    raise SystemExit(7)
if mode == "malformed":
    print("not-json")
    raise SystemExit(0)

assert sys.argv[1] == "--execute-linear-request"
request_path = Path(sys.argv[2])
request = json.loads(request_path.read_text(encoding="utf-8"))
assert set(request) == {
    "schema_version", "request", "request_id", "operation", "input", "weight", "output"
}
assert request["schema_version"] == 1
assert request["request"] == "lokahi-ane-linear-request"
assert request["operation"] == "fp16_linear"
assert request["input"]["shape"] == [64, 256]
assert request["input"]["dtype"] == "float16"
assert request["input"]["byte_count"] == 32768
assert request["weight"]["shape"] == [256, 256]
assert request["weight"]["dtype"] == "float16"
assert request["weight"]["layout"] == "out_in"
assert request["weight"]["byte_count"] == 131072
assert request["output"]["shape"] == [64, 256]
assert request["output"]["byte_count"] == 32768

input_path = Path(request["input"]["path"])
weight_path = Path(request["weight"]["path"])
output_path = Path(request["output"]["path"])
assert input_path.parent == request_path.parent
assert weight_path.parent == request_path.parent
assert output_path.parent == request_path.parent
assert input_path.stat().st_size == 32768
assert weight_path.stat().st_size == 131072
assert not output_path.exists()

def report(**updates):
    value = {
        "schema_version": 1,
        "report": "lokahi-ane-linear-result",
        "request_id": request["request_id"],
        "operation": "fp16_linear",
        "input_shape": [64, 256],
        "weight_shape": [256, 256],
        "weight_layout": "out_in",
        "dtype": "float16",
        "last_successful_stage": "output_written",
        "success": True,
        "compile_succeeded": True,
        "load_succeeded": True,
        "dispatch_succeeded": True,
        "output_written": True,
        "output_byte_count": 32768,
        "dispatch_ms": 0.25,
        "error": "",
        "cleanup_error": "",
    }
    value.update(updates)
    return value

if mode == "structured_failure":
    print(json.dumps(report(
        last_successful_stage="surface_validated",
        success=False,
        compile_succeeded=False,
        load_succeeded=False,
        dispatch_succeeded=False,
        output_written=False,
        output_byte_count=0,
        dispatch_ms=None,
        error="compile: synthetic compiler failure",
    )))
    raise SystemExit(0)

x = np.fromfile(input_path, dtype=np.float16).reshape(64, 256)
w = np.fromfile(weight_path, dtype=np.float16).reshape(256, 256)
# Emulate the native logical [token,width] -> physical [channel,spatial]
# IOSurface copy, convolution, and physical -> logical output copy.
surface_input = x.T.copy()
surface_output = w.astype(np.float32) @ surface_input.astype(np.float32)
output = surface_output.T.astype(np.float16)
if mode == "nonfinite_output":
    output.fill(np.nan)
if mode == "short_output":
    output_path.write_bytes(b"\x00\x00")
else:
    output_path.write_bytes(output.tobytes(order="C"))
result = report()
if mode == "wrong_request_id":
    result["request_id"] = "0" * 32
if mode == "extra_report_key":
    result["untrusted"] = True
print(json.dumps(result))
"""


@contextmanager
def _fake_worker(mode: str):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fake-ane-worker"
        source = textwrap.dedent(_FAKE_WORKER).replace("MODE", repr(mode), 1)
        path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        yield path


def _success_report(request_id: str = "a" * 32) -> dict:
    return {
        "schema_version": 1,
        "report": "lokahi-ane-linear-result",
        "request_id": request_id,
        "operation": "fp16_linear",
        "input_shape": [64, 256],
        "weight_shape": [256, 256],
        "weight_layout": "out_in",
        "dtype": "float16",
        "last_successful_stage": "output_written",
        "success": True,
        "compile_succeeded": True,
        "load_succeeded": True,
        "dispatch_succeeded": True,
        "output_written": True,
        "output_byte_count": 32768,
        "dispatch_ms": 0.25,
        "error": "",
        "cleanup_error": "",
    }


class ANEProjectionReportTest(unittest.TestCase):
    def test_valid_success_report(self):
        report = ANEProjectionReport.from_json(
            json.dumps(_success_report()).encode(), expected_request_id="a" * 32
        )

        self.assertTrue(report.success)
        self.assertEqual(report.output_byte_count, 32768)
        self.assertEqual(report.last_successful_stage, "output_written")

    def test_contradictory_progress_fails_closed(self):
        document = _success_report()
        document["compile_succeeded"] = False
        with self.assertRaisesRegex(ANEProjectionError, "load claim"):
            ANEProjectionReport.from_json(
                json.dumps(document).encode(), expected_request_id="a" * 32
            )

    def test_extra_report_key_fails_closed(self):
        document = _success_report()
        document["untrusted"] = True
        with self.assertRaisesRegex(ANEProjectionError, "invalid keys"):
            ANEProjectionReport.from_json(
                json.dumps(document).encode(), expected_request_id="a" * 32
            )


class ANEProjectionExecutorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.input = rng.normal(0.0, 0.1, (64, 256)).astype(np.float16)
        self.weight = rng.normal(0.0, 0.03, (256, 256)).astype(np.float16)

    def test_success_protocol_matches_cpu_reference(self):
        expected = (
            self.input.astype(np.float32) @ self.weight.astype(np.float32).T
        ).astype(np.float16)
        with _fake_worker("success") as worker:
            executor = ANEProjectionExecutor(worker, timeout_seconds=2)
            actual = executor.execute(self.input, self.weight)

        self.assertEqual(actual.shape, (64, 256))
        self.assertEqual(actual.dtype, np.float16)
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual, expected)
        self.assertIsNotNone(executor.last_report)
        self.assertTrue(executor.last_report.success)

    def test_structured_worker_failure_is_rejected(self):
        with _fake_worker("structured_failure") as worker:
            executor = ANEProjectionExecutor(worker, timeout_seconds=2)
            with self.assertRaisesRegex(ANEProjectionError, "synthetic compiler failure"):
                executor.execute(self.input, self.weight)

        self.assertIsNotNone(executor.last_report)
        self.assertFalse(executor.last_report.success)

    def test_output_size_is_checked_independently_of_report(self):
        with _fake_worker("short_output") as worker:
            executor = ANEProjectionExecutor(worker, timeout_seconds=2)
            with self.assertRaisesRegex(ANEProjectionError, "exact size"):
                executor.execute(self.input, self.weight)

    def test_nonfinite_output_is_rejected(self):
        with _fake_worker("nonfinite_output") as worker:
            executor = ANEProjectionExecutor(worker, timeout_seconds=2)
            with self.assertRaisesRegex(ANEProjectionError, "non-finite"):
                executor.execute(self.input, self.weight)

    def test_report_must_bind_to_request_identifier(self):
        with _fake_worker("wrong_request_id") as worker:
            executor = ANEProjectionExecutor(worker, timeout_seconds=2)
            with self.assertRaisesRegex(ANEProjectionError, "identifier does not match"):
                executor.execute(self.input, self.weight)

    def test_malformed_and_extra_reports_fail_closed(self):
        for mode, message in (
            ("malformed", "valid JSON"),
            ("extra_report_key", "invalid keys"),
        ):
            with self.subTest(mode=mode), _fake_worker(mode) as worker:
                executor = ANEProjectionExecutor(worker, timeout_seconds=2)
                with self.assertRaisesRegex(ANEProjectionError, message):
                    executor.execute(self.input, self.weight)

    def test_timeout_and_nonzero_exit_fail_closed(self):
        for mode, timeout, message in (
            ("timeout", 0.01, "timed out"),
            ("exit", 2, "status 7"),
        ):
            with self.subTest(mode=mode), _fake_worker(mode) as worker:
                executor = ANEProjectionExecutor(worker, timeout_seconds=timeout)
                with self.assertRaisesRegex(ANEProjectionError, message):
                    executor.execute(self.input, self.weight)

    def test_inputs_must_match_the_exact_contiguous_fp16_envelope(self):
        invalid_cases = (
            (self.input.astype(np.float32), self.weight, "dtype float16"),
            (self.input[:63], self.weight, "exact shape"),
            (np.asfortranarray(self.input), self.weight, "C-contiguous"),
            (self.input, self.weight[:255], "exact shape"),
        )
        executor = ANEProjectionExecutor("/worker/is/not/reached")
        for input_tensor, weight, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                (TypeError, ValueError), message
            ):
                executor.execute(input_tensor, weight)

    def test_nonfinite_input_is_rejected_before_worker_launch(self):
        input_tensor = self.input.copy()
        input_tensor[0, 0] = np.inf
        executor = ANEProjectionExecutor("/worker/is/not/reached")

        with self.assertRaisesRegex(ANEProjectionError, "must be finite"):
            executor.execute(input_tensor, self.weight)


if __name__ == "__main__":
    unittest.main()
