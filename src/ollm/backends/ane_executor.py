"""Bounded callable protocol for one qualified private-ANE projection.

This module intentionally does not register a planner operation. It can execute
only the fixed FP16 envelope proven by the native capability worker: logical
``[64, 256] @ [256, 256].T`` with an ``out_in`` weight layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np


ANE_LINEAR_SCHEMA_VERSION = 1
ANE_LINEAR_REQUEST_NAME = "strata-ane-linear-request"
ANE_LINEAR_RESULT_NAME = "strata-ane-linear-result"
ANE_LINEAR_OPERATION = "fp16_linear"
ANE_LINEAR_INPUT_SHAPE = (64, 256)
ANE_LINEAR_WEIGHT_SHAPE = (256, 256)
ANE_LINEAR_DTYPE = np.dtype(np.float16)
ANE_LINEAR_INPUT_BYTES = math.prod(ANE_LINEAR_INPUT_SHAPE) * ANE_LINEAR_DTYPE.itemsize
ANE_LINEAR_WEIGHT_BYTES = math.prod(ANE_LINEAR_WEIGHT_SHAPE) * ANE_LINEAR_DTYPE.itemsize
_MAX_REPORT_BYTES = 16 * 1024

_REPORT_KEYS = {
    "schema_version",
    "report",
    "request_id",
    "operation",
    "input_shape",
    "weight_shape",
    "weight_layout",
    "dtype",
    "last_successful_stage",
    "success",
    "compile_succeeded",
    "load_succeeded",
    "dispatch_succeeded",
    "output_written",
    "output_byte_count",
    "dispatch_ms",
    "error",
    "cleanup_error",
}


class ANEProjectionError(RuntimeError):
    """Raised when the fixed-shape ANE request cannot be trusted."""


@dataclass(frozen=True)
class ANEProjectionReport:
    """Strictly validated native response for a single projection request."""

    raw: Mapping[str, Any]
    request_id: str
    last_successful_stage: str
    success: bool
    compile_succeeded: bool
    load_succeeded: bool
    dispatch_succeeded: bool
    output_written: bool
    output_byte_count: int
    dispatch_ms: float | None
    error: str
    cleanup_error: str

    @classmethod
    def from_json(cls, payload: bytes, *, expected_request_id: str) -> "ANEProjectionReport":
        if not payload or len(payload) > _MAX_REPORT_BYTES:
            raise ANEProjectionError("ANE linear worker emitted an empty or oversized report")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ANEProjectionError("ANE linear worker did not emit one valid JSON document") from exc
        if not isinstance(document, dict):
            raise ANEProjectionError("ANE linear report must be a JSON object")
        _require_exact_keys(document, _REPORT_KEYS, "report")
        if _require_int(document["schema_version"], "schema_version") != ANE_LINEAR_SCHEMA_VERSION:
            raise ANEProjectionError("unsupported ANE linear report schema version")
        if _require_str(document["report"], "report") != ANE_LINEAR_RESULT_NAME:
            raise ANEProjectionError("unexpected ANE linear report identity")
        request_id = _require_str(document["request_id"], "request_id")
        if request_id != expected_request_id:
            raise ANEProjectionError("ANE linear report request identifier does not match")
        if _require_str(document["operation"], "operation") != ANE_LINEAR_OPERATION:
            raise ANEProjectionError("unexpected ANE linear operation")
        if _require_int_list(document["input_shape"], "input_shape") != list(
            ANE_LINEAR_INPUT_SHAPE
        ):
            raise ANEProjectionError("unexpected ANE linear input shape")
        if _require_int_list(document["weight_shape"], "weight_shape") != list(
            ANE_LINEAR_WEIGHT_SHAPE
        ):
            raise ANEProjectionError("unexpected ANE linear weight shape")
        if _require_str(document["weight_layout"], "weight_layout") != "out_in":
            raise ANEProjectionError("unexpected ANE linear weight layout")
        if _require_str(document["dtype"], "dtype") != "float16":
            raise ANEProjectionError("unexpected ANE linear dtype")

        last_stage = _require_str(document["last_successful_stage"], "last_successful_stage")
        success = _require_bool(document["success"], "success")
        compile_succeeded = _require_bool(
            document["compile_succeeded"], "compile_succeeded"
        )
        load_succeeded = _require_bool(document["load_succeeded"], "load_succeeded")
        dispatch_succeeded = _require_bool(
            document["dispatch_succeeded"], "dispatch_succeeded"
        )
        output_written = _require_bool(document["output_written"], "output_written")
        output_byte_count = _require_int(document["output_byte_count"], "output_byte_count")
        dispatch_ms = _require_optional_number(document["dispatch_ms"], "dispatch_ms")
        error = _require_str(document["error"], "error")
        cleanup_error = _require_str(document["cleanup_error"], "cleanup_error")

        if load_succeeded and not compile_succeeded:
            raise ANEProjectionError("ANE linear load claim lacks a successful compile")
        if dispatch_succeeded and not load_succeeded:
            raise ANEProjectionError("ANE linear dispatch claim lacks a successful load")
        if output_written and not dispatch_succeeded:
            raise ANEProjectionError("ANE linear output claim lacks a successful dispatch")
        if dispatch_succeeded and dispatch_ms is None:
            raise ANEProjectionError("ANE linear dispatch success lacks timing evidence")
        if dispatch_ms is not None and dispatch_ms < 0:
            raise ANEProjectionError("ANE linear dispatch timing must not be negative")
        expected_byte_count = ANE_LINEAR_INPUT_BYTES if output_written else 0
        if output_byte_count != expected_byte_count:
            raise ANEProjectionError("ANE linear output byte count contradicts output state")

        expected_stage = None
        if compile_succeeded:
            expected_stage = "compile_succeeded"
        if load_succeeded:
            expected_stage = "load_succeeded"
        if dispatch_succeeded:
            expected_stage = "dispatch_succeeded"
        if output_written:
            expected_stage = "output_written"
        if expected_stage is not None and last_stage != expected_stage:
            raise ANEProjectionError("ANE linear last-successful-stage claim is inconsistent")
        if expected_stage is None and last_stage not in {
            "none",
            "request_validated",
            "surface_validated",
        }:
            raise ANEProjectionError("ANE linear pre-execution stage is invalid")

        expected_success = (
            compile_succeeded
            and load_succeeded
            and dispatch_succeeded
            and output_written
            and not error
            and not cleanup_error
        )
        if success != expected_success:
            raise ANEProjectionError("ANE linear success conclusion is inconsistent")
        if not success and not error and not cleanup_error:
            raise ANEProjectionError("failed ANE linear request lacks error evidence")

        return cls(
            raw=document,
            request_id=request_id,
            last_successful_stage=last_stage,
            success=success,
            compile_succeeded=compile_succeeded,
            load_succeeded=load_succeeded,
            dispatch_succeeded=dispatch_succeeded,
            output_written=output_written,
            output_byte_count=output_byte_count,
            dispatch_ms=dispatch_ms,
            error=error,
            cleanup_error=cleanup_error,
        )


class ANEProjectionExecutor:
    """Execute only Strata's exact qualified ``[64, 256]`` FP16 projection.

    Availability of this callable is not evidence that arbitrary model segments
    or shapes can run on ANE. The planner intentionally does not consume it yet.
    """

    def __init__(
        self,
        worker_path: os.PathLike[str] | str | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.worker_path = (
            Path(worker_path) if worker_path is not None else _default_worker_path()
        )
        self.timeout_seconds = float(timeout_seconds)
        self._last_report: ANEProjectionReport | None = None

    @property
    def last_report(self) -> ANEProjectionReport | None:
        return self._last_report

    def execute(self, input_tensor: np.ndarray, out_in_weight: np.ndarray) -> np.ndarray:
        """Run the fixed projection and return a contiguous ``[64, 256]`` FP16 array."""
        _validate_array(input_tensor, ANE_LINEAR_INPUT_SHAPE, "input_tensor")
        _validate_array(out_in_weight, ANE_LINEAR_WEIGHT_SHAPE, "out_in_weight")
        if not np.isfinite(input_tensor).all() or not np.isfinite(out_in_weight).all():
            raise ANEProjectionError("ANE linear input and weight values must be finite")

        worker = self.worker_path.expanduser()
        if not worker.is_file():
            raise ANEProjectionError(f"ANE linear worker does not exist: {worker}")
        if not os.access(worker, os.X_OK):
            raise ANEProjectionError(f"ANE linear worker is not executable: {worker}")

        self._last_report = None
        with tempfile.TemporaryDirectory(prefix="strata-ane-linear-") as directory:
            request_directory = Path(directory)
            input_path = request_directory / "input.fp16"
            weight_path = request_directory / "weight.fp16"
            output_path = request_directory / "output.fp16"
            request_path = request_directory / "request.json"
            input_path.write_bytes(input_tensor.tobytes(order="C"))
            weight_path.write_bytes(out_in_weight.tobytes(order="C"))
            request_id = secrets.token_hex(16)
            request = _build_request(
                request_id=request_id,
                input_path=input_path,
                weight_path=weight_path,
                output_path=output_path,
            )
            request_path.write_text(
                json.dumps(request, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )

            try:
                completed = subprocess.run(
                    [str(worker), "--execute-linear-request", str(request_path)],
                    check=False,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise ANEProjectionError(
                    f"ANE linear worker timed out after {self.timeout_seconds:g} seconds"
                ) from exc
            except OSError as exc:
                raise ANEProjectionError(f"failed to start ANE linear worker: {exc}") from exc

            if completed.returncode != 0:
                stderr = completed.stderr[:512].decode("utf-8", errors="replace").strip()
                detail = f": {stderr}" if stderr else ""
                raise ANEProjectionError(
                    f"ANE linear worker exited with status {completed.returncode}{detail}"
                )
            report = ANEProjectionReport.from_json(
                completed.stdout, expected_request_id=request_id
            )
            self._last_report = report
            if not report.success:
                detail = report.error or report.cleanup_error
                raise ANEProjectionError(
                    f"ANE linear request failed after {report.last_successful_stage}: {detail}"
                )
            output_bytes = _read_exact_regular_file(output_path, ANE_LINEAR_INPUT_BYTES)
            output = np.frombuffer(output_bytes, dtype=ANE_LINEAR_DTYPE).copy()
            output = output.reshape(ANE_LINEAR_INPUT_SHAPE)
            if not np.isfinite(output).all():
                raise ANEProjectionError("ANE linear output contained non-finite values")
            return output

    __call__ = execute


def _build_request(
    *, request_id: str, input_path: Path, weight_path: Path, output_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": ANE_LINEAR_SCHEMA_VERSION,
        "request": ANE_LINEAR_REQUEST_NAME,
        "request_id": request_id,
        "operation": ANE_LINEAR_OPERATION,
        "input": {
            "path": str(input_path),
            "shape": list(ANE_LINEAR_INPUT_SHAPE),
            "dtype": "float16",
            "byte_count": ANE_LINEAR_INPUT_BYTES,
        },
        "weight": {
            "path": str(weight_path),
            "shape": list(ANE_LINEAR_WEIGHT_SHAPE),
            "dtype": "float16",
            "layout": "out_in",
            "byte_count": ANE_LINEAR_WEIGHT_BYTES,
        },
        "output": {
            "path": str(output_path),
            "shape": list(ANE_LINEAR_INPUT_SHAPE),
            "dtype": "float16",
            "byte_count": ANE_LINEAR_INPUT_BYTES,
        },
    }


def _validate_array(value: Any, shape: tuple[int, int], name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != ANE_LINEAR_DTYPE:
        raise TypeError(f"{name} must have dtype float16")
    if value.shape != shape:
        raise ValueError(f"{name} must have exact shape {shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _read_exact_regular_file(path: Path, expected_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ANEProjectionError(f"cannot open ANE linear output: {exc}") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size != expected_bytes:
            raise ANEProjectionError("ANE linear output is not a regular file of the exact size")
        chunks: list[bytes] = []
        remaining = expected_bytes
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise ANEProjectionError("ANE linear output changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ANEProjectionError("ANE linear output changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _default_worker_path() -> Path:
    override = os.environ.get("STRATA_ANE_WORKER")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "native" / "ane" / "build" / "strata-ane-probe"


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ANEProjectionError(
            f"{path} has invalid keys; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ANEProjectionError(f"{path} must be a boolean")
    return value


def _require_int(value: Any, path: str) -> int:
    if type(value) is not int:
        raise ANEProjectionError(f"{path} must be an integer")
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ANEProjectionError(f"{path} must be a string")
    return value


def _require_int_list(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ANEProjectionError(f"{path} must be an array of integers")
    return value


def _require_optional_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ANEProjectionError(f"{path} must be a finite number or null")
    return float(value)


__all__ = [
    "ANEProjectionError",
    "ANEProjectionExecutor",
    "ANEProjectionReport",
]
