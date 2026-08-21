"""Fail-closed adapter for Strata's native Metal capability probe."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional

from ..core.capabilities import RuntimeCapabilities
from ..core.hardware import ComputeUnit
from ..core.ir import InferencePhase, OperationKind
from .base import Backend, BackendError


@dataclass(frozen=True)
class MetalProbeResult:
    """Validated subset of the native probe's evidence envelope."""

    probe_path: str
    dispatch_success: bool = False
    device_name: str = ""
    operation: str = ""
    dtype: str = ""
    numerical_max_error: float | None = None
    average_dispatch_wall_time_ms: float | None = None
    failure_reason: str | None = None
    raw: Mapping[str, Any] | None = None

    def is_usable(self, maximum_error: float) -> bool:
        return (
            self.dispatch_success
            and self.operation == "rmsnorm_residual_f32"
            and self.dtype == "float32"
            and self.numerical_max_error is not None
            and self.numerical_max_error <= maximum_error
            and bool(self.device_name)
        )


class MetalBackend(Backend):
    """Capability adapter for a separately built native Metal executable.

    This backend intentionally does not advertise an attention kernel or model
    movement yet. It becomes available only after the native executable reports
    a successful dispatch with acceptable numerical error.
    """

    name = "metal"

    def __init__(
        self,
        probe_path: str | os.PathLike[str] | None = None,
        *,
        timeout_seconds: float = 15.0,
        maximum_error: float = 1.0e-4,
    ) -> None:
        self._configured_probe_path = probe_path
        self._timeout_seconds = timeout_seconds
        self._maximum_error = maximum_error
        self._cached_probe_result: MetalProbeResult | None = None

    def _probe_path(self) -> Path:
        if self._configured_probe_path is not None:
            return Path(self._configured_probe_path).expanduser()
        configured = os.environ.get("STRATA_METAL_PROBE")
        if configured:
            return Path(configured).expanduser()
        repository_root = Path(__file__).resolve().parents[3]
        return repository_root / "native" / "metal" / "build" / "strata-metal-probe"

    def probe(self, *, refresh: bool = False) -> MetalProbeResult:
        if self._cached_probe_result is not None and not refresh:
            return self._cached_probe_result

        path = self._probe_path()
        if not path.is_file() or not os.access(path, os.X_OK):
            result = MetalProbeResult(
                probe_path=str(path),
                failure_reason="native Metal probe is missing or not executable",
            )
            self._cached_probe_result = result
            return result

        try:
            completed = subprocess.run(
                [str(path)],
                capture_output=True,
                check=False,
                text=True,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            result = MetalProbeResult(
                probe_path=str(path),
                failure_reason=f"native Metal probe failed to execute: {exc}",
            )
            self._cached_probe_result = result
            return result

        try:
            payload = json.loads(completed.stdout)
            if not isinstance(payload, dict):
                raise ValueError("root JSON value is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            result = MetalProbeResult(
                probe_path=str(path),
                failure_reason=f"native Metal probe returned invalid JSON: {exc}",
            )
            self._cached_probe_result = result
            return result

        failure_reason = payload.get("failure_reason")
        if completed.returncode != 0:
            failure_reason = failure_reason or (
                f"native Metal probe exited with status {completed.returncode}"
            )
        timing = payload.get("timing")
        if not isinstance(timing, dict):
            timing = {}

        contract_valid = (
            payload.get("schema_version") == 1
            and payload.get("backend") == "metal"
        )
        if not contract_valid:
            failure_reason = failure_reason or (
                "native Metal probe returned an unsupported evidence envelope"
            )
        result = MetalProbeResult(
            probe_path=str(path),
            dispatch_success=contract_valid
            and completed.returncode == 0
            and payload.get("dispatch_success") is True,
            device_name=_string_value(payload.get("device_name")),
            operation=_string_value(payload.get("operation")),
            dtype=_string_value(payload.get("dtype")),
            numerical_max_error=_optional_nonnegative_float(
                payload.get("numerical_max_error")
            ),
            average_dispatch_wall_time_ms=_optional_nonnegative_float(
                timing.get("average_dispatch_wall_time_ms")
            ),
            failure_reason=_string_value(failure_reason) or None,
            raw=payload,
        )
        if result.dispatch_success and not result.is_usable(self._maximum_error):
            result = MetalProbeResult(
                **{
                    **result.__dict__,
                    "failure_reason": (
                        result.failure_reason
                        or "native Metal evidence did not satisfy the capability contract"
                    ),
                }
            )
        self._cached_probe_result = result
        return result

    def is_available(self) -> bool:
        return self.probe().is_usable(self._maximum_error)

    def resolve_device(self, device_request: Optional[str]) -> str:
        result = self.probe()
        if not result.is_usable(self._maximum_error):
            raise BackendError(result.failure_reason or "native Metal backend is unavailable")
        if device_request not in (None, "", "auto", "gpu", "0", result.device_name):
            raise BackendError(f"Unsupported Metal device specification: {device_request}")
        return result.device_name

    def move_model_to_device(self, model: Any, device: Any) -> Any:
        raise BackendError("native Metal model execution is not implemented")

    def create_kv_cache(self, cache_dir: str, stats: Optional[Any]):
        raise BackendError("native Metal KV cache is not implemented")

    def attention_kernel(self):
        raise BackendError("native Metal attention is not implemented")

    def capabilities(self) -> RuntimeCapabilities:
        if not self.is_available():
            return RuntimeCapabilities()
        return RuntimeCapabilities(
            compute_units=(ComputeUnit.GPU,),
            supported_phases=(InferencePhase.PREFILL, InferencePhase.DECODE),
            supported_operations=(
                OperationKind.NORMALIZATION,
                OperationKind.RESIDUAL,
            ),
            supported_dtypes=("float32",),
            supports_dynamic_shapes=False,
        )


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _optional_nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        return None
    return result
