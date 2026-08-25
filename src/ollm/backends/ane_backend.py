"""Fail-closed adapter for the isolated native ANE capability worker.

Discovery of Apple's private ANE runtime is deliberately separate from graph
execution. A valid discovery report identifies the ANE surface, but the backend
is not available as a compute unit until compile, dispatch, and numerical
verification exist in the native worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Tuple

from ..core.capabilities import RuntimeCapabilities
from ..core.hardware import ComputeUnit
from .base import Backend, BackendError


ANE_PROBE_SCHEMA_VERSION = 2
ANE_PROBE_NAME = "strata-ane-capability"
_MAX_REPORT_BYTES = 1_048_576

_CLASS_NAMES = (
    "_ANEClient",
    "_ANECompiler",
    "_ANEInMemoryModelDescriptor",
    "_ANEModel",
    "_ANEInMemoryModel",
    "_ANERequest",
    "_ANEIOSurfaceObject",
)
_FRAMEWORK_NAMES = ("AppleNeuralEngine", "ANECompiler")
_SURFACE_BOOL_FIELDS = (
    "private_frameworks_loaded",
    "ane_client_present",
    "ane_client_entrypoints_present",
    "ane_compiler_present",
    "compiler_equivalent_present",
    "descriptor_present",
    "descriptor_factory_present",
    "model_lifecycle_entrypoints_present",
    "satisfied",
    "request_class_present",
    "iosurface_object_class_present",
    "request_factory_present",
    "iosurface_object_factory_present",
    "execution_surface_satisfied",
)
_EXECUTION_BOOL_FIELDS = (
    "requested",
    "mil_generated",
    "descriptor_created",
    "model_created",
    "artifacts_written",
    "compile_attempted",
    "compile_succeeded",
    "load_attempted",
    "load_succeeded",
    "iosurfaces_created",
    "request_created",
    "dispatch_attempted",
    "dispatch_succeeded",
    "output_finite",
    "numeric_verified",
    "execution_verified",
)
_EXECUTION_STRING_FIELDS = (
    "operation",
    "input_dtype",
    "compute_dtype",
    "output_dtype",
    "last_successful_stage",
    "failure_stage",
    "error",
    "cleanup_error",
)
_EXECUTION_KEYS = set(_EXECUTION_BOOL_FIELDS) | set(_EXECUTION_STRING_FIELDS) | {
    "shape",
    "tolerance",
    "max_abs_error",
    "dispatch_ms",
}


class ANEProbeError(BackendError):
    """Raised when the native capability worker cannot produce trusted data."""


@dataclass(frozen=True)
class ANEProbeReport:
    """Validated result from one invocation of the native ANE worker."""

    raw: Mapping[str, Any]
    chip: str
    macos_version: str
    macos_build: str
    architecture: str
    descriptor_class: str
    compiler_class: str
    required_surface_satisfied: bool
    execution_surface_satisfied: bool
    execution_requested: bool
    last_successful_stage: str
    compile_attempted: bool
    compile_succeeded: bool
    load_succeeded: bool
    dispatch_attempted: bool
    dispatch_succeeded: bool
    execution_verified: bool
    numeric_verified: bool
    max_abs_error: float | None
    dispatch_ms: float | None
    execution_error: str
    cleanup_error: str
    errors: Tuple[str, ...]

    @property
    def apple_silicon_supported(self) -> bool:
        """Whether the reported host can physically provide an Apple ANE."""
        return self.architecture == "arm64" and self.chip.startswith("Apple ")

    @property
    def runtime_fingerprint(self) -> str:
        """Stable cache key for the observed private runtime qualification.

        Timings and numerical measurements are deliberately excluded so
        repeated successful probes on the same runtime produce the same key.
        """
        identity = {
            "architecture": self.architecture,
            "chip": self.chip,
            "compiler_class": self.compiler_class,
            "descriptor_class": self.descriptor_class,
            "execution_surface_satisfied": self.execution_surface_satisfied,
            "macos_build": self.macos_build,
            "macos_version": self.macos_version,
            "probe": ANE_PROBE_NAME,
            "schema_version": ANE_PROBE_SCHEMA_VERSION,
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def qualification_verified(self) -> bool:
        """Whether execution, numerical readback, and cleanup all succeeded."""
        return (
            self.apple_silicon_supported
            and self.required_surface_satisfied
            and self.execution_verified
            and self.numeric_verified
            and not self.execution_error
            and not self.cleanup_error
            and not self.errors
        )

    @classmethod
    def from_json(cls, payload: str) -> "ANEProbeReport":
        if not payload or len(payload.encode("utf-8")) > _MAX_REPORT_BYTES:
            raise ANEProbeError("ANE probe emitted an empty or oversized report")
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ANEProbeError("ANE probe did not emit one valid JSON document") from exc
        if not isinstance(document, dict):
            raise ANEProbeError("ANE probe report must be a JSON object")

        _require_exact_keys(
            document,
            {
                "schema_version",
                "probe",
                "platform",
                "frameworks",
                "objective_c",
                "required_surface",
                "execution",
                "errors",
            },
            "report",
        )
        if _require_int(document["schema_version"], "schema_version") != ANE_PROBE_SCHEMA_VERSION:
            raise ANEProbeError("unsupported ANE probe schema version")
        if _require_str(document["probe"], "probe") != ANE_PROBE_NAME:
            raise ANEProbeError("unexpected ANE probe identity")

        platform = _require_mapping(document["platform"], "platform")
        _require_exact_keys(
            platform,
            {"os", "version", "build", "architecture", "chip"},
            "platform",
        )
        if _require_str(platform["os"], "platform.os") != "macOS":
            raise ANEProbeError("ANE probe did not report macOS")
        macos_version = _require_nonempty_str(platform["version"], "platform.version")
        macos_build = _require_nonempty_str(platform["build"], "platform.build")
        architecture = _require_nonempty_str(platform["architecture"], "platform.architecture")
        chip = _require_nonempty_str(platform["chip"], "platform.chip")

        frameworks = _require_mapping(document["frameworks"], "frameworks")
        _require_exact_keys(frameworks, set(_FRAMEWORK_NAMES), "frameworks")
        framework_loaded = {}
        for name in _FRAMEWORK_NAMES:
            entry = _require_mapping(frameworks[name], f"frameworks.{name}")
            _require_exact_keys(entry, {"path", "loaded", "error"}, f"frameworks.{name}")
            _require_nonempty_str(entry["path"], f"frameworks.{name}.path")
            framework_loaded[name] = _require_bool(entry["loaded"], f"frameworks.{name}.loaded")
            _require_str(entry["error"], f"frameworks.{name}.error")

        objective_c = _require_mapping(document["objective_c"], "objective_c")
        _require_exact_keys(
            objective_c,
            {"classes", "descriptor_candidates", "descriptor_class"},
            "objective_c",
        )
        classes = _require_mapping(objective_c["classes"], "objective_c.classes")
        _require_exact_keys(classes, set(_CLASS_NAMES), "objective_c.classes")
        classes_present = {}
        instance_methods = {}
        class_methods = {}
        for name in _CLASS_NAMES:
            entry = _require_mapping(classes[name], f"objective_c.classes.{name}")
            _require_exact_keys(
                entry,
                {"present", "instance_methods", "class_methods"},
                f"objective_c.classes.{name}",
            )
            classes_present[name] = _require_bool(
                entry["present"], f"objective_c.classes.{name}.present"
            )
            instance_methods[name] = set(_require_str_list(
                entry["instance_methods"], f"objective_c.classes.{name}.instance_methods"
            ))
            class_methods[name] = set(_require_str_list(
                entry["class_methods"], f"objective_c.classes.{name}.class_methods"
            ))
        descriptor_candidates = _require_str_list(
            objective_c["descriptor_candidates"], "objective_c.descriptor_candidates"
        )
        descriptor_class = _require_str(
            objective_c["descriptor_class"], "objective_c.descriptor_class"
        )

        surface = _require_mapping(document["required_surface"], "required_surface")
        _require_exact_keys(
            surface,
            set(_SURFACE_BOOL_FIELDS) | {"descriptor_class", "compiler_class"},
            "required_surface",
        )
        surface_flags = {
            name: _require_bool(surface[name], f"required_surface.{name}")
            for name in _SURFACE_BOOL_FIELDS
        }
        surface_descriptor = _require_str(
            surface["descriptor_class"], "required_surface.descriptor_class"
        )
        compiler_class = _require_str(
            surface["compiler_class"], "required_surface.compiler_class"
        )

        expected_frameworks = all(framework_loaded.values())
        if surface_flags["private_frameworks_loaded"] != expected_frameworks:
            raise ANEProbeError("ANE framework summary contradicts framework observations")
        if surface_flags["ane_client_present"] != classes_present["_ANEClient"]:
            raise ANEProbeError("ANE client summary contradicts class observations")
        if surface_flags["ane_compiler_present"] != classes_present["_ANECompiler"]:
            raise ANEProbeError("ANE compiler summary contradicts class observations")
        if descriptor_class != surface_descriptor:
            raise ANEProbeError("ANE descriptor summaries disagree")
        if surface_flags["descriptor_present"]:
            if (
                descriptor_class not in descriptor_candidates
                or descriptor_class not in classes_present
                or not classes_present[descriptor_class]
            ):
                raise ANEProbeError("ANE descriptor class was not actually observed")
        elif descriptor_class:
            raise ANEProbeError("ANE descriptor class is set while descriptor is absent")

        observed_client_entrypoints = (
            "sharedConnection" in class_methods["_ANEClient"]
            and "loadModel:options:qos:error:" in instance_methods["_ANEClient"]
            and "evaluateWithModel:options:request:qos:error:"
            in instance_methods["_ANEClient"]
        )
        if surface_flags["ane_client_entrypoints_present"] != observed_client_entrypoints:
            raise ANEProbeError("ANE client entrypoint summary contradicts method observations")
        observed_descriptor_factory = (
            "modelWithMILText:weights:optionsPlist:"
            in class_methods["_ANEInMemoryModelDescriptor"]
        )
        if surface_flags["descriptor_factory_present"] != observed_descriptor_factory:
            raise ANEProbeError("ANE descriptor factory summary contradicts method observations")
        observed_in_memory_compiler = (
            "inMemoryModelWithDescriptor:" in class_methods["_ANEInMemoryModel"]
            and "compileWithQoS:options:error:" in instance_methods["_ANEInMemoryModel"]
        )
        observed_compiler_equivalent = (
            classes_present["_ANECompiler"] or observed_in_memory_compiler
        )
        if surface_flags["compiler_equivalent_present"] != observed_compiler_equivalent:
            raise ANEProbeError("ANE compiler summary contradicts method observations")
        if surface_flags["compiler_equivalent_present"]:
            if compiler_class not in classes_present or not classes_present[compiler_class]:
                raise ANEProbeError("ANE compiler equivalent was not actually observed")
        elif compiler_class:
            raise ANEProbeError("ANE compiler class is set while compiler surface is absent")
        observed_model_lifecycle = {
            "loadWithQoS:options:error:",
            "evaluateWithQoS:options:request:error:",
            "unloadWithQoS:error:",
        }.issubset(instance_methods["_ANEInMemoryModel"])
        if surface_flags["model_lifecycle_entrypoints_present"] != observed_model_lifecycle:
            raise ANEProbeError("ANE lifecycle summary contradicts method observations")

        observed_request_factory = (
            "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"
            in class_methods["_ANERequest"]
        )
        observed_iosurface_factory = (
            "objectWithIOSurface:" in class_methods["_ANEIOSurfaceObject"]
        )
        if surface_flags["request_class_present"] != classes_present["_ANERequest"]:
            raise ANEProbeError("ANE request summary contradicts class observations")
        if (
            surface_flags["iosurface_object_class_present"]
            != classes_present["_ANEIOSurfaceObject"]
        ):
            raise ANEProbeError("ANE IOSurface summary contradicts class observations")
        if surface_flags["request_factory_present"] != observed_request_factory:
            raise ANEProbeError("ANE request factory summary contradicts method observations")
        if surface_flags["iosurface_object_factory_present"] != observed_iosurface_factory:
            raise ANEProbeError("ANE IOSurface factory summary contradicts method observations")

        expected_surface = all(
            surface_flags[name]
            for name in (
                "private_frameworks_loaded",
                "ane_client_present",
                "ane_client_entrypoints_present",
                "compiler_equivalent_present",
                "descriptor_present",
                "descriptor_factory_present",
                "model_lifecycle_entrypoints_present",
            )
        )
        if surface_flags["satisfied"] != expected_surface:
            raise ANEProbeError("ANE required-surface conclusion is internally inconsistent")

        expected_execution_surface = expected_surface and all(
            surface_flags[name]
            for name in (
                "request_class_present",
                "iosurface_object_class_present",
                "request_factory_present",
                "iosurface_object_factory_present",
            )
        )
        if surface_flags["execution_surface_satisfied"] != expected_execution_surface:
            raise ANEProbeError("ANE execution-surface conclusion is internally inconsistent")

        execution = _require_mapping(document["execution"], "execution")
        _require_exact_keys(execution, _EXECUTION_KEYS, "execution")
        execution_flags = {
            name: _require_bool(execution[name], f"execution.{name}")
            for name in _EXECUTION_BOOL_FIELDS
        }
        execution_strings = {
            name: _require_str(execution[name], f"execution.{name}")
            for name in _EXECUTION_STRING_FIELDS
        }
        shape = _require_int_list(execution["shape"], "execution.shape")
        if shape != [1, 256, 1, 64]:
            raise ANEProbeError("ANE projection proof has an unexpected tensor shape")
        if execution_strings["operation"] != "fp16_projection":
            raise ANEProbeError("ANE execution proof has an unexpected operation")
        for field in ("input_dtype", "compute_dtype", "output_dtype"):
            if execution_strings[field] != "float16":
                raise ANEProbeError(f"execution.{field} must be float16")
        tolerance = _require_number(execution["tolerance"], "execution.tolerance")
        if tolerance <= 0:
            raise ANEProbeError("execution.tolerance must be positive")
        max_abs_error = _require_optional_number(
            execution["max_abs_error"], "execution.max_abs_error"
        )
        dispatch_ms = _require_optional_number(execution["dispatch_ms"], "execution.dispatch_ms")
        if max_abs_error is not None and max_abs_error < 0:
            raise ANEProbeError("execution.max_abs_error must not be negative")
        if dispatch_ms is not None and dispatch_ms < 0:
            raise ANEProbeError("execution.dispatch_ms must not be negative")

        _validate_execution_progress(
            execution_flags=execution_flags,
            execution_strings=execution_strings,
            execution_surface_satisfied=surface_flags["execution_surface_satisfied"],
            tolerance=tolerance,
            max_abs_error=max_abs_error,
            dispatch_ms=dispatch_ms,
        )

        errors = tuple(_require_str_list(document["errors"], "errors"))
        return cls(
            raw=document,
            chip=chip,
            macos_version=macos_version,
            macos_build=macos_build,
            architecture=architecture,
            descriptor_class=descriptor_class,
            compiler_class=compiler_class,
            required_surface_satisfied=surface_flags["satisfied"],
            execution_surface_satisfied=surface_flags["execution_surface_satisfied"],
            execution_requested=execution_flags["requested"],
            last_successful_stage=execution_strings["last_successful_stage"],
            compile_attempted=execution_flags["compile_attempted"],
            compile_succeeded=execution_flags["compile_succeeded"],
            load_succeeded=execution_flags["load_succeeded"],
            dispatch_attempted=execution_flags["dispatch_attempted"],
            dispatch_succeeded=execution_flags["dispatch_succeeded"],
            execution_verified=execution_flags["execution_verified"],
            numeric_verified=execution_flags["numeric_verified"],
            max_abs_error=max_abs_error,
            dispatch_ms=dispatch_ms,
            execution_error=execution_strings["error"],
            cleanup_error=execution_strings["cleanup_error"],
            errors=errors,
        )


@dataclass(frozen=True)
class ANEDevice:
    """Resolved private-ANE device observation, not an execution handle."""

    report: ANEProbeReport


class ANEBackend(Backend):
    """Fail-closed backend backed by an out-of-process private ANE probe."""

    name = "ane"

    def __init__(
        self,
        worker_path: os.PathLike[str] | str | None = None,
        *,
        timeout_seconds: float = 30.0,
        attempt_execution: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.worker_path = Path(worker_path) if worker_path is not None else _default_worker_path()
        self.timeout_seconds = float(timeout_seconds)
        self.attempt_execution = bool(attempt_execution)
        self._cached_report: ANEProbeReport | None = None
        self._cached_error: ANEProbeError | None = None

    def probe(self, *, refresh: bool = False) -> ANEProbeReport:
        if refresh:
            self._cached_report = None
            self._cached_error = None
        if self._cached_report is not None:
            return self._cached_report
        if self._cached_error is not None:
            raise self._cached_error

        try:
            report = self._run_worker()
        except ANEProbeError as exc:
            self._cached_error = exc
            raise
        self._cached_report = report
        return report

    def _run_worker(self) -> ANEProbeReport:
        path = self.worker_path.expanduser()
        if not path.is_file():
            raise ANEProbeError(f"ANE worker does not exist: {path}")
        if not os.access(path, os.X_OK):
            raise ANEProbeError(f"ANE worker is not executable: {path}")
        try:
            command = [str(path)]
            if self.attempt_execution:
                command.append("--execute-projection")
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ANEProbeError(
                f"ANE worker timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise ANEProbeError(f"failed to start ANE worker: {exc}") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()[:512]
            detail = f": {stderr}" if stderr else ""
            raise ANEProbeError(
                f"ANE worker exited with status {completed.returncode}{detail}"
            )
        return ANEProbeReport.from_json(completed.stdout)

    def is_discovered(self) -> bool:
        """Return whether the required private surface exists on Apple Silicon."""
        try:
            report = self.probe()
            return report.apple_silicon_supported and report.required_surface_satisfied
        except ANEProbeError:
            return False

    def is_available(self) -> bool:
        """Return whether the fixed projection qualification proof passed.

        This qualifies the local private runtime. It does not imply that a
        general StrataIR segment executor has been implemented.
        """
        try:
            report = self.probe()
            return report.qualification_verified
        except ANEProbeError:
            return False

    def resolve_device(self, device_request: Optional[str]) -> ANEDevice:
        if device_request not in (None, "", "auto", "ane"):
            raise BackendError(f"Unsupported ANE device specification: {device_request}")
        try:
            report = self.probe()
        except ANEProbeError as exc:
            raise BackendError(f"ANE capability probe failed: {exc}") from exc
        if not report.apple_silicon_supported:
            raise BackendError("ANE backend requires a native Apple Silicon process")
        if not report.required_surface_satisfied:
            raise BackendError("required private ANE runtime surface is unavailable")
        if not report.qualification_verified:
            raise BackendError(
                "private ANE surface is discovered but qualification is incomplete"
            )
        return ANEDevice(report=report)

    def capabilities(self) -> RuntimeCapabilities:
        try:
            report = self.probe()
        except ANEProbeError:
            return RuntimeCapabilities()
        if not report.apple_silicon_supported or not report.required_surface_satisfied:
            return RuntimeCapabilities()
        if not report.qualification_verified:
            return RuntimeCapabilities(requires_private_api=True)
        return RuntimeCapabilities(
            compute_units=(ComputeUnit.ANE,),
            supported_dtypes=("float16",),
            supports_shared_iosurface=True,
            requires_private_api=True,
        )

    def move_model_to_device(self, model: Any, device: Any) -> Any:
        raise BackendError("ANE projection proof does not implement general model placement")

    def create_kv_cache(self, cache_dir: str, stats: Optional[Any]):
        raise BackendError("ANE projection proof does not implement a KV cache")

    def attention_kernel(self):
        raise BackendError("ANE projection proof does not implement attention")


def _default_worker_path() -> Path:
    override = os.environ.get("STRATA_ANE_WORKER")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "native" / "ane" / "build" / "strata-ane-probe"


def _validate_execution_progress(
    *,
    execution_flags: Mapping[str, bool],
    execution_strings: Mapping[str, str],
    execution_surface_satisfied: bool,
    tolerance: float,
    max_abs_error: float | None,
    dispatch_ms: float | None,
) -> None:
    requested = execution_flags["requested"]
    progress_chain = (
        "mil_generated",
        "descriptor_created",
        "model_created",
        "artifacts_written",
        "compile_attempted",
        "compile_succeeded",
        "load_attempted",
        "load_succeeded",
        "iosurfaces_created",
        "request_created",
        "dispatch_attempted",
        "dispatch_succeeded",
    )
    for index, field in enumerate(progress_chain[1:], start=1):
        if execution_flags[field] and not execution_flags[progress_chain[index - 1]]:
            raise ANEProbeError(
                f"execution.{field} is true before {progress_chain[index - 1]}"
            )
    if any(execution_flags[field] for field in progress_chain) and not requested:
        raise ANEProbeError("ANE execution progress exists without a requested proof")
    if execution_flags["output_finite"] and not execution_flags["dispatch_succeeded"]:
        raise ANEProbeError("finite ANE output requires successful dispatch")
    if (dispatch_ms is not None) != execution_flags["dispatch_attempted"]:
        raise ANEProbeError("ANE dispatch timing contradicts dispatch state")
    if (max_abs_error is not None) != execution_flags["output_finite"]:
        raise ANEProbeError("ANE numerical error contradicts output state")

    expected_numeric = (
        execution_flags["output_finite"]
        and max_abs_error is not None
        and max_abs_error <= tolerance
    )
    if execution_flags["numeric_verified"] != expected_numeric:
        raise ANEProbeError("ANE numeric-verification conclusion is inconsistent")
    expected_execution = (
        execution_flags["compile_succeeded"]
        and execution_flags["load_succeeded"]
        and execution_flags["dispatch_succeeded"]
        and execution_flags["numeric_verified"]
    )
    if execution_flags["execution_verified"] != expected_execution:
        raise ANEProbeError("ANE execution-verification conclusion is inconsistent")

    expected_stage = "discovery"
    if requested and execution_surface_satisfied:
        expected_stage = "surface_validated"
    stage_for_flag = {
        "mil_generated": "mil_generated",
        "descriptor_created": "descriptor_created",
        "model_created": "model_created",
        "artifacts_written": "artifacts_written",
        "compile_succeeded": "compile_succeeded",
        "load_succeeded": "load_succeeded",
        "iosurfaces_created": "iosurfaces_created",
        "request_created": "request_created",
        "dispatch_succeeded": "dispatch_succeeded",
        "numeric_verified": "numeric_verified",
    }
    for field, stage in stage_for_flag.items():
        if execution_flags[field]:
            expected_stage = stage
    if execution_strings["last_successful_stage"] != expected_stage:
        raise ANEProbeError("ANE last-successful-stage conclusion is inconsistent")

    failure_stage = execution_strings["failure_stage"]
    error = execution_strings["error"]
    if not requested:
        if failure_stage or error:
            raise ANEProbeError("unrequested ANE proof must not report a failure")
    elif execution_flags["execution_verified"]:
        if failure_stage or error:
            raise ANEProbeError("verified ANE execution must not report a failure")
    elif not failure_stage or not error:
        raise ANEProbeError("failed ANE execution must report its failure stage and error")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ANEProbeError(f"{path} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ANEProbeError(f"{path} has invalid keys; missing={missing}, extra={extra}")


def _require_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ANEProbeError(f"{path} must be a boolean")
    return value


def _require_int(value: Any, path: str) -> int:
    if type(value) is not int:
        raise ANEProbeError(f"{path} must be an integer")
    return value


def _require_number(value: Any, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ANEProbeError(f"{path} must be a finite number")
    return float(value)


def _require_optional_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _require_number(value, path)


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ANEProbeError(f"{path} must be a string")
    return value


def _require_nonempty_str(value: Any, path: str) -> str:
    text = _require_str(value, path)
    if not text:
        raise ANEProbeError(f"{path} must not be empty")
    return text


def _require_str_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ANEProbeError(f"{path} must be an array of strings")
    return value


def _require_int_list(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ANEProbeError(f"{path} must be an array of integers")
    return value


__all__ = [
    "ANEBackend",
    "ANEDevice",
    "ANEProbeError",
    "ANEProbeReport",
]
