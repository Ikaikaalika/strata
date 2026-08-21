"""Fail-closed adapter for the isolated native ANE capability worker.

Discovery of Apple's private ANE runtime is deliberately separate from graph
execution. A valid discovery report identifies the ANE surface, but the backend
is not available as a compute unit until compile, dispatch, and numerical
verification exist in the native worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Tuple

from ..core.capabilities import RuntimeCapabilities
from ..core.hardware import ComputeUnit
from .base import Backend, BackendError


ANE_PROBE_SCHEMA_VERSION = 1
ANE_PROBE_NAME = "strata-ane-capability"
_MAX_REPORT_BYTES = 1_048_576

_CLASS_NAMES = (
    "_ANEClient",
    "_ANECompiler",
    "_ANEInMemoryModelDescriptor",
    "_ANEModel",
    "_ANEInMemoryModel",
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
)
_EXECUTION_FIELDS = (
    "compile_attempted",
    "dispatch_attempted",
    "execution_verified",
    "numeric_verified",
)


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
    compile_attempted: bool
    dispatch_attempted: bool
    execution_verified: bool
    numeric_verified: bool
    errors: Tuple[str, ...]

    @property
    def apple_silicon_supported(self) -> bool:
        """Whether the reported host can physically provide an Apple ANE."""
        return self.architecture == "arm64" and self.chip.startswith("Apple ")

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

        execution = _require_mapping(document["execution"], "execution")
        _require_exact_keys(execution, set(_EXECUTION_FIELDS), "execution")
        execution_flags = {
            name: _require_bool(execution[name], f"execution.{name}")
            for name in _EXECUTION_FIELDS
        }
        if execution_flags["execution_verified"] and not (
            execution_flags["compile_attempted"]
            and execution_flags["dispatch_attempted"]
            and execution_flags["numeric_verified"]
        ):
            raise ANEProbeError("ANE execution claim lacks compile, dispatch, or numeric proof")
        if execution_flags["numeric_verified"] and not execution_flags["execution_verified"]:
            raise ANEProbeError("ANE numeric verification requires verified execution")

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
            compile_attempted=execution_flags["compile_attempted"],
            dispatch_attempted=execution_flags["dispatch_attempted"],
            execution_verified=execution_flags["execution_verified"],
            numeric_verified=execution_flags["numeric_verified"],
            errors=errors,
        )


@dataclass(frozen=True)
class ANEDevice:
    """Resolved private-ANE device observation, not an execution handle."""

    report: ANEProbeReport


class ANEBackend(Backend):
    """Discovery-only backend for an out-of-process private ANE worker."""

    name = "ane"

    def __init__(
        self,
        worker_path: os.PathLike[str] | str | None = None,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.worker_path = Path(worker_path) if worker_path is not None else _default_worker_path()
        self.timeout_seconds = float(timeout_seconds)
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
            completed = subprocess.run(
                [str(path)],
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
        """Return whether generated ANE work has been numerically verified."""
        try:
            report = self.probe()
            return (
                report.apple_silicon_supported
                and report.required_surface_satisfied
                and report.execution_verified
                and report.numeric_verified
            )
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
        if not report.execution_verified or not report.numeric_verified:
            raise BackendError(
                "private ANE surface is discovered but execution is not numerically verified"
            )
        return ANEDevice(report=report)

    def capabilities(self) -> RuntimeCapabilities:
        try:
            report = self.probe()
        except ANEProbeError:
            return RuntimeCapabilities()
        if not report.apple_silicon_supported or not report.required_surface_satisfied:
            return RuntimeCapabilities()
        if not report.execution_verified or not report.numeric_verified:
            return RuntimeCapabilities(requires_private_api=True)
        return RuntimeCapabilities(
            compute_units=(ComputeUnit.ANE,),
            requires_private_api=True,
        )

    def move_model_to_device(self, model: Any, device: Any) -> Any:
        raise BackendError("ANE graph execution has not been numerically verified")

    def create_kv_cache(self, cache_dir: str, stats: Optional[Any]):
        raise BackendError("ANE KV-cache execution has not been numerically verified")

    def attention_kernel(self):
        raise BackendError("ANE graph execution has not been numerically verified")


def _default_worker_path() -> Path:
    override = os.environ.get("STRATA_ANE_WORKER")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "native" / "ane" / "build" / "strata-ane-probe"


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


__all__ = [
    "ANEBackend",
    "ANEDevice",
    "ANEProbeError",
    "ANEProbeReport",
]
