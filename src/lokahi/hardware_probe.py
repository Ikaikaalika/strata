"""Local Apple Silicon discovery for adaptive-plan cache keys."""
from __future__ import annotations

import platform
import subprocess
from typing import Callable, Mapping, Optional, Sequence

from .core.hardware import ComputeUnit, HardwareProfile


CommandRunner = Callable[[Sequence[str]], str]
MLXInfoProvider = Callable[[], Mapping[str, object]]


def _run_text(command: Sequence[str]) -> str:
    completed = subprocess.run(
        tuple(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip()


def _mlx_device_info() -> Mapping[str, object]:
    try:
        import mlx.core as mx
    except ModuleNotFoundError:
        return {}
    if hasattr(mx, "device_info"):
        return mx.device_info()
    if hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
        return mx.metal.device_info()
    return {}


def detect_apple_hardware(
    *,
    ane_runtime_fingerprint: Optional[str] = None,
    ane_execution_verified: bool = False,
    storage_read_bytes_per_second: Optional[float] = None,
    command_runner: CommandRunner = _run_text,
    mlx_info_provider: MLXInfoProvider = _mlx_device_info,
) -> HardwareProfile:
    """Return facts used to select and invalidate hardware execution plans.

    ANE is included as an executable compute unit only after a generated graph
    has dispatched and passed a numerical comparison. Runtime discovery alone
    may still be recorded in ``ane_runtime_fingerprint`` for cache invalidation.
    """
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"}:
        raise RuntimeError(f"Lokahi hardware probe requires Apple Silicon, got {machine!r}")
    if ane_execution_verified and not ane_runtime_fingerprint:
        raise ValueError("verified ANE execution requires a runtime fingerprint")

    chip = command_runner(("sysctl", "-n", "machdep.cpu.brand_string"))
    unified_memory = int(command_runner(("sysctl", "-n", "hw.memsize")))
    logical_cores = int(command_runner(("sysctl", "-n", "hw.logicalcpu")))
    macos_version = command_runner(("sw_vers", "-productVersion"))
    macos_build = command_runner(("sw_vers", "-buildVersion"))

    mlx_info = mlx_info_provider()
    working_set_value = mlx_info.get("max_recommended_working_set_size")
    if working_set_value is None:
        working_set_value = mlx_info.get("memory_size")
    working_set = int(working_set_value) if working_set_value else None

    compute_units = [ComputeUnit.CPU, ComputeUnit.GPU]
    if ane_execution_verified:
        compute_units.append(ComputeUnit.ANE)

    return HardwareProfile(
        chip=chip,
        macos_version=macos_version,
        macos_build=macos_build,
        unified_memory_bytes=unified_memory,
        cpu_logical_cores=logical_cores,
        compute_units=tuple(compute_units),
        gpu_recommended_working_set_bytes=working_set,
        ane_runtime_fingerprint=ane_runtime_fingerprint,
        storage_read_bytes_per_second=storage_read_bytes_per_second,
    )
