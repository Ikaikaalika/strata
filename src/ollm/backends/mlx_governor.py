"""Apple Silicon integration for Strata's runtime-neutral memory governor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..core.capabilities import RuntimeCapabilities
from ..core.execution_plan import ExecutionPlan
from ..core.hardware import ComputeUnit
from ..core.ir import InferencePhase, OperationKind
from ..scheduling.dense_pipeline import DenseLayerPipeline
from ..scheduling.prefetch_scheduler import PrefetchScheduler
from ..scheduling.residency_manager import ResidencyManager
from ..storage.manifest import decoder_model_spec
from ..storage.tensor_store import LoaderGroupTensorStore


@dataclass(frozen=True)
class MLXHardwareProfile:
    device_name: str
    architecture: str
    unified_memory_bytes: int
    recommended_working_set_bytes: int
    active_memory_bytes: int
    peak_memory_bytes: int
    cache_memory_bytes: int


def mlx_capabilities() -> RuntimeCapabilities:
    import mlx.core as mx

    return RuntimeCapabilities(
        compute_units=(ComputeUnit.GPU,),
        supported_phases=(InferencePhase.PREFILL, InferencePhase.DECODE),
        supported_operations=(
            OperationKind.EMBEDDING,
            OperationKind.NORMALIZATION,
            OperationKind.LINEAR,
            OperationKind.ROPE,
            OperationKind.ATTENTION,
            OperationKind.RESIDUAL,
            OperationKind.ACTIVATION,
            OperationKind.MLP,
            OperationKind.LOGITS,
        ),
        supported_dtypes=("float16", "bfloat16", "float32"),
        supports_weight_paging=True,
        supports_async_prefetch=True,
        supports_external_kv_cache=True,
        supports_quantized_weights=False,
        supports_graph_capture=hasattr(mx, "compile"),
        supports_memory_counters=all(
            hasattr(mx, name)
            for name in ("get_active_memory", "get_peak_memory", "get_cache_memory")
        ),
        supports_dynamic_shapes=True,
    )


def detect_mlx_hardware() -> MLXHardwareProfile:
    import mlx.core as mx

    if hasattr(mx, "device_info"):
        info = mx.device_info()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
        info = mx.metal.device_info()
    else:
        info = {}
    return MLXHardwareProfile(
        device_name=str(info.get("device_name", "unknown")),
        architecture=str(info.get("architecture", "unknown")),
        unified_memory_bytes=int(info.get("memory_size", 0)),
        recommended_working_set_bytes=int(
            info.get("max_recommended_working_set_size", info.get("memory_size", 0))
        ),
        active_memory_bytes=int(mx.get_active_memory()),
        peak_memory_bytes=int(mx.get_peak_memory()),
        cache_memory_bytes=int(mx.get_cache_memory()),
    )


def suggested_residency_budget(
    profile: Optional[MLXHardwareProfile] = None,
    *,
    headroom_fraction: float = 0.75,
    safety_margin_bytes: int = 512 * 1024 * 1024,
) -> int:
    """Suggest, but do not apply, a conservative warm-weight budget."""
    if not 0 < headroom_fraction <= 1:
        raise ValueError("headroom_fraction must be in (0, 1]")
    if safety_margin_bytes < 0:
        raise ValueError("safety_margin_bytes must be non-negative")
    profile = profile or detect_mlx_hardware()
    available = (
        profile.recommended_working_set_bytes
        - profile.active_memory_bytes
        - safety_margin_bytes
    )
    return max(0, int(available * headroom_fraction))


def synchronize_mlx(value: Any) -> None:
    import mlx.core as mx

    mx.eval(value)
    mx.synchronize()


def mlx_memory_probe() -> Mapping[str, int]:
    import mlx.core as mx

    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
    }


def build_mlx_dense_pipeline(
    *,
    model_id: str,
    layers: Sequence[Any],
    loader: Any,
    budget_bytes: int,
    tracer: Optional[Any] = None,
    prefetch_distance: int = 1,
) -> DenseLayerPipeline:
    """Build a persistent, budgeted MLX decoder-layer pipeline."""
    layer_manifests = [layer._layer_param_manifest_names() for layer in layers]
    model_spec = decoder_model_spec(model_id, layer_manifests, loader)
    plan = ExecutionPlan.dense(
        model_spec,
        prefetch_distance=prefetch_distance,
    )

    def on_evict(group_id: str, value: Any) -> None:
        if tracer is not None:
            tracer.record_operation(
                "weight_evict",
                0.0,
                phase="runtime",
                group_id=group_id,
                cache_status="lru_evict",
            )

    residency = ResidencyManager(budget_bytes, on_evict=on_evict)
    store = LoaderGroupTensorStore(loader)
    scheduler = PrefetchScheduler(store, residency, tracer=tracer)
    return DenseLayerPipeline(
        model_spec,
        plan,
        scheduler,
        tracer=tracer,
        synchronize=synchronize_mlx,
        memory_probe=mlx_memory_probe,
    )
