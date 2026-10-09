"""Quantized Llama laboratory runtime with explicit layer-weight residency.

This is a Python/MLX correctness and paging laboratory, not the target native
Lokahi hot path. It exists to qualify exact artifacts and residency behavior
before the same contracts move into the native runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.ssd_offload import SSDOffloadMode, SSDOffloadPolicy


@dataclass(frozen=True)
class PagedRuntimeMetadata:
    snapshot: str
    model_type: str
    quantization_bits: int
    quantization_group_size: int
    total_layer_weight_bytes: int
    pinned_global_weight_bytes: int
    layer_residency_budget_bytes: int
    prefetch_distance: int
    io_workers: int
    pinned_layer_count: int
    pinned_layer_weight_bytes: int
    storage_target_id: str | None


def select_pinned_layer_group_ids(
    groups: tuple[Any, ...],
    *,
    count: int,
    layer_budget_bytes: int,
    prefetch_distance: int,
) -> tuple[str, ...]:
    """Select a deterministic pinned prefix while preserving rolling capacity."""

    if count < 0 or count >= len(groups):
        raise ValueError("pinned layer count must leave at least one rolling layer")
    selected = groups[:count]
    rolling = groups[count:]
    pinned_bytes = sum(group.nbytes for group in selected)
    largest_rolling = max(group.nbytes for group in rolling)
    rolling_slots = 1 + min(prefetch_distance, max(0, len(rolling) - 1))
    conservative_required = pinned_bytes + rolling_slots * largest_rolling
    if conservative_required > layer_budget_bytes:
        raise ValueError(
            "pinned layer set leaves insufficient budget for current and prefetched rolling layers"
        )
    return tuple(group.group_id for group in selected)


def maximum_pinned_layer_count(
    groups: tuple[Any, ...],
    *,
    layer_budget_bytes: int,
    prefetch_distance: int,
) -> int:
    """Choose the largest pinned prefix that preserves the rolling window."""

    for count in range(len(groups) - 1, -1, -1):
        try:
            select_pinned_layer_group_ids(
                groups,
                count=count,
                layer_budget_bytes=layer_budget_bytes,
                prefetch_distance=prefetch_distance,
            )
        except ValueError:
            continue
        return count
    raise ValueError("layer budget cannot hold one rolling layer")


class PagedMLXLlamaRuntime:
    """Own one custom Llama model and its bounded layer residency pipeline."""

    def __init__(
        self,
        *,
        model: Any,
        loader: Any,
        config: Any,
        metadata: PagedRuntimeMetadata,
        pinned_leases: tuple[Any, ...] = (),
    ) -> None:
        self.model = model
        self.loader = loader
        self.config = config
        self.metadata = metadata
        self._pinned_leases = pinned_leases

    @classmethod
    def load(
        cls,
        snapshot: str | Path,
        policy: SSDOffloadPolicy,
        pinned_layer_count: int | None = None,
    ) -> "PagedMLXLlamaRuntime":
        if policy.mode is SSDOffloadMode.DISABLED:
            raise ValueError("paged Llama runtime requires SSD offload auto or required")
        if policy.max_resident_weight_bytes is None:
            raise ValueError(
                "paged Llama runtime requires max_resident_weight_bytes so paging is measurable"
            )
        path = Path(snapshot).expanduser().resolve()
        forbidden = Path("/Volumes/Tyler HDD").resolve()
        try:
            path.relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise ValueError("/Volumes/Tyler HDD is forbidden for paged model weights")
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("snapshot must be a local model directory")

        import mlx.core as mx
        from transformers import AutoConfig

        from ..llama_mlx import MLXLlamaForCausalLM
        from ..mlx_loader import MLXMoEWeightsLoader

        config = AutoConfig.from_pretrained(str(path), local_files_only=True)
        if getattr(config, "model_type", None) != "llama":
            raise ValueError("paged MLX laboratory currently supports Llama only")
        quantization = getattr(config, "quantization", None)
        if not isinstance(quantization, dict):
            raise ValueError("paged MLX Llama requires an MLX quantized artifact")
        group_size = int(quantization.get("group_size", 0))
        bits = int(quantization.get("bits", 0))
        if group_size <= 0 or bits <= 0:
            raise ValueError("invalid quantization metadata")

        loader = MLXMoEWeightsLoader(str(path))
        pinned_names = [
            "model.embed_tokens.weight",
            "model.embed_tokens.scales",
            "model.embed_tokens.biases",
            "model.norm.weight",
        ]
        if not getattr(config, "tie_word_embeddings", False):
            pinned_names.extend(
                ["lm_head.weight", "lm_head.scales", "lm_head.biases"]
            )
        pinned_bytes = sum(loader.tensor_metadata_any(name)["nbytes"] for name in pinned_names)
        layer_budget = policy.max_resident_weight_bytes - pinned_bytes
        if layer_budget <= 0:
            raise ValueError(
                "max_resident_weight_bytes does not cover pinned embedding/norm weights"
            )

        model = MLXLlamaForCausalLM(
            config,
            weight_loader=loader,
            memory_budget_bytes=layer_budget,
            prefetch_distance=policy.prefetch_distance,
            prefetch_workers=policy.io_workers,
        )
        model.model.embed_tokens_weight = loader.load_tensor("model.embed_tokens.weight")
        model.model.embed_tokens_scales = loader.load_tensor("model.embed_tokens.scales")
        model.model.embed_tokens_biases = loader.load_tensor("model.embed_tokens.biases")
        model.model.norm.weight = loader.load_tensor("model.norm.weight")
        if not getattr(config, "tie_word_embeddings", False):
            model.model.lm_head_weight = loader.load_tensor("lm_head.weight")
            model.model.lm_head_scales = loader.load_tensor("lm_head.scales")
            model.model.lm_head_biases = loader.load_tensor("lm_head.biases")
        mx.eval(
            model.model.embed_tokens_weight,
            model.model.embed_tokens_scales,
            model.model.embed_tokens_biases,
            model.model.norm.weight,
        )

        pipeline = model.model.layer_pipeline
        assert pipeline is not None
        total_layer_bytes = pipeline.model_spec.total_weight_bytes
        largest_layer_bytes = max(group.nbytes for group in pipeline.model_spec.groups)
        if layer_budget < largest_layer_bytes:
            model.close()
            raise ValueError(
                "layer residency budget is smaller than the largest atomic layer group"
            )
        if policy.mode is SSDOffloadMode.REQUIRED and layer_budget >= total_layer_bytes:
            model.close()
            raise ValueError(
                "required SSD offload needs a resident cap smaller than total layer weights"
            )

        groups = pipeline.model_spec.groups
        if pinned_layer_count is None:
            pinned_layer_count = maximum_pinned_layer_count(
                groups,
                layer_budget_bytes=layer_budget,
                prefetch_distance=policy.prefetch_distance,
            )
        pinned_group_ids = select_pinned_layer_group_ids(
            groups,
            count=pinned_layer_count,
            layer_budget_bytes=layer_budget,
            prefetch_distance=policy.prefetch_distance,
        )
        groups_by_id = pipeline.model_spec.groups_by_id()
        leases = []
        try:
            for group_id in pinned_group_ids:
                leases.append(
                    pipeline.scheduler.acquire(
                        groups_by_id[group_id],
                        phase="model_load",
                    )
                )
        except BaseException:
            for lease in reversed(leases):
                lease.release()
            model.close()
            raise
        pinned_layer_bytes = sum(
            groups_by_id[group_id].nbytes for group_id in pinned_group_ids
        )

        return cls(
            model=model,
            loader=loader,
            config=config,
            metadata=PagedRuntimeMetadata(
                snapshot=str(path),
                model_type=str(config.model_type),
                quantization_bits=bits,
                quantization_group_size=group_size,
                total_layer_weight_bytes=total_layer_bytes,
                pinned_global_weight_bytes=pinned_bytes,
                layer_residency_budget_bytes=layer_budget,
                prefetch_distance=policy.prefetch_distance,
                io_workers=policy.io_workers,
                pinned_layer_count=len(pinned_group_ids),
                pinned_layer_weight_bytes=pinned_layer_bytes,
                storage_target_id=policy.storage_target_id,
            ),
            pinned_leases=tuple(leases),
        )

    def scheduler_snapshot(self) -> Any:
        pipeline = self.model.model.layer_pipeline
        if pipeline is None:
            raise RuntimeError("paged runtime has no layer pipeline")
        return pipeline.scheduler.snapshot()

    def residency_snapshot(self) -> Any:
        pipeline = self.model.model.layer_pipeline
        if pipeline is None:
            raise RuntimeError("paged runtime has no layer pipeline")
        return pipeline.scheduler.residency.snapshot()

    def close(self) -> None:
        for lease in reversed(self._pinned_leases):
            lease.release()
        self._pinned_leases = ()
        self.model.close()

    def __enter__(self) -> "PagedMLXLlamaRuntime":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "PagedMLXLlamaRuntime",
    "PagedRuntimeMetadata",
    "maximum_pinned_layer_count",
    "select_pinned_layer_group_ids",
]
