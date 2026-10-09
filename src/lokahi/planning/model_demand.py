"""Architecture-aware lower-bound memory estimates for request admission."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.model_manifest import (
    ArchitectureClass,
    AttentionPattern,
    PortableModelManifest,
)


@dataclass(frozen=True)
class KVCacheEstimate:
    """Logical KV bytes before backend padding, alignment, or allocator overhead."""

    batch_size: int
    context_tokens: int
    bytes_per_element: int
    total_bytes: int
    all_dense_bytes: int
    dense_layer_count: int
    windowed_layer_count: int

    @property
    def bytes_saved_vs_all_dense(self) -> int:
        return self.all_dense_bytes - self.total_bytes

    @property
    def fraction_saved_vs_all_dense(self) -> float:
        if self.all_dense_bytes == 0:
            return 0.0
        return self.bytes_saved_vs_all_dense / self.all_dense_bytes


def estimate_weight_bytes_per_decode_token(
    manifest: PortableModelManifest,
    *,
    rolling_weight_bytes: int,
    selected_expert_bytes: int | None = None,
) -> int:
    """Conservative storage traffic for a nonresident decode iteration.

    Dense and hybrid transformers traverse their rolling layer set every token;
    a cyclic scan can defeat ordinary LRU unless that entire set fits. Sparse
    MoE instead uses the explicitly supplied selected-expert working set.
    """

    if rolling_weight_bytes <= 0:
        raise ValueError("rolling_weight_bytes must be positive")
    architecture_class = manifest.architecture.architecture_class
    if architecture_class is ArchitectureClass.SYSTEM_HOSTED:
        raise ValueError("system-hosted model weight traffic is opaque")
    if architecture_class is ArchitectureClass.SPARSE_MOE_TRANSFORMER:
        if selected_expert_bytes is None or selected_expert_bytes <= 0:
            raise ValueError("sparse MoE paging requires selected_expert_bytes")
        return selected_expert_bytes
    if selected_expert_bytes is not None:
        raise ValueError("selected_expert_bytes is valid only for sparse MoE")
    return rolling_weight_bytes


def estimate_kv_cache_bytes(
    manifest: PortableModelManifest,
    *,
    batch_size: int,
    context_tokens: int,
    bytes_per_element: int = 2,
) -> KVCacheEstimate:
    """Estimate logical attention KV bytes from layer-exact semantics.

    This is a deterministic planning estimate, not measured peak memory. Hybrid
    recurrent/linear state is rejected until its state schema is represented.
    """

    if min(batch_size, context_tokens, bytes_per_element) <= 0:
        raise ValueError(
            "batch_size, context_tokens, and bytes_per_element must be positive"
        )
    architecture = manifest.architecture
    if architecture.architecture_class is ArchitectureClass.SYSTEM_HOSTED:
        raise ValueError("system-hosted model KV state is opaque")
    if (
        architecture.max_context_tokens is None
        or context_tokens > architecture.max_context_tokens
    ):
        raise ValueError("context_tokens exceeds the model manifest limit")
    if architecture.layer_count is None or architecture.attention is None:
        raise ValueError("model manifest does not define attention state")

    attention = architecture.attention
    per_token_layer_bytes = (
        2
        * batch_size
        * attention.kv_heads
        * attention.head_dim
        * bytes_per_element
    )
    total_layer_tokens = 0
    dense_layers = 0
    windowed_layers = 0
    for layer_index in range(architecture.layer_count):
        pattern = attention.schedule.pattern_for_layer(
            layer_index, architecture.layer_count
        )
        if pattern is AttentionPattern.DENSE:
            layer_tokens = context_tokens
            dense_layers += 1
        elif pattern in {
            AttentionPattern.SLIDING_WINDOW,
            AttentionPattern.BANDED,
        }:
            assert attention.schedule.window_tokens is not None
            layer_tokens = min(context_tokens, attention.schedule.window_tokens)
            windowed_layers += 1
        else:
            raise ValueError(
                f"attention pattern {pattern.value!r} requires a recurrent state schema"
            )
        total_layer_tokens += layer_tokens

    all_dense_bytes = (
        per_token_layer_bytes * architecture.layer_count * context_tokens
    )
    return KVCacheEstimate(
        batch_size=batch_size,
        context_tokens=context_tokens,
        bytes_per_element=bytes_per_element,
        total_bytes=per_token_layer_bytes * total_layer_tokens,
        all_dense_bytes=all_dense_bytes,
        dense_layer_count=dense_layers,
        windowed_layer_count=windowed_layers,
    )
