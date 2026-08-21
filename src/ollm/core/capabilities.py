"""Capabilities declared by a compute runtime adapter."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    supports_weight_paging: bool = False
    supports_expert_paging: bool = False
    supports_async_prefetch: bool = False
    supports_external_kv_cache: bool = False
    supports_quantized_weights: bool = False
    supports_graph_capture: bool = False
    supports_memory_counters: bool = False
