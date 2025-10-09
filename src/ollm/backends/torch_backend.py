"""Torch backend implementation."""
from __future__ import annotations

from typing import Any, Optional

import torch

from .base import Backend


class TorchBackend(Backend):
    name = "torch"

    def resolve_device(self, device_request: Optional[str]) -> torch.device:
        if device_request in (None, "", "auto"):
            return self._auto_device()
        return torch.device(device_request)

    def move_model_to_device(self, model: Any, device: torch.device) -> Any:
        return model.to(device)

    def create_kv_cache(self, cache_dir: str, stats: Optional[Any]):
        from ..kvcache import KVCache

        return KVCache(cache_dir=cache_dir, stats=stats)

    def _auto_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def attention_kernel(self):
        from ..attention import online_chunked_grouped_attention_rope_no_mask

        return online_chunked_grouped_attention_rope_no_mask
