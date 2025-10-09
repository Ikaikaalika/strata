"""MLX backend skeleton."""
from __future__ import annotations

from typing import Any, Optional

from .base import Backend, BackendError
from .mlx_ops import require_mx, online_chunked_grouped_attention_rope_no_mask_mx

try:  # pragma: no cover - optional dependency
    import mlx.core as mx
except ModuleNotFoundError:  # pragma: no cover
    mx = None


class MLXBackend(Backend):
    name = "mlx"

    def is_available(self) -> bool:
        return mx is not None

    def resolve_device(self, device_request: Optional[str]) -> Any:
        if mx is None:
            raise BackendError("MLX backend is not available (mlx library not installed)")

        if device_request in (None, "", "auto"):
            return mx.default_device()

        # MLX devices are integers (GPU=0) or cpu references
        if device_request.lower() in {"gpu", "gpu:0", "0"}:
            return mx.gpu
        if device_request.lower() in {"cpu"}:
            return mx.cpu

        raise BackendError(f"Unsupported MLX device specification: {device_request}")

    def move_model_to_device(self, model: Any, device: Any) -> Any:
        raise BackendError("MLX backend does not yet support moving models; implementation pending")

    def create_kv_cache(self, cache_dir: str, stats: Optional[Any]):
        raise BackendError("MLX backend KV cache not implemented yet")

    def attention_kernel(self):
        require_mx()
        return online_chunked_grouped_attention_rope_no_mask_mx
