"""MLX backend implementation for Apple Silicon."""
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
        """
        Move model to MLX device.

        For MLX, models are already on the appropriate device after loading.
        This method is mainly for compatibility with the backend API.
        MLX uses unified memory, so explicit device movement is less critical.

        Args:
            model: Model instance
            device: MLX device (mx.gpu or mx.cpu)

        Returns:
            The model (unchanged, as MLX handles device placement automatically)
        """
        if mx is None:
            raise BackendError("MLX backend is not available")

        # MLX models don't need explicit device movement due to unified memory
        # The device is set during weight loading
        # This is a no-op that maintains API compatibility
        return model

    def create_kv_cache(self, cache_dir: str, stats: Optional[Any], config: Optional[Any] = None, model_id: Optional[str] = None):
        """
        Create MLX-compatible KV cache with disk offloading.

        Args:
            cache_dir: Directory for cache files
            stats: Stats tracker
            config: Model config (required)
            model_id: Model identifier for specialized cache selection

        Returns:
            MLX KV cache instance
        """
        if mx is None:
            raise BackendError("MLX backend is not available")

        from ..mlx_kvcache import create_mlx_kv_cache

        if config is None:
            raise BackendError("config is required to create MLX KV cache")

        return create_mlx_kv_cache(
            model_id=model_id or "default",
            config=config,
            cache_dir=cache_dir,
            stats=stats
        )

    def attention_kernel(self):
        require_mx()
        return online_chunked_grouped_attention_rope_no_mask_mx
