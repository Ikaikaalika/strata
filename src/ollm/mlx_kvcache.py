"""
MLX-compatible KV cache implementation with disk offloading support.
Mirrors the Torch KVCache API but uses MLX arrays and numpy serialization.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class MLXKVCache:
    """
    MLX-compatible KV cache with disk offloading for long-context inference.

    Similar to transformers.Cache but:
    - Uses MLX arrays instead of Torch tensors
    - Supports disk offloading for memory efficiency
    - Uses .npz format instead of .pt pickles
    """

    def __init__(
        self,
        config: Any,
        cache_dir: str = "./kvcache",
        stats: Optional[Any] = None
    ):
        """
        Initialize MLX KV cache.

        Args:
            config: Model config object (must have num_hidden_layers)
            cache_dir: Directory for disk cache
            stats: Optional stats tracker
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MLXKVCache")

        self.config = config
        self.cache_dir = cache_dir
        self.stats = stats

        # Cache storage: layer_idx -> {"key": mx.array, "value": mx.array}
        self._cache: Dict[int, Dict[str, mx.array]] = {}
        self._disk_offloaded: Dict[int, str] = {}  # layer_idx -> filename

        os.makedirs(cache_dir, exist_ok=True)

        # Track sequence length
        self._seen_tokens = 0

    def update(
        self,
        key_states: mx.array,
        value_states: mx.array,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update the cache with new key/value states.

        Args:
            key_states: New key states [batch, num_heads, seq_len, head_dim]
            value_states: New value states [batch, num_heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_kwargs: Additional arguments (sin, cos, cache_position, etc.)

        Returns:
            Tuple of (updated_keys, updated_values)
        """
        # Load from disk if offloaded
        if layer_idx in self._disk_offloaded:
            self._load_layer_from_disk(layer_idx)

        # Initialize layer cache if needed
        if layer_idx not in self._cache:
            self._cache[layer_idx] = {"key": key_states, "value": value_states}
        else:
            # Concatenate with existing cache
            self._cache[layer_idx]["key"] = mx.concatenate(
                [self._cache[layer_idx]["key"], key_states],
                axis=2  # Concatenate along sequence dimension
            )
            self._cache[layer_idx]["value"] = mx.concatenate(
                [self._cache[layer_idx]["value"], value_states],
                axis=2
            )

        # Update sequence length
        self._seen_tokens = self._cache[layer_idx]["key"].shape[2]

        return self._cache[layer_idx]["key"], self._cache[layer_idx]["value"]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Get the sequence length of the cache.

        Args:
            layer_idx: Layer index (default 0)

        Returns:
            Sequence length
        """
        if layer_idx in self._cache:
            return self._cache[layer_idx]["key"].shape[2]
        elif layer_idx in self._disk_offloaded:
            # Need to check disk
            return self._seen_tokens
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        """
        Get maximum cache shape (for compatibility with transformers API).

        Returns:
            None (dynamic cache has no max)
        """
        return None

    def save_to_disk(self, layer_idx: int, filename: Optional[str] = None):
        """
        Save a layer's cache to disk and free memory.

        Args:
            layer_idx: Layer index to save
            filename: Optional custom filename
        """
        if layer_idx not in self._cache:
            return

        if filename is None:
            filename = os.path.join(
                self.cache_dir,
                f"kvcache_layer_{layer_idx}.npz"
            )

        t0 = time.perf_counter()

        # Convert MLX arrays to numpy
        key_np = np.array(self._cache[layer_idx]["key"])
        value_np = np.array(self._cache[layer_idx]["value"])

        # Save to disk
        np.savez_compressed(
            filename,
            key=key_np,
            value=value_np,
            seen_tokens=self._seen_tokens
        )

        if self.stats:
            self.stats.set("kv_save_to_disk", t0)

        # Track offloaded location and free memory
        self._disk_offloaded[layer_idx] = filename
        del self._cache[layer_idx]

    def _load_layer_from_disk(self, layer_idx: int):
        """
        Load a layer's cache from disk.

        Args:
            layer_idx: Layer index to load
        """
        if layer_idx not in self._disk_offloaded:
            return

        filename = self._disk_offloaded[layer_idx]
        t0 = time.perf_counter()

        # Load from disk
        data = np.load(filename)
        key_np = data["key"]
        value_np = data["value"]
        if "seen_tokens" in data:
            self._seen_tokens = int(data["seen_tokens"])

        # Convert to MLX
        self._cache[layer_idx] = {
            "key": mx.array(key_np),
            "value": mx.array(value_np)
        }

        if self.stats:
            self.stats.set("kv_load_from_disk", t0)

        # Remove from offloaded tracking
        del self._disk_offloaded[layer_idx]

    def offload_layers_to_disk(self, layer_indices: Optional[List[int]] = None):
        """
        Offload specified layers to disk.

        Args:
            layer_indices: List of layer indices. If None, offload all.
        """
        if layer_indices is None:
            layer_indices = list(self._cache.keys())

        for idx in layer_indices:
            if idx in self._cache:
                self.save_to_disk(idx)

    def clear(self):
        """Clear all cache (memory and disk)."""
        # Clear memory
        self._cache.clear()

        # Delete disk files
        for filename in self._disk_offloaded.values():
            if os.path.exists(filename):
                os.remove(filename)

        self._disk_offloaded.clear()
        self._seen_tokens = 0

    def __len__(self) -> int:
        """Return number of cached layers."""
        return len(self._cache) + len(self._disk_offloaded)

    def __getitem__(self, layer_idx: int) -> Dict[str, mx.array]:
        """
        Get cache for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Dictionary with "key" and "value" arrays
        """
        if layer_idx in self._disk_offloaded:
            self._load_layer_from_disk(layer_idx)

        return self._cache.get(layer_idx, {"key": None, "value": None})

    def __contains__(self, layer_idx: int) -> bool:
        """Check if a layer is cached."""
        return layer_idx in self._cache or layer_idx in self._disk_offloaded


class MLXLlamaDiskCache(MLXKVCache):
    """
    Specialized disk cache for Llama models using MLX.
    Automatically offloads older layers to disk during generation.
    """

    def __init__(
        self,
        config: Any,
        cache_dir: str = "./kvcache",
        stats: Optional[Any] = None,
        max_layers_in_memory: int = 4
    ):
        """
        Initialize Llama disk cache.

        Args:
            config: Model config
            cache_dir: Directory for cache files
            stats: Stats tracker
            max_layers_in_memory: Maximum number of layers to keep in memory
        """
        super().__init__(config, cache_dir, stats)
        self.max_layers_in_memory = max_layers_in_memory

    def update(
        self,
        key_states: mx.array,
        value_states: mx.array,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache with automatic disk offloading.

        Args:
            key_states: New key states
            value_states: New value states
            layer_idx: Layer index
            cache_kwargs: Additional arguments

        Returns:
            Tuple of (updated_keys, updated_values)
        """
        # Update cache normally
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Auto-offload if too many layers in memory
        if len(self._cache) > self.max_layers_in_memory:
            # Offload oldest layers (lowest indices)
            layers_to_offload = sorted(self._cache.keys())[:-self.max_layers_in_memory]
            for idx in layers_to_offload:
                if idx != layer_idx:  # Don't offload the layer we just updated
                    self.save_to_disk(idx)

        return result


class MLXQwen3NextDiskCache(MLXKVCache):
    """
    Specialized disk cache for Qwen3-Next models.
    Handles the specific caching requirements for Qwen3-Next architecture.
    """

    def __init__(
        self,
        config: Any,
        cache_dir: str = "./kvcache",
        stats: Optional[Any] = None
    ):
        """
        Initialize Qwen3-Next cache.

        Args:
            config: Model config
            cache_dir: Cache directory
            stats: Stats tracker
        """
        super().__init__(config, cache_dir, stats)

        # Qwen3-Next specific configuration
        if hasattr(config, 'num_hidden_layers'):
            self.num_layers = config.num_hidden_layers
        else:
            self.num_layers = 80  # Default for Qwen3-Next-80B


def create_mlx_kv_cache(
    model_id: str,
    config: Any,
    cache_dir: str = "./kvcache",
    stats: Optional[Any] = None
) -> MLXKVCache:
    """
    Factory function to create the appropriate MLX KV cache for a model.

    Args:
        model_id: Model identifier (e.g., "llama3-1B-chat")
        config: Model config
        cache_dir: Cache directory
        stats: Stats tracker

    Returns:
        Appropriate MLX KV cache instance
    """
    if model_id.startswith("llama"):
        return MLXLlamaDiskCache(config, cache_dir, stats)
    elif model_id.startswith("qwen"):
        return MLXQwen3NextDiskCache(config, cache_dir, stats)
    elif model_id.startswith("gemma"):
        return MLXKVCache(config, cache_dir, stats)
    else:
        # Default to base class
        return MLXKVCache(config, cache_dir, stats)
