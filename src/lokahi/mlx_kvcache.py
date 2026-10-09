"""
MLX-compatible KV cache implementation with disk offloading support.
Mirrors the Torch KVCache API but uses MLX arrays and numpy serialization.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
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
        cache_dir: Optional[str] = None,
        stats: Optional[Any] = None
    ):
        """
        Initialize MLX KV cache.

        Args:
            config: Model config object (must have num_hidden_layers)
            cache_dir: Directory for disk cache. ``None`` creates a memory-only
                cache and does not touch the filesystem.
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
        self._layer_lengths: Dict[int, int] = {}
        self._managed_disk_files: set[str] = set()

        if cache_dir is not None:
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
        self._validate_layer_index(layer_idx)
        self._validate_state_shapes(key_states, value_states)

        # Load from disk if offloaded
        if layer_idx in self._disk_offloaded:
            self._load_layer_from_disk(layer_idx)

        previous_length = self._layer_lengths.get(layer_idx, 0)
        self._validate_cache_position(
            cache_kwargs,
            previous_length=previous_length,
            new_length=key_states.shape[2],
        )

        if layer_idx in self._cache:
            cached_key = self._cache[layer_idx]["key"]
            cached_value = self._cache[layer_idx]["value"]
            if cached_key.shape[:2] + cached_key.shape[3:] != key_states.shape[:2] + key_states.shape[3:]:
                raise ValueError("new key states must match cached batch, head, and feature dimensions")
            if cached_value.shape[:2] + cached_value.shape[3:] != value_states.shape[:2] + value_states.shape[3:]:
                raise ValueError("new value states must match cached batch, head, and feature dimensions")

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
        layer_length = self._cache[layer_idx]["key"].shape[2]
        self._layer_lengths[layer_idx] = layer_length
        self._seen_tokens = max(self._layer_lengths.values(), default=0)

        return self._cache[layer_idx]["key"], self._cache[layer_idx]["value"]

    def _validate_layer_index(self, layer_idx: int) -> None:
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must be a non-negative integer")
        layer_count = getattr(self.config, "num_hidden_layers", None)
        if layer_count is not None and layer_idx >= layer_count:
            raise ValueError(
                f"layer_idx {layer_idx} is outside configured layer count {layer_count}"
            )

    @staticmethod
    def _validate_state_shapes(key_states: mx.array, value_states: mx.array) -> None:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("key and value states must have rank 4")
        if key_states.shape != value_states.shape:
            raise ValueError("key and value states must have identical shapes")

    @staticmethod
    def _validate_cache_position(
        cache_kwargs: Optional[Dict[str, Any]],
        *,
        previous_length: int,
        new_length: int,
    ) -> None:
        if not cache_kwargs or cache_kwargs.get("cache_position") is None:
            return
        positions = np.asarray(cache_kwargs["cache_position"]).reshape(-1)
        expected = np.arange(previous_length, previous_length + new_length)
        if positions.shape != expected.shape or not np.array_equal(positions, expected):
            raise ValueError(
                "cache_position must append exactly after the cached sequence; "
                f"expected {expected.tolist()}, got {positions.tolist()}"
            )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Get the sequence length of the cache.

        Args:
            layer_idx: Layer index (default 0)

        Returns:
            Sequence length
        """
        if layer_idx is None:
            return max(self._layer_lengths.values(), default=0)
        return self._layer_lengths.get(layer_idx, 0)

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
            if self.cache_dir is None:
                raise ValueError("filename is required for a memory-only KV cache")
            filename = os.path.join(self.cache_dir, f"kvcache_layer_{layer_idx}.npz")

        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()

        # Convert MLX arrays to numpy
        key_np = np.array(self._cache[layer_idx]["key"])
        value_np = np.array(self._cache[layer_idx]["value"])

        # Save to disk
        np.savez_compressed(
            destination,
            key=key_np,
            value=value_np,
            seen_tokens=self._seen_tokens
        )

        if self.stats:
            self.stats.set("kv_save_to_disk", t0)

        # Track offloaded location and free memory
        self._disk_offloaded[layer_idx] = os.fspath(destination)
        self._managed_disk_files.add(os.fspath(destination))
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
        with np.load(filename) as data:
            key_np = data["key"]
            value_np = data["value"]
            if "seen_tokens" in data:
                self._seen_tokens = int(data["seen_tokens"])

        # Convert to MLX
        self._cache[layer_idx] = {
            "key": mx.array(key_np),
            "value": mx.array(value_np)
        }
        self._layer_lengths[layer_idx] = self._cache[layer_idx]["key"].shape[2]

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
        for filename in self._managed_disk_files:
            if os.path.exists(filename):
                os.remove(filename)

        self._disk_offloaded.clear()
        self._managed_disk_files.clear()
        self._layer_lengths.clear()
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
        if max_layers_in_memory <= 0:
            raise ValueError("max_layers_in_memory must be positive")
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
