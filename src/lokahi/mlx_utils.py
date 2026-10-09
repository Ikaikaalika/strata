"""
MLX utility functions.
"""

import time
from typing import Any, Optional

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


def tensor_size_gb(tensor: Any) -> float:
    """
    Calculate MLX tensor size in GB.

    Args:
        tensor: MLX array

    Returns:
        Size in gigabytes
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available")
    # MLX array size
    nbytes = tensor.nbytes if hasattr(tensor, 'nbytes') else tensor.size * tensor.itemsize
    return nbytes / (1024 ** 3)


def get_dtype_size(dtype: Any) -> int:
    """
    Get size in bytes for an MLX dtype.

    Args:
        dtype: MLX data type

    Returns:
        Size in bytes
    """
    # MLX dtype sizes
    dtype_map = {
        mx.float32: 4,
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.int32: 4,
        mx.int16: 2,
        mx.int8: 1,
    }
    return dtype_map.get(dtype, 4)


class Stats:
    """
    Performance statistics tracker.
    Backend-agnostic timing and logging.
    """

    def __init__(self):
        self.timings = {}
        self.counts = {}

    def set(self, name: str, start_time: float):
        """
        Record a timing.

        Args:
            name: Operation name
            start_time: Start time from time.perf_counter()
        """
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms

        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0

        self.timings[name].append(elapsed)
        self.counts[name] += 1

    def get_avg(self, name: str) -> float:
        """Get average time for an operation."""
        if name not in self.timings or not self.timings[name]:
            return 0.0
        return sum(self.timings[name]) / len(self.timings[name])

    def get_total(self, name: str) -> float:
        """Get total time for an operation."""
        if name not in self.timings:
            return 0.0
        return sum(self.timings[name])

    def print_and_clean(self) -> str:
        """
        Print statistics and clear.

        Returns:
            Formatted stats string
        """
        if not self.timings:
            return "No stats recorded"

        lines = []
        for name in sorted(self.timings.keys()):
            avg = self.get_avg(name)
            total = self.get_total(name)
            count = self.counts[name]
            lines.append(f"{name}: {avg:.2f}ms avg, {total:.2f}ms total, {count}x")

        result = " | ".join(lines)

        # Clear
        self.timings.clear()
        self.counts.clear()

        return result

    def clear(self):
        """Clear all statistics."""
        self.timings.clear()
        self.counts.clear()




def file_get_contents(path: str) -> str:
    """
    Read file contents.

    Args:
        path: File path

    Returns:
        File contents as string
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_hf_config(model_dir: str) -> Any:
    """
    Load HuggingFace model config.

    Args:
        model_dir: Model directory

    Returns:
        Config object
    """
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_dir)


def create_position_ids(
    seq_len: int,
    past_seen_tokens: int = 0
) -> Any:
    """
    Create position IDs tensor for MLX.

    Args:
        seq_len: Sequence length
        past_seen_tokens: Number of previously seen tokens

    Returns:
        Position IDs MLX array
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available")
    position_ids = mx.arange(
        past_seen_tokens,
        past_seen_tokens + seq_len,
        dtype=mx.int32
    )
    return mx.expand_dims(position_ids, axis=0)
