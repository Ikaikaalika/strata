"""Logical tensor references that do not depend on a compute runtime."""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Optional, Tuple


_DTYPE_BITS = {
    "bool": 8,
    "int8": 8,
    "uint8": 8,
    "int16": 16,
    "uint16": 16,
    "float16": 16,
    "bfloat16": 16,
    "int32": 32,
    "uint32": 32,
    "float32": 32,
    "int64": 64,
    "uint64": 64,
    "float64": 64,
    "int4": 4,
    "uint4": 4,
}


@dataclass(frozen=True)
class TensorRef:
    """A tensor's logical identity, storage identity, shape, and byte cost."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    storage_key: Optional[str] = None
    storage_nbytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be positive")
        if self.storage_nbytes is not None and self.storage_nbytes <= 0:
            raise ValueError("storage_nbytes must be positive")
        if self.storage_nbytes is None and self.normalized_dtype not in _DTYPE_BITS:
            raise ValueError(
                f"unknown dtype {self.dtype!r}; provide storage_nbytes explicitly"
            )

    @property
    def normalized_dtype(self) -> str:
        value = self.dtype.lower().replace("mlx.core.", "")
        return value.replace("torch.", "")

    @property
    def element_count(self) -> int:
        return reduce(mul, self.shape, 1)

    @property
    def nbytes(self) -> int:
        if self.storage_nbytes is not None:
            return self.storage_nbytes
        bits = self.element_count * _DTYPE_BITS[self.normalized_dtype]
        return (bits + 7) // 8

    @property
    def key(self) -> str:
        return self.storage_key or self.name
