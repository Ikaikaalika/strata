"""Minimal NumPy safetensors reader and writer.

The format is an 8-byte little-endian header length, a JSON header mapping
tensor names to ``dtype``, ``shape`` and ``data_offsets``, then the raw
little-endian tensor bytes. BF16 tensors are returned as ``uint16`` bit
patterns; use :func:`bf16_to_f32` to widen them.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Mapping

import numpy as np

_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}
_MAX_HEADER_BYTES = 100 * 1024 * 1024


class SafetensorsError(ValueError):
    """Raised for malformed safetensors files."""


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Widen BF16 bit patterns stored as ``uint16`` to ``float32``."""

    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round ``float32`` values to BF16 (nearest even) as ``uint16`` bits."""

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    rounded = ((bits + rounding) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(values, dtype=np.float32))
    return np.where(nan, np.uint16(0x7FC0), rounded)


def read_safetensors(path: Path) -> tuple[dict[str, tuple[str, np.ndarray]], dict[str, str]]:
    """Return ``{name: (dtype, array)}`` and the ``__metadata__`` map."""

    data = Path(path).read_bytes()
    if len(data) < 8:
        raise SafetensorsError(f"{path}: file too small")
    (header_len,) = struct.unpack("<Q", data[:8])
    if header_len > _MAX_HEADER_BYTES or 8 + header_len > len(data):
        raise SafetensorsError(f"{path}: invalid header length")
    header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    metadata = header.pop("__metadata__", {}) or {}
    payload = memoryview(data)[8 + header_len :]
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    for name, entry in header.items():
        dtype_name = entry["dtype"]
        if dtype_name not in _DTYPES:
            raise SafetensorsError(f"{path}: unsupported dtype {dtype_name} for {name}")
        dtype = _DTYPES[dtype_name]
        begin, end = entry["data_offsets"]
        shape = tuple(int(dim) for dim in entry["shape"])
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if end - begin != count * dtype.itemsize or end > len(payload) or begin > end:
            raise SafetensorsError(f"{path}: bad data offsets for {name}")
        array = np.frombuffer(payload[begin:end], dtype=dtype).reshape(shape).copy()
        tensors[name] = (dtype_name, array)
    return tensors, dict(metadata)


def write_safetensors(
    path: Path,
    tensors: Mapping[str, tuple[str, np.ndarray]],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write ``{name: (dtype, array)}``.

    The format forbids holes in the data buffer, so tensors are written in
    descending element-size order; every tensor then starts at an offset that
    is a multiple of its own element size.
    """

    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    for name, (dtype_name, _) in tensors.items():
        if dtype_name not in _DTYPES:
            raise SafetensorsError(f"unsupported dtype {dtype_name} for {name}")
    order = sorted(tensors, key=lambda name: (-_DTYPES[tensors[name][0]].itemsize, name))
    chunks: list[bytes] = []
    offset = 0
    for name in order:
        dtype_name, array = tensors[name]
        raw = np.ascontiguousarray(array, dtype=_DTYPES[dtype_name]).tobytes()
        header[name] = {
            "dtype": dtype_name,
            "shape": list(np.shape(array)),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for chunk in chunks:
            handle.write(chunk)
