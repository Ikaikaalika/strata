"""Versioned, aligned, runtime-neutral storage for immutable model weights.

The format is intentionally independent of NumPy, MLX, Metal, and ANE. Tensor
payloads are opaque bytes; shape, dtype, and quantization metadata belong to a
higher-level model manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


FORMAT_NAME = "strata-weight-pack"
FORMAT_VERSION = 1
DEFAULT_ALIGNMENT = 4096

_MAGIC = b"STRATAPK"
_HEADER_SIZE = 128
_HEADER_STRUCT = struct.Struct("<8sIIQQQ32s")
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MANIFEST_KEYS = {
    "format",
    "version",
    "alignment",
    "data_offset",
    "file_size",
    "tensors",
}
_TENSOR_KEYS = {"name", "offset", "length", "sha256"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class WeightPackError(RuntimeError):
    """Base error for weight-pack operations."""


class WeightPackFormatError(WeightPackError):
    """Raised when a pack's version or structural invariants are invalid."""


class WeightPackChecksumError(WeightPackError):
    """Raised when immutable payload bytes do not match their manifest."""


@dataclass(frozen=True)
class WeightPackTensor:
    """One opaque tensor payload in a weight pack."""

    name: str
    offset: int
    length: int
    sha256: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WeightPackTensor":
        _require_exact_keys(payload, _TENSOR_KEYS, "tensor record")
        name = payload["name"]
        offset = payload["offset"]
        length = payload["length"]
        checksum = payload["sha256"]
        if not isinstance(name, str) or not name:
            raise WeightPackFormatError("tensor name must be a non-empty string")
        if not _is_integer(offset) or offset < 0:
            raise WeightPackFormatError(f"tensor {name!r} offset must be non-negative")
        if not _is_integer(length) or length <= 0:
            raise WeightPackFormatError(f"tensor {name!r} length must be positive")
        if not isinstance(checksum, str) or not _SHA256_PATTERN.fullmatch(checksum):
            raise WeightPackFormatError(
                f"tensor {name!r} sha256 must be 64 lowercase hexadecimal characters"
            )
        return cls(name=name, offset=offset, length=length, sha256=checksum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offset": self.offset,
            "length": self.length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class WeightPackManifest:
    """Strict version-one manifest and its canonical aligned layout."""

    alignment: int
    data_offset: int
    file_size: int
    tensors: tuple[WeightPackTensor, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WeightPackManifest":
        _require_exact_keys(payload, _MANIFEST_KEYS, "weight-pack manifest")
        if payload["format"] != FORMAT_NAME:
            raise WeightPackFormatError("unsupported weight-pack format name")
        if not _is_integer(payload["version"]) or payload["version"] != FORMAT_VERSION:
            raise WeightPackFormatError(
                f"unsupported weight-pack version {payload['version']!r}"
            )

        alignment = payload["alignment"]
        data_offset = payload["data_offset"]
        file_size = payload["file_size"]
        tensor_payloads = payload["tensors"]
        if not _is_integer(alignment) or not _valid_alignment(alignment):
            raise WeightPackFormatError(
                "alignment must be a power of two between 64 and 1048576 bytes"
            )
        if not _is_integer(data_offset) or data_offset < _HEADER_SIZE:
            raise WeightPackFormatError("data_offset is before the fixed header")
        if data_offset % alignment:
            raise WeightPackFormatError("data_offset is not aligned")
        if not _is_integer(file_size) or file_size <= data_offset:
            raise WeightPackFormatError("file_size must extend beyond data_offset")
        if not isinstance(tensor_payloads, list) or not tensor_payloads:
            raise WeightPackFormatError("manifest must contain at least one tensor")

        tensors = tuple(
            WeightPackTensor.from_dict(item)
            if isinstance(item, Mapping)
            else _raise_tensor_mapping_error()
            for item in tensor_payloads
        )
        manifest = cls(
            alignment=alignment,
            data_offset=data_offset,
            file_size=file_size,
            tensors=tensors,
        )
        manifest._validate_layout()
        return manifest

    def _validate_layout(self) -> None:
        names: set[str] = set()
        previous_end = self.data_offset
        previous_name: str | None = None
        for tensor in self.tensors:
            if tensor.name in names:
                raise WeightPackFormatError(f"duplicate tensor name {tensor.name!r}")
            names.add(tensor.name)
            if tensor.offset < previous_end:
                raise WeightPackFormatError(
                    f"tensor {tensor.name!r} overlaps tensor {previous_name!r}"
                )
            expected_offset = _align_up(previous_end, self.alignment)
            if tensor.offset != expected_offset:
                raise WeightPackFormatError(
                    f"tensor {tensor.name!r} has non-canonical offset {tensor.offset}; "
                    f"expected {expected_offset}"
                )
            tensor_end = tensor.offset + tensor.length
            if tensor_end > self.file_size:
                raise WeightPackFormatError(
                    f"tensor {tensor.name!r} extends beyond declared file bounds"
                )
            previous_name = tensor.name
            previous_end = tensor_end
        if previous_end != self.file_size:
            raise WeightPackFormatError(
                "declared file_size must equal the end of the final tensor"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "alignment": self.alignment,
            "data_offset": self.data_offset,
            "file_size": self.file_size,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    def tensor(self, name: str) -> WeightPackTensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)


class MappedTensor:
    """Context-managed read-only view into a mapped pack payload."""

    def __init__(self, file_handle: BinaryIO, mapping: mmap.mmap, tensor: WeightPackTensor):
        self._file_handle = file_handle
        self._mapping = mapping
        self._view: memoryview | None = memoryview(mapping)[
            tensor.offset : tensor.offset + tensor.length
        ]
        self.tensor = tensor

    def __enter__(self) -> memoryview:
        return self.view

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._view is not None:
            self._view.release()
            self._view = None
        try:
            self._mapping.close()
        finally:
            self._file_handle.close()

    @property
    def view(self) -> memoryview:
        if self._view is None:
            raise WeightPackError("mapped tensor is already closed")
        return self._view


class WeightPack:
    """Validated reader for an immutable Strata weight pack."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.manifest = _read_manifest(self.path)
        self._tensors = {tensor.name: tensor for tensor in self.manifest.tensors}

    def tensor_info(self, name: str) -> WeightPackTensor:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"weight pack has no tensor {name!r}") from exc

    def read_tensor(self, name: str, *, verify_checksum: bool = True) -> bytes:
        """Read exactly one tensor range without reading the whole pack."""
        tensor = self.tensor_info(name)
        data = _read_exact_range(self.path, tensor.offset, tensor.length)
        if verify_checksum:
            _verify_checksum(tensor, data)
        return data

    def mmap_tensor(self, name: str, *, verify_checksum: bool = True) -> MappedTensor:
        """Map the pack read-only and return a context-managed tensor slice."""
        tensor = self.tensor_info(name)
        file_handle = self.path.open("rb")
        try:
            mapping = mmap.mmap(file_handle.fileno(), length=0, access=mmap.ACCESS_READ)
            mapped = MappedTensor(file_handle, mapping, tensor)
            try:
                if verify_checksum:
                    _verify_checksum(tensor, mapped.view)
            except Exception:
                mapped.close()
                raise
            return mapped
        except Exception:
            file_handle.close()
            raise

    def verify_all(self) -> None:
        for tensor in self.manifest.tensors:
            self.read_tensor(tensor.name, verify_checksum=True)


def write_weight_pack(
    path: str | os.PathLike[str],
    tensors: Iterable[tuple[str, bytes | bytearray | memoryview]],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> WeightPackManifest:
    """Atomically create or replace a pack from ordered opaque payloads.

    The temporary file is created beside the destination, flushed, and fsynced
    before ``os.replace``. If preparation or writing fails, an existing target
    remains unchanged and the temporary file is removed.
    """
    if not _valid_alignment(alignment):
        raise ValueError("alignment must be a power of two from 64 to 1048576")
    payloads = _normalize_payloads(tensors)
    manifest, manifest_bytes = _build_manifest(payloads, alignment)
    header = _build_header(manifest, manifest_bytes)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as output:
            output.write(header)
            output.write(manifest_bytes)
            _write_zeros(output, manifest.data_offset - output.tell())
            for tensor, (_, payload) in zip(manifest.tensors, payloads):
                _write_zeros(output, tensor.offset - output.tell())
                output.write(payload)
            if output.tell() != manifest.file_size:
                raise WeightPackError("internal error: emitted pack length is inconsistent")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        _fsync_directory(target.parent)
        return manifest
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _normalize_payloads(
    tensors: Iterable[tuple[str, bytes | bytearray | memoryview]],
) -> tuple[tuple[str, bytes], ...]:
    normalized: list[tuple[str, bytes]] = []
    names: set[str] = set()
    for item in tensors:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("each tensor must be a (name, bytes-like payload) tuple")
        name, payload = item
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be non-empty strings")
        if name in names:
            raise ValueError(f"duplicate tensor name {name!r}")
        names.add(name)
        try:
            payload_bytes = memoryview(payload).tobytes()
        except (TypeError, ValueError) as exc:
            raise TypeError(f"tensor {name!r} payload must be bytes-like") from exc
        if not payload_bytes:
            raise ValueError(f"tensor {name!r} payload must not be empty")
        normalized.append((name, payload_bytes))
    if not normalized:
        raise ValueError("at least one tensor payload is required")
    return tuple(normalized)


def _build_manifest(
    payloads: Sequence[tuple[str, bytes]], alignment: int
) -> tuple[WeightPackManifest, bytes]:
    checksums = [hashlib.sha256(payload).hexdigest() for _, payload in payloads]
    data_offset = _align_up(_HEADER_SIZE, alignment)
    for _ in range(16):
        tensors: list[WeightPackTensor] = []
        current_offset = data_offset
        for (name, payload), checksum in zip(payloads, checksums):
            current_offset = _align_up(current_offset, alignment)
            tensors.append(
                WeightPackTensor(
                    name=name,
                    offset=current_offset,
                    length=len(payload),
                    sha256=checksum,
                )
            )
            current_offset += len(payload)
        manifest = WeightPackManifest(
            alignment=alignment,
            data_offset=data_offset,
            file_size=current_offset,
            tensors=tuple(tensors),
        )
        manifest._validate_layout()
        manifest_bytes = _canonical_json(manifest.to_dict())
        next_data_offset = _align_up(_HEADER_SIZE + len(manifest_bytes), alignment)
        if next_data_offset == data_offset:
            return manifest, manifest_bytes
        data_offset = next_data_offset
    raise WeightPackError("weight-pack manifest layout did not converge")


def _build_header(manifest: WeightPackManifest, manifest_bytes: bytes) -> bytes:
    packed = _HEADER_STRUCT.pack(
        _MAGIC,
        FORMAT_VERSION,
        _HEADER_SIZE,
        len(manifest_bytes),
        manifest.data_offset,
        manifest.file_size,
        hashlib.sha256(manifest_bytes).digest(),
    )
    return packed + bytes(_HEADER_SIZE - len(packed))


def _read_manifest(path: Path) -> WeightPackManifest:
    try:
        actual_size = path.stat().st_size
        with path.open("rb") as source:
            header = source.read(_HEADER_SIZE)
            if len(header) != _HEADER_SIZE:
                raise WeightPackFormatError("weight-pack header is truncated")
            values = _HEADER_STRUCT.unpack(header[: _HEADER_STRUCT.size])
            magic, version, header_size, manifest_length, data_offset, file_size, digest = values
            if magic != _MAGIC:
                raise WeightPackFormatError("weight-pack magic is invalid")
            if version != FORMAT_VERSION:
                raise WeightPackFormatError(f"unsupported header version {version}")
            if header_size != _HEADER_SIZE:
                raise WeightPackFormatError("unsupported fixed header size")
            if any(header[_HEADER_STRUCT.size :]):
                raise WeightPackFormatError("weight-pack reserved header bytes are nonzero")
            if not 0 < manifest_length <= _MAX_MANIFEST_BYTES:
                raise WeightPackFormatError("manifest length is invalid")
            if _HEADER_SIZE + manifest_length > data_offset:
                raise WeightPackFormatError("manifest extends into tensor payload area")
            if file_size != actual_size:
                raise WeightPackFormatError(
                    "declared file size does not match the physical file"
                )
            manifest_bytes = source.read(manifest_length)
            if len(manifest_bytes) != manifest_length:
                raise WeightPackFormatError("weight-pack manifest is truncated")
    except OSError as exc:
        raise WeightPackError(f"failed to read weight pack {path}: {exc}") from exc

    if hashlib.sha256(manifest_bytes).digest() != digest:
        raise WeightPackChecksumError("weight-pack manifest checksum mismatch")
    payload = _strict_json_loads(manifest_bytes)
    if not isinstance(payload, Mapping):
        raise WeightPackFormatError("weight-pack manifest root must be an object")
    if _canonical_json(payload) != manifest_bytes:
        raise WeightPackFormatError("weight-pack manifest JSON is not canonical")
    manifest = WeightPackManifest.from_dict(payload)
    if manifest.data_offset != data_offset or manifest.file_size != file_size:
        raise WeightPackFormatError("header and manifest layout fields disagree")
    return manifest


def _strict_json_loads(data: bytes) -> Any:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise WeightPackFormatError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightPackFormatError(f"manifest JSON is invalid: {exc}") from exc


def _read_exact_range(path: Path, offset: int, length: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            if hasattr(os, "pread"):
                chunks: list[bytes] = []
                remaining = length
                position = offset
                while remaining:
                    chunk = os.pread(descriptor, remaining, position)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    position += len(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
            else:  # pragma: no cover - platform compatibility path
                os.close(descriptor)
                descriptor = -1
                with path.open("rb") as source:
                    source.seek(offset)
                    data = source.read(length)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except OSError as exc:
        raise WeightPackError(f"failed to read tensor range: {exc}") from exc
    if len(data) != length:
        raise WeightPackFormatError(
            f"tensor range is truncated: expected {length} bytes, read {len(data)}"
        )
    return data


def _verify_checksum(tensor: WeightPackTensor, data: bytes | memoryview) -> None:
    checksum = hashlib.sha256(data).hexdigest()
    if checksum != tensor.sha256:
        raise WeightPackChecksumError(
            f"tensor {tensor.name!r} checksum mismatch"
        )


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _write_zeros(output: BinaryIO, count: int) -> None:
    if count < 0:
        raise WeightPackError("internal error: attempted to write negative padding")
    block = bytes(min(64 * 1024, count))
    remaining = count
    while remaining:
        chunk = min(remaining, len(block))
        output.write(block[:chunk])
        remaining -= chunk


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some platforms/filesystems do not permit directory fsync. The file was
        # still fully fsynced before the atomic replacement.
        pass


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], description: str
) -> None:
    actual = set(payload.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise WeightPackFormatError(
            f"{description} keys are invalid; missing={missing}, unknown={unknown}"
        )


def _valid_alignment(value: Any) -> bool:
    return (
        _is_integer(value)
        and 64 <= value <= 1024 * 1024
        and value & (value - 1) == 0
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _raise_tensor_mapping_error():
    raise WeightPackFormatError("each tensor record must be an object")
