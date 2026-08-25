"""Weight-pack adapter for Strata's group prefetch and residency pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import os
from threading import RLock
from typing import Any, Callable, Mapping

from ..core.model_spec import WeightGroup
from ..core.tensor_ref import TensorRef
from .weight_pack import WeightPack


WeightDecoder = Callable[[TensorRef, bytes], Any]


class WeightPackStoreError(RuntimeError):
    """Raised when a model reference cannot be satisfied by a weight pack."""


@dataclass(frozen=True)
class ImmutableWeightBytes:
    """Default runtime-neutral decoded value with measurable byte residency."""

    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("ImmutableWeightBytes data must be bytes")

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def view(self) -> memoryview:
        """Return a read-only zero-copy view over the immutable payload."""
        return memoryview(self.data)

    def __bytes__(self) -> bytes:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]


def immutable_bytes_decoder(
    tensor_ref: TensorRef, payload: bytes
) -> ImmutableWeightBytes:
    """Default decoder; tensor metadata remains available to custom decoders."""
    del tensor_ref
    return ImmutableWeightBytes(payload)


class WeightPackGroupTensorStore:
    """Load checksum-verified tensor groups from one immutable weight pack.

    Reads use :meth:`WeightPack.read_tensor`, so each tensor is an exact-range
    read whose SHA-256 is checked before the decoder sees it. No mmap is retained
    by the store or returned runtime values.
    """

    def __init__(
        self,
        pack: WeightPack | str | os.PathLike[str],
        decoder: WeightDecoder | None = None,
    ) -> None:
        if isinstance(pack, WeightPack):
            self.pack = pack
        elif isinstance(pack, (str, os.PathLike)):
            self.pack = WeightPack(pack)
        else:
            raise TypeError("pack must be a WeightPack or filesystem path")
        if decoder is not None and not callable(decoder):
            raise TypeError("decoder must be callable")
        self.decoder = decoder or immutable_bytes_decoder
        self._closed = False
        self._lock = RLock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def load_group(self, group: WeightGroup) -> Mapping[str, Any]:
        """Read, verify, size-check, and decode an atomic weight group."""
        if not isinstance(group, WeightGroup):
            raise TypeError("group must be a WeightGroup")
        storage_keys = [tensor_ref.key for tensor_ref in group.tensors]
        seen_keys: set[str] = set()
        duplicate_keys: set[str] = set()
        for storage_key in storage_keys:
            if storage_key in seen_keys:
                duplicate_keys.add(storage_key)
            seen_keys.add(storage_key)
        if duplicate_keys:
            raise WeightPackStoreError(
                f"group {group.group_id!r} contains duplicate storage keys: "
                f"{sorted(duplicate_keys)}"
            )

        with self._lock:
            if self._closed:
                raise WeightPackStoreError("weight-pack store is closed")

            # Check every reference before the first read, then verify every
            # payload before the first decoder call. Pack incompatibility or
            # corruption therefore cannot partially decode a group.
            records = []
            for tensor_ref in group.tensors:
                try:
                    record = self.pack.tensor_info(tensor_ref.key)
                except KeyError as exc:
                    raise WeightPackStoreError(
                        f"group {group.group_id!r} references missing packed tensor "
                        f"{tensor_ref.key!r}"
                    ) from exc
                if record.length != tensor_ref.nbytes:
                    raise WeightPackStoreError(
                        f"packed tensor {tensor_ref.key!r} length {record.length} "
                        f"does not match TensorRef.nbytes {tensor_ref.nbytes}"
                    )
                records.append((tensor_ref, record.name))

            verified_payloads = [
                (
                    tensor_ref,
                    self.pack.read_tensor(storage_key, verify_checksum=True),
                )
                for tensor_ref, storage_key in records
            ]
            decoded: dict[str, Any] = {}
            for tensor_ref, payload in verified_payloads:
                decoded[tensor_ref.name] = self.decoder(tensor_ref, payload)
            return decoded

    def close(self) -> None:
        """Reject future loads after any in-progress load has completed."""
        with self._lock:
            self._closed = True

    def __enter__(self) -> "WeightPackGroupTensorStore":
        with self._lock:
            if self._closed:
                raise WeightPackStoreError("weight-pack store is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
