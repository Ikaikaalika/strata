"""Small, runtime-neutral tensor and operation trace records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TraceEvent:
    """One observed tensor or timed operation."""

    kind: str
    name: str
    phase: str
    layer: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    byte_size: Optional[int] = None
    duration_ms: Optional[float] = None
    active_memory_bytes: Optional[int] = None
    peak_memory_bytes: Optional[int] = None
    group_id: Optional[str] = None
    cache_status: Optional[str] = None
    bytes_transferred: Optional[int] = None
    bandwidth_bytes_per_second: Optional[float] = None


class TensorTracer:
    """Collect deterministic trace metadata without owning runtime tensors.

    The tracer stores only scalar metadata, so tracing does not keep model
    activations alive. Runtime integrations may supply active and peak memory
    readings after synchronizing their device work.
    """

    def __init__(self) -> None:
        self.events: List[TraceEvent] = []
        self._lock = RLock()

    def record_tensor(
        self,
        name: str,
        tensor: Any,
        *,
        phase: str,
        layer: Optional[int] = None,
    ) -> TraceEvent:
        shape = tuple(int(dimension) for dimension in tensor.shape)
        byte_size = getattr(tensor, "nbytes", None)
        if byte_size is None:
            size = int(getattr(tensor, "size", 0))
            itemsize = int(getattr(tensor, "itemsize", 0))
            byte_size = size * itemsize
        event = TraceEvent(
            kind="tensor",
            name=name,
            phase=phase,
            layer=layer,
            shape=shape,
            dtype=str(tensor.dtype),
            byte_size=int(byte_size),
        )
        with self._lock:
            self.events.append(event)
        return event

    def record_operation(
        self,
        name: str,
        duration_ms: float,
        *,
        phase: str,
        layer: Optional[int] = None,
        active_memory_bytes: Optional[int] = None,
        peak_memory_bytes: Optional[int] = None,
        group_id: Optional[str] = None,
        cache_status: Optional[str] = None,
        bytes_transferred: Optional[int] = None,
        bandwidth_bytes_per_second: Optional[float] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            kind="operation",
            name=name,
            phase=phase,
            layer=layer,
            duration_ms=float(duration_ms),
            active_memory_bytes=(
                None if active_memory_bytes is None else int(active_memory_bytes)
            ),
            peak_memory_bytes=(
                None if peak_memory_bytes is None else int(peak_memory_bytes)
            ),
            group_id=group_id,
            cache_status=cache_status,
            bytes_transferred=(
                None if bytes_transferred is None else int(bytes_transferred)
            ),
            bandwidth_bytes_per_second=(
                None
                if bandwidth_bytes_per_second is None
                else float(bandwidth_bytes_per_second)
            ),
        )
        with self._lock:
            self.events.append(event)
        return event

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def to_dicts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(event) for event in self.events]
