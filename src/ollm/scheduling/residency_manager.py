"""Thread-safe, byte-budgeted LRU residency state."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple


class BudgetExceededError(RuntimeError):
    """Raised when a weight group cannot fit inside the residency budget."""


class ResidencyState(str, Enum):
    LOADING = "loading"
    RESIDENT = "resident"


@dataclass
class _Entry:
    key: str
    byte_size: int
    state: ResidencyState
    last_used: int
    value: Any = None
    pin_count: int = 0


@dataclass(frozen=True)
class ResidencySnapshot:
    budget_bytes: int
    used_bytes: int
    loading_bytes: int
    resident_bytes: int
    resident_keys: Tuple[str, ...]
    loading_keys: Tuple[str, ...]
    pinned_keys: Tuple[str, ...]


class ResidencyManager:
    """Reserve, pin, and evict atomic weight groups under a hard byte budget."""

    def __init__(
        self,
        budget_bytes: int,
        *,
        on_evict: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.budget_bytes = int(budget_bytes)
        self._on_evict = on_evict
        self._entries: Dict[str, _Entry] = {}
        self._tick = 0
        self._lock = RLock()

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    def _used_bytes(self) -> int:
        return sum(entry.byte_size for entry in self._entries.values())

    def _evict_for(
        self,
        byte_size: int,
        *,
        exclude: Optional[str] = None,
    ) -> List[Tuple[str, Any]]:
        evicted: List[Tuple[str, Any]] = []
        while self._used_bytes() + byte_size > self.budget_bytes:
            candidates = [
                entry
                for entry in self._entries.values()
                if entry.key != exclude
                and entry.state is ResidencyState.RESIDENT
                and entry.pin_count == 0
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda entry: entry.last_used)
            del self._entries[victim.key]
            evicted.append((victim.key, victim.value))
        return evicted

    def _notify_evictions(self, evicted: List[Tuple[str, Any]]) -> None:
        if self._on_evict is None:
            return
        for key, value in evicted:
            self._on_evict(key, value)

    def reserve(self, key: str, byte_size: int) -> bool:
        """Reserve bytes for an asynchronous or synchronous group load."""
        if not key:
            raise ValueError("residency key must not be empty")
        if byte_size <= 0:
            raise ValueError("byte_size must be positive")
        if byte_size > self.budget_bytes:
            return False

        with self._lock:
            if key in self._entries:
                return True
            evicted = self._evict_for(byte_size)
            if self._used_bytes() + byte_size > self.budget_bytes:
                reserved = False
            else:
                self._entries[key] = _Entry(
                    key=key,
                    byte_size=int(byte_size),
                    state=ResidencyState.LOADING,
                    last_used=self._next_tick(),
                )
                reserved = True

        self._notify_evictions(evicted)
        return reserved

    def commit(self, key: str, value: Any, *, byte_size: Optional[int] = None) -> None:
        """Turn a loading reservation into a resident value."""
        with self._lock:
            if key not in self._entries:
                raise KeyError(f"no reservation exists for {key!r}")
            entry = self._entries[key]
            if entry.state is not ResidencyState.LOADING:
                raise RuntimeError(f"residency key {key!r} is already committed")

            actual_size = entry.byte_size if byte_size is None else int(byte_size)
            if actual_size <= 0:
                raise ValueError("committed byte_size must be positive")
            delta = actual_size - entry.byte_size
            evicted = self._evict_for(max(delta, 0), exclude=key)
            if self._used_bytes() + delta > self.budget_bytes:
                del self._entries[key]
                failed = True
            else:
                entry.byte_size = actual_size
                entry.value = value
                entry.state = ResidencyState.RESIDENT
                entry.last_used = self._next_tick()
                failed = False

        self._notify_evictions(evicted)
        if failed:
            raise BudgetExceededError(
                f"loaded group {key!r} requires {actual_size} bytes, exceeding "
                f"the {self.budget_bytes}-byte residency budget"
            )

    def fail(self, key: str) -> None:
        """Release a failed loading reservation."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.state is ResidencyState.LOADING:
                del self._entries[key]

    def is_resident(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and entry.state is ResidencyState.RESIDENT

    def is_loading(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and entry.state is ResidencyState.LOADING

    def pin(self, key: str) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state is not ResidencyState.RESIDENT:
                raise KeyError(f"residency key {key!r} is not resident")
            entry.pin_count += 1
            entry.last_used = self._next_tick()
            return entry.value

    def release(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state is not ResidencyState.RESIDENT:
                raise KeyError(f"residency key {key!r} is not resident")
            if entry.pin_count <= 0:
                raise RuntimeError(f"residency key {key!r} is not pinned")
            entry.pin_count -= 1
            entry.last_used = self._next_tick()

    def evict(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.state is ResidencyState.LOADING or entry.pin_count > 0:
                return False
            del self._entries[key]
            evicted = [(key, entry.value)]
        self._notify_evictions(evicted)
        return True

    def clear(self) -> None:
        with self._lock:
            pinned = [entry.key for entry in self._entries.values() if entry.pin_count]
            if pinned:
                raise RuntimeError(f"cannot clear pinned residency entries: {pinned}")
            evicted = [
                (entry.key, entry.value)
                for entry in self._entries.values()
                if entry.state is ResidencyState.RESIDENT
            ]
            self._entries.clear()
        self._notify_evictions(evicted)

    def snapshot(self) -> ResidencySnapshot:
        with self._lock:
            loading = tuple(
                entry.key
                for entry in self._entries.values()
                if entry.state is ResidencyState.LOADING
            )
            resident = tuple(
                entry.key
                for entry in self._entries.values()
                if entry.state is ResidencyState.RESIDENT
            )
            pinned = tuple(
                entry.key for entry in self._entries.values() if entry.pin_count > 0
            )
            loading_bytes = sum(
                entry.byte_size
                for entry in self._entries.values()
                if entry.state is ResidencyState.LOADING
            )
            resident_bytes = sum(
                entry.byte_size
                for entry in self._entries.values()
                if entry.state is ResidencyState.RESIDENT
            )
            return ResidencySnapshot(
                budget_bytes=self.budget_bytes,
                used_bytes=loading_bytes + resident_bytes,
                loading_bytes=loading_bytes,
                resident_bytes=resident_bytes,
                resident_keys=resident,
                loading_keys=loading,
                pinned_keys=pinned,
            )
