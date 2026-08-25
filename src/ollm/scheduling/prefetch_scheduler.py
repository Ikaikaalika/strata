"""Exact weight-group prefetch with measured stall time."""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import Any, Mapping, Optional

from ..core.model_spec import WeightGroup
from ..storage.tensor_store import GroupTensorStore
from .residency_manager import BudgetExceededError, ResidencyManager


def _bundle_nbytes(bundle: Mapping[str, Any], fallback: int) -> int:
    sizes = [getattr(value, "nbytes", None) for value in bundle.values()]
    if sizes and all(size is not None for size in sizes):
        measured = sum(int(size) for size in sizes)
        if measured > 0:
            return measured
    return fallback


@dataclass(frozen=True)
class SchedulerSnapshot:
    resident_hits: int
    cold_misses: int
    prefetch_ready_hits: int
    prefetch_waits: int
    prefetch_skips: int
    bytes_loaded: int
    load_time_ms: float
    stall_time_ms: float


class WeightLease:
    """Pinned weight-group value returned by the prefetch scheduler."""

    def __init__(
        self,
        scheduler: "PrefetchScheduler",
        group: WeightGroup,
        value: Mapping[str, Any],
        cache_status: str,
    ) -> None:
        self.scheduler = scheduler
        self.group = group
        self.value = value
        self.cache_status = cache_status
        self._released = False

    def release(self) -> None:
        if not self._released:
            self.scheduler.release(self.group)
            self._released = True

    def __enter__(self) -> Mapping[str, Any]:
        return self.value

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class PrefetchScheduler:
    """Load dense weight groups ahead of demand using bounded I/O workers."""

    def __init__(
        self,
        store: GroupTensorStore,
        residency: ResidencyManager,
        *,
        tracer: Optional[Any] = None,
        max_workers: int = 1,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.store = store
        self.residency = residency
        self.tracer = tracer
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="strata-prefetch",
        )
        self._futures: dict[str, Future] = {}
        self._lock = RLock()
        self._resident_hits = 0
        self._cold_misses = 0
        self._prefetch_ready_hits = 0
        self._prefetch_waits = 0
        self._prefetch_skips = 0
        self._bytes_loaded = 0
        self._load_time_ms = 0.0
        self._stall_time_ms = 0.0

    def _trace_operation(self, name: str, duration_ms: float, **kwargs: Any) -> None:
        if self.tracer is not None:
            self.tracer.record_operation(name, duration_ms, **kwargs)

    def _load_reserved(self, group: WeightGroup, phase: str, status: str):
        started = time.perf_counter()
        try:
            bundle = self.store.load_group(group)
            actual_nbytes = _bundle_nbytes(bundle, group.nbytes)
            self.residency.commit(
                group.group_id,
                bundle,
                byte_size=actual_nbytes,
            )
        except BaseException:
            self.residency.fail(group.group_id)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._trace_operation(
                "weight_load",
                elapsed_ms,
                phase=phase,
                group_id=group.group_id,
                cache_status="error",
                bytes_transferred=0,
            )
            raise

        elapsed_seconds = time.perf_counter() - started
        elapsed_ms = elapsed_seconds * 1000.0
        bandwidth = actual_nbytes / elapsed_seconds if elapsed_seconds > 0 else None
        with self._lock:
            self._bytes_loaded += actual_nbytes
            self._load_time_ms += elapsed_ms
        self._trace_operation(
            "weight_load",
            elapsed_ms,
            phase=phase,
            group_id=group.group_id,
            cache_status=status,
            bytes_transferred=actual_nbytes,
            bandwidth_bytes_per_second=bandwidth,
        )
        return bundle

    def prefetch(self, group: WeightGroup, *, phase: str) -> bool:
        """Attempt to reserve and asynchronously load a future weight group."""
        with self._lock:
            if self.residency.is_resident(group.group_id):
                return True
            if group.group_id in self._futures:
                return True
            if not self.residency.reserve(group.group_id, group.nbytes):
                self._prefetch_skips += 1
                self._trace_operation(
                    "weight_prefetch",
                    0.0,
                    phase=phase,
                    group_id=group.group_id,
                    cache_status="budget_skip",
                )
                return False
            self._futures[group.group_id] = self._executor.submit(
                self._load_reserved,
                group,
                phase,
                "prefetch",
            )
            return True

    def acquire(self, group: WeightGroup, *, phase: str) -> WeightLease:
        """Pin a resident group, waiting or loading synchronously on a miss."""
        started = time.perf_counter()

        with self._lock:
            future = self._futures.get(group.group_id)

        if future is not None:
            ready_before_wait = future.done()
            try:
                future.result()
            finally:
                with self._lock:
                    self._futures.pop(group.group_id, None)
            with self._lock:
                if ready_before_wait:
                    self._prefetch_ready_hits += 1
                    cache_status = "prefetch_ready"
                else:
                    self._prefetch_waits += 1
                    cache_status = "prefetch_wait"
            try:
                value = self.residency.pin(group.group_id)
            except KeyError:
                # With multiple workers and a tight budget, another completed
                # prefetch may evict this unpinned result between future.result
                # and pin(). Correctness must not depend on completion order:
                # reload the demanded group synchronously under the same cap.
                if not self.residency.reserve(group.group_id, group.nbytes):
                    snapshot = self.residency.snapshot()
                    raise BudgetExceededError(
                        f"cannot reload evicted prefetch {group.group_id!r} "
                        f"inside a {snapshot.budget_bytes}-byte budget"
                    )
                self._load_reserved(group, phase, "prefetch_evicted_reload")
                value = self.residency.pin(group.group_id)
                cache_status = "prefetch_evicted_reload"
                with self._lock:
                    self._cold_misses += 1
        elif self.residency.is_resident(group.group_id):
            value = self.residency.pin(group.group_id)
            cache_status = "resident_hit"
            with self._lock:
                self._resident_hits += 1
        else:
            if not self.residency.reserve(group.group_id, group.nbytes):
                snapshot = self.residency.snapshot()
                raise BudgetExceededError(
                    f"cannot acquire {group.group_id!r} ({group.nbytes} bytes) "
                    f"inside a {snapshot.budget_bytes}-byte budget; "
                    f"pinned={list(snapshot.pinned_keys)}"
                )
            self._load_reserved(group, phase, "demand")
            value = self.residency.pin(group.group_id)
            cache_status = "cold_miss"
            with self._lock:
                self._cold_misses += 1

        stall_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._stall_time_ms += stall_ms
        self._trace_operation(
            "weight_acquire",
            stall_ms,
            phase=phase,
            group_id=group.group_id,
            cache_status=cache_status,
        )
        return WeightLease(self, group, value, cache_status)

    def release(self, group: WeightGroup) -> None:
        self.residency.release(group.group_id)

    def snapshot(self) -> SchedulerSnapshot:
        with self._lock:
            return SchedulerSnapshot(
                resident_hits=self._resident_hits,
                cold_misses=self._cold_misses,
                prefetch_ready_hits=self._prefetch_ready_hits,
                prefetch_waits=self._prefetch_waits,
                prefetch_skips=self._prefetch_skips,
                bytes_loaded=self._bytes_loaded,
                load_time_ms=self._load_time_ms,
                stall_time_ms=self._stall_time_ms,
            )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "PrefetchScheduler":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
