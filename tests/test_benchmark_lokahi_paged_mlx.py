from __future__ import annotations

from types import SimpleNamespace

from benchmarks.benchmark_lokahi_paged_mlx import _scheduler_delta


def test_scheduler_delta_preserves_paging_counters_and_timings() -> None:
    before = SimpleNamespace(
        resident_hits=1,
        cold_misses=2,
        prefetch_ready_hits=3,
        prefetch_waits=4,
        prefetch_skips=5,
        bytes_loaded=100,
        load_time_ms=10.0,
        stall_time_ms=2.0,
    )
    after = SimpleNamespace(
        resident_hits=3,
        cold_misses=5,
        prefetch_ready_hits=7,
        prefetch_waits=9,
        prefetch_skips=11,
        bytes_loaded=250,
        load_time_ms=25.0,
        stall_time_ms=6.0,
    )

    assert _scheduler_delta(before, after) == {
        "resident_hits": 2,
        "cold_misses": 3,
        "prefetch_ready_hits": 4,
        "prefetch_waits": 5,
        "prefetch_skips": 6,
        "bytes_loaded": 150,
        "load_time_ms": 15.0,
        "stall_time_ms": 4.0,
    }
