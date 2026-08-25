"""Dense layer execution with exact next-layer prefetch."""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional

from ..core.execution_plan import ExecutionPlan
from ..core.model_spec import ModelSpec, WeightGroup
from .prefetch_scheduler import PrefetchScheduler


class DenseLayerPipeline:
    """Execute an ordered dense model plan under a residency budget."""

    def __init__(
        self,
        model_spec: ModelSpec,
        plan: ExecutionPlan,
        scheduler: PrefetchScheduler,
        *,
        tracer: Optional[Any] = None,
        synchronize: Optional[Callable[[Any], None]] = None,
        memory_probe: Optional[Callable[[], Mapping[str, int]]] = None,
    ) -> None:
        if plan.model_id != model_spec.model_id:
            raise ValueError("execution plan and model spec IDs must match")
        self.model_spec = model_spec
        self.plan = plan
        self.scheduler = scheduler
        self.tracer = tracer
        self.synchronize = synchronize
        self.memory_probe = memory_probe
        self._groups = model_spec.groups_by_id()

        unknown = {
            group_id
            for step in plan.steps
            for group_id in (step.group_id, *step.prefetch_group_ids)
            if group_id not in self._groups
        }
        if unknown:
            raise ValueError(f"execution plan references unknown groups: {sorted(unknown)}")

    def run(
        self,
        state: Any,
        execute_layer: Callable[[Any, WeightGroup, Mapping[str, Any]], Any],
        *,
        phase: str,
        release_weights: Optional[
            Callable[[WeightGroup, Mapping[str, Any]], None]
        ] = None,
    ) -> Any:
        """Run all plan steps, overlapping next-group I/O with current compute."""
        for step in self.plan.steps:
            group = self._groups[step.group_id]
            lease = self.scheduler.acquire(group, phase=phase)
            weights = lease.value
            try:
                for prefetch_group_id in step.prefetch_group_ids:
                    self.scheduler.prefetch(
                        self._groups[prefetch_group_id],
                        phase=phase,
                    )

                started = time.perf_counter()
                state = execute_layer(state, group, weights)
                if self.synchronize is not None:
                    self.synchronize(state)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                memory = self.memory_probe() if self.memory_probe is not None else {}
                if self.tracer is not None:
                    self.tracer.record_operation(
                        "layer_compute",
                        elapsed_ms,
                        phase=phase,
                        layer=group.order,
                        group_id=group.group_id,
                        cache_status=lease.cache_status,
                        active_memory_bytes=memory.get("active_memory_bytes"),
                        peak_memory_bytes=memory.get("peak_memory_bytes"),
                    )
            finally:
                if release_weights is not None:
                    release_weights(group, weights)
                lease.release()

        return state

    def close(self) -> None:
        self.scheduler.close()
