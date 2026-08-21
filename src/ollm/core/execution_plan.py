"""Static execution plans produced from model weight groups."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .model_spec import ModelSpec


@dataclass(frozen=True)
class PlanStep:
    ordinal: int
    group_id: str
    prefetch_group_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    model_id: str
    steps: Tuple[PlanStep, ...]

    @classmethod
    def dense(
        cls,
        model_spec: ModelSpec,
        *,
        prefetch_distance: int = 1,
    ) -> "ExecutionPlan":
        if prefetch_distance < 0:
            raise ValueError("prefetch_distance must be non-negative")
        groups = model_spec.groups
        steps = []
        for index, group in enumerate(groups):
            prefetch = tuple(
                candidate.group_id
                for candidate in groups[index + 1 : index + 1 + prefetch_distance]
            )
            steps.append(
                PlanStep(
                    ordinal=index,
                    group_id=group.group_id,
                    prefetch_group_ids=prefetch,
                )
            )
        return cls(model_id=model_spec.model_id, steps=tuple(steps))
