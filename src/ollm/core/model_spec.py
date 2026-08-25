"""Runtime-neutral model and weight-group descriptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .tensor_ref import TensorRef


@dataclass(frozen=True)
class WeightGroup:
    """Weights that become resident and evict together."""

    group_id: str
    order: int
    tensors: Tuple[TensorRef, ...]
    kind: str = "dense_layer"

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if self.order < 0:
            raise ValueError("group order must be non-negative")
        if not self.tensors:
            raise ValueError("weight groups must contain at least one tensor")
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tensor name in group {self.group_id!r}")

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors)


@dataclass(frozen=True)
class ModelSpec:
    """Ordered model weight groups consumed by an execution planner."""

    model_id: str
    groups: Tuple[WeightGroup, ...]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.groups:
            raise ValueError("model spec must contain at least one weight group")
        ids = [group.group_id for group in self.groups]
        orders = [group.order for group in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("model weight-group IDs must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError("model weight-group order values must be unique")

    @classmethod
    def from_groups(cls, model_id: str, groups: Iterable[WeightGroup]) -> "ModelSpec":
        return cls(model_id=model_id, groups=tuple(sorted(groups, key=lambda group: group.order)))

    @property
    def total_weight_bytes(self) -> int:
        return sum(group.nbytes for group in self.groups)

    def groups_by_id(self) -> Dict[str, WeightGroup]:
        return {group.group_id: group for group in self.groups}
