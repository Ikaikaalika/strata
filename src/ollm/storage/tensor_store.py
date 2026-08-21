"""Tensor-store protocols and adapters."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol, runtime_checkable

from ..core.model_spec import WeightGroup


@runtime_checkable
class GroupTensorStore(Protocol):
    """Load one atomic weight group from its cold storage tier."""

    def load_group(self, group: WeightGroup) -> Mapping[str, Any]: ...


class LoaderGroupTensorStore:
    """Adapt the project's existing single-parameter loaders to group loading."""

    def __init__(self, loader: Any):
        if not hasattr(loader, "load_param_to_device"):
            raise TypeError("loader must define load_param_to_device(name)")
        self.loader = loader

    def load_group(self, group: WeightGroup) -> Mapping[str, Any]:
        tensors: Dict[str, Any] = {}
        for tensor_ref in group.tensors:
            tensors[tensor_ref.name] = self.loader.load_param_to_device(tensor_ref.key)
        return tensors
