"""Storage adapters for runtime-neutral Strata weight groups."""

from .tensor_store import GroupTensorStore, LoaderGroupTensorStore

__all__ = ["GroupTensorStore", "LoaderGroupTensorStore"]
