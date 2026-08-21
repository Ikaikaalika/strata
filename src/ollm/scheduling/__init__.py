"""Memory residency and prefetch scheduling for Strata."""

from .dense_pipeline import DenseLayerPipeline
from .prefetch_scheduler import PrefetchScheduler, WeightLease
from .residency_manager import (
    BudgetExceededError,
    ResidencyManager,
    ResidencySnapshot,
)

__all__ = [
    "BudgetExceededError",
    "DenseLayerPipeline",
    "PrefetchScheduler",
    "ResidencyManager",
    "ResidencySnapshot",
    "WeightLease",
]
