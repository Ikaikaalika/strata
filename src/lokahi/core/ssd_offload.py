"""User-facing SSD weight-offload policy mapped to admission contracts."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from .platform import ResidencyPreference, ServiceObjective


class SSDOffloadMode(str, Enum):
    """How explicitly the caller permits Lokahi-managed weight paging."""

    DISABLED = "disabled"
    AUTO = "auto"
    REQUIRED = "required"


@dataclass(frozen=True)
class SSDOffloadPolicy:
    """Executable form of the public ``ssd_offload`` parameter.

    Paths deliberately do not enter this contract. ``storage_target_id`` is an
    opaque identifier that the host resolves to a user-approved, qualified SSD.
    """

    mode: SSDOffloadMode = SSDOffloadMode.DISABLED
    storage_target_id: Optional[str] = None
    max_resident_weight_bytes: Optional[int] = None
    prefetch_distance: int = 1
    io_workers: int = 1

    def __post_init__(self) -> None:
        if self.storage_target_id == "":
            raise ValueError("storage_target_id must not be empty")
        if (
            self.max_resident_weight_bytes is not None
            and self.max_resident_weight_bytes <= 0
        ):
            raise ValueError("max_resident_weight_bytes must be positive")
        if self.prefetch_distance < 0:
            raise ValueError("prefetch_distance must be non-negative")
        if self.io_workers <= 0:
            raise ValueError("io_workers must be positive")
        if self.mode is SSDOffloadMode.DISABLED and (
            self.storage_target_id is not None
            or self.max_resident_weight_bytes is not None
        ):
            raise ValueError(
                "disabled SSD offload cannot select storage or a weight window"
            )

    @property
    def residency_preference(self) -> ResidencyPreference:
        return {
            SSDOffloadMode.DISABLED: ResidencyPreference.FULL,
            SSDOffloadMode.AUTO: ResidencyPreference.AUTO,
            SSDOffloadMode.REQUIRED: ResidencyPreference.PAGED,
        }[self.mode]

    @property
    def allow_weight_spill(self) -> bool:
        return self.mode is not SSDOffloadMode.DISABLED

    def apply(self, objective: ServiceObjective) -> ServiceObjective:
        """Return a request objective with this policy applied atomically."""

        return replace(
            objective,
            residency_preference=self.residency_preference,
            preferred_storage_target_id=self.storage_target_id,
            max_resident_weight_bytes=self.max_resident_weight_bytes,
            allow_weight_spill=self.allow_weight_spill,
        )


__all__ = ["SSDOffloadMode", "SSDOffloadPolicy"]
