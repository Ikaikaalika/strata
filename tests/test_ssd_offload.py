from __future__ import annotations

import pytest

from lokahi.core.platform import (
    ResidencyPreference,
    ServiceObjective,
    WorkloadClass,
)
from lokahi.core.ssd_offload import SSDOffloadMode, SSDOffloadPolicy


def _objective() -> ServiceObjective:
    return ServiceObjective(
        workload_class=WorkloadClass.INTERACTIVE,
        context_tokens=512,
        max_output_tokens=128,
    )


@pytest.mark.parametrize(
    ("mode", "preference", "spill"),
    [
        (SSDOffloadMode.DISABLED, ResidencyPreference.FULL, False),
        (SSDOffloadMode.AUTO, ResidencyPreference.AUTO, True),
        (SSDOffloadMode.REQUIRED, ResidencyPreference.PAGED, True),
    ],
)
def test_ssd_offload_mode_maps_to_fail_closed_admission_policy(
    mode: SSDOffloadMode,
    preference: ResidencyPreference,
    spill: bool,
) -> None:
    result = SSDOffloadPolicy(mode=mode).apply(_objective())

    assert result.residency_preference is preference
    assert result.allow_weight_spill is spill


def test_ssd_offload_applies_opaque_target_window_and_prefetch() -> None:
    policy = SSDOffloadPolicy(
        mode=SSDOffloadMode.REQUIRED,
        storage_target_id="approved-internal-ssd",
        max_resident_weight_bytes=512 * 1024 * 1024,
        prefetch_distance=2,
        io_workers=3,
    )

    result = policy.apply(_objective())

    assert result.preferred_storage_target_id == "approved-internal-ssd"
    assert result.max_resident_weight_bytes == 512 * 1024 * 1024
    assert policy.prefetch_distance == 2
    assert policy.io_workers == 3


@pytest.mark.parametrize("io_workers", (0, -1))
def test_ssd_offload_rejects_non_positive_io_workers(io_workers: int) -> None:
    with pytest.raises(ValueError, match="io_workers"):
        SSDOffloadPolicy(mode=SSDOffloadMode.AUTO, io_workers=io_workers)


def test_disabled_ssd_offload_rejects_storage_parameters() -> None:
    with pytest.raises(ValueError, match="disabled"):
        SSDOffloadPolicy(
            mode=SSDOffloadMode.DISABLED,
            storage_target_id="approved-internal-ssd",
        )
