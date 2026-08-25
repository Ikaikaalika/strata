from __future__ import annotations

import pytest

from ollm.core import (
    ANEExecutionMode,
    BackendTarget,
    ComputeUnit,
    OptimizationMode,
    ResidencyPreference,
    RuntimeCapabilities,
    RuntimePerformancePolicy,
    SSDOffloadMode,
    SSDOffloadPolicy,
    ServiceObjective,
    WorkloadClass,
)


def _objective() -> ServiceObjective:
    return ServiceObjective(
        workload_class=WorkloadClass.INTERACTIVE,
        context_tokens=512,
        max_output_tokens=128,
        allow_weight_spill=True,
    )


def test_maximum_speed_forces_full_residency_and_disables_spill() -> None:
    policy = RuntimePerformancePolicy.maximum_speed()

    objective = policy.apply(_objective())

    assert policy.ssd_offload.mode is SSDOffloadMode.DISABLED
    assert objective.residency_preference is ResidencyPreference.FULL
    assert objective.allow_weight_spill is False
    assert objective.max_resident_weight_bytes is None
    assert objective.preferred_storage_target_id is None


def test_maximum_speed_rejects_paged_ssd_configuration() -> None:
    with pytest.raises(ValueError, match="maximum_speed"):
        RuntimePerformancePolicy(
            optimization_mode=OptimizationMode.MAXIMUM_SPEED,
            ssd_offload=SSDOffloadPolicy(
                mode=SSDOffloadMode.REQUIRED,
                storage_target_id="approved-internal-ssd",
                max_resident_weight_bytes=512 * 1024 * 1024,
            ),
        )


def test_capacity_mode_retains_explicit_ssd_toggle() -> None:
    policy = RuntimePerformancePolicy(
        optimization_mode=OptimizationMode.CAPACITY,
        ssd_offload=SSDOffloadPolicy(
            mode=SSDOffloadMode.REQUIRED,
            storage_target_id="approved-internal-ssd",
            max_resident_weight_bytes=512 * 1024 * 1024,
        ),
    )

    objective = policy.apply(_objective())

    assert objective.residency_preference is ResidencyPreference.PAGED
    assert objective.allow_weight_spill is True


def test_ane_toggle_maps_public_coreml_and_private_research_separately() -> None:
    disabled = RuntimePerformancePolicy.maximum_speed(
        ane_execution=ANEExecutionMode.DISABLED
    )
    public = RuntimePerformancePolicy.maximum_speed()
    private = RuntimePerformancePolicy.maximum_speed(
        ane_execution=ANEExecutionMode.PRIVATE_RESEARCH,
        allow_private_apis=True,
    )
    direct_ane = RuntimeCapabilities(
        compute_units=(ComputeUnit.ANE,),
        requires_private_api=True,
    )

    assert disabled.coreml_compute_units == "cpuAndGPU"
    assert public.coreml_compute_units == "all"
    assert public.target_rejection_reason(BackendTarget.ANE, direct_ane) is not None
    assert private.target_rejection_reason(BackendTarget.ANE, direct_ane) is None


def test_policy_identity_changes_with_speed_margin_and_ane_mode() -> None:
    base = RuntimePerformancePolicy.maximum_speed()
    stricter = RuntimePerformancePolicy.maximum_speed(minimum_speedup_percent=10)
    no_ane = RuntimePerformancePolicy.maximum_speed(
        ane_execution=ANEExecutionMode.DISABLED
    )

    assert len({base.identity_digest, stricter.identity_digest, no_ane.identity_digest}) == 3


def test_public_parameters_round_trip_maximum_speed_toggles() -> None:
    parameters = {
        "optimization_mode": "maximum_speed",
        "ssd_offload": {"mode": "disabled"},
        "ane_execution": "public_auto",
        "allow_private_apis": False,
        "allowed_targets": ["mlx", "metal", "coreml", "cpu", "ane"],
        "minimum_speedup_percent": 7.5,
        "require_mlx_fallback": True,
    }

    policy = RuntimePerformancePolicy.from_parameters(parameters)
    decoded = RuntimePerformancePolicy.from_parameters(policy.to_parameters())

    assert decoded == policy
    assert decoded.ssd_offload.mode is SSDOffloadMode.DISABLED
    assert decoded.minimum_speedup_percent == 7.5


def test_public_parameters_reject_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown runtime policy"):
        RuntimePerformancePolicy.from_parameters({"ssd": "off"})
