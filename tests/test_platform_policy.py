import unittest

from lokahi.core import (
    AdaptiveAdmissionPolicy,
    AdmissionReason,
    ComputeUnit,
    DeploymentMode,
    HardwareProfile,
    LivePlatformState,
    MemoryPressure,
    PowerSource,
    ResidencyMode,
    ResidencyPreference,
    RuntimeDemand,
    ServiceObjective,
    StorageMedium,
    StorageTarget,
    ThermalState,
    WorkloadClass,
)


GIB = 1024**3


def hardware():
    return HardwareProfile(
        chip="Apple M1",
        macos_version="26.5",
        macos_build="25F90",
        unified_memory_bytes=16 * GIB,
        cpu_logical_cores=8,
        compute_units=(ComputeUnit.CPU, ComputeUnit.GPU),
        gpu_recommended_working_set_bytes=12 * GIB,
    )


def objective(**overrides):
    values = dict(
        workload_class=WorkloadClass.INTERACTIVE,
        context_tokens=2048,
        max_output_tokens=512,
    )
    values.update(overrides)
    return ServiceObjective(**values)


def demand(**overrides):
    values = dict(
        model_weight_bytes=6 * GIB,
        kv_cache_bytes=GIB,
        activation_bytes=512 * 1024**2,
        temporary_bytes=512 * 1024**2,
        minimum_weight_window_bytes=GIB,
    )
    values.update(overrides)
    return RuntimeDemand(**values)


def platform(profile, **overrides):
    values = dict(
        hardware_fingerprint=profile.fingerprint,
        available_memory_bytes=12 * GIB,
        gpu_allocated_bytes=GIB,
        memory_pressure=MemoryPressure.NORMAL,
        thermal_state=ThermalState.NOMINAL,
        power_source=PowerSource.AC,
        low_power_mode=False,
    )
    values.update(overrides)
    return LivePlatformState(**values)


class AdaptiveAdmissionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.hardware = hardware()
        self.policy = AdaptiveAdmissionPolicy(
            system_reserve_bytes=GIB,
            max_unified_memory_fraction=0.85,
        )

    def test_fully_resident_request_is_admitted(self):
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware),
            objective=objective(),
            demand=demand(),
        )
        self.assertTrue(decision.admitted)
        self.assertIs(decision.residency_mode, ResidencyMode.FULL)
        self.assertEqual(decision.reasons, (AdmissionReason.FULL_RESIDENCY,))
        self.assertFalse(decision.private_backends_allowed)

    def test_paging_selects_fastest_approved_measured_ssd_and_never_hdd(self):
        targets = (
            StorageTarget("checkout-hdd", StorageMedium.HDD, True, 100 * GIB, 3e9),
            StorageTarget("slow-ssd", StorageMedium.EXTERNAL_SSD, True, 100 * GIB, 1e9),
            StorageTarget("fast-ssd", StorageMedium.INTERNAL_SSD, True, 100 * GIB, 2e9),
            StorageTarget("unapproved", StorageMedium.INTERNAL_SSD, False, 100 * GIB, 4e9),
        )
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                available_memory_bytes=6 * GIB,
                storage_targets=targets,
            ),
            objective=objective(allow_weight_spill=True),
            demand=demand(model_weight_bytes=10 * GIB, storage_bytes_required=10 * GIB),
        )
        self.assertTrue(decision.admitted)
        self.assertIs(decision.residency_mode, ResidencyMode.PAGED)
        self.assertEqual(decision.storage_target_id, "fast-ssd")

    def test_auto_paging_rejects_storage_ceiling_below_decode_objective(self):
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                available_memory_bytes=6 * GIB,
                storage_targets=(
                    StorageTarget(
                        "approved-ssd",
                        StorageMedium.INTERNAL_SSD,
                        True,
                        100 * GIB,
                        2 * GIB,
                    ),
                ),
            ),
            objective=objective(
                allow_weight_spill=True,
                min_decode_tokens_per_second=8.0,
            ),
            demand=demand(
                model_weight_bytes=10 * GIB,
                paging_bytes_per_decode_token=GIB,
            ),
        )

        self.assertFalse(decision.admitted)
        self.assertEqual(
            decision.reasons,
            (AdmissionReason.SSD_THROUGHPUT_BOUND,),
        )

    def test_required_paging_admits_capacity_mode_and_discloses_ceiling(self):
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                available_memory_bytes=6 * GIB,
                storage_targets=(
                    StorageTarget(
                        "approved-ssd",
                        StorageMedium.INTERNAL_SSD,
                        True,
                        100 * GIB,
                        2 * GIB,
                    ),
                ),
            ),
            objective=objective(
                residency_preference=ResidencyPreference.PAGED,
                min_decode_tokens_per_second=8.0,
            ),
            demand=demand(
                model_weight_bytes=10 * GIB,
                paging_bytes_per_decode_token=GIB,
            ),
        )

        self.assertTrue(decision.admitted)
        self.assertEqual(decision.estimated_storage_decode_ceiling_tps, 2.0)
        self.assertEqual(
            decision.reasons,
            (AdmissionReason.SSD_PAGING, AdmissionReason.SSD_THROUGHPUT_BOUND),
        )

    def test_explicit_paged_mode_uses_ssd_even_when_full_residency_fits(self):
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                storage_targets=(
                    StorageTarget(
                        "approved-ssd",
                        StorageMedium.INTERNAL_SSD,
                        True,
                        100 * GIB,
                        2e9,
                    ),
                ),
            ),
            objective=objective(
                residency_preference=ResidencyPreference.PAGED,
                max_resident_weight_bytes=2 * GIB,
            ),
            demand=demand(storage_bytes_required=6 * GIB),
        )
        self.assertTrue(decision.admitted)
        self.assertIs(decision.residency_mode, ResidencyMode.PAGED)
        self.assertEqual(decision.storage_target_id, "approved-ssd")
        self.assertEqual(decision.weight_residency_budget_bytes, 2 * GIB)

    def test_paged_mode_can_select_one_approved_storage_target(self):
        targets = (
            StorageTarget(
                "internal-ssd",
                StorageMedium.INTERNAL_SSD,
                True,
                100 * GIB,
                3e9,
            ),
            StorageTarget(
                "selected-external-ssd",
                StorageMedium.EXTERNAL_SSD,
                True,
                100 * GIB,
                1e9,
            ),
        )
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware, storage_targets=targets),
            objective=objective(
                residency_preference=ResidencyPreference.PAGED,
                preferred_storage_target_id="selected-external-ssd",
            ),
            demand=demand(storage_bytes_required=6 * GIB),
        )

        self.assertTrue(decision.admitted)
        self.assertEqual(decision.storage_target_id, "selected-external-ssd")

    def test_weight_cap_below_minimum_window_is_rejected(self):
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                storage_targets=(
                    StorageTarget(
                        "approved-ssd",
                        StorageMedium.INTERNAL_SSD,
                        True,
                        100 * GIB,
                        2e9,
                    ),
                ),
            ),
            objective=objective(
                residency_preference=ResidencyPreference.PAGED,
                max_resident_weight_bytes=512 * 1024**2,
            ),
            demand=demand(minimum_weight_window_bytes=GIB),
        )

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reasons, (AdmissionReason.INSUFFICIENT_MEMORY,))

    def test_explicit_full_mode_never_falls_back_to_ssd(self):
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                available_memory_bytes=6 * GIB,
                storage_targets=(
                    StorageTarget(
                        "approved-ssd",
                        StorageMedium.INTERNAL_SSD,
                        True,
                        100 * GIB,
                        2e9,
                    ),
                ),
            ),
            objective=objective(
                residency_preference=ResidencyPreference.FULL,
                allow_weight_spill=True,
            ),
            demand=demand(model_weight_bytes=10 * GIB),
        )
        self.assertFalse(decision.admitted)
        self.assertEqual(
            decision.reasons,
            (AdmissionReason.FULL_RESIDENCY_REQUIRED,),
        )

    def test_hdd_only_never_qualifies_as_paging_target(self):
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                available_memory_bytes=6 * GIB,
                storage_targets=(
                    StorageTarget(
                        "checkout-hdd", StorageMedium.HDD, True, 100 * GIB, 3e9
                    ),
                ),
            ),
            objective=objective(allow_weight_spill=True),
            demand=demand(model_weight_bytes=10 * GIB),
        )
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reasons, (AdmissionReason.NO_QUALIFIED_SSD,))

    def test_critical_pressure_and_serious_thermal_state_fail_closed(self):
        pressure = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                memory_pressure=MemoryPressure.CRITICAL,
            ),
            objective=objective(),
            demand=demand(),
        )
        thermal = self.policy.decide(
            hardware=self.hardware,
            platform=platform(
                self.hardware,
                thermal_state=ThermalState.SERIOUS,
            ),
            objective=objective(),
            demand=demand(),
        )
        self.assertEqual(pressure.reasons, (AdmissionReason.MEMORY_PRESSURE,))
        self.assertEqual(thermal.reasons, (AdmissionReason.THERMAL_PRESSURE,))

    def test_state_for_another_hardware_fingerprint_is_rejected(self):
        decision = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware, hardware_fingerprint="stale"),
            objective=objective(),
            demand=demand(),
        )
        self.assertEqual(
            decision.reasons,
            (AdmissionReason.HARDWARE_STATE_MISMATCH,),
        )

    def test_supported_mode_disables_private_backends(self):
        supported = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware),
            objective=objective(deployment_mode=DeploymentMode.SUPPORTED),
            demand=demand(),
        )
        research = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware),
            objective=objective(deployment_mode=DeploymentMode.RESEARCH),
            demand=demand(),
        )
        self.assertFalse(supported.private_backends_allowed)
        self.assertTrue(research.private_backends_allowed)

    def test_battery_requires_explicit_permission(self):
        rejected = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware, power_source=PowerSource.BATTERY),
            objective=objective(),
            demand=demand(),
        )
        admitted = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware, power_source=PowerSource.BATTERY),
            objective=objective(allow_battery=True),
            demand=demand(),
        )
        self.assertEqual(rejected.reasons, (AdmissionReason.POWER_POLICY,))
        self.assertTrue(admitted.admitted)

    def test_low_power_mode_requires_explicit_permission(self):
        rejected = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware, low_power_mode=True),
            objective=objective(),
            demand=demand(),
        )
        admitted = self.policy.decide(
            hardware=self.hardware,
            platform=platform(self.hardware, low_power_mode=True),
            objective=objective(allow_low_power_mode=True),
            demand=demand(),
        )
        self.assertEqual(rejected.reasons, (AdmissionReason.POWER_POLICY,))
        self.assertTrue(admitted.admitted)


if __name__ == "__main__":
    unittest.main()
