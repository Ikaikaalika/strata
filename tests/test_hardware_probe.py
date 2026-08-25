import unittest
from unittest import mock

from ollm.core import ComputeUnit
from ollm.hardware_probe import detect_apple_hardware


RESPONSES = {
    ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M1",
    ("sysctl", "-n", "hw.memsize"): str(16 * 1024**3),
    ("sysctl", "-n", "hw.logicalcpu"): "8",
    ("sw_vers", "-productVersion"): "26.5",
    ("sw_vers", "-buildVersion"): "25F90",
}


def fake_runner(command):
    return RESPONSES[tuple(command)]


class AppleHardwareProbeTest(unittest.TestCase):
    @mock.patch("ollm.hardware_probe.platform.machine", return_value="arm64")
    def test_discovered_ane_surface_is_not_execution_capability(self, _machine):
        profile = detect_apple_hardware(
            ane_runtime_fingerprint="surface-only",
            ane_execution_verified=False,
            command_runner=fake_runner,
            mlx_info_provider=lambda: {
                "max_recommended_working_set_size": 12 * 1024**3
            },
        )
        self.assertEqual(profile.chip, "Apple M1")
        self.assertEqual(profile.compute_units, (ComputeUnit.CPU, ComputeUnit.GPU))
        self.assertEqual(profile.ane_runtime_fingerprint, "surface-only")

    @mock.patch("ollm.hardware_probe.platform.machine", return_value="arm64")
    def test_numerically_verified_ane_is_executable(self, _machine):
        profile = detect_apple_hardware(
            ane_runtime_fingerprint="verified-runtime",
            ane_execution_verified=True,
            storage_read_bytes_per_second=2.0e9,
            command_runner=fake_runner,
            mlx_info_provider=lambda: {"memory_size": 16 * 1024**3},
        )
        self.assertIn(ComputeUnit.ANE, profile.compute_units)
        self.assertEqual(profile.storage_read_bytes_per_second, 2.0e9)

    @mock.patch("ollm.hardware_probe.platform.machine", return_value="arm64")
    def test_verified_ane_requires_runtime_fingerprint(self, _machine):
        with self.assertRaisesRegex(ValueError, "runtime fingerprint"):
            detect_apple_hardware(
                ane_execution_verified=True,
                command_runner=fake_runner,
                mlx_info_provider=dict,
            )

    @mock.patch("ollm.hardware_probe.platform.machine", return_value="x86_64")
    def test_non_apple_silicon_fails_closed(self, _machine):
        with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
            detect_apple_hardware(
                command_runner=fake_runner,
                mlx_info_provider=dict,
            )


if __name__ == "__main__":
    unittest.main()
