"""Offline fixture, numerical, reserve and receipt gates; no performance proof."""
import copy
from pathlib import Path

import numpy as np
import pytest

from benchmarks.benchmark_native_q4 import (
    ATOL, RTOL, KERNELS, RESERVE_BYTES, SHAPES, fixture, numerical_gate,
    require_ssd_reserve, validate_native_receipt,
)


@pytest.mark.parametrize("shape", SHAPES)
def test_packed_affine_formula_and_dtype(shape):
    x, packed, scales, biases, expected = fixture(shape)
    rows, width, outputs = shape
    assert x.dtype == scales.dtype == biases.dtype == expected.dtype == np.float16
    assert packed.dtype == np.uint32
    # Decode actual packed words; check against the separately constructed code
    # matrix and FP64 dot product rather than trusting a fixture output digest.
    expanded = np.empty((outputs, width), dtype=np.float64)
    for k in range(width):
        code = (packed[:, k // 8] >> (4 * (k % 8))) & 15
        expanded[:, k] = scales[:, k // 64].astype(np.float64) * code + biases[:, k // 64]
        assert np.array_equal(code, (k * 5 + np.arange(outputs) * 3) % 16)
    independent = (x.astype(np.float64) @ expanded.T).astype(np.float16)
    assert np.array_equal(expected, independent)
    assert expected.shape == (rows, outputs)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 2])
def test_numerical_gate_rejects_unwritten_nonfinite_and_wrong_outputs(value):
    with pytest.raises(ValueError):
        numerical_gate(np.array([value]), np.array([0]))


def test_reserve_rejects_shortfall_and_failed_drive_without_access(monkeypatch, tmp_path):
    class Usage:
        free = RESERVE_BYTES - 1
    monkeypatch.setattr("benchmarks.benchmark_native_q4.shutil.disk_usage", lambda p: Usage())
    with pytest.raises(ValueError, match="reserve requires"):
        require_ssd_reserve(tmp_path)
    with pytest.raises(ValueError, match="real approved"):
        require_ssd_reserve(Path("/Volumes/Tyler HDD/strata"))
    Usage.free = RESERVE_BYTES
    assert require_ssd_reserve(tmp_path) == RESERVE_BYTES


def test_reserve_rejects_other_volume_before_reading_free_space(monkeypatch, tmp_path):
    monkeypatch.setattr("benchmarks.benchmark_native_q4._device_id",
                        lambda path: 1 if path == Path.home() else 2)
    def forbidden_usage(path):
        pytest.fail("external-volume free space must not count toward the reserve")
    monkeypatch.setattr("benchmarks.benchmark_native_q4.shutil.disk_usage", forbidden_usage)
    with pytest.raises(ValueError, match="internal home volume"):
        require_ssd_reserve(tmp_path)


def test_reserve_rejects_symlink_ancestor_before_touching_failed_mount(tmp_path):
    link = tmp_path / "failed-drive-link"
    link.symlink_to("/Volumes/Tyler HDD", target_is_directory=True)
    with pytest.raises(ValueError, match="real approved"):
        require_ssd_reserve(link / "strata" / "build")


def passing_receipt():
    return {"schema_version": 1, "execution_verified": True,
            "promotion_eligible": False, "ssd_offload": "disabled",
            "evidence_kind": "hardware_component", "fixture_id": "integer_affine_q4_v1",
            "dtype": "fp16_input_scale_bias_output_fp32_accumulation",
            "bits": 4, "group_size": 64, "warmups": 5, "iterations": 20,
            **{name + "_sha256": "a" * 64 for name in ("source", "contract", "host", "binary", "metallib")},
            "cells": [{"shape": list(shape), "variants": [
                {"kernel": kernel, "wall_ms": [0.1] * 20, "gpu_ms": [None] * 20,
                 "numerical": {"passed": True, "max_abs": 0, "max_scaled": 0, "atol": ATOL, "rtol": RTOL}}
                for kernel in sorted(KERNELS)]} for shape in SHAPES]}


def test_receipt_accepts_unavailable_gpu_counter_without_inventing_zero():
    validate_native_receipt(passing_receipt())


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(promotion_eligible=True),
    lambda r: r.update(bits=8),
    lambda r: r.update(group_size=32),
    lambda r: r.update(dtype="float32"),
    lambda r: r.update(source_sha256="unknown"),
    lambda r: r.update(iterations=0),
    lambda r: r["cells"].pop(),
    lambda r: r["cells"].append(copy.deepcopy(r["cells"][0])),
    lambda r: r["cells"][0]["variants"].pop(),
    lambda r: r["cells"][0]["variants"][0]["numerical"].update(atol=1),
    lambda r: r["cells"][0]["variants"][0]["numerical"].update(max_scaled=2),
    lambda r: r["cells"][0]["variants"][0]["numerical"].update(max_abs=float("nan")),
    lambda r: r["cells"][0]["variants"][0]["wall_ms"].__setitem__(0, float("nan")),
    lambda r: r["cells"][0]["variants"][0]["wall_ms"].__setitem__(0, True),
    lambda r: r["cells"][0]["variants"][0]["gpu_ms"].__setitem__(0, 0),
])
def test_receipt_rejects_incomplete_invalid_or_relaxed_evidence(mutation):
    receipt = passing_receipt()
    mutation(receipt)
    with pytest.raises(ValueError):
        validate_native_receipt(receipt)
