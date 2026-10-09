from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.benchmark_mlx_lm import (
    _artifact_metadata,
    mlx_lazy_load_for_ssd_offload,
    summarize_repetitions,
    validate_local_snapshot,
)


def test_mlx_ssd_offload_parameter_is_explicit_about_os_management() -> None:
    assert mlx_lazy_load_for_ssd_offload("disabled") is False
    assert mlx_lazy_load_for_ssd_offload("os-managed") is True
    with pytest.raises(ValueError, match="unsupported"):
        mlx_lazy_load_for_ssd_offload("lokahi-paged")


def test_artifact_metadata_records_shape_quantization_and_weight_hash(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"fixture","hidden_size":16,"quantization":{"bits":4}}',
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"fixture")

    artifact = _artifact_metadata(tmp_path)

    assert artifact["model_type"] == "fixture"
    assert artifact["hidden_size"] == 16
    assert artifact["weight_quantization"] == {"bits": 4}
    assert artifact["weights"] == [
        {
            "filename": "model.safetensors",
            "bytes": 7,
            "sha256": "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d",
        }
    ]


def test_snapshot_validation_requires_exact_revision_and_approved_root(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    root = tmp_path / "approved"
    snapshot = root / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"fixture")

    assert validate_local_snapshot(snapshot, root, revision) == snapshot.resolve()
    with pytest.raises(ValueError, match="exact pinned"):
        validate_local_snapshot(snapshot, root, "b" * 40)


def test_mlx_summary_preserves_latency_and_throughput_directions() -> None:
    repetitions = [
        {
            "ttft_ms": 10.0 + index,
            "inter_token_latency_ms": 2.0 + index,
            "prompt_tokens_per_second": 100.0 + index,
            "decode_tokens_per_second": 50.0 + index,
            "aggregate_tokens_per_second_including_prefill": 40.0 + index,
            "peak_memory_gb": 1.0 + index,
        }
        for index in range(5)
    ]

    summary = summarize_repetitions(repetitions)

    assert summary["ttft_ms"]["median"] == 12.0
    assert summary["decode_tokens_per_second"]["median"] == 52.0
    assert summary["peak_memory_gb"]["p95"] == 5.0
