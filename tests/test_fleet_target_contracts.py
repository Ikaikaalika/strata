"""Offline metadata/target integrity; no device-capacity or execution claim."""
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "benchmarks/targets"
AUDIT = json.loads((TARGETS / "commoncompute_artifact_audit_20260907.json").read_text())
RELEASES = json.loads((TARGETS / "family_metadata_20260907.json").read_text())
PROFILES = json.loads((TARGETS / "fleet_optimization_profiles_20260907.json").read_text())


@pytest.mark.parametrize("model", AUDIT["models"], ids=lambda m: m["id"])
def test_complete_unique_shard_accounting(model):
    assert re.fullmatch(r"[0-9a-f]{40}", model["sha"])
    files = model["root_weight_files"]
    assert files and len({f["path"] for f in files}) == len(files)
    assert all(f["size"] > 0 and re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) for f in files)
    assert sum(f["size"] for f in files) == model["root_weight_bytes"]


def test_family_targets_resolve_to_exact_audit_pins():
    catalog = {m["catalog"]["id"] for m in AUDIT["models"] if m["catalog"]}
    releases = {m["id"] for m in RELEASES["models"]}
    assert len(catalog) == 33 and len(releases) == 16
    assert {f["family"] for f in PROFILES["families"]} == {"qwen", "gemma", "glm", "deepseek", "gpt_oss", "llama"}
    for family in PROFILES["families"]:
        assert set(family["catalog_ids"]) <= catalog
        assert set(family["upstream"]) <= releases


def test_metadata_is_not_native_promotion():
    assert not PROFILES["native_runtime_qualified"]
    assert not PROFILES["promotion"]["component_results_can_promote"]
    assert PROFILES["ssd_offload"] == "disabled"
    assert all(not r["native_full_model_qualified"] and not r["weights_downloaded"] for r in RELEASES["models"])
    tiers = PROFILES["hardware"]["memory_tiers_gib"]
    assert tiers == sorted(set(tiers)) and tiers[0] == 16 and tiers[-1] == 512


def test_qwen_catalog_underestimate_is_not_the_measured_size():
    qwen = next(m for m in AUDIT["models"] if m["id"] == "mlx-community/Qwen3.8-27B-4bit")
    assert len(qwen["root_weight_files"]) == 3
    assert qwen["root_weight_bytes"] == 16054541349
    assert qwen["root_weight_bytes"] > 3 * qwen["catalog"]["catalog_weights_gb"] * 1e9


def test_private_fleet_evidence_is_ignored():
    assert "/.local/" in (ROOT / ".gitignore").read_text().splitlines()
