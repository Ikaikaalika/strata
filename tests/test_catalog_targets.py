"""Catalog metadata consistency and fail-closed qualification boundaries."""
import copy
import json

import pytest

from benchmarks.catalog_targets import DEFAULT_TARGET, exact_catalog_pair, load_catalog_targets


def test_current_snapshot_and_proposal_are_distinct():
    target = load_catalog_targets()
    assert len(target["catalog_models"]) == 40
    assert sum(m["workloadId"] == "mlx_llm" for m in target["catalog_models"]) == 30
    assert sum(m["workloadId"] == "mlx_vlm" for m in target["catalog_models"]) == 10
    assert len(target["recommended_targets"]) == 20
    additions = {m["model_id"] for m in target["recommended_targets"] if m["catalog_state"] == "proposed_addition"}
    assert additions == {"gemma-4-e4b", "ministral-3-8b", "glm-4.7-flash", "glm-ocr"}
    assert target["native_candidate"]["output_dtype"] == "float16"


def test_exact_workload_identity_never_aliases_or_enables_proposals():
    target = load_catalog_targets()
    assert exact_catalog_pair(target, "qwen3-4b", "mlx_llm")["id"] == "qwen3-4b"
    for model_id, workload in (("qwen3-4b", "mlx_vlm"), ("qwen", "mlx_llm"), ("glm-4.7-flash", "mlx_llm")):
        with pytest.raises(ValueError, match="unobserved"):
            exact_catalog_pair(target, model_id, workload)


@pytest.mark.parametrize("mutation", [
    lambda c: c["catalog_models"].append(copy.deepcopy(c["catalog_models"][0])),
    lambda c: c["artifact_pins"].pop(),
    lambda c: c["artifact_pins"][0].update(revision="main"),
    lambda c: c["recommended_targets"][0].update(pin=None),
    lambda c: c["recommended_targets"][0]["pin"].update(repo="different/model"),
    lambda c: c["recommended_targets"][0]["pin"].update(revision="0" * 40),
    lambda c: c["recommended_targets"][0].update(native_full_model_qualified=True),
    lambda c: c["recommended_targets"][0].update(catalog_state="proposed_addition"),
    lambda c: c["recommended_targets"][-1].update(catalog_state="observed_preview"),
    lambda c: c["policy"].update(metadata_can_promote_runtime=True),
    lambda c: c["policy"].update(ssd_offload="auto"),
    lambda c: c["policy"].update(baseline_runtimes={"mlx_llm": "mlx_lm", "mlx_vlm": "mlx_lm"}),
    lambda c: c["native_candidate"].update(promotion_eligible=True),
    lambda c: c["experimental_targets"][0].update(native_full_model_qualified=True),
    lambda c: c["catalog_models"][0].update(minMemoryGb=True),
    lambda c: c["catalog_models"][0].update(weightsGb=-1),
])
def test_metadata_cannot_become_execution_evidence(tmp_path, mutation):
    payload = json.loads(DEFAULT_TARGET.read_text())
    mutation(payload)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_catalog_targets(path)
