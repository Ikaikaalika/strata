"""Offline CommonCompute catalog targets; metadata never grants native support."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

DEFAULT_TARGET = Path(__file__).parent / "targets/commoncompute_catalog_20260930.json"
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")


def load_catalog_targets(path: Path = DEFAULT_TARGET) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected catalog target schema v1")
    if payload.get("evidence_kind") != "public_catalog_metadata":
        raise ValueError("catalog evidence must remain metadata")
    policy = payload["policy"]
    if any(policy.get(key) is not False for key in (
        "native_runtime_qualified", "metadata_can_promote_runtime",
        "component_results_can_promote_runtime", "automatic_catalog_retirement",
    )) or policy.get("ssd_offload") != "disabled":
        raise ValueError("catalog target cannot promote or retire a runtime")
    if policy.get("baseline_runtimes") != {"mlx_llm": "mlx_lm", "mlx_vlm": "mlx_vlm"}:
        raise ValueError("text and vision require their own MLX controls")
    observed: set[tuple[str, str]] = set()
    for model in payload["catalog_models"]:
        pair = (model["id"], model["workloadId"])
        if not pair[0] or pair[1] not in {"mlx_llm", "mlx_vlm"} or pair in observed:
            raise ValueError("invalid or duplicate exact catalog pair")
        observed.add(pair)
        for field in ("minMemoryGb", "weightsGb"):
            value = model[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("catalog memory fields must be finite positive numbers")
    if not observed:
        raise ValueError("catalog must be nonempty")
    pins: dict[tuple[str, str], dict[str, str]] = {}
    for pin in payload["artifact_pins"]:
        pair = (pin["model_id"], pin["workload_id"])
        if pair not in observed or pair in pins or not pin["repo"] or not _SHA40.fullmatch(pin["revision"]):
            raise ValueError("observed pairs require unique source artifact pins")
        pins[pair] = {"repo": pin["repo"], "revision": pin["revision"]}
    if set(pins) != observed:
        raise ValueError("artifact pins do not cover the observed catalog")
    recommended: set[tuple[str, str]] = set()
    for target in payload["recommended_targets"]:
        pair = (target["model_id"], target["workload_id"])
        if not pair[0] or pair[1] not in {"mlx_llm", "mlx_vlm"} or pair in recommended:
            raise ValueError("invalid or duplicate recommended pair")
        recommended.add(pair)
        expected = "observed_preview" if pair in observed else "proposed_addition"
        if target["catalog_state"] != expected or target["native_full_model_qualified"] is not False:
            raise ValueError("recommendations cannot imply observed availability or native qualification")
        pin = target["pin"]
        if pin is not None and (not pin["repo"] or not _SHA40.fullmatch(pin["revision"])):
            raise ValueError("invalid recommended artifact pin")
        if pair in observed and pin != pins[pair]:
            raise ValueError("recommended observed pair must retain its source artifact pin")
    if not recommended or payload["native_candidate"]["promotion_eligible"] is not False:
        raise ValueError("component target cannot promote a model")
    for target in payload["experimental_targets"]:
        if target["native_full_model_qualified"] is not False:
            raise ValueError("experimental targets cannot imply native qualification")
    return payload


def exact_catalog_pair(payload: dict[str, Any], model_id: str, workload_id: str) -> dict[str, Any]:
    for model in payload["catalog_models"]:
        if (model["id"], model["workloadId"]) == (model_id, workload_id):
            return model
    raise ValueError(f"unobserved catalog pair: {model_id}/{workload_id}")
