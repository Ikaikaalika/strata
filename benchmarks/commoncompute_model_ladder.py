"""Validation and lookup for the Common Compute model qualification ladder."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


class ModelLadderError(ValueError):
    pass


_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def load_model_ladder(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelLadderError(f"could not load model ladder {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ModelLadderError("model ladder must be a schema-v1 object")
    models = payload.get("local_priority")
    if not isinstance(models, list) or not models:
        raise ModelLadderError("local_priority must be a non-empty array")

    ids: set[str] = set()
    priorities: set[int] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ModelLadderError("local_priority entries must be objects")
        model_id = model.get("model_id")
        priority = model.get("priority")
        revision = model.get("revision")
        if not isinstance(model_id, str) or not model_id:
            raise ModelLadderError("every local model requires model_id")
        if model_id in ids:
            raise ModelLadderError(f"duplicate local model_id {model_id!r}")
        ids.add(model_id)
        if isinstance(priority, bool) or not isinstance(priority, int) or priority <= 0:
            raise ModelLadderError(f"model {model_id!r} has invalid priority")
        if priority in priorities:
            raise ModelLadderError(f"duplicate model priority {priority}")
        priorities.add(priority)
        if not isinstance(revision, str) or _SHA40.fullmatch(revision) is None:
            raise ModelLadderError(f"model {model_id!r} revision is not a 40-hex SHA")
        if model.get("status") != "preview":
            raise ModelLadderError(f"local model {model_id!r} is not preview")
        if model.get("min_memory_gb") > 16:
            raise ModelLadderError(f"local model {model_id!r} exceeds the M1 memory tier")

    if priorities != set(range(1, len(models) + 1)):
        raise ModelLadderError("local priorities must be contiguous starting at one")

    remote = payload.get("darkbloom_remote_targets")
    if not isinstance(remote, list) or not remote:
        raise ModelLadderError("darkbloom_remote_targets must be a non-empty array")
    for model in remote:
        common_compute = model.get("commoncompute_revision")
        darkbloom = model.get("darkbloom_revision")
        for name, revision in (
            ("commoncompute_revision", common_compute),
            ("darkbloom_revision", darkbloom),
        ):
            if not isinstance(revision, str) or _SHA40.fullmatch(revision) is None:
                raise ModelLadderError(f"{model.get('model_id')!r} has invalid {name}")
        if model.get("exact_revision_match") is not (common_compute == darkbloom):
            raise ModelLadderError(
                f"{model.get('model_id')!r} exact_revision_match is inconsistent"
            )
    return payload


def model_entry(ladder: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for model in ladder["local_priority"]:
        if model["model_id"] == model_id:
            return model
    raise ModelLadderError(f"model {model_id!r} is not in the local ladder")
