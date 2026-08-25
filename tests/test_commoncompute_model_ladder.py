from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.commoncompute_model_ladder import (
    ModelLadderError,
    load_model_ladder,
    model_entry,
)


LADDER = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "targets"
    / "commoncompute_m1_model_ladder_v1.json"
)


def test_checked_in_ladder_prioritizes_current_m1_models() -> None:
    ladder = load_model_ladder(LADDER)

    assert len(ladder["local_priority"]) == 8
    assert model_entry(ladder, "qwen3-0.6b")["priority"] == 1
    assert model_entry(ladder, "llama-3.2-3b")["purpose"].startswith(
        "Common Compute default"
    )
    assert all(model["min_memory_gb"] <= 16 for model in ladder["local_priority"])


def test_darkbloom_revision_compatibility_is_explicit() -> None:
    ladder = load_model_ladder(LADDER)
    targets = {model["model_id"]: model for model in ladder["darkbloom_remote_targets"]}

    assert targets["gpt-oss-20b"]["exact_revision_match"] is True
    assert targets["gemma-4-26b-a4b"]["exact_revision_match"] is False


def test_ladder_rejects_revision_match_drift(tmp_path: Path) -> None:
    payload = json.loads(LADDER.read_text(encoding="utf-8"))
    payload["darkbloom_remote_targets"][1]["exact_revision_match"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelLadderError, match="inconsistent"):
        load_model_ladder(changed)
