from __future__ import annotations

from pathlib import Path

import pytest

from ollm.core.ssd_offload import SSDOffloadMode, SSDOffloadPolicy
from ollm.runtime.paged_mlx_llama import (
    PagedMLXLlamaRuntime,
    maximum_pinned_layer_count,
    select_pinned_layer_group_ids,
)


class _Group:
    def __init__(self, group_id: str, nbytes: int) -> None:
        self.group_id = group_id
        self.nbytes = nbytes


def test_pinned_layer_selection_preserves_current_and_prefetch_window() -> None:
    groups = tuple(_Group(f"layer.{index}", 10) for index in range(6))

    assert select_pinned_layer_group_ids(
        groups,
        count=2,
        layer_budget_bytes=40,
        prefetch_distance=1,
    ) == ("layer.0", "layer.1")
    with pytest.raises(ValueError, match="insufficient"):
        select_pinned_layer_group_ids(
            groups,
            count=3,
            layer_budget_bytes=40,
            prefetch_distance=1,
        )

    assert maximum_pinned_layer_count(
        groups,
        layer_budget_bytes=40,
        prefetch_distance=1,
    ) == 2


def test_paged_runtime_rejects_disabled_policy_before_model_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires SSD offload"):
        PagedMLXLlamaRuntime.load(tmp_path, SSDOffloadPolicy())


def test_paged_runtime_requires_explicit_resident_weight_cap(tmp_path: Path) -> None:
    policy = SSDOffloadPolicy(mode=SSDOffloadMode.REQUIRED)
    with pytest.raises(ValueError, match="max_resident_weight_bytes"):
        PagedMLXLlamaRuntime.load(tmp_path, policy)


def test_paged_runtime_rejects_hdd_model_path() -> None:
    policy = SSDOffloadPolicy(
        mode=SSDOffloadMode.REQUIRED,
        max_resident_weight_bytes=512 * 1024 * 1024,
    )
    with pytest.raises(ValueError, match="Tyler HDD"):
        PagedMLXLlamaRuntime.load(Path.cwd(), policy)
