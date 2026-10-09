from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.generated_model_fixtures import (
    generated_dense_manifest,
    generated_gpt_oss_20b_manifest,
)
from lokahi.core import AttentionPattern, AttentionSchedule
from lokahi.planning import (
    estimate_kv_cache_bytes,
    estimate_weight_bytes_per_decode_token,
)


def test_dense_paging_models_the_complete_cyclic_layer_scan() -> None:
    assert (
        estimate_weight_bytes_per_decode_token(
            generated_dense_manifest(),
            rolling_weight_bytes=547_487_744,
        )
        == 547_487_744
    )


def test_sparse_moe_paging_uses_selected_expert_working_set() -> None:
    manifest = generated_gpt_oss_20b_manifest()
    assert (
        estimate_weight_bytes_per_decode_token(
            manifest,
            rolling_weight_bytes=10_000,
            selected_expert_bytes=2_000,
        )
        == 2_000
    )
    with pytest.raises(ValueError, match="selected_expert_bytes"):
        estimate_weight_bytes_per_decode_token(
            manifest,
            rolling_weight_bytes=10_000,
        )


def test_gpt_oss_schedule_reduces_logical_long_context_kv() -> None:
    manifest = generated_gpt_oss_20b_manifest()
    estimate = estimate_kv_cache_bytes(
        manifest,
        batch_size=1,
        context_tokens=131_072,
    )

    assert estimate.dense_layer_count == 12
    assert estimate.windowed_layer_count == 12
    assert estimate.total_bytes == 3_224_371_200
    assert estimate.all_dense_bytes == 6_442_450_944
    assert estimate.fraction_saved_vs_all_dense == pytest.approx(0.49951171875)


def test_kv_estimate_rejects_context_above_manifest_limit() -> None:
    manifest = generated_gpt_oss_20b_manifest()

    with pytest.raises(ValueError, match="exceeds"):
        estimate_kv_cache_bytes(
            manifest,
            batch_size=1,
            context_tokens=131_073,
        )


def test_kv_estimate_rejects_unmodeled_recurrent_state() -> None:
    manifest = generated_gpt_oss_20b_manifest()
    assert manifest.architecture.attention is not None
    recurrent_attention = replace(
        manifest.architecture.attention,
        schedule=AttentionSchedule(patterns=(AttentionPattern.RECURRENT,)),
    )
    recurrent = replace(
        manifest,
        architecture=replace(manifest.architecture, attention=recurrent_attention),
    )

    with pytest.raises(ValueError, match="state schema"):
        estimate_kv_cache_bytes(recurrent, batch_size=1, context_tokens=128)
