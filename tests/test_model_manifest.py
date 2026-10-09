from __future__ import annotations

from dataclasses import replace

import pytest

from lokahi.core import (
    ArchitectureClass,
    ArchitectureSpec,
    ArtifactKind,
    AttentionPattern,
    AttentionSchedule,
    AttentionSpec,
    FeedForwardKind,
    FeedForwardSpec,
    PortableModelManifest,
    PromptProtocol,
    TokenizerSpec,
)


def gpt_oss_manifest() -> PortableModelManifest:
    return PortableModelManifest(
        schema_version=1,
        model_id="gpt-oss-20b",
        artifact_kind=ArtifactKind.LOCAL_WEIGHTS,
        artifact_revision="fixture-revision",
        artifact_digest="fixture-artifact-digest",
        weight_format="safetensors",
        quantization="mxfp4-moe",
        weights_bytes=12_800_000_000,
        architecture=ArchitectureSpec(
            model_type="gpt_oss",
            architecture_class=ArchitectureClass.SPARSE_MOE_TRANSFORMER,
            layer_count=24,
            hidden_size=2880,
            max_context_tokens=131_072,
            normalization="rms_norm_pre",
            attention=AttentionSpec(
                query_heads=64,
                kv_heads=8,
                head_dim=64,
                schedule=AttentionSchedule(
                    patterns=(AttentionPattern.BANDED, AttentionPattern.DENSE),
                    window_tokens=128,
                ),
                position_encoding="rope_yarn",
                rope_scaling="yarn",
                attention_sinks=1,
            ),
            feed_forward=FeedForwardSpec(
                kind=FeedForwardKind.SPARSE_MOE,
                intermediate_size=2880,
                activation="gated_swiglu_clamped",
                expert_count=32,
                active_expert_count=4,
                expert_weight_format="mxfp4",
            ),
        ),
        tokenizer=TokenizerSpec(
            tokenizer_id="o200k_harmony",
            revision="fixture-tokenizer-revision",
            digest="fixture-tokenizer-digest",
            vocab_size=201_088,
            prompt_protocol=PromptProtocol.HARMONY,
        ),
    )


def test_gpt_oss_manifest_exposes_semantics_without_model_name_branching() -> None:
    manifest = gpt_oss_manifest()

    assert manifest.architecture.feed_forward is not None
    assert manifest.architecture.feed_forward.active_expert_count == 4
    assert manifest.architecture.attention is not None
    assert manifest.architecture.attention.schedule.pattern_for_layer(0, 24) is AttentionPattern.BANDED
    assert manifest.architecture.attention.schedule.pattern_for_layer(1, 24) is AttentionPattern.DENSE
    assert len(manifest.identity_digest) == 64


def test_manifest_identity_changes_when_execution_semantics_change() -> None:
    manifest = gpt_oss_manifest()
    changed_architecture = replace(manifest.architecture, max_context_tokens=65_536)
    changed = replace(manifest, architecture=changed_architecture)

    assert changed.identity_digest != manifest.identity_digest


def test_manifest_json_round_trip_is_canonical() -> None:
    manifest = gpt_oss_manifest()
    decoded = PortableModelManifest.from_json(manifest.to_json())

    assert decoded == manifest
    assert decoded.to_json() == manifest.to_json()
    assert decoded.identity_digest == manifest.identity_digest


def test_manifest_json_rejects_unknown_semantic_fields() -> None:
    manifest = gpt_oss_manifest().to_dict()
    manifest["backend"] = "metal"

    with pytest.raises(ValueError, match="unknown=.*backend"):
        PortableModelManifest.from_dict(manifest)


def test_sparse_moe_requires_valid_router_cardinality() -> None:
    with pytest.raises(ValueError, match="active_expert_count"):
        FeedForwardSpec(
            kind=FeedForwardKind.SPARSE_MOE,
            intermediate_size=1024,
            activation="swiglu",
            expert_count=8,
            active_expert_count=9,
        )


def test_system_model_cannot_claim_local_weights() -> None:
    manifest = gpt_oss_manifest()
    system_architecture = ArchitectureSpec(
        model_type="apple_foundation_system",
        architecture_class=ArchitectureClass.SYSTEM_HOSTED,
    )

    with pytest.raises(ValueError, match="locally managed weights"):
        replace(
            manifest,
            artifact_kind=ArtifactKind.SYSTEM_MODEL,
            weights_bytes=1,
            architecture=system_architecture,
        )


def test_system_model_keeps_tokenizer_and_weights_opaque() -> None:
    manifest = PortableModelManifest(
        schema_version=1,
        model_id="apple-foundation-3b",
        artifact_kind=ArtifactKind.SYSTEM_MODEL,
        artifact_revision="fixture-os-build",
        artifact_digest="fixture-system-runtime-digest",
        weight_format="system-managed",
        quantization="system-managed",
        weights_bytes=0,
        architecture=ArchitectureSpec(
            model_type="apple_foundation_system",
            architecture_class=ArchitectureClass.SYSTEM_HOSTED,
        ),
        tokenizer=TokenizerSpec(
            tokenizer_id="apple-foundation-system",
            revision="fixture-os-build",
            digest="fixture-system-tokenizer-digest",
            vocab_size=None,
            prompt_protocol=PromptProtocol.SYSTEM_API,
        ),
    )

    assert manifest.weights_bytes == 0
    assert manifest.tokenizer.vocab_size is None
