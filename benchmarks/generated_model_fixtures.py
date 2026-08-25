"""Small/generated model manifests for offline benchmark contracts."""
from __future__ import annotations

from ollm.core import (
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


def generated_dense_manifest() -> PortableModelManifest:
    return PortableModelManifest(
        schema_version=1,
        model_id="generated-dense-v1",
        artifact_kind=ArtifactKind.LOCAL_WEIGHTS,
        artifact_revision="generated-fixture-v1",
        artifact_digest="generated-fixture-no-real-weights",
        weight_format="generated",
        quantization="float16",
        weights_bytes=8 * 1024 * 1024,
        architecture=ArchitectureSpec(
            model_type="generated_dense",
            architecture_class=ArchitectureClass.DENSE_TRANSFORMER,
            layer_count=4,
            hidden_size=256,
            max_context_tokens=8192,
            normalization="rms_norm_pre",
            attention=AttentionSpec(
                query_heads=8,
                kv_heads=2,
                head_dim=32,
                schedule=AttentionSchedule(patterns=(AttentionPattern.DENSE,)),
                position_encoding="rope",
            ),
            feed_forward=FeedForwardSpec(
                kind=FeedForwardKind.DENSE_GATED,
                intermediate_size=512,
                activation="swiglu",
            ),
        ),
        tokenizer=TokenizerSpec(
            tokenizer_id="generated-tokenizer",
            revision="generated-fixture-v1",
            digest="generated-tokenizer-no-real-vocabulary",
            vocab_size=256,
            prompt_protocol=PromptProtocol.CHAT_TEMPLATE,
            template_digest="generated-template-v1",
        ),
    )


def generated_gpt_oss_20b_manifest() -> PortableModelManifest:
    return PortableModelManifest(
        schema_version=1,
        model_id="generated-gpt-oss-20b-v1",
        artifact_kind=ArtifactKind.LOCAL_WEIGHTS,
        artifact_revision="generated-fixture-v1",
        artifact_digest="generated-fixture-no-real-gpt-oss-weights",
        weight_format="generated-safetensors-layout",
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
            revision="generated-fixture-v1",
            digest="generated-tokenizer-no-real-vocabulary",
            vocab_size=201_088,
            prompt_protocol=PromptProtocol.HARMONY,
        ),
    )

