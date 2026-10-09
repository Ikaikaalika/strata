"""Versioned semantic model contracts for architecture-driven planning."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Tuple, Type, TypeVar


EnumValue = TypeVar("EnumValue", bound=Enum)


class ArtifactKind(str, Enum):
    """How the runtime obtains execution semantics and weights."""

    LOCAL_WEIGHTS = "local_weights"
    SYSTEM_MODEL = "system_model"


class ArchitectureClass(str, Enum):
    """Top-level lowering strategy; never inferred from a catalog name."""

    DENSE_TRANSFORMER = "dense_transformer"
    SPARSE_MOE_TRANSFORMER = "sparse_moe_transformer"
    HYBRID_TRANSFORMER = "hybrid_transformer"
    SYSTEM_HOSTED = "system_hosted"


class AttentionPattern(str, Enum):
    DENSE = "dense"
    SLIDING_WINDOW = "sliding_window"
    BANDED = "banded"
    LINEAR = "linear"
    RECURRENT = "recurrent"


class FeedForwardKind(str, Enum):
    DENSE_GATED = "dense_gated"
    SPARSE_MOE = "sparse_moe"


class PromptProtocol(str, Enum):
    CHAT_TEMPLATE = "chat_template"
    HARMONY = "harmony"
    SYSTEM_API = "system_api"


@dataclass(frozen=True)
class AttentionSchedule:
    """A repeating or layer-exact attention schedule."""

    patterns: Tuple[AttentionPattern, ...]
    repeats: bool = True
    window_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ValueError("attention schedule must contain at least one pattern")
        windowed = {AttentionPattern.SLIDING_WINDOW, AttentionPattern.BANDED}
        if any(pattern in windowed for pattern in self.patterns):
            if self.window_tokens is None or self.window_tokens <= 0:
                raise ValueError("windowed attention requires positive window_tokens")
        elif self.window_tokens is not None:
            raise ValueError("window_tokens requires a windowed attention pattern")

    def pattern_for_layer(self, layer_index: int, layer_count: int) -> AttentionPattern:
        if layer_index < 0 or layer_index >= layer_count:
            raise IndexError("layer index is outside the architecture")
        if self.repeats:
            return self.patterns[layer_index % len(self.patterns)]
        if len(self.patterns) != layer_count:
            raise ValueError("non-repeating attention schedule must describe every layer")
        return self.patterns[layer_index]


@dataclass(frozen=True)
class AttentionSpec:
    query_heads: int
    kv_heads: int
    head_dim: int
    schedule: AttentionSchedule
    position_encoding: str
    rope_scaling: str | None = None
    attention_sinks: int = 0

    def __post_init__(self) -> None:
        if min(self.query_heads, self.kv_heads, self.head_dim) <= 0:
            raise ValueError("attention head geometry must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if not self.position_encoding:
            raise ValueError("position_encoding must not be empty")
        if self.attention_sinks < 0:
            raise ValueError("attention_sinks must be non-negative")


@dataclass(frozen=True)
class FeedForwardSpec:
    kind: FeedForwardKind
    intermediate_size: int
    activation: str
    expert_count: int | None = None
    active_expert_count: int | None = None
    shared_expert_count: int = 0
    expert_weight_format: str | None = None

    def __post_init__(self) -> None:
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if not self.activation:
            raise ValueError("activation must not be empty")
        if self.shared_expert_count < 0:
            raise ValueError("shared_expert_count must be non-negative")
        if self.kind is FeedForwardKind.SPARSE_MOE:
            if self.expert_count is None or self.expert_count <= 0:
                raise ValueError("sparse MoE requires a positive expert_count")
            if self.active_expert_count is None or not (
                0 < self.active_expert_count <= self.expert_count
            ):
                raise ValueError("sparse MoE requires a valid active_expert_count")
        elif any(
            value is not None
            for value in (self.expert_count, self.active_expert_count, self.expert_weight_format)
        ) or self.shared_expert_count:
            raise ValueError("dense feed-forward blocks cannot declare experts")


@dataclass(frozen=True)
class ArchitectureSpec:
    """Model semantics required to lower a model without name-based branching."""

    model_type: str
    architecture_class: ArchitectureClass
    layer_count: int | None = None
    hidden_size: int | None = None
    max_context_tokens: int | None = None
    normalization: str | None = None
    attention: AttentionSpec | None = None
    feed_forward: FeedForwardSpec | None = None
    feature_flags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_type:
            raise ValueError("model_type must not be empty")
        if len(self.feature_flags) != len(set(self.feature_flags)):
            raise ValueError("architecture feature flags must be unique")
        if self.architecture_class is ArchitectureClass.SYSTEM_HOSTED:
            if any(
                value is not None
                for value in (
                    self.layer_count,
                    self.hidden_size,
                    self.max_context_tokens,
                    self.normalization,
                    self.attention,
                    self.feed_forward,
                )
            ):
                raise ValueError("system-hosted architecture internals must remain opaque")
            return
        if any(
            value is None
            for value in (
                self.layer_count,
                self.hidden_size,
                self.max_context_tokens,
                self.normalization,
                self.attention,
                self.feed_forward,
            )
        ):
            raise ValueError("local architectures require complete execution semantics")
        assert self.layer_count is not None
        assert self.hidden_size is not None
        assert self.max_context_tokens is not None
        if min(self.layer_count, self.hidden_size, self.max_context_tokens) <= 0:
            raise ValueError("architecture dimensions must be positive")
        assert self.feed_forward is not None
        if (
            self.architecture_class is ArchitectureClass.SPARSE_MOE_TRANSFORMER
            and self.feed_forward.kind is not FeedForwardKind.SPARSE_MOE
        ):
            raise ValueError("sparse MoE architecture requires sparse MoE feed-forward semantics")
        if (
            self.architecture_class is ArchitectureClass.DENSE_TRANSFORMER
            and self.feed_forward.kind is not FeedForwardKind.DENSE_GATED
        ):
            raise ValueError("dense architecture requires dense feed-forward semantics")


@dataclass(frozen=True)
class TokenizerSpec:
    tokenizer_id: str
    revision: str
    digest: str
    vocab_size: int | None
    prompt_protocol: PromptProtocol
    template_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.tokenizer_id or not self.revision or not self.digest:
            raise ValueError("tokenizer identity must be pinned")
        if self.prompt_protocol is PromptProtocol.SYSTEM_API:
            if self.vocab_size is not None or self.template_digest is not None:
                raise ValueError("system API tokenizer internals must remain opaque")
            return
        if self.vocab_size is None or self.vocab_size <= 0:
            raise ValueError("local tokenizer vocab_size must be positive")
        if (
            self.prompt_protocol is PromptProtocol.CHAT_TEMPLATE
            and not self.template_digest
        ):
            raise ValueError("chat-template protocol requires template_digest")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = sorted(keys - value.keys())
    unknown = sorted(value.keys() - keys)
    if missing or unknown:
        raise ValueError(f"{name} keys mismatch: missing={missing}, unknown={unknown}")


def _integer(value: Any, name: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _text(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _enum(enum_type: Type[EnumValue], value: Any, name: str) -> EnumValue:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has unsupported value {value!r}") from exc


@dataclass(frozen=True)
class PortableModelManifest:
    """Immutable model input shared by standalone and Common Compute adapters."""

    schema_version: int
    model_id: str
    artifact_kind: ArtifactKind
    artifact_revision: str
    artifact_digest: str
    weight_format: str
    quantization: str
    weights_bytes: int
    architecture: ArchitectureSpec
    tokenizer: TokenizerSpec
    capabilities: Tuple[str, ...] = ("text.generate",)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported portable model manifest schema")
        if not all(
            (self.model_id, self.artifact_revision, self.artifact_digest, self.weight_format)
        ):
            raise ValueError("model artifact identity must be pinned")
        if not self.quantization:
            raise ValueError("quantization must not be empty")
        if not self.capabilities or len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("model capabilities must be non-empty and unique")
        if self.artifact_kind is ArtifactKind.SYSTEM_MODEL:
            if self.weights_bytes != 0:
                raise ValueError("system models cannot declare locally managed weights")
            if self.architecture.architecture_class is not ArchitectureClass.SYSTEM_HOSTED:
                raise ValueError("system model requires a system-hosted architecture")
            if self.tokenizer.prompt_protocol is not PromptProtocol.SYSTEM_API:
                raise ValueError("system model requires the system API prompt protocol")
        else:
            if self.weights_bytes <= 0:
                raise ValueError("local model weights must have positive byte size")
            if self.architecture.architecture_class is ArchitectureClass.SYSTEM_HOSTED:
                raise ValueError("local weights cannot use a system-hosted architecture")
            if self.tokenizer.prompt_protocol is PromptProtocol.SYSTEM_API:
                raise ValueError("local weights cannot use the system API prompt protocol")

    @property
    def identity_digest(self) -> str:
        """Stable identity for cache, plan, and evidence keys."""

        payload = json.dumps(
            _json_value(asdict(self)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible schema representation."""

        return _json_value(asdict(self))

    def to_json(self) -> str:
        """Return compact canonical JSON used by cross-language golden tests."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "PortableModelManifest":
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model manifest is not valid JSON: {exc}") from exc
        return cls.from_dict(_object(decoded, "manifest"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortableModelManifest":
        """Decode schema v1 strictly so semantic drift fails closed."""

        top_keys = {
            "schema_version",
            "model_id",
            "artifact_kind",
            "artifact_revision",
            "artifact_digest",
            "weight_format",
            "quantization",
            "weights_bytes",
            "architecture",
            "tokenizer",
            "capabilities",
        }
        _exact_keys(payload, top_keys, "manifest")

        raw_architecture = _object(payload["architecture"], "architecture")
        architecture_keys = {
            "model_type",
            "architecture_class",
            "layer_count",
            "hidden_size",
            "max_context_tokens",
            "normalization",
            "attention",
            "feed_forward",
            "feature_flags",
        }
        _exact_keys(raw_architecture, architecture_keys, "architecture")

        attention = None
        if raw_architecture["attention"] is not None:
            raw_attention = _object(raw_architecture["attention"], "attention")
            attention_keys = {
                "query_heads",
                "kv_heads",
                "head_dim",
                "schedule",
                "position_encoding",
                "rope_scaling",
                "attention_sinks",
            }
            _exact_keys(raw_attention, attention_keys, "attention")
            raw_schedule = _object(raw_attention["schedule"], "attention.schedule")
            _exact_keys(
                raw_schedule,
                {"patterns", "repeats", "window_tokens"},
                "attention.schedule",
            )
            raw_patterns = raw_schedule["patterns"]
            if not isinstance(raw_patterns, list):
                raise ValueError("attention.schedule.patterns must be an array")
            repeats = raw_schedule["repeats"]
            if not isinstance(repeats, bool):
                raise ValueError("attention.schedule.repeats must be a boolean")
            attention = AttentionSpec(
                query_heads=int(_integer(raw_attention["query_heads"], "attention.query_heads")),
                kv_heads=int(_integer(raw_attention["kv_heads"], "attention.kv_heads")),
                head_dim=int(_integer(raw_attention["head_dim"], "attention.head_dim")),
                schedule=AttentionSchedule(
                    patterns=tuple(
                        _enum(AttentionPattern, pattern, "attention pattern")
                        for pattern in raw_patterns
                    ),
                    repeats=repeats,
                    window_tokens=_integer(
                        raw_schedule["window_tokens"],
                        "attention.schedule.window_tokens",
                        optional=True,
                    ),
                ),
                position_encoding=str(
                    _text(raw_attention["position_encoding"], "attention.position_encoding")
                ),
                rope_scaling=_text(
                    raw_attention["rope_scaling"],
                    "attention.rope_scaling",
                    optional=True,
                ),
                attention_sinks=int(
                    _integer(raw_attention["attention_sinks"], "attention.attention_sinks")
                ),
            )

        feed_forward = None
        if raw_architecture["feed_forward"] is not None:
            raw_feed_forward = _object(
                raw_architecture["feed_forward"], "feed_forward"
            )
            feed_forward_keys = {
                "kind",
                "intermediate_size",
                "activation",
                "expert_count",
                "active_expert_count",
                "shared_expert_count",
                "expert_weight_format",
            }
            _exact_keys(raw_feed_forward, feed_forward_keys, "feed_forward")
            feed_forward = FeedForwardSpec(
                kind=_enum(FeedForwardKind, raw_feed_forward["kind"], "feed_forward.kind"),
                intermediate_size=int(
                    _integer(
                        raw_feed_forward["intermediate_size"],
                        "feed_forward.intermediate_size",
                    )
                ),
                activation=str(
                    _text(raw_feed_forward["activation"], "feed_forward.activation")
                ),
                expert_count=_integer(
                    raw_feed_forward["expert_count"],
                    "feed_forward.expert_count",
                    optional=True,
                ),
                active_expert_count=_integer(
                    raw_feed_forward["active_expert_count"],
                    "feed_forward.active_expert_count",
                    optional=True,
                ),
                shared_expert_count=int(
                    _integer(
                        raw_feed_forward["shared_expert_count"],
                        "feed_forward.shared_expert_count",
                    )
                ),
                expert_weight_format=_text(
                    raw_feed_forward["expert_weight_format"],
                    "feed_forward.expert_weight_format",
                    optional=True,
                ),
            )

        feature_flags = raw_architecture["feature_flags"]
        if not isinstance(feature_flags, list) or not all(
            isinstance(value, str) and value for value in feature_flags
        ):
            raise ValueError("architecture.feature_flags must be a string array")
        architecture = ArchitectureSpec(
            model_type=str(_text(raw_architecture["model_type"], "architecture.model_type")),
            architecture_class=_enum(
                ArchitectureClass,
                raw_architecture["architecture_class"],
                "architecture.architecture_class",
            ),
            layer_count=_integer(
                raw_architecture["layer_count"], "architecture.layer_count", optional=True
            ),
            hidden_size=_integer(
                raw_architecture["hidden_size"], "architecture.hidden_size", optional=True
            ),
            max_context_tokens=_integer(
                raw_architecture["max_context_tokens"],
                "architecture.max_context_tokens",
                optional=True,
            ),
            normalization=_text(
                raw_architecture["normalization"],
                "architecture.normalization",
                optional=True,
            ),
            attention=attention,
            feed_forward=feed_forward,
            feature_flags=tuple(feature_flags),
        )

        raw_tokenizer = _object(payload["tokenizer"], "tokenizer")
        tokenizer_keys = {
            "tokenizer_id",
            "revision",
            "digest",
            "vocab_size",
            "prompt_protocol",
            "template_digest",
        }
        _exact_keys(raw_tokenizer, tokenizer_keys, "tokenizer")
        tokenizer = TokenizerSpec(
            tokenizer_id=str(_text(raw_tokenizer["tokenizer_id"], "tokenizer.tokenizer_id")),
            revision=str(_text(raw_tokenizer["revision"], "tokenizer.revision")),
            digest=str(_text(raw_tokenizer["digest"], "tokenizer.digest")),
            vocab_size=_integer(
                raw_tokenizer["vocab_size"], "tokenizer.vocab_size", optional=True
            ),
            prompt_protocol=_enum(
                PromptProtocol,
                raw_tokenizer["prompt_protocol"],
                "tokenizer.prompt_protocol",
            ),
            template_digest=_text(
                raw_tokenizer["template_digest"],
                "tokenizer.template_digest",
                optional=True,
            ),
        )

        capabilities = payload["capabilities"]
        if not isinstance(capabilities, list) or not all(
            isinstance(value, str) and value for value in capabilities
        ):
            raise ValueError("manifest.capabilities must be a string array")
        return cls(
            schema_version=int(_integer(payload["schema_version"], "manifest.schema_version")),
            model_id=str(_text(payload["model_id"], "manifest.model_id")),
            artifact_kind=_enum(
                ArtifactKind, payload["artifact_kind"], "manifest.artifact_kind"
            ),
            artifact_revision=str(
                _text(payload["artifact_revision"], "manifest.artifact_revision")
            ),
            artifact_digest=str(
                _text(payload["artifact_digest"], "manifest.artifact_digest")
            ),
            weight_format=str(_text(payload["weight_format"], "manifest.weight_format")),
            quantization=str(_text(payload["quantization"], "manifest.quantization")),
            weights_bytes=int(_integer(payload["weights_bytes"], "manifest.weights_bytes")),
            architecture=architecture,
            tokenizer=tokenizer,
            capabilities=tuple(capabilities),
        )
