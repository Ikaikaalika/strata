"""Evidence-gated resident MLX route selection and token iteration.

The direct route removes continuous-batching bookkeeping for a single request.
It is not a new compute backend: both routes execute through the pinned MLX-LM
compatibility backend on the GPU.  Unknown models, shapes, versions, or
concurrent workloads fail safely to ``BatchGenerator``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


class ResidentMLXRoute(str, Enum):
    BATCH_GENERATOR = "batch_generator"
    DIRECT_SINGLE_SEQUENCE = "direct_single_sequence"


@dataclass(frozen=True)
class ResidentMLXRouteProfile:
    target_id: str
    model_id: str
    revision: str
    mlx_version: str
    mlx_lm_version: str
    prompt_tokens: int
    output_tokens: int
    prefill_step_size: int
    route: ResidentMLXRoute
    exact_greedy_token_parity: bool
    end_to_end_improvement_percent: float
    evidence_path: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResidentMLXRouteProfile":
        required = {
            "target_id",
            "model_id",
            "revision",
            "mlx_version",
            "mlx_lm_version",
            "prompt_tokens",
            "output_tokens",
            "prefill_step_size",
            "route",
            "exact_greedy_token_parity",
            "end_to_end_improvement_percent",
            "evidence_path",
        }
        missing = required - set(value)
        unknown = set(value) - required
        if missing or unknown:
            raise ValueError(
                f"invalid resident MLX profile keys; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        profile = cls(
            target_id=str(value["target_id"]),
            model_id=str(value["model_id"]),
            revision=str(value["revision"]),
            mlx_version=str(value["mlx_version"]),
            mlx_lm_version=str(value["mlx_lm_version"]),
            prompt_tokens=int(value["prompt_tokens"]),
            output_tokens=int(value["output_tokens"]),
            prefill_step_size=int(value["prefill_step_size"]),
            route=ResidentMLXRoute(str(value["route"])),
            exact_greedy_token_parity=value["exact_greedy_token_parity"],
            end_to_end_improvement_percent=float(
                value["end_to_end_improvement_percent"]
            ),
            evidence_path=str(value["evidence_path"]),
        )
        if not all(
            (
                profile.target_id,
                profile.model_id,
                profile.revision,
                profile.mlx_version,
                profile.mlx_lm_version,
                profile.evidence_path,
            )
        ):
            raise ValueError("resident MLX profile strings must not be empty")
        if min(
            profile.prompt_tokens,
            profile.output_tokens,
            profile.prefill_step_size,
        ) <= 0:
            raise ValueError("resident MLX profile dimensions must be positive")
        if not isinstance(profile.exact_greedy_token_parity, bool):
            raise TypeError("exact_greedy_token_parity must be a boolean")
        return profile


def load_resident_mlx_profiles(path: Path) -> tuple[ResidentMLXRouteProfile, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "profiles"}:
        raise ValueError("resident MLX profile document has invalid top-level keys")
    if payload["schema_version"] != 1 or not isinstance(payload["profiles"], list):
        raise ValueError("unsupported resident MLX profile schema")
    profiles = tuple(
        ResidentMLXRouteProfile.from_mapping(item) for item in payload["profiles"]
    )
    identities = [
        (
            profile.target_id,
            profile.model_id,
            profile.revision,
            profile.prompt_tokens,
            profile.output_tokens,
        )
        for profile in profiles
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("resident MLX profiles must have unique identities")
    return profiles


def select_resident_mlx_route(
    profiles: Sequence[ResidentMLXRouteProfile],
    *,
    target_id: str,
    model_id: str,
    revision: str,
    mlx_version: str,
    mlx_lm_version: str,
    prompt_tokens: int,
    maximum_output_tokens: int,
    batch_size: int,
    active_sequences: int,
    ssd_offload_disabled: bool,
    minimum_improvement_percent: float = 2.0,
) -> tuple[ResidentMLXRoute, int]:
    """Select a validated batch-one route or return the safe batch fallback.

    Returns ``(route, prefill_step_size)``.  A step size of 2048 is the pinned
    MLX-LM default when no direct-route profile qualifies.
    """

    if min(prompt_tokens, maximum_output_tokens, batch_size, active_sequences) <= 0:
        raise ValueError("request dimensions and sequence counts must be positive")
    if minimum_improvement_percent < 0:
        raise ValueError("minimum_improvement_percent must not be negative")
    fallback = (ResidentMLXRoute.BATCH_GENERATOR, 2048)
    if not ssd_offload_disabled or batch_size != 1 or active_sequences != 1:
        return fallback
    for profile in profiles:
        if (
            profile.target_id == target_id
            and profile.model_id == model_id
            and profile.revision == revision
            and profile.mlx_version == mlx_version
            and profile.mlx_lm_version == mlx_lm_version
            and profile.prompt_tokens == prompt_tokens
            and maximum_output_tokens == profile.output_tokens
            and profile.route is ResidentMLXRoute.DIRECT_SINGLE_SEQUENCE
            and profile.exact_greedy_token_parity
            and profile.end_to_end_improvement_percent
            >= minimum_improvement_percent
        ):
            return profile.route, profile.prefill_step_size
    return fallback


def iter_resident_mlx_tokens(
    *,
    route: ResidentMLXRoute,
    model: Any,
    prompt: Sequence[int],
    maximum_output_tokens: int,
    sampler: Any,
    prefill_step_size: int = 2048,
) -> Iterator[int]:
    """Execute one selected route without importing MLX until it is used."""

    if not prompt or maximum_output_tokens <= 0 or prefill_step_size <= 0:
        raise ValueError("prompt, output count, and prefill step must be positive")
    if route is ResidentMLXRoute.DIRECT_SINGLE_SEQUENCE:
        import mlx.core as mx
        from mlx_lm.generate import generate_step, wired_limit

        with wired_limit(model):
            for token, _ in generate_step(
                mx.array(prompt),
                model,
                max_tokens=maximum_output_tokens,
                sampler=sampler,
                prefill_step_size=prefill_step_size,
                kv_bits=None,
            ):
                yield int(token)
        return

    if route is ResidentMLXRoute.BATCH_GENERATOR:
        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator(
            model,
            max_tokens=maximum_output_tokens,
            stop_tokens=None,
            sampler=sampler,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=prefill_step_size,
        )
        uid = generator.insert([list(prompt)], [maximum_output_tokens])[0]
        try:
            while responses := generator.next_generated():
                for response in responses:
                    if response.uid == uid:
                        yield int(response.token)
        finally:
            generator.close()
        return
    raise ValueError(f"unsupported resident MLX route: {route!r}")


__all__ = [
    "ResidentMLXRoute",
    "ResidentMLXRouteProfile",
    "iter_resident_mlx_tokens",
    "load_resident_mlx_profiles",
    "select_resident_mlx_route",
]
