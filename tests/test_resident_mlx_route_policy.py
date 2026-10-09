from __future__ import annotations

import json
from pathlib import Path

from lokahi.runtime.resident_mlx import (
    ResidentMLXRoute,
    load_resident_mlx_profiles,
    select_resident_mlx_route,
)


ROOT = Path(__file__).parents[1]
PROFILES = ROOT / "benchmarks/targets/apple_m1_resident_mlx_route_profiles_v1.json"


def _select(model_id: str, revision: str, **overrides):
    values = {
        "target_id": "commoncompute-apple-m1-16gb-model-ladder-v1",
        "model_id": model_id,
        "revision": revision,
        "mlx_version": "0.31.1",
        "mlx_lm_version": "0.31.2",
        "prompt_tokens": 512,
        "maximum_output_tokens": 128,
        "batch_size": 1,
        "active_sequences": 1,
        "ssd_offload_disabled": True,
    }
    values.update(overrides)
    return select_resident_mlx_route(load_resident_mlx_profiles(PROFILES), **values)


def test_qwen_and_gemma_use_validated_direct_route() -> None:
    qwen = _select("qwen3-0.6b", "73e3e38d981303bc594367cd910ea6eb48349da8")
    gemma = _select("gemma-3-1b", "15fed4eafb456c6fcb2a1165f19ac609670ed14b")

    assert qwen == (ResidentMLXRoute.DIRECT_SINGLE_SEQUENCE, 2048)
    assert gemma == (ResidentMLXRoute.DIRECT_SINGLE_SEQUENCE, 2048)


def test_llama_and_unvalidated_envelopes_fail_to_batch_route() -> None:
    llama = _select("llama-3.2-1b", "08231374eeacb049a0eade7922910865b8fce912")
    short_llama = _select(
        "llama-3.2-1b",
        "08231374eeacb049a0eade7922910865b8fce912",
        maximum_output_tokens=32,
    )
    concurrent = _select(
        "qwen3-0.6b",
        "73e3e38d981303bc594367cd910ea6eb48349da8",
        active_sequences=2,
    )
    paged = _select(
        "qwen3-0.6b",
        "73e3e38d981303bc594367cd910ea6eb48349da8",
        ssd_offload_disabled=False,
    )
    wrong_shape = _select(
        "gemma-3-1b",
        "15fed4eafb456c6fcb2a1165f19ac609670ed14b",
        prompt_tokens=2048,
    )

    assert llama[0] is ResidentMLXRoute.BATCH_GENERATOR
    assert short_llama[0] is ResidentMLXRoute.DIRECT_SINGLE_SEQUENCE
    assert concurrent[0] is ResidentMLXRoute.BATCH_GENERATOR
    assert paged[0] is ResidentMLXRoute.BATCH_GENERATOR
    assert wrong_shape[0] is ResidentMLXRoute.BATCH_GENERATOR


def test_profile_metrics_match_checked_in_hardware_evidence() -> None:
    for profile in load_resident_mlx_profiles(PROFILES):
        evidence = json.loads((ROOT / profile.evidence_path).read_text(encoding="utf-8"))
        comparison = evidence["comparison"]
        assert evidence["evidence_kind"] == "hardware"
        assert comparison["exact_greedy_token_parity"] is profile.exact_greedy_token_parity
        measured = comparison["median_improvement_percent"][
            "end_to_end_output_tokens_per_second"
        ]
        assert measured == profile.end_to_end_improvement_percent
        assert comparison["batch_one_route_promotion_passed"] is True
