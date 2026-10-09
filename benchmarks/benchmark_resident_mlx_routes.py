#!/usr/bin/env python3
"""Compare resident MLX execution routes for one pinned Common Compute model.

This is a paired hardware benchmark.  It loads the model once with ``lazy=False``
and alternates run order between MLX-LM's continuous ``BatchGenerator`` control
and Lokahi's batch-one route using ``generate_step`` directly.  No route uses
Lokahi-managed or OS-managed SSD offload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable, Mapping

try:
    from benchmarks.benchmark_mlx_lm import (
        DEFAULT_LADDER,
        _artifact_metadata,
        _hardware,
        _prompt_tokens,
        validate_local_snapshot,
    )
    from benchmarks.commoncompute_model_ladder import load_model_ladder, model_entry
except ModuleNotFoundError:  # Direct ``python benchmarks/...`` execution.
    from benchmark_mlx_lm import (
        DEFAULT_LADDER,
        _artifact_metadata,
        _hardware,
        _prompt_tokens,
        validate_local_snapshot,
    )
    from commoncompute_model_ladder import load_model_ladder, model_entry


METRIC_DIRECTIONS = {
    "ttft_ms": "lower",
    "inter_token_latency_ms": "lower",
    "wall_time_seconds": "lower",
    "end_to_end_output_tokens_per_second": "higher",
}


def _distribution(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("metric samples must be finite and non-empty")
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def summarize_runs(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    return {
        metric: _distribution([float(run[metric]) for run in runs])
        for metric in METRIC_DIRECTIONS
    }


def compare_route_runs(
    control_runs: list[Mapping[str, Any]],
    candidate_runs: list[Mapping[str, Any]],
    *,
    minimum_end_to_end_improvement_percent: float = 2.0,
) -> dict[str, Any]:
    """Compare paired routes and apply a conservative batch-one promotion gate."""

    if len(control_runs) != len(candidate_runs) or not control_runs:
        raise ValueError("control and candidate runs must have the same non-zero length")
    exact_tokens = all(
        control["token_ids"] == candidate["token_ids"]
        for control, candidate in zip(control_runs, candidate_runs)
    )
    control_invariant = all(
        run["token_ids"] == control_runs[0]["token_ids"] for run in control_runs
    )
    candidate_invariant = all(
        run["token_ids"] == candidate_runs[0]["token_ids"] for run in candidate_runs
    )
    control_summary = summarize_runs(control_runs)
    candidate_summary = summarize_runs(candidate_runs)
    improvements: dict[str, float] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        control = float(control_summary[metric]["median"])
        candidate = float(candidate_summary[metric]["median"])
        improvements[metric] = (
            100.0 * (candidate - control) / control
            if direction == "higher"
            else 100.0 * (control - candidate) / control
        )
    gate = (
        exact_tokens
        and control_invariant
        and candidate_invariant
        and improvements["end_to_end_output_tokens_per_second"]
        >= minimum_end_to_end_improvement_percent
        and improvements["ttft_ms"] >= 0.0
        and improvements["inter_token_latency_ms"] >= -2.0
    )
    return {
        "exact_greedy_token_parity": exact_tokens,
        "control_repetition_token_invariance": control_invariant,
        "candidate_repetition_token_invariance": candidate_invariant,
        "control_summary": control_summary,
        "candidate_summary": candidate_summary,
        "median_improvement_percent": improvements,
        "promotion_thresholds": {
            "minimum_end_to_end_improvement_percent": minimum_end_to_end_improvement_percent,
            "minimum_ttft_improvement_percent": 0.0,
            "minimum_inter_token_improvement_percent": -2.0,
        },
        "batch_one_route_promotion_passed": gate,
    }


def _timed_tokens(token_iterator: Any, output_tokens: int) -> dict[str, Any]:
    token_ids: list[int] = []
    arrival_times: list[float] = []
    started = time.perf_counter()
    for token in token_iterator:
        token_ids.append(int(token))
        arrival_times.append(time.perf_counter())
    ended = time.perf_counter()
    if len(token_ids) != output_tokens:
        raise RuntimeError("route did not produce the requested token count")
    intervals = [
        (right - left) * 1000.0
        for left, right in zip(arrival_times, arrival_times[1:])
    ]
    wall = ended - started
    return {
        "ttft_ms": (arrival_times[0] - started) * 1000.0,
        "inter_token_latency_ms": statistics.median(intervals),
        "wall_time_seconds": wall,
        "end_to_end_output_tokens_per_second": output_tokens / wall,
        "token_ids": token_ids,
    }


def _batch_run(
    *,
    model: Any,
    prompt: list[int],
    output_tokens: int,
    sampler: Any,
    batch_generator_type: Any,
    prefill_step_size: int,
) -> dict[str, Any]:
    generator = batch_generator_type(
        model,
        max_tokens=output_tokens,
        stop_tokens=None,
        sampler=sampler,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=prefill_step_size,
    )
    uid = generator.insert([list(prompt)], [output_tokens])[0]

    def tokens():
        try:
            while responses := generator.next_generated():
                for response in responses:
                    if response.uid == uid:
                        yield int(response.token)
        finally:
            generator.close()

    return _timed_tokens(tokens(), output_tokens)


def _direct_run(
    *,
    mx: Any,
    model: Any,
    prompt: list[int],
    output_tokens: int,
    sampler: Any,
    generate_step: Callable[..., Any],
    wired_limit: Callable[..., Any],
    prefill_step_size: int,
) -> dict[str, Any]:
    def tokens():
        with wired_limit(model):
            for token, _ in generate_step(
                mx.array(prompt),
                model,
                max_tokens=output_tokens,
                sampler=sampler,
                prefill_step_size=prefill_step_size,
                kv_bits=None,
            ):
                yield int(token)

    return _timed_tokens(tokens(), output_tokens)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--approved-ssd-root", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--minimum-improvement-percent", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.prompt_tokens,
        args.output_tokens,
        args.repetitions,
        args.prefill_step_size,
    ) <= 0 or args.warmups < 0:
        parser.error("token counts, repetitions, and step size must be positive")

    ladder = load_model_ladder(args.ladder)
    entry = model_entry(ladder, args.model_id)
    snapshot = validate_local_snapshot(
        args.model_path,
        args.approved_ssd_root,
        entry["revision"],
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator, generate_step, wired_limit
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    versions = {"mlx": metadata.version("mlx"), "mlx_lm": metadata.version("mlx-lm")}
    for package, expected in (
        ("mlx", ladder["mlx_lm_baseline"]["mlx_version"]),
        ("mlx_lm", ladder["mlx_lm_baseline"]["mlx_lm_version"]),
    ):
        if versions[package] != expected:
            raise RuntimeError(f"{package} version {versions[package]} != pinned {expected}")

    model, tokenizer = load(str(snapshot), lazy=False)
    prompt = _prompt_tokens(tokenizer, args.prompt_tokens)
    sampler = make_sampler(temp=0.0)
    shared = {
        "model": model,
        "prompt": prompt,
        "output_tokens": args.output_tokens,
        "sampler": sampler,
        "prefill_step_size": args.prefill_step_size,
    }
    control = lambda: _batch_run(batch_generator_type=BatchGenerator, **shared)
    candidate = lambda: _direct_run(
        mx=mx,
        generate_step=generate_step,
        wired_limit=wired_limit,
        **shared,
    )
    for index in range(args.warmups):
        first, second = (control, candidate) if index % 2 == 0 else (candidate, control)
        first()
        second()

    control_runs: list[dict[str, Any]] = []
    candidate_runs: list[dict[str, Any]] = []
    run_order: list[list[str]] = []
    for index in range(args.repetitions):
        ordered = (
            (("control", control), ("candidate", candidate))
            if index % 2 == 0
            else (("candidate", candidate), ("control", control))
        )
        run_order.append([name for name, _ in ordered])
        for name, route in ordered:
            result = route()
            (control_runs if name == "control" else candidate_runs).append(result)

    comparison = compare_route_runs(
        control_runs,
        candidate_runs,
        minimum_end_to_end_improvement_percent=args.minimum_improvement_percent,
    )
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "hardware",
        "runner": "lokahi-resident-mlx-route-comparison",
        "target_id": ladder["target_id"],
        "model": {
            "model_id": entry["model_id"],
            "repo": entry["repo"],
            "revision": entry["revision"],
            "local_snapshot": str(snapshot),
            "artifact": _artifact_metadata(snapshot),
        },
        "hardware": _hardware(),
        "software": {"python": platform.python_version(), **versions},
        "workload": {
            "prompt_tokens_per_sequence": args.prompt_tokens,
            "output_tokens_per_sequence": args.output_tokens,
            "batch_size": 1,
            "sampling": "greedy",
            "warmups_per_route": args.warmups,
            "repetitions_per_route": args.repetitions,
            "prefix_cache": False,
            "prefill_step_size": args.prefill_step_size,
        },
        "weight_residency": {
            "ssd_offload_parameter": "disabled",
            "model_load_lazy": False,
            "manager": "mlx-lm",
            "lokahi_managed_paging": False,
            "semantics": "fully materialized warm model; no explicit or OS-managed offload",
        },
        "routes": {
            "control": "MLX-LM BatchGenerator continuous-batching path at batch one",
            "candidate": "Lokahi batch-one policy using MLX-LM generate_step directly",
        },
        "measurement": {
            "run_order": run_order,
            "timing_boundary": "host wall clock immediately before route iteration through requested final token",
            "model_load_timed": False,
            "download_timed": False,
        },
        "comparison": comparison,
        "control_runs": control_runs,
        "candidate_runs": candidate_runs,
        "claim_boundary": (
            "Paired warm hardware evidence for Lokahi route selection over the pinned "
            "MLX-LM compatibility backend; not a native Metal, ANE, SSD, provider, or "
            "Darkbloom comparison."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if not output.parent.is_dir():
            raise ValueError("output parent directory must already exist")
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if comparison["batch_one_route_promotion_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
