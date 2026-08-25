#!/usr/bin/env python3
"""Benchmark Strata's explicit quantized Llama layer pager against MLX-LM."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

try:
    from benchmarks.benchmark_mlx_lm import (
        _artifact_metadata,
        _hardware,
        _prompt_tokens,
        summarize_repetitions,
        validate_local_snapshot,
    )
    from benchmarks.commoncompute_model_ladder import load_model_ladder, model_entry
except ModuleNotFoundError:
    from benchmark_mlx_lm import (
        _artifact_metadata,
        _hardware,
        _prompt_tokens,
        summarize_repetitions,
        validate_local_snapshot,
    )
    from commoncompute_model_ladder import load_model_ladder, model_entry

from ollm.core.ssd_offload import SSDOffloadMode, SSDOffloadPolicy
from ollm.mlx_kvcache import MLXKVCache
from ollm.runtime import PagedMLXLlamaRuntime


DEFAULT_LADDER = Path(__file__).parent / "targets" / "commoncompute_m1_model_ladder_v1.json"


def _scheduler_delta(before: Any, after: Any) -> dict[str, Any]:
    return {
        field: getattr(after, field) - getattr(before, field)
        for field in (
            "resident_hits",
            "cold_misses",
            "prefetch_ready_hits",
            "prefetch_waits",
            "prefetch_skips",
            "bytes_loaded",
            "load_time_ms",
            "stall_time_ms",
        )
    }


def _run(
    runtime: PagedMLXLlamaRuntime,
    prompt: list[int],
    output_tokens: int,
) -> dict[str, Any]:
    import mlx.core as mx

    mx.reset_peak_memory()
    before = runtime.scheduler_snapshot()
    cache = MLXKVCache(runtime.config, cache_dir=None)
    model_input = mx.array([prompt], dtype=mx.int32)
    generated: list[int] = []
    arrivals: list[float] = []
    started = time.perf_counter()
    for _ in range(output_tokens):
        logits = runtime.model(model_input, past_key_values=cache, use_cache=True)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)
        arrivals.append(time.perf_counter())
        generated.append(int(next_token[0, 0].item()))
        model_input = next_token
    ended = time.perf_counter()
    after = runtime.scheduler_snapshot()
    intervals = [right - left for left, right in zip(arrivals, arrivals[1:])]
    decode_tps = (
        (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
        if len(arrivals) > 1 and arrivals[-1] > arrivals[0]
        else 0.0
    )
    return {
        "ttft_ms": (arrivals[0] - started) * 1000.0,
        "inter_token_latency_ms": (
            statistics.median(intervals) * 1000.0 if intervals else 0.0
        ),
        "prompt_tokens_per_second": len(prompt) / (arrivals[0] - started),
        "decode_tokens_per_second": decode_tps,
        "aggregate_tokens_per_second_including_prefill": output_tokens / (ended - started),
        "peak_memory_gb": float(mx.get_peak_memory()) / 1_000_000_000.0,
        "wall_time_seconds": ended - started,
        "token_ids": [generated],
        "weight_scheduler": _scheduler_delta(before, after),
        "residency_after": asdict(runtime.residency_snapshot()),
    }


def _baseline_reference(path: Path, entry: dict[str, Any], workload: dict[str, Any]) -> list[list[int]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("runner") != "mlx-lm-baseline":
        raise ValueError("baseline report must be an MLX-LM result")
    model = report.get("model", {})
    if model.get("repo") != entry["repo"] or model.get("revision") != entry["revision"]:
        raise ValueError("baseline report does not match the model artifact")
    for key, value in workload.items():
        if report.get("workload", {}).get(key) != value:
            raise ValueError(f"baseline workload does not match {key}")
    return report["repetitions"][0]["token_ids"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--model-id", default="llama-3.2-1b")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--approved-ssd-root", type=Path, required=True)
    parser.add_argument("--storage-target-id", default="approved-internal-ssd")
    parser.add_argument("--ssd-offload", choices=("auto", "required"), default="required")
    parser.add_argument("--max-resident-weight-mib", type=int, required=True)
    parser.add_argument("--prefetch-distance", type=int, default=1)
    parser.add_argument("--io-workers", type=int, default=1)
    parser.add_argument(
        "--pinned-layer-count",
        default="auto",
        help="non-negative integer or auto",
    )
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.max_resident_weight_mib,
        args.prompt_tokens,
        args.output_tokens,
        args.repetitions,
        args.io_workers,
    ) <= 0 or args.warmups < 0 or args.prefetch_distance < 0:
        parser.error("budgets, token counts, and repetitions must be positive")
    if args.pinned_layer_count == "auto":
        pinned_layer_count = None
    else:
        try:
            pinned_layer_count = int(args.pinned_layer_count)
        except ValueError:
            parser.error("pinned layer count must be a non-negative integer or auto")
        if pinned_layer_count < 0:
            parser.error("pinned layer count must be non-negative")

    ladder = load_model_ladder(args.ladder)
    entry = model_entry(ladder, args.model_id)
    if args.model_id != "llama-3.2-1b":
        raise ValueError("paged MLX laboratory currently supports llama-3.2-1b only")
    snapshot = validate_local_snapshot(
        args.model_path, args.approved_ssd_root, entry["revision"]
    )
    workload_match = {
        "prompt_tokens_per_sequence": args.prompt_tokens,
        "output_tokens_per_sequence": args.output_tokens,
        "batch_size": 1,
        "sampling": "greedy",
        "prefix_cache": False,
    }
    reference_tokens = _baseline_reference(args.baseline, entry, workload_match)

    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(snapshot)
    prompt = _prompt_tokens(tokenizer, args.prompt_tokens)
    policy = SSDOffloadPolicy(
        mode=SSDOffloadMode(args.ssd_offload),
        storage_target_id=args.storage_target_id,
        max_resident_weight_bytes=args.max_resident_weight_mib * 1024 * 1024,
        prefetch_distance=args.prefetch_distance,
        io_workers=args.io_workers,
    )
    with PagedMLXLlamaRuntime.load(
        snapshot,
        policy,
        pinned_layer_count=pinned_layer_count,
    ) as runtime:
        for _ in range(args.warmups):
            _run(runtime, prompt, args.output_tokens)
        repetitions = [
            _run(runtime, prompt, args.output_tokens) for _ in range(args.repetitions)
        ]
        runtime_metadata = asdict(runtime.metadata)

    token_invariance = all(
        item["token_ids"] == repetitions[0]["token_ids"] for item in repetitions
    )
    oracle_parity = repetitions[0]["token_ids"] == reference_tokens
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "hardware",
        "runner": "strata-paged-mlx-lab",
        "target_id": ladder["target_id"],
        "model": {
            "model_id": entry["model_id"],
            "repo": entry["repo"],
            "revision": entry["revision"],
            "local_snapshot": str(snapshot),
            "artifact": _artifact_metadata(snapshot),
        },
        "hardware": _hardware(),
        "software": {
            "python": platform.python_version(),
            "mlx": metadata.version("mlx"),
            "mlx_lm": metadata.version("mlx-lm"),
        },
        "workload": {
            **workload_match,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
        },
        "weight_residency": {
            "ssd_offload_parameter": args.ssd_offload,
            "strata_managed_paging": True,
            "storage_target_id": args.storage_target_id,
            "runtime": runtime_metadata,
            "cache_state": "uncontrolled warm file/page-cache state",
        },
        "correctness": {
            "requested_token_count_passed": all(
                len(item["token_ids"][0]) == args.output_tokens for item in repetitions
            ),
            "repetition_token_invariance": token_invariance,
            "mlx_lm_exact_greedy_token_parity": oracle_parity,
            "numerical_error": "not measured; exact greedy tokens only",
        },
        "measurement": {
            "units": {
                "latency": "milliseconds",
                "throughput": "tokens per second",
                "memory": "decimal gigabytes from MLX peak allocation counter",
                "weight_bytes": "bytes",
            },
            "timing_boundaries": {
                "ttft_ms": "before custom Strata model prefill to materialized first sampled token",
                "inter_token_latency_ms": "median interval between materialized sampled tokens",
                "prompt_tokens_per_second": "prompt length divided by host-observed TTFT",
                "decode_tokens_per_second": "post-first generated tokens divided by host-observed decode interval",
                "weight_scheduler": "cumulative exact-range load work and acquire-visible stall inside each repetition",
            },
            "download_timed": False,
            "model_load_timed": False,
        },
        "summary": summarize_repetitions(repetitions),
        "repetitions": repetitions,
        "claim_boundary": (
            "Strata Python/MLX quantized Llama paging laboratory with an explicit resident cap; "
            "not native Strata, cold-SSD throughput, direct SSD execution, ANE, or Common Compute evidence."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if not output.parent.is_dir():
            raise ValueError("output parent directory must exist")
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if token_invariance and oracle_parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
