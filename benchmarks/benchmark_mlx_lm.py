#!/usr/bin/env python3
"""Offline-only MLX-LM baseline runner for pinned Common Compute artifacts.

The runner refuses repository IDs and requires a local Hugging Face snapshot
whose directory name is the expected 40-hex revision. It never downloads.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Mapping

try:
    from benchmarks.commoncompute_model_ladder import load_model_ladder, model_entry
except ModuleNotFoundError:  # Direct ``python benchmarks/...`` execution.
    from commoncompute_model_ladder import load_model_ladder, model_entry


DEFAULT_LADDER = (
    Path(__file__).parent / "targets" / "commoncompute_m1_model_ladder_v1.json"
)


def mlx_lazy_load_for_ssd_offload(mode: str) -> bool:
    """Resolve the MLX control's storage parameter without overstating it.

    ``os-managed`` asks MLX-LM to retain lazy file-backed arrays. It does not
    provide Lokahi-managed residency, guaranteed eviction, or SSD I/O evidence.
    """

    if mode == "disabled":
        return False
    if mode == "os-managed":
        return True
    raise ValueError(f"unsupported MLX SSD offload mode: {mode}")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_local_snapshot(
    model_path: Path,
    approved_ssd_root: Path,
    expected_revision: str,
) -> Path:
    resolved_model = model_path.expanduser().resolve()
    resolved_root = approved_ssd_root.expanduser().resolve()
    forbidden = Path("/Volumes/Tyler HDD").resolve()
    if _inside(resolved_model, forbidden):
        raise ValueError("/Volumes/Tyler HDD is forbidden for model artifacts")
    if not resolved_root.is_dir():
        raise ValueError("approved SSD root does not exist")
    if not resolved_model.is_dir() or not _inside(resolved_model, resolved_root):
        raise ValueError("model path must be an existing directory inside the approved SSD root")
    if resolved_model.name != expected_revision:
        raise ValueError("model path must end in the exact pinned snapshot revision")
    for required in ("config.json",):
        if not (resolved_model / required).is_file():
            raise ValueError(f"model snapshot is missing {required}")
    if not any(resolved_model.glob("*.safetensors")):
        raise ValueError("model snapshot contains no safetensors weights")
    return resolved_model


def _distribution(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("benchmark samples must be finite and non-empty")
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "mean": statistics.fmean(ordered),
    }


def summarize_repetitions(repetitions: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not repetitions:
        raise ValueError("at least one repetition is required")
    metric_names = (
        "ttft_ms",
        "inter_token_latency_ms",
        "prompt_tokens_per_second",
        "decode_tokens_per_second",
        "aggregate_tokens_per_second_including_prefill",
        "peak_memory_gb",
    )
    return {
        name: _distribution([float(repetition[name]) for repetition in repetitions])
        for name in metric_names
    }


def _hardware() -> dict[str, Any]:
    def sysctl(name: str) -> str:
        return subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    thermal_raw = subprocess.run(
        [
            "/usr/bin/xcrun",
            "swift",
            "-e",
            "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    thermal_states = {"0": "nominal", "1": "fair", "2": "serious", "3": "critical"}
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "unified_memory_bytes": int(sysctl("hw.memsize")),
        "architecture": platform.machine(),
        "macos_version": platform.mac_ver()[0],
        "macos_build": sysctl("kern.osversion"),
        "thermal_state_after_run": thermal_states.get(thermal_raw, f"unknown-{thermal_raw}"),
    }


def _artifact_metadata(snapshot: Path) -> dict[str, Any]:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    weights = sorted(snapshot.glob("*.safetensors"))
    weight_records = []
    for weight in weights:
        digest = hashlib.sha256()
        with weight.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        weight_records.append(
            {
                "filename": weight.name,
                "bytes": weight.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    quantization = config.get("quantization") or config.get("quantization_config")
    return {
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "activation_dtype": config.get("torch_dtype"),
        "weight_quantization": quantization,
        "weights": weight_records,
    }


def _prompt_tokens(tokenizer: Any, length: int) -> list[int]:
    encoded = tokenizer.encode(" benchmark", add_special_tokens=False)
    eos = set(tokenizer.eos_token_ids)
    seed = next((int(token) for token in encoded if int(token) not in eos), None)
    if seed is None:
        raise RuntimeError("tokenizer did not provide a non-EOS fixture token")
    bos = getattr(tokenizer, "bos_token_id", None)
    tokens = [int(bos)] if bos is not None and int(bos) not in eos else []
    tokens.extend([seed] * (length - len(tokens)))
    if len(tokens) != length:
        raise RuntimeError("could not construct the exact prompt length")
    return tokens


def _run_batch(
    *,
    mx: Any,
    model: Any,
    batch_generator_type: Any,
    sampler: Any,
    prompt: list[int],
    batch_size: int,
    output_tokens: int,
) -> dict[str, Any]:
    mx.reset_peak_memory()
    generator = batch_generator_type(
        model,
        max_tokens=output_tokens,
        stop_tokens=None,
        sampler=sampler,
        completion_batch_size=batch_size,
        prefill_batch_size=batch_size,
    )
    uids = generator.insert(
        [list(prompt) for _ in range(batch_size)],
        [output_tokens] * batch_size,
    )
    token_times = {uid: [] for uid in uids}
    token_ids = {uid: [] for uid in uids}
    started = time.perf_counter()
    try:
        with generator.stats() as stats:
            while responses := generator.next_generated():
                observed = time.perf_counter()
                for response in responses:
                    token_times[response.uid].append(observed)
                    token_ids[response.uid].append(int(response.token))
    finally:
        generator.close()
    ended = time.perf_counter()

    if any(len(tokens) != output_tokens for tokens in token_ids.values()):
        raise RuntimeError("MLX-LM did not produce the requested token count")
    ttft = [(times[0] - started) * 1000.0 for times in token_times.values()]
    inter_token = [
        (right - left) * 1000.0
        for times in token_times.values()
        for left, right in zip(times, times[1:])
    ]
    wall_seconds = ended - started
    return {
        "ttft_ms": statistics.median(ttft),
        "inter_token_latency_ms": statistics.median(inter_token),
        "prompt_tokens_per_second": float(stats.prompt_tps),
        "decode_tokens_per_second": float(stats.generation_tps),
        "aggregate_tokens_per_second_including_prefill": (
            batch_size * output_tokens / wall_seconds
        ),
        "peak_memory_gb": float(stats.peak_memory),
        "wall_time_seconds": wall_seconds,
        "token_ids": [token_ids[uid] for uid in uids],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--approved-ssd-root", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--ssd-offload",
        choices=("disabled", "os-managed"),
        default="disabled",
        help=(
            "MLX control only: disabled fully materializes weights; os-managed "
            "uses MLX-LM lazy loading and does not claim Lokahi-managed paging"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ladder = load_model_ladder(args.ladder)
    if args.list:
        print(json.dumps({"models": ladder["local_priority"]}, sort_keys=True))
        return 0
    if not args.model_id or args.model_path is None or args.approved_ssd_root is None:
        parser.error("a run requires --model-id, --model-path, and --approved-ssd-root")
    if min(
        args.prompt_tokens,
        args.output_tokens,
        args.batch_size,
        args.repetitions,
    ) <= 0 or args.warmups < 0:
        parser.error("token counts, batch size, and repetitions must be positive")

    entry = model_entry(ladder, args.model_id)
    snapshot = validate_local_snapshot(
        args.model_path,
        args.approved_ssd_root,
        entry["revision"],
    )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    runtime_versions = {
        "mlx": metadata.version("mlx"),
        "mlx_lm": metadata.version("mlx-lm"),
    }
    for package, expected in (
        ("mlx", ladder["mlx_lm_baseline"]["mlx_version"]),
        ("mlx_lm", ladder["mlx_lm_baseline"]["mlx_lm_version"]),
    ):
        if runtime_versions[package] != expected:
            raise RuntimeError(
                f"{package} version {runtime_versions[package]} does not match "
                f"the ladder baseline {expected}"
            )

    lazy_load = mlx_lazy_load_for_ssd_offload(args.ssd_offload)
    model, tokenizer = load(str(snapshot), lazy=lazy_load)
    prompt = _prompt_tokens(tokenizer, args.prompt_tokens)
    sampler = make_sampler(temp=0.0)
    run_arguments = {
        "mx": mx,
        "model": model,
        "batch_generator_type": BatchGenerator,
        "sampler": sampler,
        "prompt": prompt,
        "batch_size": args.batch_size,
        "output_tokens": args.output_tokens,
    }
    for _ in range(args.warmups):
        _run_batch(**run_arguments)
    repetitions = [_run_batch(**run_arguments) for _ in range(args.repetitions)]
    reference_tokens = repetitions[0]["token_ids"]
    arrival_invariant = all(
        repetition["token_ids"] == reference_tokens for repetition in repetitions
    )

    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "hardware",
        "runner": "mlx-lm-baseline",
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
            **runtime_versions,
        },
        "workload": {
            "prompt_tokens_per_sequence": args.prompt_tokens,
            "output_tokens_per_sequence": args.output_tokens,
            "batch_size": args.batch_size,
            "sampling": "greedy",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "prefix_cache": False,
        },
        "weight_residency": {
            "ssd_offload_parameter": args.ssd_offload,
            "model_load_lazy": lazy_load,
            "manager": "mlx-macos" if lazy_load else "mlx-lm",
            "lokahi_managed_paging": False,
            "semantics": (
                "MLX-LM lazy file-backed loading with residency controlled by MLX and macOS; warmups may materialize the complete model"
                if lazy_load
                else "MLX-LM fully materialized warm model"
            ),
        },
        "correctness": {
            "requested_token_count_passed": True,
            "repetition_token_invariance": arrival_invariant,
            "numerical_error": "not measured; exact greedy token invariance only",
        },
        "measurement": {
            "units": {
                "latency": "milliseconds",
                "throughput": "tokens per second",
                "memory": "decimal gigabytes as reported by MLX-LM",
            },
            "timing_boundaries": {
                "ttft_ms": "immediately before BatchGenerator.next_generated iteration to first observed response token",
                "inter_token_latency_ms": "median host-observed interval between consecutive response tokens",
                "prompt_tokens_per_second": "MLX-LM BatchGenerator prompt processing statistic",
                "decode_tokens_per_second": "MLX-LM BatchGenerator generation statistic",
                "aggregate_tokens_per_second_including_prefill": "requested output tokens divided by host wall time from before generation iteration through completion",
            },
            "model_load_timed": False,
            "download_timed": False,
        },
        "summary": summarize_repetitions(repetitions),
        "repetitions": repetitions,
        "claim_boundary": (
            "Warm local MLX-LM baseline on one exact pinned artifact; not Lokahi, "
            "Common Compute provider, Darkbloom, ANE, or SSD throughput evidence."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if not output.parent.is_dir():
            raise ValueError("output parent directory must already exist")
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if arrival_invariant else 1


if __name__ == "__main__":
    raise SystemExit(main())
