#!/usr/bin/env python3
"""Benchmark the native Lokahi engine on one pinned, locally staged artifact.

The report uses the MLX-LM baseline schema (``benchmark_mlx_lm.py``) so
``compare_mlx_baseline.py`` can score it against the recorded control. The
prompt is built with the same rule as the control: BOS followed by a repeated
non-EOS token. Timing comes from inside the native engine; Python only
prepares inputs and writes the report. Never downloads.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from benchmarks.benchmark_mlx_lm import (
        DEFAULT_LADDER,
        _artifact_metadata,
        _hardware,
        _prompt_tokens,
        summarize_repetitions,
        validate_local_snapshot,
    )
    from benchmarks.commoncompute_model_ladder import load_model_ladder, model_entry
except ModuleNotFoundError:  # Direct ``python benchmarks/...`` execution.
    from benchmark_mlx_lm import (  # type: ignore[no-redef]
        DEFAULT_LADDER,
        _artifact_metadata,
        _hardware,
        _prompt_tokens,
        summarize_repetitions,
        validate_local_snapshot,
    )
    from commoncompute_model_ladder import load_model_ladder, model_entry  # type: ignore[no-redef]


def _load_tokenizer(snapshot: Path) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(snapshot)


def run_native_bench(
    cli: Path,
    snapshot: Path,
    prompt: list[int],
    output_tokens: int,
    warmups: int,
    repetitions: int,
    backend: str,
    max_context: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lokahi-bench-") as directory:
        tokens_file = Path(directory) / "prompt.txt"
        tokens_file.write_text(",".join(map(str, prompt)), encoding="utf-8")
        result = subprocess.run(
            [
                str(cli), "bench",
                "--model", str(snapshot),
                "--backend", backend,
                "--tokens-file", str(tokens_file),
                "--max-new", str(output_tokens),
                "--warmups", str(warmups),
                "--repetitions", str(repetitions),
                "--max-context", str(max_context),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"native bench failed: {result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--approved-ssd-root", type=Path, required=True)
    parser.add_argument("--native-cli", type=Path, required=True)
    parser.add_argument("--backend", default="metal", choices=("metal", "cpu"))
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, args.repetitions) <= 0 or args.warmups < 0:
        parser.error("token counts and repetitions must be positive")

    ladder = load_model_ladder(args.ladder)
    entry = model_entry(ladder, args.model_id)
    snapshot = validate_local_snapshot(args.model_path, args.approved_ssd_root, entry["revision"])
    prompt = _prompt_tokens(_load_tokenizer(snapshot), args.prompt_tokens)
    native = run_native_bench(
        args.native_cli.expanduser().resolve(),
        snapshot,
        prompt,
        args.output_tokens,
        args.warmups,
        args.repetitions,
        args.backend,
        max_context=args.prompt_tokens + args.output_tokens,
    )
    repetitions = native["repetitions"]
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "hardware",
        "runner": native["runner"],
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
            "lokahi_backend": native["backend"],
            "lokahi_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "unknown",
        },
        "workload": {
            "prompt_tokens_per_sequence": args.prompt_tokens,
            "output_tokens_per_sequence": args.output_tokens,
            "batch_size": 1,
            "sampling": "greedy",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "prefix_cache": False,
        },
        "weight_residency": {
            "ssd_offload_parameter": "disabled",
            "model_load_lazy": False,
            "manager": "lokahi-native",
            "lokahi_managed_paging": False,
            "semantics": "weights copied once into resident GPU buffers at load",
        },
        "correctness": {
            "requested_token_count_passed": all(
                len(rep["token_ids"]) == args.output_tokens for rep in repetitions
            ),
            "repetition_token_invariance": native["repetition_token_invariance"],
            "numerical_error": "not measured here; see tests/test_native_engine.py for oracle parity",
        },
        "measurement": {
            "units": {
                "latency": "milliseconds",
                "throughput": "tokens per second",
                "memory": "decimal gigabytes of process peak resident set size",
            },
            "timing_boundaries": {
                "ttft_ms": "native: prompt submission to first sampled token visible on the host",
                "inter_token_latency_ms": "native: median host-observed interval between tokens",
                "prompt_tokens_per_second": "prompt tokens divided by ttft",
                "decode_tokens_per_second": "(output tokens - 1) divided by first-to-last token time",
                "aggregate_tokens_per_second_including_prefill": "output tokens divided by ttft plus decode time",
            },
            "model_load_timed": False,
            "load_seconds": native["load_seconds"],
            "download_timed": False,
        },
        "summary": summarize_repetitions(repetitions),
        "repetitions": repetitions,
        "claim_boundary": (
            "Warm native Lokahi engine on one exact pinned artifact; comparable to the "
            "MLX-LM baseline only for the same hardware, artifact, and workload."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if native["repetition_token_invariance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
