#!/usr/bin/env python3
"""Synthetic manifest and architecture-aware KV planning benchmark.

This does not execute a model, measure token throughput, or measure physical
memory. It verifies that dense and GPT-OSS-style manifests round-trip strictly
and quantifies logical KV bytes implied by their attention schedules.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from generated_model_fixtures import (
    generated_dense_manifest,
    generated_gpt_oss_20b_manifest,
)
from ollm.core import PortableModelManifest
from ollm.planning import estimate_kv_cache_bytes


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return ordered[index]


def _parse_timing(manifest: PortableModelManifest, iterations: int) -> dict:
    payload = manifest.to_json()
    expected_identity = manifest.identity_digest
    for _ in range(10):
        PortableModelManifest.from_json(payload)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        decoded = PortableModelManifest.from_json(payload)
        decoded_identity = decoded.identity_digest
        ended = time.perf_counter_ns()
        if decoded_identity != expected_identity:
            raise RuntimeError("manifest identity changed during round trip")
        samples.append((ended - started) / 1000.0)
    return {
        "iterations": iterations,
        "median_microseconds": statistics.median(samples),
        "p95_microseconds": _percentile(samples, 0.95),
        "maximum_microseconds": max(samples),
        "timing_boundary": "strict JSON decode, semantic validation, and identity digest",
    }


def _kv_rows(manifest: PortableModelManifest, batches: list[int]) -> list[dict]:
    assert manifest.architecture.max_context_tokens is not None
    contexts = sorted(
        {
            min(128, manifest.architecture.max_context_tokens),
            min(8192, manifest.architecture.max_context_tokens),
            manifest.architecture.max_context_tokens,
        }
    )
    rows = []
    for batch_size in batches:
        for context_tokens in contexts:
            estimate = estimate_kv_cache_bytes(
                manifest,
                batch_size=batch_size,
                context_tokens=context_tokens,
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "context_tokens": context_tokens,
                    "logical_kv_bytes": estimate.total_bytes,
                    "all_dense_logical_kv_bytes": estimate.all_dense_bytes,
                    "bytes_saved_vs_all_dense": estimate.bytes_saved_vs_all_dense,
                    "fraction_saved_vs_all_dense": estimate.fraction_saved_vs_all_dense,
                    "dense_layers": estimate.dense_layer_count,
                    "windowed_layers": estimate.windowed_layer_count,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batches", default="1,8")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    try:
        batches = [int(value) for value in args.batches.split(",")]
    except ValueError as exc:
        parser.error(f"--batches must be comma-separated integers: {exc}")
    if not batches or min(batches) <= 0:
        parser.error("--batches must contain positive integers")

    manifests = [generated_dense_manifest(), generated_gpt_oss_20b_manifest()]
    report = {
        "schema_version": 1,
        "kind": "synthetic_model_manifest_and_kv_planning",
        "evidence_kind": "synthetic",
        "models": [
            {
                "model_id": manifest.model_id,
                "manifest_identity": manifest.identity_digest,
                "architecture_class": manifest.architecture.architecture_class.value,
                "manifest_parse": _parse_timing(manifest, args.iterations),
                "kv_estimates": _kv_rows(manifest, batches),
            }
            for manifest in manifests
        ],
        "claim_boundary": (
            "Generated manifests, strict host-side parsing, and logical KV arithmetic only; "
            "not physical peak memory, accelerator use, SSD evidence, or token throughput."
        ),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
