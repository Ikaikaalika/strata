#!/usr/bin/env python3
"""Synthetic overlap benchmark for Strata Governor.

This benchmark does not claim real SSD or model throughput. It answers a
narrow question: given measured layer-load and layer-compute durations, how
much demand stall can exact next-layer prefetch remove?
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from ollm.core import ExecutionPlan, ModelSpec, TensorRef, WeightGroup
from ollm.scheduling import DenseLayerPipeline, PrefetchScheduler, ResidencyManager


class SyntheticTensor:
    def __init__(self, value: int, nbytes: int):
        self.value = value
        self.nbytes = nbytes


class SyntheticStore:
    def __init__(self, load_seconds: float):
        self.load_seconds = load_seconds
        self.loads = 0

    def load_group(self, group):
        time.sleep(self.load_seconds)
        self.loads += 1
        return {
            "weight": SyntheticTensor(group.order + 1, group.nbytes),
        }


def build_spec(layer_count: int, layer_bytes: int) -> ModelSpec:
    groups = []
    for layer_index in range(layer_count):
        groups.append(
            WeightGroup(
                group_id=f"decoder.layer.{layer_index}",
                order=layer_index,
                tensors=(
                    TensorRef(
                        name="weight",
                        shape=(1,),
                        dtype="int8",
                        storage_nbytes=layer_bytes,
                    ),
                ),
            )
        )
    return ModelSpec.from_groups("synthetic-dense", groups)


def sequential_baseline(spec, store, passes, compute_seconds):
    started = time.perf_counter()
    state = 0
    for _ in range(passes):
        for group in spec.groups:
            weights = store.load_group(group)
            time.sleep(compute_seconds)
            state += weights["weight"].value
    return state, (time.perf_counter() - started) * 1000.0


def governed_run(spec, store, passes, compute_seconds, budget_bytes):
    residency = ResidencyManager(budget_bytes)
    scheduler = PrefetchScheduler(store, residency)
    pipeline = DenseLayerPipeline(spec, ExecutionPlan.dense(spec), scheduler)

    def compute(state, group, weights):
        time.sleep(compute_seconds)
        return state + weights["weight"].value

    started = time.perf_counter()
    state = 0
    try:
        for pass_index in range(passes):
            state = pipeline.run(
                state,
                compute,
                phase="prefill" if pass_index == 0 else "decode",
            )
    finally:
        pipeline.close()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return state, elapsed_ms, scheduler.snapshot(), residency.snapshot()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--layer-mb", type=float, default=64.0)
    parser.add_argument("--load-ms", type=float, default=12.0)
    parser.add_argument("--compute-ms", type=float, default=18.0)
    parser.add_argument("--budget-layers", type=int, default=2)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    if min(args.layers, args.budget_layers, args.passes) <= 0:
        parser.error("layers, budget-layers, and passes must be positive")
    if min(args.layer_mb, args.load_ms, args.compute_ms) < 0:
        parser.error("layer-mb, load-ms, and compute-ms must be non-negative")

    layer_bytes = max(1, int(args.layer_mb * 1024 * 1024))
    spec = build_spec(args.layers, layer_bytes)
    baseline_store = SyntheticStore(args.load_ms / 1000.0)
    governed_store = SyntheticStore(args.load_ms / 1000.0)

    baseline_state, baseline_ms = sequential_baseline(
        spec,
        baseline_store,
        args.passes,
        args.compute_ms / 1000.0,
    )
    governed_state, governed_ms, scheduler, residency = governed_run(
        spec,
        governed_store,
        args.passes,
        args.compute_ms / 1000.0,
        layer_bytes * args.budget_layers,
    )
    if baseline_state != governed_state:
        raise RuntimeError("governed and sequential executions diverged")

    report = {
        "kind": "synthetic_overlap_model",
        "inputs": vars(args),
        "baseline_ms": baseline_ms,
        "governed_ms": governed_ms,
        "modeled_speedup": baseline_ms / governed_ms,
        "scheduler": asdict(scheduler),
        "residency": asdict(residency),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
