"""Serialized direct-ANE / MLX dense-chain experiment; not LLM qualification."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def fixture(tokens: int, depth: int):
    x = (((np.arange(tokens * 256) * 13) % 67 - 33) / 64).astype(np.float16).reshape(tokens, 256)
    weights = []
    for layer in range(depth):
        state = 730 + layer
        values = []
        for _ in range(256 * 256):
            state ^= (state << 13) & 0xffffffff
            state ^= state >> 17
            state ^= (state << 5) & 0xffffffff
            values.append(((state % 65) - 32) / 512)
        weights.append(np.array(values, dtype=np.float16).reshape(256, 256))
    return x, weights


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_output(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("invalid output shape or nonfinite values")
    error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
    if error > .002:
        raise ValueError(f"MLX correctness gate failed: {error}")
    return error


def run_mlx(x, weights, compiled):
    import mlx.core as mx

    arrays = tuple(mx.array(w) for w in weights)
    resident = mx.array(x)
    mx.eval(resident, *arrays)

    def forward(a):
        for w in arrays:
            a = mx.maximum(a @ w.T, 0)
        return a

    execute = mx.compile(forward) if compiled else forward
    samples = {"resident_compute_ms": [], "resident_with_io_ms": []}
    output = None
    for boundary in samples:
        for iteration in range(25):
            begin = time.perf_counter_ns()
            arg = mx.array(x) if boundary == "resident_with_io_ms" else resident
            out = execute(arg)
            mx.eval(out)
            if boundary == "resident_with_io_ms":
                output = np.array(out)
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            if iteration >= 5:
                samples[boundary].append(elapsed)
        output = np.array(out)
    return {"variant": "mlx_compiled" if compiled else "mlx_eager", **samples}, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 256, 512])
    args = parser.parse_args()
    if any(t not in (64, 256, 512) for t in args.tokens):
        parser.error("unqualified shape")
    if args.output.exists():
        parser.error("refusing to overwrite receipt")
    root = Path(__file__).resolve().parents[1]
    report = {"schema_version": 1, "benchmark": "strata-ane-prefill-chain-paired",
              "evidence_kind": "hardware", "comparison_scope": "generated_dense_relu_chain",
              "promotion_eligible": False, "success": False, "depth": 4, "width": 256,
              "warmups": 5, "iterations": 20, "rounds": args.rounds,
              "software": {n: importlib.metadata.version(n) for n in ("mlx", "mlx-lm", "numpy")},
              "sources": {str(p.relative_to(root)): digest(p) for p in
                          [Path(__file__), root/"native/ane/prefill_chain_bench.m", root/"native/ane/strata_ane_probe.m"]},
              "probe_sha256": digest(args.probe), "records": [], "summaries": [],
              "claim_boundary": "Generated chain hardware timing; not a real-model, complete-prefill, ANE LLM or Darkbloom win",
              "timing_policy": "Compare native dispatch against MLX resident compute, and native I/O-inclusive against MLX I/O-inclusive; compilation/load excluded and reported separately"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="strata-ane-chain-", dir="/private/tmp") as scratch:
        try:
            for tokens in args.tokens:
                x, weights = fixture(tokens, 4)
                for round_id in range(args.rounds):
                    order = ["split", "fused", "mlx_eager", "mlx_compiled"]
                    # Rotate and reverse to reduce fixed order bias; no concurrent runs.
                    offset = round_id % len(order)
                    order = order[offset:] + order[:offset]
                    if round_id % 2:
                        order.reverse()
                    outputs = {}
                    for variant in order:
                        if variant.startswith("mlx"):
                            record, output = run_mlx(x, weights, variant == "mlx_compiled")
                        else:
                            output_path = Path(scratch)/f"{tokens}-{round_id}-{variant}.bin"
                            run = subprocess.run([str(args.probe.resolve()), str(tokens), "4", variant, str(output_path)],
                                                 capture_output=True, text=True, timeout=60)
                            record = json.loads(run.stdout)
                            if run.returncode or not record.get("success") or not record.get("cleanup_ok"):
                                report["records"].append(record)
                                raise RuntimeError(f"native gate failed: {record.get('error', run.stderr)}")
                            if record.get("verified_direct_ane_dispatches", 0) <= 0:
                                raise RuntimeError("no verified ANE dispatches")
                            output = np.fromfile(output_path, dtype=np.float16).reshape(tokens, 256)
                            record["resident_compute_ms"] = record["dispatch_ms_samples"]
                            record["resident_with_io_ms"] = record["resident_with_io_ms_samples"]
                        outputs[variant] = output
                        record.update(tokens=tokens, round=round_id, order=order)
                        report["records"].append(record)
                    for record in report["records"][-4:]:
                        record["mlx_max_abs_error"] = compare_output(outputs[record["variant"]], outputs["mlx_compiled"])
                    print(json.dumps({"tokens": tokens, "round": round_id, "correctness": "pass",
                                      "with_io_ms": {r["variant"]: statistics.median(r["resident_with_io_ms"]) for r in report["records"][-4:]}}), flush=True)
                entries = [r for r in report["records"] if r["tokens"] == tokens]
                summary = {"tokens": tokens}
                for boundary in ("resident_compute_ms", "resident_with_io_ms"):
                    medians = {v: statistics.median(statistics.median(r[boundary]) for r in entries if r["variant"] == v)
                               for v in ("split", "fused", "mlx_eager", "mlx_compiled")}
                    summary[boundary] = medians
                    summary[boundary+"_split_over_fused"] = medians["split"] / medians["fused"]
                    summary[boundary+"_mlx_compiled_over_fused"] = medians["mlx_compiled"] / medians["fused"]
                report["summaries"].append(summary)
            report["success"] = True
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
