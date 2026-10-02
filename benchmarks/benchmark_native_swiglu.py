"""Bounded model-connected native FFN iteration; no serving promotion/downloads.

Run generated FP16/BF16 Qwen3 logits/tokens/KV gates first, then compare three
native phase candidates against unchanged MLX-LM on the pinned local canary.
The entire hardware worker is isolated and deadline-bounded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from ollm.runtime.native_swiglu import VARIANTS, experimental_qwen3_swiglu, metal_source
if __package__ in (None, ""):
    from benchmark_native_q4 import SHAPES, fixture, require_ssd_reserve
else:
    from benchmarks.benchmark_native_q4 import SHAPES, fixture, require_ssd_reserve

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL, MAX_RELATIVE_L2 = 0.005, 0.02, 0.01
PROMPT_LENGTHS = (128, 512, 2048)
ROUTES = ("mlx_lm", *VARIANTS)


def source_identity():
    return {"metal_sha256": hashlib.sha256(metal_source().encode()).hexdigest(),
            "adapter_sha256": hashlib.sha256((ROOT / "src/ollm/runtime/native_swiglu.py").read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def require_stable_sources(identity):
    if source_identity() != identity:
        raise ValueError("source changed during the experiment; reject mixed evidence")


def round_storage(value, dtype):
    value = np.asarray(value, dtype=np.float32)
    if dtype == "float16":
        return value.astype(np.float16).astype(np.float32)
    if dtype != "bfloat16":
        raise ValueError("unsupported reference storage type")
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def affine_reference(x, packed, scales, biases):
    width = x.shape[-1]
    k = np.arange(width)
    codes = (packed[:, k // 8] >> (4 * (k % 8))) & 15
    weight = scales[:, k // 64].astype(np.float64) * codes + biases[:, k // 64]
    return x.astype(np.float64) @ weight.T


def primitive_gate():
    import mlx.core as mx
    from mlx_lm.models.activations import swiglu
    from ollm.runtime.native_swiglu import _activation
    from types import SimpleNamespace
    rows = []
    for dtype in (mx.float16, mx.bfloat16):
        dtype_name = "float16" if dtype == mx.float16 else "bfloat16"
        for shape in SHAPES:
            x, q, s, b, _ = fixture(shape)
            uq, us, ub = [np.flip(a, axis=0).copy() for a in (q, s, b)]
            g_arrays = [mx.array(q), mx.array(s, dtype=dtype), mx.array(b, dtype=dtype)]
            u_arrays = [mx.array(uq), mx.array(us, dtype=dtype), mx.array(ub, dtype=dtype)]
            x_array = mx.array(x[None, ...], dtype=dtype)
            mlp = SimpleNamespace(gate_proj=SimpleNamespace(weight=g_arrays[0], scales=g_arrays[1], biases=g_arrays[2]),
                                  up_proj=SimpleNamespace(weight=u_arrays[0], scales=u_arrays[1], biases=u_arrays[2]))
            # Independent FP64 formula and software storage rounding. No MLX
            # dequantizer, activation or native helper computes the oracle.
            gate = round_storage(affine_reference(x, q, s, b), dtype_name).astype(np.float64)
            up = round_storage(affine_reference(x, uq, us, ub), dtype_name).astype(np.float64)
            expected = round_storage((gate / (1 + np.exp(-gate))) * up, dtype_name)[None, ...]
            projections = [mx.quantized_matmul(x_array, *arrays, transpose=True, bits=4, group_size=64,
                                               mode="affine") for arrays in (g_arrays, u_arrays)]
            baseline = swiglu(*projections).astype(mx.float32)
            mx.eval(baseline)
            error_gate(np.array(baseline), expected)
            for kind in ("simd", "tiled"):
                actual = _activation(x_array, mlp, shape[1], shape[2], kind).astype(mx.float32)
                mx.eval(actual)
                rows.append({"shape": list(shape), "dtype": dtype_name, "kernel": kind,
                             "independent_arithmetic": error_gate(np.array(actual), expected),
                             "mlx_parity": error_gate(np.array(actual), np.array(baseline))})
    return rows


def error_gate(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not a.size:
        raise ValueError("invalid model correctness evidence")
    delta = a - b
    scaled = float(np.max(np.abs(delta) / (ATOL + RTOL * np.abs(b))))
    norm = float(np.linalg.norm(b.ravel()))
    error_norm = float(np.linalg.norm(delta.ravel()))
    relative = error_norm / norm if norm else (0.0 if error_norm == 0 else math.inf)
    if scaled > 1 or relative > MAX_RELATIVE_L2:
        raise ValueError(f"frozen model numerical gate failed: scaled={scaled}, relative_l2={relative}")
    return {"max_scaled": scaled, "relative_l2": relative,
            "atol": ATOL, "rtol": RTOL, "max_relative_l2": MAX_RELATIVE_L2}


def trace(model):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    x = mx.array([[5, 11, 3, 17, 44, 91, 23]], dtype=mx.int32)
    tokens, logits = [], []
    for _ in range(4):
        y = model(x, cache=cache)
        last = y[:, -1, :].astype(mx.float32)
        mx.eval(last)
        logits.append(np.array(last))
        token = int(mx.argmax(last[0]).item())
        tokens.append(token)
        x = mx.array([[token]], dtype=mx.int32)
    states = [[np.array(a.astype(mx.float32)) for a in c.state] for c in cache]
    return tokens, logits, states, [c.offset for c in cache]


def compare_trace(candidate, control):
    if candidate[0] != control[0] or candidate[3] != control[3]:
        raise ValueError("greedy tokens or KV offsets diverged")
    if len(candidate[1]) != len(control[1]) or len(candidate[2]) != len(control[2]):
        raise ValueError("trace structure diverged")
    errors = [error_gate(a, b) for a, b in zip(candidate[1], control[1])]
    for a_layer, b_layer in zip(candidate[2], control[2]):
        if len(a_layer) != len(b_layer):
            raise ValueError("KV state structure diverged")
        errors.extend(error_gate(a, b) for a, b in zip(a_layer, b_layer))
    return {"exact_greedy_token_parity": True, "kv_offset_parity": True,
            "logit_and_kv_errors": errors}


def require_graph_calls(stats, route):
    if route in ("simd_decode", "fused_both") and stats.simd_graph_calls <= 0:
        raise ValueError("candidate did not construct its declared decode path")
    if route in ("tiled_prefill", "fused_both") and stats.tiled_graph_calls <= 0:
        raise ValueError("candidate did not construct its declared prefill path")


def generated_gate():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3 import Model, ModelArgs
    receipts = []
    for dtype in (mx.float16, mx.bfloat16):
        mx.random.seed(730)
        model = Model(ModelArgs(
            model_type="qwen3", hidden_size=128, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, rms_norm_eps=1e-6, vocab_size=128,
            max_position_embeddings=1024, rope_theta=10000, tie_word_embeddings=False))
        model.set_dtype(dtype)
        nn.quantize(model, group_size=64, bits=4, mode="affine")
        mx.eval(model.parameters())
        from mlx.utils import tree_flatten
        before = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
        control = trace(model)
        for route in VARIANTS:
            with experimental_qwen3_swiglu(model, variant=route, enabled=True) as stats:
                during = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
                if before != during:
                    raise ValueError("adapter changed parameter accounting")
                candidate = trace(model)
                parity = compare_trace(candidate, control)
                require_graph_calls(stats, route)
            receipts.append({"dtype": str(dtype), "route": route, **parity,
                             "parameter_bytes_unchanged": True, "graph_calls": asdict(stats)})
        del model
        mx.clear_cache()
    return receipts


def measured_request(model, tokenizer, prompt, route, output_tokens):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.synchronize()
    mx.reset_peak_memory()
    with experimental_qwen3_swiglu(model, variant=route if route != "mlx_lm" else "fused_both",
                                   enabled=route != "mlx_lm") as stats:
        started = time.perf_counter()
        arrivals, tokens = [], []
        for response in stream_generate(model, tokenizer, prompt, max_tokens=output_tokens,
                                        sampler=make_sampler(temp=0), prefill_step_size=512):
            arrivals.append(time.perf_counter())
            tokens.append(int(response.token))
        mx.synchronize()
        ended = time.perf_counter()
        if len(tokens) != output_tokens or len(arrivals) < 2:
            raise ValueError("bounded fixed-work request did not complete")
        if route != "mlx_lm":
            require_graph_calls(stats, route)
        decode = (len(tokens) - 1) / (arrivals[-1] - arrivals[0])
        ttft = arrivals[0] - started
        record = {"route": route, "prompt_tokens": len(prompt), "output_token_ids": tokens,
                  "arrival_seconds": [t - started for t in arrivals],
                  "wall_seconds": ended - started, "ttft_ms": ttft * 1000,
                  "prefill_proxy_tps": len(prompt) / ttft, "decode_tps": decode,
                  "goodput_tps": len(tokens) / (ended - started),
                  "inter_token_ms": [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])],
                  "peak_mlx_bytes": mx.get_peak_memory(), "graph_calls": asdict(stats)}
    # Scope teardown excluded from request wall time, identical context envelope.
    return record


def summarize(records):
    for record in records:
        for key in ("ttft_ms", "prefill_proxy_tps", "decode_tps", "goodput_tps", "peak_mlx_bytes"):
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid full-request metric")
        if record["route"] not in ROUTES or record["prompt_tokens"] not in PROMPT_LENGTHS:
            raise ValueError("record outside bounded workload matrix")
        if not record["output_token_ids"]:
            raise ValueError("missing complete-request output")
    cells = []
    for length in PROMPT_LENGTHS:
        control = [r for r in records if r["prompt_tokens"] == length and r["route"] == "mlx_lm" and not r["warmup"]]
        if not control:
            raise ValueError("missing MLX control cell")
        reference = control[0]["output_token_ids"]
        for route in ROUTES:
            rows = [r for r in records if r["prompt_tokens"] == length and r["route"] == route and not r["warmup"]]
            if len(rows) != len(control) or any(r["output_token_ids"] != reference for r in rows):
                raise ValueError("missing repetitions or full-request token divergence")
            medians = {key: statistics.median(r[key] for r in rows) for key in (
                "ttft_ms", "prefill_proxy_tps", "decode_tps", "goodput_tps", "peak_mlx_bytes")}
            p95_ttft = float(np.percentile([r["ttft_ms"] for r in rows], 95))
            ratios = {key: medians[key] / statistics.median(r[key] for r in control)
                      for key in ("prefill_proxy_tps", "decode_tps", "goodput_tps")}
            cells.append({"prompt_tokens": length, "route": route, "medians": medians,
                          "p95_ttft_ms": p95_ttft, "ratios_vs_mlx": ratios,
                          "exact_greedy_token_parity": True})
    # Descriptive shortlist only. Prefill is a TTFT proxy, serving/energy/ANE are
    # missing, and three controlled repeated-text prompts are not held-out evals.
    shortlist = []
    for route in VARIANTS:
        selected = [c for c in cells if c["route"] == route]
        def passes(cell):
            baseline = next(c for c in cells if c["route"] == "mlx_lm" and c["prompt_tokens"] == cell["prompt_tokens"])
            return (cell["ratios_vs_mlx"]["prefill_proxy_tps"] >= 1.25 and
                    cell["ratios_vs_mlx"]["decode_tps"] >= 1.25 and
                    cell["ratios_vs_mlx"]["goodput_tps"] >= 1 and
                    cell["medians"]["peak_mlx_bytes"] <= baseline["medians"]["peak_mlx_bytes"] and
                    cell["p95_ttft_ms"] <= baseline["p95_ttft_ms"])
        if all(passes(c) for c in selected):
            shortlist.append(route)
    frontiers = []
    for length in PROMPT_LENGTHS:
        points = [c for c in cells if c["prompt_tokens"] == length]
        def costs(point):
            m = point["medians"]
            return (m["ttft_ms"], point["p95_ttft_ms"], m["peak_mlx_bytes"], -m["decode_tps"], -m["goodput_tps"])
        winners = []
        for point in points:
            cost = costs(point)
            if not any(all(a <= b for a, b in zip(costs(other), cost)) and
                       any(a < b for a, b in zip(costs(other), cost)) for other in points):
                winners.append(point["route"])
        frontiers.append({"prompt_tokens": length, "routes": sorted(winners),
                          "profile": "resident interactive/throughput research; not energy or serving qualification"})
    return {"cells": cells, "development_speed_shortlist": shortlist,
            "development_pareto_frontiers": frontiers,
            "promotion_eligible": False, "full_model_serving_gate_passed": False}


def worker(args):
    # Both parent and worker check before MLX import, model hashes or loading.
    require_ssd_reserve(Path.home() / "Library/Application Support/Strata")
    from ollm.runtime.native_swiglu import _require_versions
    _require_versions()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from benchmarks.benchmark_mlx_lm import _hardware
    identity = source_identity()
    report = {"schema_version": 1, "evidence_kind": "generated_model_correctness",
              "success": False, "promotion_eligible": False, "ssd_offload": "disabled",
              "ane_dispatches": 0, "created_at": datetime.now(timezone.utc).isoformat(),
              "versions": {n: version(n) for n in ("mlx", "mlx-lm")},
              **identity,
              "native_contract": {"bits": 4, "mode": "affine", "group_size": 64,
                                  "storage": ["float16", "bfloat16"], "accumulation": "float32",
                                  "rows_max": 512, "width_max": 8192, "outputs_max": 16384,
                                  "resident_operator_bytes_max": 64 * 1024**2, "math_mode": "safe"},
              "hardware_before": _hardware(), "energy_joules": None, "records": []}
    report["primitive_gates"] = primitive_gate()
    report["generated_gates"] = generated_gate()
    if args.check_generated:
        require_stable_sources(identity)
        report["hardware_after"] = _hardware()
        report["success"] = True
        return report
    from benchmarks.recovered_mlx_baseline import artifact, prompt_tokens
    import mlx.core as mx
    from mlx_lm import load
    report["evidence_kind"] = "paired_complete_request_experiment"
    path, report["artifact"] = artifact("Qwen3-0.6B-4bit")
    report["hardware_before"] = _hardware()
    model, tokenizer = load(str(path), lazy=False, tokenizer_config={"trust_remote_code": False})
    # This is pinned, synthetic fixed-work benchmarking, not a user-facing EOS policy.
    tokenizer._eos_token_ids = set()
    control = trace(model)
    report["real_weight_trace_gates"] = []
    for route in VARIANTS:
        with experimental_qwen3_swiglu(model, variant=route, enabled=True) as stats:
            parity = compare_trace(trace(model), control)
            require_graph_calls(stats, route)
        report["real_weight_trace_gates"].append({"route": route, **parity, "graph_calls": asdict(stats)})
    for length in PROMPT_LENGTHS:
        prompt = prompt_tokens(tokenizer, length)
        for round_ in range(-1, args.rounds):
            require_ssd_reserve(Path.home() / "Library/Application Support/Strata")
            offset = (round_ + 1) % len(ROUTES)
            order = ROUTES[offset:] + ROUTES[:offset]
            for route in order:
                mx.random.seed(730)
                record = measured_request(model, tokenizer, prompt, route, args.output_tokens)
                record.update(round=round_, warmup=round_ < 0, order=list(order))
                report["records"].append(record)
    report["summary"] = summarize(report["records"])
    report["hardware_after"] = _hardware()
    require_stable_sources(identity)
    report.update(success=True, output_tokens=args.output_tokens, rounds=args.rounds,
                  prefill_step_size=512, workload="warm uncached synthetic repeated text, batch one",
                  timing_boundary="stream_generate to final completion; model load/JIT warmup and context installation excluded",
                  prefill_metric_boundary="prompt/TTFT proxy includes first-token and API processing, not pure phase compute",
                  missing_gates=["held_out_quality", "ANE_on_off", "serving_reliability", "energy", "thermal_soak", "fleet"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-generated", action="store_true")
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--output-tokens", type=int, choices=(64, 128), default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_ssd_reserve(Path.home() / "Library/Application Support/Strata")
    if args.worker:
        try:
            report = worker(args)
        except Exception as error:
            report = {"schema_version": 1, "success": False, "promotion_eligible": False,
                      "evidence_kind": "failed_experiment", "ssd_offload": "disabled",
                      "failure_reason": str(error), "failure_type": type(error).__name__,
                      "created_at": datetime.now(timezone.utc).isoformat(), **source_identity()}
        print(json.dumps(report, allow_nan=False))
        return 0 if report["success"] else 1
    if args.output and args.output.exists():
        raise ValueError("existing receipt; choose a fresh output")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--rounds", str(args.rounds),
               "--output-tokens", str(args.output_tokens)]
    if args.check_generated:
        command.append("--check-generated")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))))
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env,
                                timeout=120 if args.check_generated else 900)
        report = json.loads(result.stdout)
        if result.returncode and report.get("success") is not False:
            raise ValueError("worker failed without a failure receipt")
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        report = {"schema_version": 1, "success": False, "promotion_eligible": False,
                  "evidence_kind": "failed_experiment", "failure_type": type(error).__name__,
                  "failure_reason": "worker exceeded its deadline or emitted malformed output",
                  **source_identity()}
    if report.get("promotion_eligible") is not False:
        raise ValueError("invalid or promotable experiment receipt")
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")
    return 0 if report.get("success") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
