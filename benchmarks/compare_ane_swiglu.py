"""Real-weight, real-activation component test. No full-model promotion."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np
from recovered_mlx_baseline import ROOT, artifact, prompt_tokens


def errors(actual, expected):
    a, b = actual.astype(np.float32), expected.astype(np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid comparison arrays")
    delta = a - b
    return {"max_abs": float(np.max(np.abs(delta))),
            "max_scaled": float(np.max(np.abs(delta) / (.01 + .02 * np.abs(b)))),
            "relative_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(b)), 1e-12))}


def timed_mlx(function, x):
    import mlx.core as mx
    function = mx.compile(function)
    resident = mx.array(x)
    mx.eval(resident)
    result = {"dispatch_ms_samples": [], "resident_with_io_ms_samples": []}
    for boundary in result:
        for i in range(25):
            begin = time.perf_counter_ns()
            y = function(mx.array(x) if boundary == "resident_with_io_ms_samples" else resident)
            mx.eval(y)
            if boundary == "resident_with_io_ms_samples":
                out = np.array(y)
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            if i >= 5:
                result[boundary].append(elapsed)
    return result, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, choices=(64, 256, 512), default=[64, 256, 512])
    parser.add_argument("--layer", type=int, choices=(0, 14, 27), default=0)
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--activation", choices=("sigmoid", "tanh"), default="sigmoid")
    parser.add_argument("--layout-tile", type=int, choices=(0, 8, 16), default=0)
    parser.add_argument("--control-layout-tile", type=int, choices=(0, 8, 16))
    args = parser.parse_args()
    if args.control_layout_tile == args.layout_tile:
        parser.error("control layout must differ from candidate")
    if args.output.exists():
        parser.error("refusing to overwrite receipt")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(ROOT / ".hf"))
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from benchmark_mlx_lm import _hardware

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    root = Path(__file__).resolve().parents[1]
    report = {"schema_version": 1, "benchmark": "qwen3-real-swiglu-paired", "success": False,
              "promotion_eligible": False, "layer": args.layer, "rounds": args.rounds, "activation_variant": args.activation, "layout_tile": args.layout_tile,
              "warmups": 5, "iterations": 20, "seed": 730, "records": [], "summaries": [],
              "control_layout_tile": args.control_layout_tile,
              "ssd_offload": "disabled", "software": {n: version(n) for n in ("mlx", "mlx-lm", "numpy")},
              "probe_sha256": sha(args.probe), "sources": {str(p.relative_to(root)): sha(p) for p in
                  [Path(__file__), root/"benchmarks/recovered_mlx_baseline.py", root/"native/ane/swiglu_bench.m", root/"native/ane/strata_ane_probe.m", root/"native/ane/tensor_layout.h"]},
              "gate": {"fp16_reference_atol": .01, "fp16_reference_rtol": .02,
                       "quantized_relative_l2_max": .025},
              "claim_boundary": "One FFN, captured Qwen3 activation, FP16-expanded weights vs same quantized artifact; excludes attention, model-wide errors, caches, API serving and energy. Compile/load reported separately, not timed as resident execution."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        path, report["artifact"] = artifact("Qwen3-0.6B-4bit")
        report["hardware_before"] = _hardware()
        model, tokenizer = load(str(path), tokenizer_config={"trust_remote_code": False})
        block = model.model.layers[args.layer]
        q = block.mlp
        weights = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = getattr(q, name)
            weights.append(mx.dequantize(layer.weight, layer.scales, layer.biases,
                group_size=layer.group_size, bits=layer.bits, mode=layer.mode).astype(mx.float16))
        mx.eval(*weights)
        weight_host = [np.array(w) for w in weights]
        if [w.shape for w in weight_host] != [(3072, 1024), (3072, 1024), (1024, 3072)]:
            raise ValueError("unqualified model dimensions")

        def dense(x):
            g, u = x @ weights[0].T, x @ weights[1].T
            return ((g * mx.sigmoid(g)) * u) @ weights[2].T

        for tokens in args.tokens:
            prompt = prompt_tokens(tokenizer, tokens)
            h = model.model.embed_tokens(mx.array([prompt]))
            mask = create_attention_mask(h)
            for previous in model.model.layers[:args.layer]:
                h = previous(h, mask)
            activation = block.post_attention_layernorm(h + block.self_attn(block.input_layernorm(h), mask))
            mx.eval(activation)
            x = np.array(activation.astype(mx.float16)).reshape(tokens, 1024)
            original_output = np.array(q(activation).astype(mx.float32)).reshape(tokens, 1024)
            reference = np.array(dense(mx.array(x)))
            quantized_reference = np.array(q(mx.array(x)))
            report.setdefault("fixtures", []).append({"tokens": tokens, "prompt_token_ids": prompt,
                "original_activation_dtype": str(activation.dtype), "tested_activation_dtype": "float16",
                "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "weight_sha256": [hashlib.sha256(w.tobytes()).hexdigest() for w in weight_host],
                "fp16_vs_original_bf16": errors(reference, original_output),
                "fp16_vs_quantized_fp16": errors(reference, quantized_reference)})
            for round_id in range(args.rounds):
                order = ["ane_fused", "mlx_fp16_compiled", "mlx_quantized_compiled"]
                if args.control_layout_tile is not None:
                    order.append("ane_layout_control")
                offset = round_id % len(order)
                order = order[offset:] + order[:offset]
                if round_id % 2:
                    order.reverse()
                rows = []
                for name in order:
                    if name in ("ane_fused", "ane_layout_control"):
                        tile = args.layout_tile if name == "ane_fused" else args.control_layout_tile
                        with tempfile.TemporaryDirectory(prefix="strata-swiglu-", dir="/private/tmp") as scratch:
                            fixture = Path(scratch)
                            x.tofile(fixture/"input.bin")
                            reference.tofile(fixture/"reference.bin")
                            for n, w in zip(("gate", "up", "down"), weight_host):
                                w.tofile(fixture/(n+".bin"))
                            run = subprocess.run([str(args.probe.resolve()), str(tokens), scratch, args.activation, str(tile)],
                                capture_output=True, text=True, timeout=120)
                            row = json.loads(run.stdout)
                            row["stderr"] = run.stderr[-4000:]
                            row.update(variant=name, tokens=tokens, round=round_id, order=order)
                            report["records"].append(row)
                            if run.returncode or not row.get("success") or not row.get("cleanup_ok") or row.get("verified_direct_ane_dispatches") != 25:
                                raise ValueError(f"native gate failed: {row.get('error', run.stderr)}")
                            out = np.fromfile(fixture/"output.bin", dtype=np.float16).reshape(tokens, 1024)
                    else:
                        row, out = timed_mlx(dense if name == "mlx_fp16_compiled" else q, x)
                        row.update(variant=name, tokens=tokens, round=round_id, order=order)
                        report["records"].append(row)
                    row["fp16_error"] = errors(out, reference)
                    row["quantized_error"] = errors(out, quantized_reference)
                    row["original_bf16_error"] = errors(out, original_output)
                    if row["fp16_error"]["max_scaled"] > 1 or row["quantized_error"]["relative_l2"] > .025:
                        raise ValueError(f"component correctness rejected: {name}")
                    rows.append(row)
                print(json.dumps({"tokens": tokens, "round": round_id, "correctness": "pass",
                    "with_io_ms": {r["variant"]: statistics.median(r["resident_with_io_ms_samples"]) for r in rows}}), flush=True)
            rows = [r for r in report["records"] if r["tokens"] == tokens]
            medians = {n: statistics.median(statistics.median(r["resident_with_io_ms_samples"]) for r in rows if r["variant"] == n) for n in order}
            report["summaries"].append({"tokens": tokens, "resident_with_io_ms": medians,
                "mlx_quantized_over_ane": medians["mlx_quantized_compiled"] / medians["ane_fused"],
                "mlx_fp16_over_ane": medians["mlx_fp16_compiled"] / medians["ane_fused"]})
        report["hardware_after"] = _hardware()
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
