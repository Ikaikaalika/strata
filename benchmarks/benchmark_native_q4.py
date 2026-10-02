#!/usr/bin/env python3
"""Paired native Metal / MLX affine-Q4 component gate, never a runtime promotion.

No downloads or real weights. Arithmetic and submission execute in Metal/native
code or the independent MLX control; Python only stages fixtures and receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 1024, 128), (7, 128, 17), (64, 1024, 128))
KERNELS = {"affine_q4_scalar", "affine_q4_simd", "affine_q4_tiled"}
RESERVE_BYTES = 40 * 1024**3
ATOL, RTOL = 0.0005, 0.005


def fixture(shape: tuple[int, int, int]):
    if shape not in SHAPES:
        raise ValueError("shape is outside the frozen fixture matrix")
    rows, width, outputs = shape
    x = (((np.arange(rows * width, dtype=np.int64) * 17) % 113 - 56) / 128).astype(np.float16).reshape(rows, width)
    scales = ((np.arange(outputs * (width // 64)) % 7 + 1) / 1024).astype(np.float16).reshape(outputs, width // 64)
    biases = (-7 * scales).astype(np.float16)
    # Explicit code matrix is independent of the packed-word decoder.
    codes = ((np.arange(width)[None, :] * 5 + np.arange(outputs)[:, None] * 3) % 16).astype(np.uint32)
    packed = np.zeros((outputs, width // 8), dtype=np.uint32)
    for bit in range(8):
        packed |= codes[:, bit::8] << (4 * bit)
    weights = np.repeat(scales.astype(np.float64), 64, axis=1) * codes + np.repeat(biases.astype(np.float64), 64, axis=1)
    reference = (x.astype(np.float64) @ weights.T).astype(np.float16)
    return x, packed, scales, biases, reference


def numerical_gate(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite or mismatched operator output")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    scaled = delta / (ATOL + RTOL * np.abs(expected.astype(np.float64)))
    maximum = float(scaled.max())
    if maximum > 1:
        raise ValueError("frozen numerical gate failed")
    return {"max_abs": float(delta.max()), "max_scaled": maximum, "atol": ATOL, "rtol": RTOL}


def _device_id(path: Path) -> int:
    return path.stat().st_dev


def require_ssd_reserve(path: Path) -> int:
    # Refuse the known failed mount before resolving any path through it.
    path = path.absolute()
    if path.is_relative_to(Path("/Volumes/Tyler HDD")):
        raise ValueError("benchmark reserve path must be a real approved SSD directory")
    # Walk from the root: do not touch descendants of a symlink to the failed
    # mount. Home is the approved internal volume on this host; free HDD space
    # must not substitute for its reserve. This is not an SSD speed claim.
    for ancestor in (*reversed(path.parents), path):
        if ancestor.is_symlink():
            raise ValueError("benchmark reserve path must be a real approved SSD directory")
    if _device_id(path) != _device_id(Path.home()):
        raise ValueError("benchmark storage must be on the approved internal home volume")
    free = shutil.disk_usage(path).free
    if free < RESERVE_BYTES:
        raise ValueError(f"internal SSD reserve requires {RESERVE_BYTES} bytes; available {free}")
    return free


def validate_native_receipt(receipt: dict) -> None:
    if receipt.get("schema_version") != 1 or receipt.get("execution_verified") is not True:
        raise ValueError("native execution did not pass")
    if receipt.get("promotion_eligible") is not False or receipt.get("ssd_offload") != "disabled":
        raise ValueError("component evidence cannot promote serving or enable paging")
    if receipt.get("evidence_kind") != "hardware_component" or receipt.get("fixture_id") != "integer_affine_q4_v1":
        raise ValueError("wrong evidence or fixture contract")
    if receipt.get("dtype") != "fp16_input_scale_bias_output_fp32_accumulation" or receipt.get("bits") != 4 or receipt.get("group_size") != 64:
        raise ValueError("wrong dtype or quantization contract")
    if receipt.get("iterations") != 20 or receipt.get("warmups") != 5:
        raise ValueError("wrong sample bounds")
    for name in ("source", "contract", "host", "binary", "metallib"):
        digest = receipt.get(name + "_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("missing source/build identity")
    seen = set()
    for cell in receipt["cells"]:
        shape = tuple(cell["shape"])
        if shape not in SHAPES or shape in seen:
            raise ValueError("unexpected or duplicate fixture shape")
        seen.add(shape)
        variants = cell["variants"]
        if len(variants) != 3 or {v["kernel"] for v in variants} != KERNELS:
            raise ValueError("native control or candidate missing")
        for variant in variants:
            numerical = variant["numerical"]
            if numerical.get("passed") is not True or numerical.get("atol") != ATOL or numerical.get("rtol") != RTOL:
                raise ValueError("native numerical gate failed or changed")
            for key in ("max_abs", "max_scaled"):
                value = numerical[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid numerical evidence")
            if numerical["max_scaled"] > 1:
                raise ValueError("native numerical gate exceeded")
            samples = variant["wall_ms"]
            if len(samples) != 20 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("invalid native wall samples")
            gpu = variant["gpu_ms"]
            if len(gpu) != 20 or any(x is not None and (isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0) for x in gpu):
                raise ValueError("invalid native device samples")
    if seen != set(SHAPES):
        raise ValueError("native fixture matrix incomplete")


def mlx_control() -> list[dict]:
    import mlx.core as mx

    cells = []
    for shape in SHAPES:
        x, q, scales, biases, reference = fixture(shape)
        arrays = [mx.array(value) for value in (x, q, scales, biases)]
        mx.eval(*arrays)

        def run():
            y = mx.quantized_matmul(arrays[0], arrays[1], arrays[2], arrays[3],
                                    transpose=True, bits=4, group_size=64, mode="affine")
            mx.eval(y)
            mx.synchronize()
            return y

        for _ in range(5):
            run()
        samples = []
        for _ in range(20):
            started = time.perf_counter()
            y = run()
            samples.append((time.perf_counter() - started) * 1000)
        actual = np.array(y)
        if actual.dtype != np.float16:
            raise ValueError("MLX control must produce FP16 output")
        cells.append({"shape": list(shape), "wall_ms": samples,
                      "numerical": numerical_gate(actual, reference)})
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path.home() / "Library/Application Support/Strata/build/q4")
    parser.add_argument("--approved-ssd-root", type=Path, default=Path.home() / "Library/Application Support/Strata")
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    free = require_ssd_reserve(args.approved_ssd_root)
    if not args.build_dir.absolute().is_relative_to(args.approved_ssd_root.absolute()):
        raise ValueError("build directory must be inside the approved SSD root")
    if args.output and args.output.exists():
        raise ValueError("receipt already exists; choose a fresh path")
    require_ssd_reserve(args.build_dir.parent)
    env = dict(os.environ, STRATA_METAL_BUILD_DIR=str(args.build_dir))
    subprocess.run(["sh", str(ROOT / "native/metal/build_q4_bench.sh"), "--build-only"],
                   env=env, check=True, timeout=60)
    identities = {"source": ROOT / "native/metal/affine_q4.metal",
                  "contract": ROOT / "native/metal/affine_q4_contract.h",
                  "host": ROOT / "native/metal/affine_q4_bench.mm",
                  "binary": args.build_dir / "strata-q4-bench",
                  "metallib": args.build_dir / "strata-q4.metallib"}
    for name, path in identities.items():
        env["STRATA_Q4_" + name.upper() + "_SHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rounds = []
    for index in range(args.rounds):
        require_ssd_reserve(args.approved_ssd_root)
        import mlx.core as mx
        mx.synchronize()

        def native():
            run = subprocess.run([str(identities["binary"]), "--benchmark", str(identities["metallib"])],
                                 env=env, check=True, capture_output=True, text=True, timeout=60)
            receipt = json.loads(run.stdout)
            validate_native_receipt(receipt)
            return receipt

        if index % 2 == 0:
            native_receipt, control = native(), mlx_control()
        else:
            control, native_receipt = mlx_control(), native()
        rounds.append({"order": "native_then_mlx" if index % 2 == 0 else "mlx_then_native",
                       "native": native_receipt, "mlx": control})
    comparisons = []
    for shape in SHAPES:
        for kernel in sorted(KERNELS):
            ratios = []
            for round_ in rounds:
                mlx_cell = next(c for c in round_["mlx"] if tuple(c["shape"]) == shape)
                native_cell = next(c for c in round_["native"]["cells"] if tuple(c["shape"]) == shape)
                variant = next(v for v in native_cell["variants"] if v["kernel"] == kernel)
                ratios.append(statistics.median(mlx_cell["wall_ms"]) / statistics.median(variant["wall_ms"]))
            comparisons.append({"shape": list(shape), "kernel": kernel,
                                "paired_component_speedup_ratios": ratios, "median_ratio": statistics.median(ratios)})
    payload = {"schema_version": 1, "evidence_kind": "paired_hardware_component",
               "observed_at": datetime.now(timezone.utc).isoformat(),
               "promotion_eligible": False, "native_full_model_qualified": False,
               "ssd_offload": "disabled", "available_ssd_bytes": free,
               "versions": {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")},
               "ttft_ms": None, "decode_tokens_per_second": None,
               "timing_boundary": "resident operator encode/evaluate/wait; no model or request execution",
               "numerical_boundary": "each implementation versus independent FP64 affine formula, rounded FP16",
               "comparisons": comparisons, "rounds": rounds}
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
