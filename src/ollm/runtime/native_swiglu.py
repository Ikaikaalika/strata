"""Opt-in model-connected Metal experiment; not a qualified serving backend.

Only exact MLX-LM Qwen3 dense MLPs are eligible. All tensor math is in the
packaged Metal source or unchanged MLX operators. No weights are expanded,
copied into NumPy, concatenated, or requantized by this binding.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from importlib.resources import files
import math
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Iterator

VARIANTS = ("simd_decode", "tiled_prefill", "fused_both")
RESERVE_BYTES = 40 * 1024**3
MAX_BYTES = 64 * 1024**2


@dataclass
class NativeSwiGLUStats:
    installed_layers: int = 0
    simd_graph_calls: int = 0
    tiled_graph_calls: int = 0
    fallback_calls: int = 0
    theoretical_intermediate_bytes_eliminated: int = 0


def select_kernel(rows: int, width: int, outputs: int, variant: str) -> str | None:
    if variant not in VARIANTS:
        raise ValueError("unknown bounded SwiGLU candidate")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (rows, width, outputs)):
        raise ValueError("operator dimensions must be integers")
    if not (1 <= rows <= 512 and 64 <= width <= 8192 and width % 64 == 0 and 1 <= outputs <= 16384):
        return None
    # Unchanged two-byte model storage, two packed matrices plus group metadata.
    resident = 2 * (outputs * width // 2 + 4 * outputs * (width // 64)) + 2 * rows * (width + outputs)
    if resident > MAX_BYTES:
        return None
    if rows == 1 and variant in ("simd_decode", "fused_both"):
        return "simd"
    if rows > 1 and variant in ("tiled_prefill", "fused_both"):
        return "tiled"
    return None


def _require_reserve() -> None:
    if shutil.disk_usage(Path.home()).free < RESERVE_BYTES:
        raise ValueError("native model experiment requires 40 GiB free internal SSD")


def _require_versions() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError("native model experiment requires Apple Silicon macOS")
    if (version("mlx"), version("mlx-lm")) != ("0.32.2", "0.31.3"):
        raise ValueError("native model experiment requires its pinned MLX/MLX-LM versions")


def _types():
    import mlx.nn as nn
    from mlx_lm.models import qwen3
    return nn, qwen3


def _validate_mlp(mlp: Any, nn: Any, qwen3: Any) -> tuple[int, int]:
    if type(mlp) is not qwen3.MLP:
        raise ValueError("only exact dense Qwen3 SwiGLU MLPs are eligible")
    gate, up = mlp.gate_proj, mlp.up_proj
    for projection in (gate, up):
        if type(projection) is not nn.QuantizedLinear:
            raise ValueError("gate/up require unmodified QuantizedLinear operators")
        if (projection.mode, projection.bits, projection.group_size) != ("affine", 4, 64):
            raise ValueError("gate/up require affine Q4 group64, not MXFP4")
        if "bias" in projection:
            raise ValueError("linear output biases are unsupported")
    outputs, words = gate.weight.shape
    width = words * 8
    if select_kernel(1, width, outputs, "fused_both") is None:
        raise ValueError("MLP lies outside the bounded weight contract")
    for projection in (gate, up):
        if (projection.weight.shape != (outputs, words) or
                projection.scales.shape != (outputs, width // 64) or
                projection.biases is None or projection.biases.shape != projection.scales.shape or
                str(projection.weight.dtype) != "mlx.core.uint32" or
                str(projection.scales.dtype) not in ("mlx.core.float16", "mlx.core.bfloat16") or
                projection.scales.dtype != projection.biases.dtype or
                projection.scales.dtype != gate.scales.dtype):
            raise ValueError("gate/up buffer layout or storage dtype is unsupported")
    return width, outputs


def metal_source() -> str:
    return files("ollm.kernels").joinpath("affine_q4_swiglu.metal").read_text(encoding="utf-8")


@lru_cache(maxsize=2)
def _kernel(kind: str):
    import mlx.core as mx
    if kind not in ("simd", "tiled"):
        raise ValueError("unknown Metal entry point")
    location = ("threadgroup_position_in_grid.xy, thread_index_in_simdgroup, threads_per_simdgroup"
                if kind == "simd" else "threadgroup_position_in_grid.xy, thread_position_in_threadgroup.xy, scratch, scratch + 256, scratch + 768")
    scratch = "" if kind == "simd" else "threadgroup float scratch[1280];\n"
    return mx.fast.metal_kernel(
        name="strata_q4_swiglu_" + kind,
        input_names=["x", "gq", "gs", "gb", "uq", "us", "ub", "dims"],
        output_names=["y"], header=metal_source(),
        source=scratch + f"strata_swiglu::{kind}<T>(x, gq, gs, gb, uq, us, ub, y, dims[0], dims[1], dims[2], {location});",
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=32)
def _dimensions(rows: int, width: int, outputs: int):
    import mlx.core as mx
    return mx.array([rows, width, outputs], dtype=mx.uint32)


def _activation(x, mlp, width, outputs, kind):
    rows = math.prod(x.shape[:-1])
    g, u = mlp.gate_proj, mlp.up_proj
    grid = (outputs * 32, rows, 1) if kind == "simd" else (((outputs + 15) // 16) * 16, ((rows + 7) // 8) * 8, 1)
    group = (32, 1, 1) if kind == "simd" else (16, 8, 1)
    return _kernel(kind)(
        inputs=[x.reshape(rows, width), g.weight, g.scales, g.biases,
                u.weight, u.scales, u.biases, _dimensions(rows, width, outputs)],
        template=[("T", x.dtype)], grid=grid, threadgroup=group,
        output_shapes=[(*x.shape[:-1], outputs)], output_dtypes=[x.dtype],
    )[0]


@contextmanager
def experimental_qwen3_swiglu(model: Any, *, variant: str = "fused_both",
                              enabled: bool = False) -> Iterator[NativeSwiGLUStats]:
    """Temporarily substitute a candidate; exclusive model ownership required.

    Fresh KV state is required after a candidate failure. Unsupported call shapes
    take the original MLP before any native work; dispatch failures propagate.
    Restoration never attempts to resume a partially failed inference request.
    This context is inference-only: do not train, save, shard, or share its model.
    """
    stats = NativeSwiGLUStats()
    if variant not in VARIANTS:
        raise ValueError("unknown bounded SwiGLU candidate")
    if not enabled:
        yield stats
        return
    _require_reserve()
    _require_versions()
    nn, qwen3 = _types()
    if type(model) is not qwen3.Model or model.model_type != "qwen3":
        raise ValueError("model adapter supports exact dense Qwen3 only")
    layers = model.model.layers
    if not 1 <= len(layers) <= 64:
        raise ValueError("model layer count is outside the bounded candidate")
    # Validate every selected layer before mutating any model structure.
    originals = [(layer, layer.mlp, *_validate_mlp(layer.mlp, nn, qwen3)) for layer in layers]

    class CandidateMLP(nn.Module):
        def __init__(self, original, width, outputs):
            super().__init__()
            self.original = original  # Retain parameters and fallback without copies.
            self._width, self._outputs = width, outputs

        def __call__(self, x):
            rows = math.prod(x.shape[:-1]) if len(x.shape) >= 2 else 0
            kind = select_kernel(rows, self._width, self._outputs, variant)
            if (len(x.shape) != 3 or x.shape[0] != 1 or x.shape[-1] != self._width or
                    x.dtype != self.original.gate_proj.scales.dtype or kind is None):
                stats.fallback_calls += 1
                return self.original(x)
            activation = _activation(x, self.original, self._width, self._outputs, kind)
            if kind == "simd":
                stats.simd_graph_calls += 1
            else:
                stats.tiled_graph_calls += 1
            stats.theoretical_intermediate_bytes_eliminated += 4 * rows * self._outputs
            return self.original.down_proj(activation)

    try:
        for layer, original, width, outputs in originals:
            layer.mlp = CandidateMLP(original, width, outputs)
            stats.installed_layers += 1
        yield stats
    finally:
        for layer, original, _, _ in originals:
            layer.mlp = original
