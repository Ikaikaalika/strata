"""Correctness evidence: the NumPy oracle against MLX and MLX-LM.

These tests pin the oracle's semantics to an independent implementation:
``mx.dequantize`` for the packed affine layout and MLX-LM's ``gemma3_text``
model for the full decoder. They need ``mlx`` and ``mlx_lm`` and therefore run
on Apple Silicon only.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lokahi.fixtures import TinyGemma3Spec, write_tiny_gemma3
from lokahi.formats import QuantSpec, dequantize_affine, pack_affine
from lokahi.oracles import Gemma3Reference

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None,
    reason="mlx and mlx_lm are required",
)

TOKENS = [2, 5, 9, 33, 100, 7, 7, 7, 400, 3, 2, 11, 90, 91, 92, 93, 17, 18, 19, 20]


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_affine_layout_matches_mx_dequantize(bits):
    import mlx.core as mx

    rng = np.random.default_rng(bits)
    rows, cols, group = 12, 256, 64
    codes = rng.integers(0, 1 << bits, size=(rows, cols), dtype=np.uint32)
    scales = rng.uniform(0.01, 0.05, size=(rows, cols // group)).astype(np.float32)
    biases = rng.uniform(-0.1, 0.1, size=(rows, cols // group)).astype(np.float32)
    words = pack_affine(codes, bits)
    ours = dequantize_affine(words, scales, biases, QuantSpec(group, bits))
    theirs = mx.dequantize(
        mx.array(words), mx.array(scales), mx.array(biases), group_size=group, bits=bits
    )
    np.testing.assert_allclose(np.array(theirs), ours, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "spec",
    [
        TinyGemma3Spec(),
        TinyGemma3Spec(
            hidden_size=256,
            num_key_value_heads=1,
            head_dim=256,
            tie_word_embeddings=True,
            rope_linear_factor=8.0,
            group_size=64,
            eight_bit_mlp_layers=(),
            seed=1,
        ),
    ],
    ids=["untied-hd64-mixed-bits", "tied-hd256-rope-scaled"],
)
def test_oracle_matches_mlx_lm_gemma3(tmp_path, spec):
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm.utils import load_model

    directory = write_tiny_gemma3(tmp_path / "model", spec)
    model, _ = load_model(directory)
    # Run MLX in float32 so the comparison isolates semantics from BF16 rounding.
    model.update(
        tree_map(
            lambda value: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value,
            model.parameters(),
        )
    )
    logits = np.array(model(mx.array([TOKENS]))[0].astype(mx.float32))
    expected = Gemma3Reference.load(directory).logits(TOKENS)
    np.testing.assert_allclose(logits, expected, rtol=0, atol=2e-3 * np.abs(expected).max())
    np.testing.assert_array_equal(logits.argmax(axis=1), expected.argmax(axis=1))
