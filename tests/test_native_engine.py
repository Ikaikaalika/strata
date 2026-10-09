"""Correctness evidence: the native engine against the independent NumPy oracle.

The native CLI is built once per session with CMake (or taken from
``LOKAHI_NATIVE_BUILD_DIR``). Every comparison uses generated fixtures, so the
results are correctness evidence only, never throughput evidence.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lokahi.fixtures import TinyGemma3Spec, write_tiny_gemma3
from lokahi.oracles import Gemma3Reference

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "native" / "engine"
TOKENS = [2, 5, 9, 33, 100, 7, 7, 7, 400, 3, 2, 11, 90, 91, 92, 93, 17, 18, 19, 20, 21, 22]


@pytest.fixture(scope="session")
def native_cli(tmp_path_factory) -> Path:
    prebuilt = os.environ.get("LOKAHI_NATIVE_BUILD_DIR")
    if prebuilt:
        cli = Path(prebuilt) / "lokahi"
        if not cli.exists():
            pytest.fail(f"LOKAHI_NATIVE_BUILD_DIR has no lokahi binary: {cli}")
        return cli
    if shutil.which("cmake") is None or shutil.which("c++") is None:
        pytest.skip("cmake and a C++ compiler are required to build the native engine")
    build = tmp_path_factory.mktemp("native-build")
    subprocess.run(
        ["cmake", "-S", str(ENGINE), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "-j", str(os.cpu_count() or 2)],
        check=True,
        capture_output=True,
    )
    return build / "lokahi"


def _backends() -> list[str]:
    requested = os.environ.get("LOKAHI_TEST_BACKENDS", "cpu")
    return [name.strip() for name in requested.split(",") if name.strip()]


def _run(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


VARIANTS = {
    "untied-hd64-mixed-bits": TinyGemma3Spec(),
    "tied-hd256-rope-scaled": TinyGemma3Spec(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        tie_word_embeddings=True,
        rope_linear_factor=8.0,
        group_size=64,
        eight_bit_mlp_layers=(),
        seed=1,
    ),
    "language-model-prefix": TinyGemma3Spec(prefix="language_model.", seed=2),
}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def fixture_model(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param)
    write_tiny_gemma3(directory, VARIANTS[request.param])
    return directory, Gemma3Reference.load(directory)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize(("prefill", "chunk"), [(len(TOKENS), 512), (5, 512), (1, 512), (17, 3)])
def test_logits_match_oracle(native_cli, fixture_model, backend, prefill, chunk, tmp_path):
    directory, oracle = fixture_model
    out = tmp_path / "logits.bin"
    report = _run(
        native_cli,
        "logits",
        "--model", str(directory),
        "--backend", backend,
        "--tokens", ",".join(map(str, TOKENS)),
        "--prefill", str(prefill),
        "--prefill-chunk", str(chunk),
        "--out", str(out),
    )
    vocab = oracle.config.vocab_size
    got = np.fromfile(out, dtype=np.float32).reshape(report["rows"], vocab)
    expected = oracle.logits(TOKENS)[prefill - 1 :]
    assert got.shape == expected.shape
    tolerance = 2e-4 if backend == "cpu" else 2e-2
    np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max(), rtol=0)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))


@pytest.mark.parametrize("backend", _backends())
def test_greedy_generation_matches_full_recompute(native_cli, fixture_model, backend):
    directory, oracle = fixture_model
    prompt = TOKENS[:6]
    report = _run(
        native_cli,
        "generate",
        "--model", str(directory),
        "--backend", backend,
        "--tokens", ",".join(map(str, prompt)),
        "--max-new", "14",
        "--no-eos-stop",
    )
    assert report["tokens"] == oracle.greedy(prompt, 14)


def test_context_overflow_fails_closed(native_cli, tmp_path):
    directory = write_tiny_gemma3(tmp_path / "model", TinyGemma3Spec())
    result = subprocess.run(
        [
            str(native_cli), "logits", "--model", str(directory), "--backend", "cpu",
            "--tokens", "1,2,3,4,5", "--max-context", "4", "--out", str(tmp_path / "x.bin"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "context length exceeded" in result.stderr


def test_rejects_unsupported_architecture(native_cli, tmp_path):
    directory = write_tiny_gemma3(tmp_path / "model", replace(TinyGemma3Spec(), extra={"model_type": "llama"}))
    result = subprocess.run(
        [str(native_cli), "info", "--model", str(directory)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "unsupported architecture" in result.stderr
