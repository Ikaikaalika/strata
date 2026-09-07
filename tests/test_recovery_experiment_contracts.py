"""Offline contracts for bounded recovery experiments; no hardware timing."""
import importlib.util
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


baseline = module("recovered_mlx_baseline")
chain = module("compare_ane_prefill_chain")
swiglu = module("compare_ane_swiglu")


@pytest.mark.parametrize("actual,expected", [
    ([float("nan")], [0]), ([float("inf")], [0]),
    ([0], [float("nan")]), ([0], [float("inf")]), ([1], [0]),
])
def test_component_gate_rejects_invalid_or_inaccurate(actual, expected):
    with pytest.raises(ValueError):
        chain.compare_output(np.array(actual), np.array(expected))


def test_component_gate_accepts_identical():
    x = np.array([0, -.25, 1.5], dtype=np.float16)
    assert chain.compare_output(x, x) == 0
    assert swiglu.errors(x, x) == {"max_abs": 0, "max_scaled": 0, "relative_l2": 0}


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        chain.compare_output(np.zeros((1, 2)), np.zeros((2, 1)))
    with pytest.raises(ValueError):
        swiglu.errors(np.zeros(2), np.zeros(3))


@pytest.mark.parametrize("side", ("actual", "expected"))
def test_swiglu_nonfinite_rejected(side):
    arrays = {"actual": np.zeros(3), "expected": np.zeros(3)}
    arrays[side][0] = float("nan")
    with pytest.raises(ValueError):
        swiglu.errors(**arrays)


def test_dense_fixture_repeatable_and_nontrivial():
    x, weights = chain.fixture(64, 2)
    y, second = chain.fixture(64, 2)
    assert x.shape == (64, 256) and x.dtype == np.float16
    assert np.array_equal(x, y)
    assert all(np.array_equal(a, b) for a, b in zip(weights, second))
    assert not np.array_equal(weights[0], weights[1])
    assert np.count_nonzero(weights[0]) > .95 * weights[0].size


@pytest.fixture
def pinned_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "MODELS", {"model": ("test/model", "a" * 40)})
    path = tmp_path / "model"
    metadata = path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    for name, data in (("config.json", b"{}"), ("model.safetensors", b"fixture-not-real-weights")):
        (path / name).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest() if name.endswith("safetensors") else hashlib.sha1(b"blob 2\0{}").hexdigest()
        (metadata / (name + ".metadata")).write_text("a" * 40 + "\n" + digest + "\n0\n")
    return path


def test_pinned_local_artifact_hashes(pinned_tree):
    path, receipt = baseline.artifact("model")
    assert path == pinned_tree
    assert len(receipt["files"]) == 2
    assert receipt["revision"] == "a" * 40


def test_tampered_bytes_rejected(pinned_tree):
    (pinned_tree / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        baseline.artifact("model")


def test_wrong_revision_rejected(pinned_tree):
    (pinned_tree / ".cache/huggingface/download/config.json.metadata").write_text("b" * 40 + "\nwrong\n")
    with pytest.raises(ValueError, match="unpinned"):
        baseline.artifact("model")


def test_model_symlink_rejected(pinned_tree, monkeypatch):
    link = pinned_tree.parent / "alias"
    link.symlink_to(pinned_tree, target_is_directory=True)
    monkeypatch.setattr(baseline, "MODELS", {"alias": ("test/model", "a" * 40)})
    with pytest.raises(ValueError, match="real directory"):
        baseline.artifact("alias")
