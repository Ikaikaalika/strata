"""Opt-in compiled CLI validation. Invalid requests must never reach ANE."""
import json
import os
from pathlib import Path
import subprocess

import pytest

BUILD = os.environ.get("STRATA_ANE_EXPERIMENT_BUILD")
pytestmark = pytest.mark.skipif(not BUILD, reason="set STRATA_ANE_EXPERIMENT_BUILD to compiled experiment directory")


def run(name, args):
    return subprocess.run([str(Path(BUILD) / name), *map(str, args)], capture_output=True, text=True, timeout=3)


@pytest.mark.parametrize("args", [[], ["1", "/private/tmp", "tanh", "16"],
    ["64", "/private/tmp", "unknown", "16"], ["64", "/private/tmp", "tanh", "32"],
    ["64x", "/private/tmp", "tanh", "16"]])
def test_swiglu_cli_bounds(args):
    result = run("swiglu-bench", args)
    assert result.returncode == 64
    assert not result.stdout


@pytest.mark.parametrize("args", [[], ["1", "4", "fused", "/private/tmp/unused.bin"],
    ["64", "8", "fused", "/private/tmp/unused.bin"], ["64", "4", "unknown", "/private/tmp/unused.bin"]])
def test_chain_cli_bounds(args):
    result = run("prefill-chain-bench", args)
    assert result.returncode == 64
    assert not result.stdout


def test_missing_fixture_rejected_before_dispatch(tmp_path):
    result = run("swiglu-bench", [64, tmp_path, "tanh", 16])
    receipt = json.loads(result.stdout)
    assert result.returncode == 1
    assert not receipt["success"]
    assert receipt["verified_direct_ane_dispatches"] == 0
    assert receipt["cleanup_ok"]
    assert not (tmp_path / "output.bin").exists()


def test_existing_output_preserved(tmp_path):
    output = tmp_path / "output.bin"
    output.write_bytes(b"owned-existing-evidence")
    result = run("swiglu-bench", [64, tmp_path, "tanh", 16])
    receipt = json.loads(result.stdout)
    assert result.returncode == 1
    assert not receipt["success"] and receipt["verified_direct_ane_dispatches"] == 0
    assert output.read_bytes() == b"owned-existing-evidence"


def test_symlink_fixture_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    result = run("swiglu-bench", [64, alias, "tanh", 16])
    receipt = json.loads(result.stdout)
    assert result.returncode == 1
    assert receipt["verified_direct_ane_dispatches"] == 0
    assert "nonsymlink" in receipt["error"]
