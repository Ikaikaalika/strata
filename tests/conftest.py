"""Shared fixtures for the Lokahi test suite."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "native" / "engine"


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
