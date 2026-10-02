"""Dry-run Make contracts: approved SSD build paths contain spaces."""
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("directory,makefile,target", [
    ("ane", "Makefile.experiments", "all"),
    ("planning", "Makefile", "test"),
    ("metal", "Makefile.q4", "test"),
])
def test_native_make_keeps_build_paths_with_spaces_intact(tmp_path, directory, makefile, target):
    make = shutil.which("make")
    if make is None:
        pytest.skip("Make is unavailable; this is a build-file parsing contract")
    build_dir = tmp_path / "Application Support" / "Strata" / "build"
    result = subprocess.run(
        [make, "-n", "-C", str(ROOT / "native" / directory), "-f", makefile,
         target, f"BUILD_DIR={build_dir}"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr.strip(), result.stderr
    compiler_outputs = []
    for line in result.stdout.splitlines():
        args = shlex.split(line)
        if "-o" in args:
            compiler_outputs.append(args[args.index("-o") + 1])
        if args[:2] == ["mkdir", "-p"]:
            assert args[2:] == [str(build_dir)]
    assert compiler_outputs
    assert all(Path(output).parent == build_dir for output in compiler_outputs)
    assert not build_dir.exists(), "Dry-run validation must not create build outputs"
