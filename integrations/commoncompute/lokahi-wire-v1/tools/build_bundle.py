#!/usr/bin/env python3
"""Build a deterministic, checksummed Lokahi Common Compute integration ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from verify_kit import KIT_ROOT, verify_fixture_tree


BUNDLE_NAME = "lokahi-commoncompute-integration-v1"
SOURCE_PATHS = (
    "VERSION",
    "README.md",
    "INTEGRATION.md",
    "schemas",
    "fixtures",
    "swift",
    "tools/verify_kit.py",
    "tools/build_bundle.py",
)


def _files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in SOURCE_PATHS:
        path = root / relative
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and "__pycache__" not in candidate.parts
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"bundle source is missing: {relative}")
    return sorted(set(files), key=lambda item: item.relative_to(root).as_posix())


def _manifest(root: Path, files: list[Path]) -> dict:
    entries = []
    for path in files:
        data = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "bundle": BUNDLE_NAME,
        "version": (root / "VERSION").read_text(encoding="utf-8").strip(),
        "files": entries,
    }


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def build_bundle(root: Path, output_dir: Path) -> tuple[Path, str]:
    verify_fixture_tree(root)
    files = _files(root)
    manifest = _manifest(root, files)
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{BUNDLE_NAME}.zip"
    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=f".{BUNDLE_NAME}.", suffix=".zip", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            prefix = f"{BUNDLE_NAME}/"
            for path in files:
                _write_member(
                    archive,
                    prefix + path.relative_to(root).as_posix(),
                    path.read_bytes(),
                )
            _write_member(archive, prefix + "BUNDLE_MANIFEST.json", manifest_data)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return output, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=KIT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=KIT_ROOT.parent / "dist",
    )
    args = parser.parse_args()
    output, digest = build_bundle(args.root.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {"success": True, "output": str(output), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
