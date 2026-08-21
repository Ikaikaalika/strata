#!/usr/bin/env python3
"""Bounded host-side weight-pack write and exact-range read benchmark.

This benchmark does not measure direct SSD-to-GPU transfer, GPU residency,
prefetch overlap, or token throughput. Operating-system and device caches are
not flushed, so its first read is not a controlled cold-cache measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from ollm.storage.weight_pack import WeightPack, write_weight_pack


MAX_SIZE_MIB = 256.0


def _read_and_checksum(pack: WeightPack, name: str) -> tuple[int, str, float]:
    started = time.perf_counter()
    payload = pack.read_tensor(name, verify_checksum=False)
    checksum = hashlib.sha256(payload).hexdigest()
    elapsed_seconds = time.perf_counter() - started
    return len(payload), checksum, elapsed_seconds


def _measurement(byte_count: int, checksum: str, elapsed_seconds: float) -> dict:
    return {
        "bytes": byte_count,
        "wall_time_ms": elapsed_seconds * 1000.0,
        "effective_bytes_per_second": (
            byte_count / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "sha256": checksum,
    }


def _run(path: Path, size_bytes: int, alignment: int) -> dict:
    pattern = hashlib.sha256(b"strata-weight-pack-benchmark-v1").digest()
    payload = (pattern * ((size_bytes + len(pattern) - 1) // len(pattern)))[:size_bytes]
    expected_checksum = hashlib.sha256(payload).hexdigest()

    write_started = time.perf_counter()
    manifest = write_weight_pack(path, [("benchmark.weight", payload)], alignment=alignment)
    write_seconds = time.perf_counter() - write_started

    pack = WeightPack(path)
    first = _read_and_checksum(pack, "benchmark.weight")
    second = _read_and_checksum(pack, "benchmark.weight")
    if first[1] != expected_checksum or second[1] != expected_checksum:
        raise RuntimeError("benchmark read checksum did not match generated payload")

    return {
        "schema_version": 1,
        "kind": "host_weight_pack_io",
        "inputs": {
            "requested_payload_bytes": size_bytes,
            "alignment_bytes": alignment,
        },
        "pack": {
            "path": str(path),
            "file_bytes": manifest.file_size,
            "payload_bytes": size_bytes,
            "tensor_count": len(manifest.tensors),
            "sha256": expected_checksum,
        },
        "write": _measurement(size_bytes, expected_checksum, write_seconds),
        "first_exact_range_read": _measurement(*first),
        "second_exact_range_read": _measurement(*second),
        "cache_caveats": {
            "cold": (
                "No cache eviction is attempted. The file was just written, so the "
                "first read is an uncontrolled cache-state observation, not a cold SSD result."
            ),
            "warm": (
                "The immediate second read is a warm-cache candidate, but the OS controls "
                "page caching and no residency assertion is made."
            ),
        },
        "claim_boundary": (
            "Host-side atomic pack write plus exact-range CPU read and SHA-256 only; "
            "not direct SSD-to-GPU, accelerator execution, or token throughput."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size-mib",
        required=True,
        type=float,
        help=f"generated payload size in MiB; must be > 0 and <= {MAX_SIZE_MIB:g}",
    )
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument(
        "--path",
        type=Path,
        help="optional persistent output path; otherwise a temporary pack is removed",
    )
    args = parser.parse_args()
    if not 0 < args.size_mib <= MAX_SIZE_MIB:
        parser.error(f"--size-mib must be > 0 and <= {MAX_SIZE_MIB:g}")
    size_bytes = max(1, int(args.size_mib * 1024 * 1024))

    if args.path is not None:
        report = _run(args.path.expanduser(), size_bytes, args.alignment)
    else:
        with tempfile.TemporaryDirectory(prefix="strata-weight-pack-benchmark-") as directory:
            report = _run(
                Path(directory) / "benchmark.strata-pack",
                size_bytes,
                args.alignment,
            )
            report["pack"]["path"] = "temporary file removed after benchmark"
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
