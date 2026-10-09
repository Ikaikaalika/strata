#!/usr/bin/env python3
"""Select a paging plan from comparable, correctness-gated hardware runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from lokahi.planning import (
    PagingTuningCandidate,
    PagingTuningObjective,
    select_paging_candidate,
)


def _fingerprint(report: dict[str, Any]) -> tuple[Any, ...]:
    model = report.get("model", {})
    hardware = report.get("hardware", {})
    workload = report.get("workload", {})
    return (
        report.get("target_id"),
        model.get("repo"),
        model.get("revision"),
        hardware.get("chip"),
        hardware.get("macos_build"),
        workload.get("prompt_tokens_per_sequence"),
        workload.get("output_tokens_per_sequence"),
        workload.get("batch_size"),
        workload.get("sampling"),
        workload.get("prefix_cache"),
    )


# Raw reports recorded before the Strata -> Lokahi rename keep their original
# runner identifier; benchmarks/results/ is immutable evidence.
PAGED_MLX_RUNNERS = frozenset({"lokahi-paged-mlx-lab", "strata-paged-mlx-lab"})


def report_to_candidate(path: Path, report: dict[str, Any]) -> PagingTuningCandidate:
    if report.get("runner") not in PAGED_MLX_RUNNERS:
        raise ValueError(f"{path}: candidate is not a Lokahi paged MLX report")
    runtime = report.get("weight_residency", {}).get("runtime", {})
    summary = report.get("summary", {})
    correctness = report.get("correctness", {})
    required_runtime = (
        "pinned_global_weight_bytes",
        "layer_residency_budget_bytes",
        "prefetch_distance",
        "io_workers",
        "pinned_layer_count",
    )
    missing = [name for name in required_runtime if name not in runtime]
    if missing:
        raise ValueError(f"{path}: runtime metadata is missing {missing}")
    total_cap = int(runtime["pinned_global_weight_bytes"]) + int(
        runtime["layer_residency_budget_bytes"]
    )
    candidate_id = (
        f"cap{total_cap // (1024 * 1024)}mib-"
        f"pd{runtime['prefetch_distance']}-io{runtime['io_workers']}-"
        f"pin{runtime['pinned_layer_count']}"
    )
    return PagingTuningCandidate(
        candidate_id=candidate_id,
        max_resident_weight_bytes=total_cap,
        prefetch_distance=int(runtime["prefetch_distance"]),
        io_workers=int(runtime["io_workers"]),
        pinned_layer_count=int(runtime["pinned_layer_count"]),
        exact_token_parity=bool(
            correctness.get("mlx_lm_exact_greedy_token_parity")
            and correctness.get("repetition_token_invariance")
            and correctness.get("requested_token_count_passed")
        ),
        decode_tokens_per_second=float(
            summary["decode_tokens_per_second"]["median"]
        ),
        ttft_ms=float(summary["ttft_ms"]["median"]),
        peak_memory_gb=float(summary["peak_memory_gb"]["median"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--max-peak-memory-gb", type=float)
    parser.add_argument("--max-ttft-ms", type=float)
    parser.add_argument("--min-decode-tokens-per-second", type=float)
    parser.add_argument("--decode-weight", type=float, default=1.0)
    parser.add_argument("--ttft-weight", type=float, default=0.0)
    parser.add_argument("--memory-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = []
    for raw_path in args.candidate:
        path = raw_path.expanduser().resolve()
        reports.append((path, json.loads(path.read_text(encoding="utf-8"))))
    fingerprints = {_fingerprint(report) for _, report in reports}
    if len(fingerprints) != 1:
        raise ValueError("candidate model, hardware, and workload fingerprints differ")

    candidates = tuple(
        report_to_candidate(path, report) for path, report in reports
    )
    objective = PagingTuningObjective(
        max_peak_memory_gb=args.max_peak_memory_gb,
        max_ttft_ms=args.max_ttft_ms,
        min_decode_tokens_per_second=args.min_decode_tokens_per_second,
        decode_weight=args.decode_weight,
        ttft_weight=args.ttft_weight,
        memory_weight=args.memory_weight,
    )
    decision = select_paging_candidate(candidates, objective)
    source_by_id = {
        candidate.candidate_id: str(path)
        for candidate, (path, _) in zip(candidates, reports)
    }
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "hardware-evidence-synthesis",
        "runner": "lokahi-paged-mlx-tuner",
        "fingerprint": list(next(iter(fingerprints))),
        "objective": asdict(objective),
        "selected": {
            **asdict(decision.selected),
            "source": source_by_id[decision.selected.candidate_id],
        },
        "evaluations": [
            {
                "candidate": asdict(evaluation.candidate),
                "eligible": evaluation.eligible,
                "score": evaluation.score,
                "rejection_reasons": list(evaluation.rejection_reasons),
                "source": source_by_id[evaluation.candidate.candidate_id],
            }
            for evaluation in decision.evaluations
        ],
        "claim_boundary": (
            "Deterministic selection from supplied comparable measurements; "
            "not a new hardware run and not transferable to another fingerprint."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if not output.parent.is_dir():
            raise ValueError("output parent directory must exist")
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
