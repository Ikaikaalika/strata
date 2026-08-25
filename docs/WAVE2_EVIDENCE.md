# Wave 2 evidence: adaptive planning, weight packs, and direct ANE execution

Wave 2 converts three architectural ideas into bounded, testable mechanisms.
The machine-readable record is
`benchmarks/results/apple_m1_wave2.json`.

## Evidence-gated adaptive planner

`AdaptivePlanner` emits separate whole-graph prefill and decode assignments.
MLX remains the fallback. A non-MLX target must satisfy the graph capability
contract and provide passing correctness evidence for the exact model hash,
graph, quantization, batch, context bucket, hardware fingerprint, phase, and
target. Synthetic performance data is ignored. An alternative displaces a
capable MLX baseline only when comparable hardware token-throughput evidence
shows it is faster.

This is the first scheduling policy, not the final partitioner. It does not yet
split one phase into multiple operation segments or account for transfer cost,
memory pressure, energy, or thermal state.

## Versioned weight pack

The version-one weight pack is a runtime-neutral single file with:

- a fixed, checksummed header
- a strict canonical JSON manifest
- aligned opaque tensor payloads
- per-tensor SHA-256 checksums
- exact-range reads and context-managed read-only `mmap`
- same-directory temporary writes followed by `fsync` and atomic replacement

A bounded 16 MiB generated payload was written in about `21.94 ms`. Two
checksum-verified host reads took about `9.70 ms` and `10.02 ms`. The file had
just been written and no cache eviction was attempted, so these are host-path
observations rather than cold-SSD bandwidth measurements. They do not prove
direct SSD-to-GPU or SSD-to-ANE transfer.

## Direct private-ANE projection proof

On the local Apple M1 running macOS 26.5.2 build 25F84, the isolated native
worker generated a MIL 1.0/ios16 program and weight blob for a fixed fp16
projection. It then completed this lifecycle:

```text
MIL + weights generated
  -> _ANEInMemoryModelDescriptor created
  -> private compile succeeded
  -> private load succeeded
  -> IOSurface request created
  -> private evaluate succeeded
  -> output compared with CPU reference
  -> private unload and temporary-artifact cleanup
```

The tensor shape was `[1, 256, 1, 64]`. The maximum absolute error was `0.0`
against the deterministic CPU calculation, within a declared tolerance of
`0.002`. One synchronous dispatch observation was about `0.336 ms`; it is not a
benchmark because there were no warmups or repeated measurements.

The proof qualifies the ANE compute unit in the fingerprinted hardware profile,
but it deliberately exposes no operations to the adaptive planner. A general
Strata segment executor, exact supported-shape envelopes, compiled-program
cache, and multi-operator corpus are still required before an LLM operation can
be scheduled there.

The implementation uses private, undocumented Apple frameworks and is expected
to break across some OS or chip revisions. Qualification is explicit opt-in,
and the runtime fingerprint changes with the macOS build and observed private
surface.

## Claim boundary

Wave 2 proves:

- deterministic, evidence-scoped backend selection policy
- recoverable, validated weight-pack storage mechanics
- direct private-ANE compile, IOSurface dispatch, and numerical readback for one
  fixed projection on this M1

It does not yet prove a full transformer block, any real model, SSD paging
overlapped with compute, end-to-end token throughput, energy improvement, or
simultaneous CPU/GPU/ANE utilization.
