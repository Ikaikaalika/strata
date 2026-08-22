# Wave 3 evidence: callable ANE segment and pack-backed residency

Wave 3 connects Wave 2's isolated mechanisms to runtime interfaces. The
machine-readable record is `benchmarks/results/apple_m1_wave3.json`.

## Exact execution envelopes

`OperationEnvelope` gives a fixed kernel an exact logical contract: operation
kind, ordered input/output shapes, and required attributes. The adaptive
planner checks the envelope before correctness or performance evidence.

The first direct-ANE envelope is:

```text
phase: prefill
operation: LINEAR
activation: [64 tokens, 256 input features], fp16
weight: [256 output features, 256 input features], fp16, out_in
output: [64 tokens, 256 output features], fp16
```

Symbolic shapes, decode, bias, other widths, other token counts, and other
layouts fail closed.

## Callable private-ANE request

`ANEProjectionExecutor` writes one bounded request in a private temporary
directory. The native worker validates exact regular-file sizes, rejects
symlink inputs and weights, requires distinct direct-child paths, and publishes
the output through an atomic rename. A request identifier binds the strict JSON
result to its caller.

The logical tensor is token-major `[64, 256]`, while the private runtime's
IOSurface is channel-major `[1, 256, 1, 64]`. The worker explicitly transposes
input into physical layout and output back into logical layout. A CPU-only
round-trip invariant guards that boundary before private execution.

One serialized Apple M1 request with deterministic random fp16 input and
weights completed compile, load, evaluate, logical readback, atomic output, and
cleanup. Against NumPy float32 accumulation followed by an fp16 cast:

```text
maximum absolute error: 0.000335693359375
mean absolute error:    0.0000614944874541834
evaluate-only dispatch: 0.31197071075439453 ms
compile-inclusive E2E:  366.167625 ms
warmups / iterations:   0 / 1
```

The dispatch value measures only the private evaluate call. The end-to-end
value includes Python request creation, process launch, MIL and weight-blob
generation, compilation, load, layout copies, readback, output publication,
unload, and temporary cleanup. This is correctness evidence plus one hardware
observation, not a throughput benchmark.

`ANEProjectionSegmentExecutor` now lowers the exact one-operation StrataIR
graph to that callable protocol. It publishes an exact prefill capability to
the planner, but the planner still requires matching correctness evidence and
comparable hardware token-throughput evidence before it can displace MLX.

## Pack-backed residency

`WeightPackGroupTensorStore` makes weight-pack ranges consumable by the existing
`PrefetchScheduler` and `ResidencyManager`. It resolves each `TensorRef` storage
key, verifies manifest length and SHA-256, then invokes a runtime decoder only
after every payload in the group has passed validation.

A deterministic three-group test exercised the real dense pipeline under an
8-byte budget. It loaded 12 bytes total, recorded one demand miss and two
prefetch outcomes, computed `1 + 2 + 3 = 6`, and evicted `layer.0` while the two
newest groups remained resident. Checksum and size failures released their
reservations instead of leaking budget.

The default decoder retains immutable host bytes. MLX, Metal, and ANE decoders
must still account for host-to-runtime copies and device allocation.

## Rejected HDD storage target

The repository lives on `/Volumes/Tyler HDD`, an external USB APFS volume behind
a device reported as `Dual SATA Bridge`. The user has confirmed that this volume
is an HDD. It is source-checkout storage only and is excluded from SSD offload,
storage qualification, and planner decisions.

A historical generated 16 MiB pack on that mount took about `268.6 ms` to write
(approximately `62.5 MB/s`). Immediate reads took about `10.19 ms` and
`10.33 ms`, but the file had just been written and no cache eviction occurred.
Those read numbers are cache-state observations from an HDD, not cold device or
SSD bandwidth. They must not be used to tune the planner or support SSD claims.
The generated file was removed after the run, and no future storage benchmark or
model/offload placement should target this volume.

## Next gate

The dominant ANE cost is now program compilation and process lifecycle, not the
single evaluate call. The next direct-ANE milestone is a restartable resident
worker with a compiled-program cache keyed by weights, exact envelope, chip,
and macOS build. Repeated correctness-checked evaluations can then establish
real end-to-end throughput.

For storage, the next gate is a decoder that creates MLX arrays from verified
pack ranges while tracing bytes, copy duration, residency, and demand stall.
Cold-media testing requires a separate user-approved SSD destination, must
fingerprint that target mount, and must control or clearly observe cache state.
