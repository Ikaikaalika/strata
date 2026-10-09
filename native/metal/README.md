# Lōkahi native Metal capability probe

This probe is a correctness-first vertical slice of the future native Metal
backend. It compiles a `float32` fused RMSNorm-plus-residual kernel into a
`.metallib`, dispatches it on generated data, waits for completion, and compares
the output with a CPU reference.

```sh
./native/metal/build_probe.sh
```

The final stdout line is machine-readable JSON. Timing is deliberately modest:
three untimed warmups followed by twenty serial dispatches. The reported wall
time includes command construction, commit, GPU execution, synchronization, and
completion handling. It excludes compilation, library loading, allocation, and
warmup.

This is not yet an optimized LLM kernel. One GPU thread performs the reduction
for one row, so the numerical and native-dispatch evidence is meaningful while
the performance number is only a baseline for a later SIMD/threadgroup
reduction.

## Native FP16 projection benchmark

The projection benchmark fixes one operator contract shared with the current
bounded ANE probe: input `[64, 256]`, `out_in` weight `[256, 256]`, FP16 storage,
FP32 accumulation, and output `[64, 256]`.

```sh
./native/metal/build_linear_bench.sh
```

It reports a native scalar CPU oracle, untiled and threadgroup-tiled direct-Metal
wall/device times, numerical error, hardware identity, revision, warmups,
iterations, and timing boundaries as JSON. The tiled variant stages a
16-output by 8-token tile over 32-wide reduction slices. A phase-program proxy
also encodes 16 independent tiled projections into one command buffer to
measure submission amortization. All variants remain learning kernels. This is
generated-fixture L2 kernel evidence, not a model tokens-per-second claim.
