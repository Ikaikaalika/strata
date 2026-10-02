# Strata native Metal capability probe

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

## Packed affine Q4 projection candidates

`affine_q4.metal` adds a scalar control, a SIMD dot-product candidate for decode,
and an 8-token/16-output tiled candidate for prefill. Weights are unsigned Q4
packed little-nibble-first into uint32, out_in row-major, group size 64, with
FP16 scales/additive biases. Input/output storage is FP16; GPU accumulation and
bounded decoded tile scratch are FP32. This is distinct from GPT-OSS MXFP4.

Build into the approved internal SSD without running a benchmark:

```sh
sh native/metal/build_q4_bench.sh --build-only
make -C native/metal -f Makefile.q4 test \
  BUILD_DIR="/Users/tylergee/Library/Application Support/Strata/build/q4-test"
```

The CPU test checks 75 arithmetic/boundary cases under UBSan. `--check` performs
untimed generated-fixture GPU verification; `--benchmark` records order-rotated
native raw samples. Both hardware modes require 40 GiB free internal SSD before
creating a Metal device. The paired native/MLX runner is
`benchmarks/benchmark_native_q4.py`; it never downloads weights and always
records `promotion_eligible=false`. See the
[dated experiment](../../docs/experiments/CATALOG_NATIVE_LATENCY_20260930.md)
for numerical/timing boundaries and the pending complete-request gate.

The next [model-connected FFN candidate](../../docs/experiments/MODEL_CONNECTED_NATIVE_FFN.md)
fuses two packed projections and SwiGLU in packaged native Metal source at
`src/ollm/kernels/affine_q4_swiglu.metal`. `check_q4_swiglu.metal` instantiates
both FP16 entries for offline compilation. The MLX-hosted Qwen3 adapter is
experimental, disabled by default, and requires separate generated and
complete-model gates; compilation is not execution or speed proof.
