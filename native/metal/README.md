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
