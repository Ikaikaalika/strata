# Wave 1 evidence: CPU, Metal, and ANE foundations

Wave 1 established three deliberately different kinds of evidence on the local
Apple M1. The structured record is
`benchmarks/results/apple_m1_wave1.json`.

## CPU correctness oracle

`CPUReferenceExecutor` executes a bounded `LokahiIR` subset with deterministic
NumPy semantics and float32 accumulation. It is the independent comparison path
for accelerator work, not the optimized Accelerate/BNNS backend.

Proven operations:

- dense linear and logits projection
- RMSNorm
- residual addition
- SiLU

## Native Metal dispatch

The generated-data probe compiled and dispatched a fused float32 RMSNorm plus
residual kernel on the Apple M1. For shape `[64, 1024]`, the measured maximum
absolute error against its CPU reference was approximately `2.38e-7`.

Three warmups preceded twenty serial measurements. Average wall time was about
`0.842 ms`, including command construction, commit, execution, synchronization,
and completion handling. Compilation, library loading, allocation, and warmup
were excluded. This is a correctness and dispatch baseline; the reduction uses
one GPU thread per row and is not performance optimized.

## Private ANE discovery

The isolated native worker loaded `AppleNeuralEngine.framework` and
`ANECompiler.framework` on macOS 26.5.2 build 25F84. It observed `_ANEClient`,
`_ANEInMemoryModelDescriptor`, and `_ANEInMemoryModel` compile/load/evaluate
lifecycle methods.

No MIL graph was compiled or dispatched. Therefore:

```text
ANE discovered: yes
ANE execution verified: no
ANE numerically verified: no
ANE schedulable by Lōkahi: no
```

The next ANE gate is a generated fp16 MIL projection with IOSurface-backed
inputs and outputs, followed by CPU-oracle comparison.
