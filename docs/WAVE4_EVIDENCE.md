# Wave 4 Evidence: Model Semantics and Phase-Program Amortization

Status: generated-fixture correctness, synthetic planning evidence, and bounded
Apple M1 L2 kernel hardware evidence recorded on 2026-08-24. This is not a real
model, token-throughput, ANE, SSD, or Common Compute benchmark.

## What was built

- strict schema-v1 JSON round trips for `PortableModelManifest`;
- generated dense and GPT-OSS-style MoE manifests with no real weights;
- layer-exact logical KV estimates for dense and windowed attention;
- a direct-Metal phase-program proxy that encodes 16 tiled projections into one
  command buffer and compute encoder;
- corrected single-dispatch wall timing that now includes command-buffer and
  encoder creation and encoding.

## Correctness evidence

The manifest suite rejects unknown schema fields and invalid model semantics.
The GPT-OSS-style schedule contains 12 windowed and 12 dense layers. At 131,072
tokens, batch 1, FP16 KV, the logical estimate is:

| Estimate | Bytes | GiB |
|---|---:|---:|
| All 24 layers dense | 6,442,450,944 | 6.0000 |
| Alternating 128-token window/dense | 3,224,371,200 | 3.0029 |
| Logical reduction | 3,218,079,744 | 2.9971 |

This is a 49.9512% logical KV reduction. It excludes allocator alignment,
backend padding, temporary score tensors, and other runtime state, so it is not
a physical peak-memory claim.

All three native Metal lanes matched the native scalar FP32-accumulation oracle
after FP16 cast with maximum and mean absolute error equal to zero for this
generated analytic fixture.

## Hardware evidence

Raw record:
[`apple_m1_phase_program_v1.json`](../benchmarks/results/apple_m1_phase_program_v1.json)

```text
chip / memory:          Apple M1 / 16 GiB unified memory
macOS:                  26.5.2 build 25F84
dtype / shape:          FP16 [64,256] @ [256,256].T
accumulation:           FP32
warmups / iterations:   5 / 50
thermal state:          nominal after run
repository state:       dirty, revision 111b3b9
```

| Lane | Median wall ms/projection | Median device ms/projection | Wall GFLOP/s |
|---|---:|---:|---:|
| Untiled, one command buffer | 0.34675 | 0.14794 | 24.19 |
| Tiled, one command buffer | 0.26238 | 0.06423 | 31.97 |
| Tiled, 16-dispatch phase proxy | 0.07493 amortized | 0.06079 amortized | 111.95 |

The phase proxy reduced amortized wall latency 71.44% relative to the corrected
single tiled dispatch. Device latency changed only modestly. The evidence says
submission/synchronization amortization is the larger available improvement at
this envelope.

## Interpretation

The phase proxy is deliberately independent repeated work. It does not model
transformer dependencies, intermediate-buffer ownership, KV updates, weight
changes, or token throughput. It establishes one architectural decision:
Strata should encode persistent multi-operation prefill and decode epochs rather
than synchronize after each primitive.

The next hardware milestone is a generated transformer-block phase program:

1. RMSNorm;
2. QKV projection;
3. RoPE and bounded attention with persistent KV;
4. output projection plus residual;
5. feed-forward projections plus residual;
6. one completion boundary for the whole block or safe multi-block epoch.

Only after that passes correctness should Strata add MXFP4 MoE expert kernels,
router grouping/deduplication, selective ANE candidates, or real-model timing.

## Reproduction

```sh
PYTHONPATH=src /usr/bin/python3 benchmarks/benchmark_model_planning.py --iterations 500
./native/metal/build_linear_bench.sh
```

The first command is synthetic evidence. The second is serialized L2 hardware
evidence. Neither command uses `/Volumes/Tyler HDD` as an SSD target.

