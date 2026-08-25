# Common Compute Model Benchmark Ladder

Status: executable no-download qualification plan for the local Apple M1 with
16 GiB unified memory, observed 2026-08-24. The live catalog and provider pins
may change; refresh both before downloading or publishing results.

## Objective

Benchmark Strata against MLX-LM on the exact artifacts Common Compute already
pins. Use the current Mac for models with an 8 GB admission floor. Use a remote
M4 Max/128 GiB Mac for the exact Darkbloom GPT-OSS comparison and any Gemma
comparison that matches Darkbloom's separate snapshot.

The machine-readable source is
[`commoncompute_m1_model_ladder_v1.json`](../benchmarks/targets/commoncompute_m1_model_ladder_v1.json).

## Captured MLX-LM baseline

Hardware evidence captured 2026-08-25 UTC on an Apple M1 with 16 GiB unified
memory, macOS 26.5.2 build 25F84, MLX 0.31.1, and MLX-LM 0.31.2. Each result
uses a 512-token prompt, 128 forced greedy decode steps, batch 1, two warmups,
and five measured repetitions. Thermal state was nominal after every run and
each model produced an identical token sequence in all five repetitions.

| Model | Prompt tok/s mean | Decode tok/s mean | TTFT mean | ITL mean | Peak MLX memory | Aggregate tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B 4-bit | 1,083.30 | 81.39 | 517.52 ms | 12.04 ms | 0.823 GB | 62.58 |
| Llama 3.2 1B 4-bit | 978.40 | 69.94 | 560.76 ms | 14.00 ms | 1.211 GB | 54.40 |
| Gemma 3 1B QAT 4-bit | 1,260.29 | 58.21 | 451.89 ms | 17.14 ms | 1.101 GB | 49.13 |

Raw reports:

- [`apple_m1_mlx_qwen3_0_6b_b1_512_128_v1.json`](../benchmarks/results/apple_m1_mlx_qwen3_0_6b_b1_512_128_v1.json)
- [`apple_m1_mlx_llama_3_2_1b_b1_512_128_v1.json`](../benchmarks/results/apple_m1_mlx_llama_3_2_1b_b1_512_128_v1.json)
- [`apple_m1_mlx_gemma_3_1b_b1_512_128_v1.json`](../benchmarks/results/apple_m1_mlx_gemma_3_1b_b1_512_128_v1.json)

These are MLX-LM baselines, not Strata performance results. They establish the
first optimization targets: Qwen currently has the strongest decode lane,
while Gemma has the strongest prompt/TTFT lane on this workload. Strata must
compare phase by phase and end to end against the matching model's own baseline
rather than select one global scheduling policy from parameter count alone.

### Resident batch-one route selection

Strata now has a measured compatibility-backend route selector for the exact
staged revisions. It alternates the MLX-LM `BatchGenerator` control and a
single-sequence `generate_step` route, loads weights once with `lazy=False`,
keeps SSD off, and compares the same greedy token IDs using one host-wall timing
boundary. This is a serving-route improvement over the continuous-batching
control, not a new GPU kernel or a claim to outperform MLX-LM's own optimized
single-sequence API.

Full hardware validation on Apple M1, macOS build 25F84, MLX 0.31.1, and
MLX-LM 0.31.2 produced:

| Model/workload | Control end-to-end | Selected route | Selected end-to-end | Improvement | TTFT improvement | ITL improvement | Decision |
|---|---:|---|---:|---:|---:|---:|---|
| Qwen3 0.6B, 512/128, B1 | 64.72 output tok/s | direct | 71.20 output tok/s | +10.01% | +6.61% | +10.33% | promote direct |
| Llama 3.2 1B, 512/32, B1 | 29.52 output tok/s | direct | 30.97 output tok/s | +4.93% | +5.59% | +2.58% | promote direct |
| Llama 3.2 1B, 512/128, B1 | 55.46 output tok/s | batch | 54.23 output tok/s direct | -2.22% direct | +5.43% direct | -3.35% direct | retain batch |
| Gemma 3 1B, 512/128, B1 | 50.97 output tok/s | direct | 56.89 output tok/s | +11.63% | +1.97% | +11.98% | promote direct |

Every row used two warmups, five repetitions per route, alternating run order,
exact token parity, and ended at nominal thermal state. The selector requires
the exact target, artifact revision, MLX/MLX-LM versions, prompt/output shape,
batch one, one active sequence, and SSD-off state. Any mismatch—including
concurrency—falls back to `BatchGenerator`.

The executable selector is `src/ollm/runtime/resident_mlx.py`; its checked-in
profiles are
[`apple_m1_resident_mlx_route_profiles_v1.json`](../benchmarks/targets/apple_m1_resident_mlx_route_profiles_v1.json).
Raw full-protocol evidence:

- [`apple_m1_strata_resident_routes_qwen3_0_6b_b1_512_128_v1.json`](../benchmarks/results/apple_m1_strata_resident_routes_qwen3_0_6b_b1_512_128_v1.json)
- [`apple_m1_strata_resident_routes_llama_3_2_1b_b1_512_32_v1.json`](../benchmarks/results/apple_m1_strata_resident_routes_llama_3_2_1b_b1_512_32_v1.json)
- [`apple_m1_strata_resident_routes_llama_3_2_1b_b1_512_128_v1.json`](../benchmarks/results/apple_m1_strata_resident_routes_llama_3_2_1b_b1_512_128_v1.json)
- [`apple_m1_strata_resident_routes_gemma_3_1b_b1_512_128_v1.json`](../benchmarks/results/apple_m1_strata_resident_routes_gemma_3_1b_b1_512_128_v1.json)

### Llama residency comparison

The first Strata paged laboratory run used the exact Llama artifact and workload
with a 256 MiB weight cap. It retained exact greedy token parity and reduced
peak MLX memory from 1.211 GB to 0.592 GB, but decode fell from 69.94 to 3.75
tok/s because all 547.5 MB of decoder-layer weights were reread for every token.
The standalone promotion gate correctly fails. This mode expands capacity; it
does not beat MLX-LM performance.

A second recursive plan pinned five layers under a 384 MiB cap. It improved
paged decode 25.44%, reduced inter-token latency 20.62%, and cut layer traffic
31.25% while preserving exact tokens. Peak memory remained 39.84% below MLX-LM,
but decode was still 93.28% slower, so full residency remains the adaptive
choice whenever this model safely fits.

A bounded parameter sweep then varied the resident cap, prefetch distance,
I/O worker count, and automatically safe pinned prefix. The full-protocol
winner uses a 640 MiB cap, zero prefetch, one worker, and fourteen pinned
layers. It reaches 6.36 tok/s: 34.46% faster than the 384 MiB profile, with
23.97% lower ITL and 81.82% less layer traffic. Peak MLX memory rises to
1.002 GB, still 17.24% below the 1.211 GB MLX-LM control. Exact greedy tokens
match, but decode remains 90.92% below MLX-LM, so this is a tuned capacity
profile rather than the speed default.

The selector supports hard memory/TTFT/decode gates and flexible relative
weights for decode speed, TTFT, and memory. Decisions are reusable only for the
matching model/hardware/OS/workload fingerprint; a new fingerprint must be
measured rather than inheriting the M1 result.

MLX-LM's `--ssd-offload os-managed` control retained the same 1.211 GB warm
peak, reduced decode 5.49%, and worsened TTFT 1.44% relative to fully
materialized MLX-LM. It is recorded as OS-managed lazy loading, not controlled
Strata paging or SSD throughput evidence.

The checked-in runtime inventory is
[`apple_m1_runtime_comparison_matrix_v1.json`](../benchmarks/targets/apple_m1_runtime_comparison_matrix_v1.json).
It fails closed: MLX-LM has L4 controls for all three staged models; Strata's
paged MLX laboratory has one non-promoted Llama result; native Metal and ANE
remain below full-model evidence; Ollama is installed but its daemon is inactive
and has no exact artifact; llama.cpp is not installed. No missing runtime is
silently treated as a completed comparison.

## Local M1 order

| Priority | Common Compute model | Weights | Why it is here |
|---:|---|---:|---|
| 1 | Qwen3 0.6B 4-bit | 0.34 GB | Fastest loader/tokenizer/benchmark-harness smoke test |
| 2 | Llama 3.2 1B Instruct 4-bit | 0.70 GB | First Strata dense adapter and native phase-program comparison |
| 3 | Gemma 3 1B QAT 4-bit | 0.73 GB | Second family; exposes tokenizer, attention, and layout assumptions |
| 4 | SmolLM3 3B 4-bit | 1.73 GB | Sustained decode and batching |
| 5 | Llama 3.2 3B Instruct 4-bit | 1.81 GB | Current Common Compute default MLX model and primary product baseline |
| 6 | Phi 4 Mini Instruct 4-bit | 2.16 GB | Phi-family adapter variance |
| 7 | Qwen3 4B Instruct 4-bit | 2.26 GB | Larger dense and long-context stress |
| 8 | Gemma 4 E2B 4-bit | 3.55 GB | Largest current 8 GB-tier artifact and Gemma 4 specialization stress |

All eight are `preview`, not verified current fleet capacity. The only live
system model is `apple-foundation-3b`; benchmark its product latency separately
because Apple keeps its weights, tokenizer internals, and SoC placement opaque.

The first three downloadable artifacts total about 1.77 GB before cache and
filesystem overhead. They are approved and staged under
`/Users/tylergee/Library/Application Support/Strata/models`; the source checkout
on `/Volumes/Tyler HDD` remains forbidden as a model cache or SSD benchmark.

## Benchmark stages

### Stage A: MLX-LM baseline

Run the same pinned snapshot through MLX 0.31.1 and MLX-LM 0.31.2. The checked-in
runner sets Hugging Face and Transformers offline mode, refuses repository IDs,
rejects `/Volumes/Tyler HDD`, and requires the local snapshot directory to end
in the exact 40-character revision.

List the ladder without downloading:

```sh
PYTHONPATH=src /usr/bin/python3 benchmarks/benchmark_mlx_lm.py --list
```

After an SSD root and snapshot exist, invoke the runtime that actually contains
MLX-LM 0.31.2:

```sh
PYTHONPATH=src /Users/tylergee/miniconda3/bin/python \
  benchmarks/benchmark_mlx_lm.py \
  --model-id llama-3.2-1b \
  --approved-ssd-root /USER/APPROVED/SSD/ROOT \
  --model-path /USER/APPROVED/SSD/ROOT/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912 \
  --prompt-tokens 512 \
  --output-tokens 128 \
  --batch-size 1 \
  --warmups 2 \
  --repetitions 5 \
  --output benchmarks/results/apple_m1_mlx_llama_3_2_1b_b1_512_128_v1.json
```

The runner records exact hardware, model revision, prompt/output/batch sizes,
TTFT, median inter-token latency, prompt and decode throughput, aggregate
throughput including prefill, peak MLX memory, greedy token invariance, thermal
state, artifact SHA-256, architecture shape and quantization metadata, units,
and explicit timing boundaries.

### Stage B: Strata compatibility baseline

Use the same local snapshot, tokenizer, prompt token IDs, greedy output length,
and warm state. Initially, Strata may call its MLX compatibility adapter. The
result must match MLX-LM's token IDs before any native segment is timed.

The measured resident selector is the compatibility floor. The first native
implementation target remains Llama 3.2 1B because it aligns with Strata's
existing Llama/MLX code and exposes the short-response versus sustained-decode
route split that the native epoch must improve.

### Stage C: Strata native phase program

Promote generated and then real Llama block segments in this order:

1. RMSNorm and QKV projection;
2. RoPE, causal attention, and persistent KV;
3. output projection and residual;
4. gated MLP and residual;
5. multi-block command-buffer epoch;
6. full prefill and incremental decode through the public runtime.

Every promotion must retain exact greedy output and improve the complete phase,
not only an isolated kernel.

### Stage D: Common Compute provider canary

Only after standalone parity, run the same snapshot through the Common Compute
XPC adapter. Measure accepted-request-to-first-token latency, streaming, cancel,
terminal receipt, peak memory, and sustained concurrency. This is a separate L5
provider result; it does not replace the standalone engine comparison.

## Workload matrix

For each local model:

| Axis | Values |
|---|---|
| Prompt tokens | 128, 512, 2,048, 8,192 where model/memory safely permit |
| Output tokens | 128 forced greedy steps |
| Batch | 1, then 4 and 8 if admission passes |
| Cache | Fresh KV for every repetition; prefix cache disabled |
| Warm state | Two warmups; five measured repetitions |
| Order | Alternate MLX-first and Strata-first runs |
| Correctness | Exact token IDs plus selected logits/KV oracle checks |
| Statistics | Median plus p95; never best-of-N performance |

Record cold model load separately. Do not include download time in inference
metrics. Reject a repetition when the host is contended, thermal state is
serious/critical, output length is wrong, or artifact/runtime identity differs.

## Darkbloom comparison boundary

Darkbloom's report used Apple M4 Max, 128 GB unified memory, macOS build 25F84,
128 greedy decode steps, and exact locally cached snapshots.

| Model | Common Compute pin | Darkbloom pin | Exact comparison? |
|---|---|---|---|
| GPT-OSS 20B MXFP4-Q8 | `773a7da...` | `773a7da...` | Yes, on matching M4 Max hardware |
| Gemma 4 26B A4B | `0d77464...` | `0e3cbab...` | No; revisions differ |

Therefore:

- this M1's small-model results are MLX-vs-Strata engineering benchmarks, not
  Darkbloom wins or losses;
- GPT-OSS can use the exact Common Compute artifact on an M4 Max/128 GiB tester;
- Gemma requires either a separate Darkbloom snapshot or a clearly labeled
  revision-mismatched comparison;
- the checked-in Darkbloom comparator remains fail-closed until every hardware,
  model, workload, correctness, and metric cell matches.

## Immediate build order

1. Expand the resident route matrix to prompt lengths 128, 2,048, and 8,192.
2. Measure batches 4/8 and concurrency before changing the batch route.
3. Make the selector run inside Strata's persistent Common Compute service.
4. Replace one Llama block at a time with the native Metal phase program.
5. Expand to Llama 3.2 3B, then architecture-family coverage.
6. Send the exact GPT-OSS matrix to a Common Compute M4 Max/128 GiB tester.
