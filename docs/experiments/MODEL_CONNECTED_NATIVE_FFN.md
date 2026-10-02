# Model-connected native FFN search

Status: native FP16 kernels compile and offline adapter/evaluator contracts pass.
Generated-kernel, BF16, complete-model correctness and speed remain unverified:
the internal-SSD 40 GiB reserve has not cleared. Disabled by default. This is
not a CommonCompute release or a universal native LLM implementation.

## Why this candidate

Parent evidence: [the packed-Q4 checkpoint](CATALOG_NATIVE_LATENCY_20260930.md)
and the historical pinned Qwen3 0.6B MLX-LM baseline in
`benchmarks/results/apple_m1_qwen3_06b_mlx_20260907_v1.json`.
The latter is historical complete-request evidence, not a current profile of
FFN bottlenecks. No new hardware win or current bottleneck attribution is known.

Structural hypothesis: a dense SwiGLU block separately computes gate/up
projections and feeds their intermediate tensors to an already-fused MLX
activation. Combine both packed projections and that activation in Metal.
The SIMD path reads each activation once for both dots; the tiled path shares
8 token rows and 16 output columns through 5 KiB of threadgroup scratch.
Store only the activated intermediate needed by the unchanged down projection.

This eliminates two logical projection-output tensors and offers fewer kernel
launches. It does not eliminate reading both weight matrices or the down
projection. Extra accumulators, low occupancy, barriers, and lost optimized MLX
GEMM behavior can make it slower. Native language alone is not a speedup.

## Implemented topology, narrowly

- `src/ollm/kernels/affine_q4_swiglu.metal`: native SIMD and tiled math; packed
  affine Q4 group64, FP32 accumulation, unchanged FP16/BF16 model storage.
- `src/ollm/runtime/native_swiglu.py`: thin, inference-only experimental binding
  through [MLX's custom Metal API](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
  Uses existing MLX arrays without converting
  weights to NumPy, concatenating matrices or creating a full expanded shadow.
  MLX may stage noncontiguous inputs; this overhead is not assumed zero.
- Only exact installed-version dense Qwen3 model/MLP classes are eligible.
  Retain the original module as a registered child so parameter accounting and
  fallback do not omit or duplicate its weights. Leave attention, KV cache,
  RoPE, normalization, down projection and sampling on MLX-LM.
- An opt-in context validates all layers before substitution and restores the
  original model after normal exit or failure. Exclusive model ownership is
  required. Do not train, save, shard or concurrently serve the patched model.
  Unsupported shapes/dtypes choose the original MLP before native graph work.
  Native failures propagate; restore the model and restart with fresh KV state,
  rather than splicing fallback into a partially failed request.

This adds a source-level connection to complete-model inference, not evidence
that the connection has executed correctly. Python handles binding and evidence;
the custom tensor hot path is Metal. It is still an MLX-hosted experiment, not
the target independent C++ engine or production XPC worker.

## Frozen candidate contract

| Candidate | One-token decode | 2..512-row prefill |
|---|---|---|
| `mlx_lm` control | Original MLX | Original MLX |
| `simd_decode` | Fused SIMD | Original MLX |
| `tiled_prefill` | Original MLX | Fused tiled |
| `fused_both` | Fused SIMD | Fused tiled |

At most three source candidates per step. Combined fusion is its own candidate,
not assumed valid from separate phase results. Bounds:

- Batch one; M <= 512, K <= 8192 divisible by 64, N <= 16384; input, output,
  both packed matrices and their group metadata together <= 64 MiB per operator.
  This is not the model/process memory budget. At most 64 model layers.
- Exact affine unsigned Q4 group64, eight little-nibble-first codes per uint32,
  out_in matrices, no linear output bias; FP16 or BF16 input/scales/bias/output.
  No dtype conversion or requantization to obtain a speed claim.
- Projection sums round to storage dtype before activation. The activation is
  a new native lowering, subject to fixed parity gates; algebraic similarity
  alone does not prove equality to MLX's compiled activation.
- MLX 0.32.2 / MLX-LM 0.31.3, Apple-Silicon macOS only. Other versions require
  fresh qualification. Kernel functions are cached in the worker; dimension
  arrays have a bounded 32-entry cache. SSD offload stays disabled.
- `simd_graph_calls` / `tiled_graph_calls` count constructed lazy graph calls,
  not measured Metal dispatches. `theoretical_intermediate_bytes_eliminated`
  is allocation accounting by formula, not measured traffic or process memory.

Qwen3 4B-sized projection envelopes can fit these bounds, but no CommonCompute
catalog artifact is qualified by that arithmetic. The first real-weight runner
allows only the already-local pinned Qwen3 0.6B canary. GPT-OSS MXFP4, Gemma
activation/norm variants, MoE/GLM, DeepSeek MLA/sparse attention, hybrid recurrence
and vision encoders require separate semantic adapters and gates.

## Deterministic iteration and rejection

`benchmarks/benchmark_native_swiglu.py` implements the evaluator, not an
autonomous source-editing proposer or production route selector:

1. Parent and worker independently enforce internal-SSD reserve before model
   import, artifact hashing or kernel/model execution. No downloads or remote
   model code; failures cannot clear resource guards.
2. Independent CPU FP64 packed-affine arithmetic plus software FP16/BF16 storage
   rounding checks both kernels and the equivalent MLX primitive on the fixed
   decode/tail/prefill cells from the prior checkpoint. Nonlinear/reference
   arithmetic belongs only to benchmark verification, not the runtime wrapper.
3. Generate a two-layer, 128-hidden/128-intermediate Qwen3 for each storage dtype.
   Check all three variants against unchanged MLX: cached prefill/decode logits,
   exact greedy tokens, KV arrays/offsets, and unchanged parameter bytes.
   Frozen checks: `atol=0.005`, `rtol=0.02`, relative L2 <= 0.01; nonfinite output
   or a changed state structure rejects the candidate. These are development
   gates, not comprehensive model-quality evaluation.
4. Verify local artifact provenance and hashes for Qwen3 0.6B revision
   `73e3e38d981303bc594367cd910ea6eb48349da8`. Run fresh-cache real-weight trace
   parity before any request timings. The maintained CommonCompute per-file
   manifest remains an additional integration gate.
5. Benchmark identical 128/512/2048-token synthetic repeated-text prompts,
   64 or 128 fixed output tokens, batch one, prefill chunks of 512. One warmup
   and 1..5 measured rounds, rotating all four route orders, one worker at a
   time. Generated-gate worker deadline 120 seconds; complete bounded matrix
   deadline 900 seconds. Timeout, malformed output and numerical failures
   produce unpromotable failure records when a fresh receipt path is supplied.
6. Record complete output IDs/arrival times, request wall time, TTFT, decode
   `(N-1)/(last-first)`, goodput, inter-token gaps, peak MLX allocation, route
   counters, software/chip/OS, thermal snapshots and source identities. Compare
   every candidate to current-run MLX, never to historical rates. Source changes
   during a worker invalidate the entire run. Energy is unavailable, not zero.
7. Calculate descriptive per-cell Pareto frontiers including MLX over TTFT/p95,
   MLX memory, decode and goodput. Shortlist only candidates reaching >=1.25x
   prefill-proxy and decode in every cell without goodput/memory/p95 regression.
   Prompt/TTFT includes first-token/API work: it is **not pure prefill compute**.
   Small-round p95 and frontiers are research observations, not SLA proof.
8. Every successful or failed receipt retains `promotion_eligible=false`.
   Held-out quality, pure-phase timing, sustained serving/cancellation, process
   memory, energy, ANE-on/off integration and fleet evidence remain required.
   Codex proposes the next bounded change from the retained evidence; missing
   evidence is not permission to select a default or broaden family coverage.

## Reproduce without downloads

Build-only validation on the approved source checkout, with output on its
approved internal SSD:

```sh
xcrun --sdk macosx metal -std=metal3.1 -O3 -fno-fast-math \
  -Wall -Wextra -Werror -I src/ollm/kernels \
  -c native/metal/check_q4_swiglu.metal \
  -o "/Users/tylergee/Library/Application Support/Strata/build/q4/swiglu-check.air"
```

Once the 40 GiB reserve passes, run generated correctness first, then the full
matrix. Use fresh output paths; binaries/caches remain on the internal SSD:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/tylergee/Documents/strata-recovered/.venv/bin/python \
  benchmarks/benchmark_native_swiglu.py --check-generated \
  --output benchmarks/results/native_ffn_generated_FRESH.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/tylergee/Documents/strata-recovered/.venv/bin/python \
  benchmarks/benchmark_native_swiglu.py --rounds 5 --output-tokens 128 \
  --output benchmarks/results/native_ffn_qwen3_06b_FRESH.json
```

These are local host paths, not portable defaults for another contributor.
The package declares its Metal asset in `pyproject.toml`; no wheel was built
because the current environment lacks the wheel/setuptools build tools. No
packages were installed to work around that gap.

## Validation actually obtained

- Both standalone FP16 kernel instantiations compiled and linked into a
  metallib with warnings as errors. This is **not dispatch or BF16 JIT proof**.
- 37 new offline adapter/reference/ranking contracts passed; the expanded full
  regression suite passed 356 tests and 22 subtests. Existing unrelated dirty
  work was preserved. Python compilation and Git whitespace checks passed.
- The generated execution command refused to run below the 40 GiB reserve.
  No generated Metal/BF16/logit/KV execution receipt or current model performance
  result exists for these candidates. Do not claim TTFT/decode improvements.

The reverse-engineered direct ANE worker remains a separate required research
lane; its layer-27 precision rejection and new-OS requalification are unchanged.
This GPU candidate never invokes ANE and cannot satisfy that integration gate.
Next decisions depend on the first execution receipts: discard regressions,
specialize winning shapes, or change algorithm/layout where the evidence points.
