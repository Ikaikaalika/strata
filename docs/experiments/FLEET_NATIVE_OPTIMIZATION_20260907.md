# Fleet targeting and native SIMD/ANE iteration

Date: 2026-09-07. Local host: Apple M1, 16 GiB, macOS 26.6.2 / 25G83.
Source and dependencies remain on the approved internal SSD. During this
experiment, no provider build, remote inference job, model download, deployment,
commit or push was performed. Subsequent public checkpoints follow the
[build-in-public guide](../BUILD_IN_PUBLIC.md) and are not product releases.
All prior recovery files and failure receipts were preserved.

## Scope and implemented changes

- Private read-only inventory of all registered CommonCompute devices, excluding
  test/Intel rows from Apple-Silicon cohorts; detailed evidence stays Git-ignored.
- Public metadata audits for 33 catalog LLM artifacts in the requested families
  and 16 upstream release targets; pinned revisions, complete root shard sizes
  and hashes. Metadata is not downloaded-file verification or execution proof.
- [Fleet optimization design](../design/COMMONCOMPUTE_FLEET_OPTIMIZATION.md) and
  [machine-readable targets](../../benchmarks/targets/fleet_optimization_profiles_20260907.json).
- `native/planning/admission.{h,cpp}`: standalone allocation-free C ABI / C++
  checked memory-budget arithmetic, per-session state/scratch limits, 16–512 GiB
  synthetic tests, conservative physical/Metal/available-memory intersection.
  Capacity success never sets native runtime qualification. SSD requests fail
  explicitly until the native pager exists. It is not wired into serving yet.
- `native/ane/tensor_layout.h`: opt-in ARM64 8x8 NEON bit-preserving transpose,
  scalar/tile-16 control retained. This is native C using integer vector
  instructions, invoked by Objective-C reverse-engineered ANE execution.
- Real-weight ANE harness now supports an interleaved layout control. Python
  orchestrates/stages fixtures; it does not implement the tested math/copy kernel.

## Bounded proposal

Parent evidence: recovery tile-16 FFN measurements and documented CPU copy cost.
Hypothesis: register transposition reduces strided scalar stores feeding the ANE.
One candidate: NEON tile 8; same fixed Qwen3 FFN weights/activation/graph, original
tanh lowering, frozen correctness tolerances, scalar tile 16 and compiled MLX
controls. Shapes: layer 0, width 1024, intermediate 3072, 64/256/512 tokens.
No larger model or different layer is qualified. Layer 27 remains rejected.

## Isolated CPU result

Hot-buffer input-and-output round-trip transpose; 100 warmups per variant,
101 alternating-order samples per variant, 32 round trips per sample. Every
sample was verified outside the timing interval. Latency is per round trip:

| Tokens | Scalar tile 16, µs | NEON tile 8, µs | Control / NEON |
|---:|---:|---:|---:|
| 64 | 50.043 | 12.654 | 3.95x |
| 256 | 198.026 | 121.495 | 1.63x |
| 512 | 395.365 | 261.387 | 1.51x |

[Raw CPU receipt](../../benchmarks/results/apple_m1_cpu_layout_neon_20260907_v1.json)
records source/binary hashes, flags and all samples. This is a CPU-cache/layout
microbenchmark, not ANE compute, model throughput or cold-memory performance.

## Real-weight direct ANE result

Five order-rotated rounds per shape, 5 warmups + 20 timed samples per round and
variant. Inputs are captured from Qwen3-0.6B layer 0; original pinned 4-bit weights
are expanded to FP16 for the native block. Both FP16-expanded and quantized MLX
controls use equivalent inputs and operations. Compilation/loading is outside
resident timing and separately recorded. Resident I/O is included.

| Tokens | NEON+ANE, ms | FP16 MLX, ms | Quantized MLX, ms | Quantized MLX / ANE |
|---:|---:|---:|---:|---:|
| 64 | 0.532 | 1.091 | 1.376 | 2.59x |
| 256 | 1.125 | 2.894 | 3.614 | 3.21x |
| 512 | 2.055 | 4.989 | 6.832 | 3.33x |

These are medians of round medians for **one FFN**, not full-model speedups.
[Interleaved receipt](../../benchmarks/results/apple_m1_qwen3_swiglu_neon8_paired_20260907_v2.json)
contains 30 native records, **750 verified direct ANE dispatches**, successful
cleanup and all numerical gates passing. Maximum scaled FP16 error was 0.164;
maximum relative L2 versus the quantized control was 0.00322. No gate was relaxed.
The [initial NEON run](../../benchmarks/results/apple_m1_qwen3_swiglu_neon8_20260907_v1.json)
is retained independently.

Important rejection: the interleaved tile-16 control had two roughly 50 ms
round medians at 512 tokens, plus substantial variability at 256. Endpoint
thermal state was nominal, which does not explain the stalls or establish energy
cost. Do not turn the resulting 512-token control/NEON ratio into a qualified
speedup. All noisy samples remain in the receipt. The isolated CPU experiment
supports the copy-path hypothesis; native default selection and full-model
promotion remain ineligible pending repeatability and end-to-end tests.

## Validation

- 400 transpose shape/tile/alignment cases passed with UBSan in each of the
  scalar and NEON builds, including empty dimensions, partial tiles, unaligned
  byte buffers and sentinels.
- 1,837 C++ native admission cases passed with warnings-as-errors and UBSan.
  These use synthetic byte budgets, not 512 GiB allocations or remote execution.
- 64 selected recovery/ANE Python tests and 12 subtests passed with the NEON
  build. The 12 compiled CLI tests also passed against the scalar build.
- 53 artifact/fleet-target contract tests passed offline.
- Git whitespace validation passed; detailed fleet files are Git-ignored.
  Existing unrelated CommonCompute changes were untouched.

## Reproduce without new dependencies or weights

From the approved recovered checkout, using fresh build/output paths:

```sh
make -C native/ane -f Makefile.experiments test \
  BUILD_DIR=/private/tmp/strata-fleet-neon-20260907 \
  LAYOUT_FLAGS=-DSTRATA_USE_NEON_TRANSPOSE
make -C native/ane -f Makefile.experiments \
  /private/tmp/strata-fleet-neon-20260907/layout-bench \
  BUILD_DIR=/private/tmp/strata-fleet-neon-20260907
/private/tmp/strata-fleet-neon-20260907/layout-bench
make -C native/planning test BUILD_DIR=/private/tmp/strata-fleet-planning-20260907
PYTHONPATH=src .venv/bin/python benchmarks/compare_ane_swiglu.py \
  --probe /private/tmp/strata-fleet-neon-20260907/swiglu-bench \
  --output benchmarks/results/new-unique-neon-receipt.json \
  --activation tanh --layout-tile 8 --control-layout-tile 16 --rounds 5
STRATA_ANE_EXPERIMENT_BUILD=/private/tmp/strata-fleet-neon-20260907 \
  PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_recovery_experiment_contracts.py tests/test_native_ane_experiment_cli.py \
  tests/test_ane_backend.py tests/test_ane_executor.py tests/test_ane_segment.py \
  tests/test_benchmark_target.py tests/test_fleet_target_contracts.py
```

Use separate build directories for different compiler flags; make does not
automatically invalidate targets when a command-line flag changes. Existing
benchmark receipts refuse overwrite. A different binary is a different candidate.

## Remaining product gates

The measured full-model baseline remains MLX-LM; no new full-model Strata versus
MLX/Darkbloom claim was established. Next: all-layer precision calibration,
native lifecycle C ABI, one real prefill substitution with GPU↔ANE synchronization
and state parity, then native GPU attention/quantized decode and measured serving
concurrency. Newest Qwen Flash-Next/GLM Flash/DeepSeek V4 need baseline and semantic
adapter work before timing. Authorize a specific fleet worker, artifact, storage
budget and benchmark window before moving beyond local experiments.
