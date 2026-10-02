# Native source publication validation — October 1, 2026

Status: local build and regression validation passed. This is a research source
checkpoint, not a runtime promotion, CommonCompute release, or measured speedup.
MLX-LM remains the full-model oracle/fallback; native candidates are opt-in and
SSD offload remains disabled for resident comparisons.

## Reviewed scope

- [Catalog and packed affine-Q4 iteration](CATALOG_NATIVE_LATENCY_20260930.md):
  frozen public metadata, standalone C++ arithmetic contracts, Objective-C++
  Metal submission, SIMD decode and tiled prefill shaders, and a paired MLX
  component evaluator. The catalog is a dated observation, not current supply.
- [Model-connected FFN follow-up](MODEL_CONNECTED_NATIVE_FFN.md): packaged
  fused gate/up/SwiGLU Metal source, exact-version dense Qwen3 experimental
  binding, bounded evaluator and offline rejection/rollback/ranking tests.
- Native ANE and admission Makefiles now escape spaces in Make target and
  prerequisite lists as well as quote paths in shell recipes. Previously an
  `Application Support` build path split into multiple targets and could put a
  compiler output at the wrong path. Three dry-run regression cases cover ANE,
  planning and Metal build files; real ANE/planning builds also passed using
  the approved path containing spaces.

Historical experiment receipts and their hashes are unchanged. No model
weights, interpreter environments, build binaries, private fleet data, storage
archive receipts, or Time Machine history are included in this checkpoint.

## Local evidence

Host: Apple M1, Apple-Silicon macOS 26.6.2 (25G83); existing MLX 0.32.2 and
MLX-LM 0.31.3 environment. Source resides on the approved replacement external
drive; generated build artifacts and Python bytecode reside on internal SSD.
No packages or models were downloaded or installed for these checks.

| Gate | Result | Evidence boundary |
| --- | --- | --- |
| Full Python regression | 359 tests and 22 subtests passed; no skips | Offline contracts and existing regression coverage |
| Packed Q4 C++ reference | 75 cases passed under UBSan | CPU arithmetic and bounds, not GPU execution |
| C/ARM64 NEON tensor layout | 400 cases passed under UBSan | Shape/tile/alignment copy correctness |
| Native C++ admission | 1,837 cases passed under UBSan | Synthetic capacity arithmetic, not fleet capacity |
| Standalone Q4 Metal + Objective-C++ host | Compiled and linked, warnings as errors | No valid Metal dispatch or numerical readback |
| Fused SwiGLU FP16 entries | Compiled and linked into metallib, warnings as errors | No BF16 JIT, generated-model or real-weight execution |
| Reverse-engineered Objective-C ANE tools | Both experiment executables compiled | CLI rejection tests do not execute valid ANE inference |
| Python compileall / Git whitespace | Passed | Syntax and patch formatting only |

## Reproduce build/regression checks

Choose an existing environment and an approved internal-SSD build location on
your host; do not use these placeholders literally. These commands build and
test contracts, not performance. No download or installation is performed.

```sh
STRATA_CHECK_BUILD="/path/to/approved/internal-ssd/strata-build"
STRATA_CHECK_PYTHON="/path/to/existing/venv/bin/python"
mkdir -p "$STRATA_CHECK_BUILD"
STRATA_METAL_BUILD_DIR="$STRATA_CHECK_BUILD/q4" \
  sh native/metal/build_q4_bench.sh --build-only
make -C native/metal -f Makefile.q4 test \
  BUILD_DIR="$STRATA_CHECK_BUILD/q4-contract"
xcrun --sdk macosx metal -std=metal3.1 -O3 -fno-fast-math \
  -Wall -Wextra -Werror -I src/ollm/kernels \
  -c native/metal/check_q4_swiglu.metal \
  -o "$STRATA_CHECK_BUILD/swiglu-check.air"
xcrun --sdk macosx metallib "$STRATA_CHECK_BUILD/swiglu-check.air" \
  -o "$STRATA_CHECK_BUILD/swiglu-check.metallib"
make -C native/ane -f Makefile.experiments test \
  BUILD_DIR="$STRATA_CHECK_BUILD/ane" LAYOUT_FLAGS=-DSTRATA_USE_NEON_TRANSPOSE
make -C native/planning test BUILD_DIR="$STRATA_CHECK_BUILD/planning"
STRATA_ANE_EXPERIMENT_BUILD="$STRATA_CHECK_BUILD/ane" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  "$STRATA_CHECK_PYTHON" -m pytest -q -p no:cacheprovider
PYTHONPYCACHEPREFIX="$STRATA_CHECK_BUILD/pycache" \
  "$STRATA_CHECK_PYTHON" -m compileall -q src tests benchmarks
git diff --check
```

## Still not qualified

The 40 GiB internal-SSD reserve remains unmet. No hardware performance run was
performed for publication. Complete-model logits/tokens/KV parity, current-run
MLX TTFT/decode comparisons, energy/thermal soak and integrated ANE-on/off gates
remain pending. Native compilation alone proves neither speed nor backend use.
The historical ANE layer-27 precision rejection is not cleared by these builds.

These candidates are not a universal native LLM/VLM runtime, independent native
serving engine, production SSD pager, persistent XPC service, continuous batcher,
or a CommonCompute worker release. Python/TypeScript/Swift production wrappers
and broader family adapters remain separately gated target work.
