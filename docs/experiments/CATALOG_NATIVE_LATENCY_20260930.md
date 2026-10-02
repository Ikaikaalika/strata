# CommonCompute catalog and native latency iteration — September 30, 2026

Status: implemented bounded native candidates and offline gates; hardware
comparison pending the internal-SSD reserve. No complete-model acceleration or
CommonCompute serving promotion is established by this iteration.

## Catalog actually observed

The public `https://api.commoncompute.ai/v1/models` response contains 30
`mlx_llm` and 10 `mlx_vlm` entries, all `preview`. The immutable observation,
source artifact pins, recommended 20-entry target and workload matrix are in
[the dated target](../../benchmarks/targets/commoncompute_catalog_20260930.json).
Public discovery is separate from authenticated provider availability and usage.

Four proposed additions are absent from this observation: `gemma-4-e4b`,
`ministral-3-8b`, `glm-4.7-flash`, and `glm-ocr`. The Ministral ID is a proposed
catalog identifier with no pinned runtime artifact yet. The other three have
source pins in the inspected CommonCompute provider catalog; none is qualified
by metadata. Artifact pins come from CommonCompute source commit
`c78b44a33006c171450cd9364976d091cf3c2cd6`, not a claim about the currently
installed provider binary. Per-file snapshot verification is still required.

Keep existing exact model IDs available during catalog migration. This work
does not change CommonCompute's release policy, defaults, deployment, or app.
Do not treat a text candidate as a qualified vision path. The existing wire v1
kit remains a text-only integration contract.

## The native change and hypothesis

Decode repeatedly consumes large weight matrices for one token. The SIMD
candidate consumes eight affine Q4 values per packed uint32 load, reuses group
scale/bias values, and reduces each output dot product across one SIMD group.
Its hypothesis is less serial reduction work and more parallel packed-weight
consumption. Its dispatch geometry can also be inefficient; MLX remains the
control and no default is selected without measurement.

Prefill offers reuse across prompt tokens. The tiled candidate stages an
8-token by 16-output by 32-reduction slice, decoding weights into bounded FP32
threadgroup scratch. Activations are shared across output columns; decoded
weights are shared across tokens. No complete FP16 weight shadow is allocated.
Staging and barriers may outweigh reuse at small shapes, so the decode,
prefill, and tail cells are measured independently.

All tensor execution is C++/Objective-C++/Metal. Python stages generated
fixtures, invokes the native process, measures the independent MLX control and
validates receipts. This operator is not yet integrated into a transformer,
KV state, sampling, or the CommonCompute XPC worker.

The frozen contract is affine unsigned Q4 with eight little-nibble-first codes
per uint32, out_in row-major weights, group size 64, FP16 input/scale/additive
bias/output, and FP32 GPU accumulation. GPT-OSS MXFP4, 8-bit weights, arbitrary
groups, expert routing, attention, recurrent state and vision encoders are
outside this contract. A catalog label containing Q4 does not prove a model's
layout matches it.

Two candidates (`affine_q4_simd`, `affine_q4_tiled`) and one native scalar
control are allowed. Each is checked against a C++ FP64 arithmetic oracle,
rounded to FP16. MLX `quantized_matmul` is independently checked against the
same mathematical operator. The frozen fixture cells are `[M,K,N]`:

| Cell | Shape | Purpose |
|---|---|---|
| Decode | `[1,1024,128]` | One token; packed projection |
| Tail | `[7,128,17]` | Partial token/output tiles |
| Prefill | `[64,1024,128]` | Weight and activation reuse |

Bounds: M <= 512, K/N <= 8192, K divisible by 64, device tensors <= 64 MiB.
Reject nonfinite input/output, wrong lengths, missing source hashes, changed
tolerances, missing cells, malformed timings and unwritten output. The native
process poisons output before each dispatch, outside the timed region.
The paired harness rejects symlinked storage and other volumes before checking
free space, and requires builds inside the approved SSD root. Recommended
observed catalog pairs must retain exactly their inspected source artifact pin.

## Reproduce

Use a user-approved internal-SSD build directory. No dependencies or weights
are downloaded. The build-only and CPU tests do not perform hardware timing:

```sh
sh native/metal/build_q4_bench.sh --build-only
make -C native/metal -f Makefile.q4 test \
  BUILD_DIR="/Users/tylergee/Library/Application Support/Strata/build/q4-test"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/tylergee/Documents/strata-recovered/.venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_catalog_targets.py tests/test_native_q4_contracts.py
```

On another host, choose its own paths/environment. Hardware checks and paired
measurements require the existing 40 GiB internal-SSD reserve:

```sh
sh native/metal/build_q4_bench.sh --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/tylergee/Documents/strata-recovered/.venv/bin/python \
  benchmarks/benchmark_native_q4.py --rounds 5 \
  --output benchmarks/results/native_q4_component_FRESH_RUN.json
```

The harness builds before measurement, records source/contract/host/binary/
metallib hashes, performs five warmups and twenty samples, rotates native
variant order, and alternates native/MLX order across fresh native processes.
Hardware runs are serialized. Wall timing covers encoding, commit/evaluation
and wait; compile/load/allocation, poisoning and CPU verification are excluded.
GPU timestamps remain null if unavailable. Device tensor bytes are not peak
process footprint; energy is unavailable. Component receipts always set
`promotion_eligible=false`; they contain no inferred TTFT or model tok/s.

## Evidence and limitations

Local validation:

- Metal and Objective-C++ compiled with warnings as errors.
- 75 C++ arithmetic/bounds cases passed under UBSan.
- Existing C/NEON layout checks passed 400 transpose cases; existing C++
  admission checks passed 1,837 synthetic budget cases. The direct-ANE
  experiment executables compiled; this is not ANE dispatch evidence.
- Full Python regression suite: 319 passed, 22 subtests passed. `compileall`
  passed with its generated bytecode redirected to the internal-SSD build root.
- The native hardware check emitted `execution_verified=false` before creating
  a Metal device because free space was below the 40 GiB reserve. The paired
  harness independently refused to build or run: its final check observed
  21,891,895,296 available bytes against 42,949,672,960 required bytes.

No native Metal numerical execution or paired speed result was produced. TTFT,
complete-model decode, peak process memory, energy, and CommonCompute serving
performance remain unmeasured by this iteration. Historical receipts are unchanged.

The new direct Metal code does not replace the reverse-engineered ANE lane.
The existing Objective-C bridge and NEON layout implementation remain in scope.
Prior Qwen3 FFN layer-27 failure remains a rejection; no tolerance was relaxed.
The macOS build has changed since those hardware receipts, so fresh ANE ABI,
compile, dispatch and all-layer numerical qualification are required before
integrating a coarse prefill segment. See
[the ANE worker](../../native/ane/README.md) and its attribution.

## Next complete-request gate

1. Complete current-version generated-fixture Metal and MLX comparison.
2. Verify the already-local Qwen3 0.6B artifact and record a fresh pinned
   MLX-LM baseline. Benchmark 128/512/2048-token prompts and 128 output tokens
   before expanding context or concurrency. No runtime download is permitted.
3. Integrate retained projection/norm/attention/state pieces into one native
   dense model path. Count submission, synchronization and state updates;
   separate cold-load, warm uncached and prefix-reuse timing.
4. Qualify direct ANE prefill/draft work with complete ANE-on/off controls.
   Keep unsupported layers on a compatible MLX plan and report backend use.
5. Gate complete-model promotion on >=1.25x matched MLX prefill and decode,
   no TTFT/p95/correctness/memory/reliability regression, cancellation and
   serving tests. Keep SSD offload disabled for resident speed qualification.
6. Extend by manifest semantics to hybrid Qwen, Gemma, GLM, GPT-OSS and VLMs,
   then collect larger-Mac evidence and integrate the qualified CommonCompute
   worker. The 16..512 GiB target matrix is not a fleet execution result.
