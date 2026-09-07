# Strata recovery and native prefill experiments — 2026-09-07

## Outcome

This is the original recovery-iteration record. The later
[fleet/native SIMD report](FLEET_NATIVE_OPTIMIZATION_20260907.md) adds NEON timing
and expanded validation; earlier receipts remain unchanged. Publication status
is governed by the [build-in-public guide](../BUILD_IN_PUBLIC.md).

Strata now has a runnable internal-SSD checkout, pinned local model baselines, and **measured direct-ANE component improvements written in Objective-C/C**. There is **no demonstrated full-model Strata speedup over MLX-LM or Darkbloom yet**. All new component receipts have `promotion_eligible: false`.

The strongest repeated component result is Qwen3 layer 0's fused SwiGLU block: **2.32–3.18x faster than the compiled quantized MLX control**, or **1.84–2.32x faster than the precision-matched expanded-FP16 MLX control**, including CPU layout/readback handling. A middle-layer smoke test passed; a final-layer test failed the unchanged numerical gate. These observations support selective, calibrated ANE use, not universal offload.

## Recovery boundaries

- Source: `/Users/tylergee/Documents/strata-recovered`, branch `codex/native-prefill-ane-recovery`, remote base `4ecda436ef2e5048c8272512d53c8d8fb1528df9`.
- Models: `/Users/tylergee/Library/Application Support/Strata/models`, a real internal-SSD directory. The former broken symlink was preserved as `models.failed-hdd-link-20260907`; its failed-HDD target was not accessed or deleted.
- Python environment and package cache: `.venv` and `.cache/pip` under the recovered checkout, ignored by Git. The observed environment is recorded in [requirements-recovery-20260907.txt](../../benchmarks/requirements-recovery-20260907.txt).
- 600 historical source candidates were reconstructed by static extraction from surviving Codex logs; 197 files were excluded from that recovery. Candidates remain in the recovery host's Git-ignored `recovery/candidates/`, separate from active native code. This is not proof that all latest changes were recovered. The local inventory at `recovery/candidates/docs/recovery/SOURCE_RECOVERY_20260907.json` is not part of the public repository.
- The previous v2 target architecture was copied byte-for-byte into [docs/design](../design/STRATA_NATIVE_RUNTIME_V2.md), SHA-256 `444e8af3440b34628633eb2f6dd6db5c52bccdcc4173a9bc9543759b6adbc072`. [The active bounded-search document](../design/BOUNDED_ARCHITECTURE_SEARCH.md) distinguishes current experiments from future serving/native-runtime work.
- During the original recovery experiment, no commits, pushes, merges, deployments, or Common Compute source changes were made. Later public checkpoints do not change these experimental results or constitute a provider release.

## Hardware and method

Apple M1; 16 GiB unified memory; macOS 26.6.2 / build 25G83. MLX 0.32.2, MLX-LM 0.31.3, Python 3.13.11. MLX and direct ANE tests ran serially, with no concurrent inference benchmark. This is an interactive Mac, not a controlled power/thermal lab; recorded endpoints were nominal, but background activity and frequency changes are not eliminated.

Private ANE execution uses Strata's native `_ANEInMemoryModelDescriptor` / `_ANEInMemoryModel` lifecycle, MIL, IOSurface, compile/load/evaluate/unload, and numerical readback. It is **actual private-API dispatch**, not inferred from Core ML configuration. The existing Strata bridge retains its reverse-engineering provenance in [THIRD_PARTY_NOTICES.md](../../native/ane/THIRD_PARTY_NOTICES.md). Community research such as [maderix/ANE](https://github.com/maderix/ANE) informed the approach; no new third-party implementation was vendored in this iteration. Undocumented APIs have no Apple stability guarantee.

Native FFN programs are compiled and loaded once per round, then reused for 5 warmups and 20 measured invocations. Real FFN input comes from the pinned model's actual layer activation on the recorded repeated-text prompt, not random activation. ANE uses expanded FP16 weights from the same quantized artifact. MLX controls separately use those FP16 weights and the original quantized projections with identical FP16 input. Error against the original BF16 block is also recorded; these controls must not be conflated.

The Python files orchestrate measurements and generate independent MLX references. New execution and layout arithmetic is in [swiglu_bench.m](../../native/ane/swiglu_bench.m), [prefill_chain_bench.m](../../native/ane/prefill_chain_bench.m), and [tensor_layout.h](../../native/ane/tensor_layout.h). No new production Metal runtime or Python/TypeScript/Swift serving wrapper is claimed.

## Experiment sequence: wins and rejected candidates

### 1. Fusion removes repeated small-graph dispatch

Generated width-256, four-stage dense linear/ReLU chain, not a transformer. Five order-rotated rounds per shape, 20 samples per round:

| Tokens | Split ANE with I/O (ms) | Fused ANE with I/O (ms) | Compiled MLX with I/O (ms) | MLX / fused ANE |
|---:|---:|---:|---:|---:|
| 64 | 0.968 | 0.216 | 0.322 | 1.49x |
| 256 | 1.048 | 0.266 | 0.486 | 1.83x |
| 512 | 1.278 | 0.478 | 0.551 | 1.15x |

Correctness passed against both an independent scalar oracle and MLX. Split ANE reuses ping-pong IOSurfaces between stages; it does not add a CPU round trip at each stage. Fusion improved over split by 2.67–4.48x including the shared input/output work. [Raw receipt](../../benchmarks/results/apple_m1_ane_prefill_chain_20260907_v1.json).

### 2. Real Qwen3 FFN: reject sigmoid, test an equivalent lowering

The direct `sigmoid` MIL lowering on layer 0 / 64 tokens failed the fixed `atol=0.01, rtol=0.02` gate: maximum scaled error 1.0903, where passing requires at most 1.0. It was rejected without loosening tolerances. [Failure receipt](../../benchmarks/results/apple_m1_qwen3_swiglu_20260907_smoke_v1.json).

The algebraically equivalent `sigmoid(x) = 0.5 * (tanh(0.5*x) + 1)` lowering passed. This establishes an observed numerical difference between compiler lowerings; it is not proof of the internal hardware cause. Repeated layer-zero measurements with the original scalar copies showed 2.26–2.90x over quantized MLX, 1.60–2.12x over FP16 MLX. [Five-round receipt](../../benchmarks/results/apple_m1_qwen3_swiglu_20260907_tanh_v3.json).

### 3. Cache-tiled CPU layout copies

The native input/output transpose now has bounded scalar/8/16 tile parameters. All layouts preserve FP16 bits; 300 shape/tile cases include non-multiple tile edges and buffer sentinels.

Layer 0, tanh lowering, tile 16, final source build, medians of five round medians:

| Tokens | ANE with I/O (ms) | FP16 MLX with I/O (ms) | Quantized MLX with I/O (ms) | Quantized MLX / ANE |
|---:|---:|---:|---:|---:|
| 64 | 0.602 | 1.106 | 1.398 | 2.32x |
| 256 | 1.181 | 2.743 | 3.753 | 3.18x |
| 512 | 2.292 | 4.976 | 6.821 | 2.98x |

Maximum scaled numerical error was 0.164 or less, relative L2 against the quantized FP16-input control about 0.32% or less, and against the original BF16 block about 0.64% or less. Each ANE round verified 25 dispatches and successful cleanup. [Final-build receipt](../../benchmarks/results/apple_m1_qwen3_swiglu_20260907_final_v5.json). The [preceding tile-16 run](../../benchmarks/results/apple_m1_qwen3_swiglu_20260907_tanh_tile16_v4.json) is retained too; differences between runs illustrate why these are descriptive component results, not release confidence bounds.

Compared with the preceding scalar-copy run, tile 16 had lower total component time at 256/512 but higher at 64. Those layout variants were measured in separate runs, so this is a tuning lead, **not a statistically qualified tile-vs-tile speedup**. Do a direct interleaved layout comparison before installing shape-specific defaults. Tile 8 is implemented/tested for correctness but not yet hardware-timed.

### 4. Cross-layer validation prevents blanket offload

- Layer 14 passed the same gates at 64/256/512 tokens in one smoke round. This is not five-round performance qualification. [Receipt](../../benchmarks/results/apple_m1_qwen3_swiglu_layer14_20260907_v1.json).
- Layer 27 failed at 64 tokens: maximum scaled error 2.426, despite a small aggregate relative L2 of 0.043%. The original BF16-versus-expanded-FP16 and quantized-versus-expanded-FP16 controls also show material pointwise differences here. The layer's conversion/accumulation contract needs investigation; no automatic ANE routing is justified. [Rejected receipt](../../benchmarks/results/apple_m1_qwen3_swiglu_layer27_20260907_v1.json).

The layer-27 fixture demonstrates why an average tensor error or early-layer pass cannot alone qualify a model. No tolerance was changed after seeing the result.

## Full-model MLX-LM baselines

Three local pinned artifacts; batch 1; 64 generated tokens; 64/512/2048 input-token cases; fixed greedy sampling; one warmup and three measured repetitions each; fresh prompt state, resident model, no SSD offload or drafting. Prompt and output token IDs, artifact hashes, raw timings, software, and hardware identity are in each receipt. Greedy repetitions agreed within each workload.

The 512-input / 64-output medians are:

| Model | MLX prefill tok/s | MLX decode tok/s | API first token (ms) | MLX-reported peak GB |
|---|---:|---:|---:|---:|
| Qwen3 0.6B 4-bit | 1,205.4 | 83.7 | 678.6 | 0.823 |
| Qwen3.5 0.8B OptiQ | 1,031.1 | 71.3 | 683.4 | 1.270 |
| Llama 3.2 3B 4-bit | 296.2 | 25.9 | 1,814.4 | 2.340 |

MLX's internal prefill metric includes processing the first token and excludes some outer API setup; API first-token timing includes that setup. Peak GB is the MLX allocator metric, **not total process RSS or complete machine residency**. Do not substitute either for the other. Fixed repeated-text prompts are reproducibility fixtures, not model-quality evaluation. Three repetitions are baselines, not robust p95/confidence evidence.

Receipts: [Qwen3](../../benchmarks/results/apple_m1_qwen3_06b_mlx_20260907_v1.json), [Qwen3.5](../../benchmarks/results/apple_m1_qwen35_08b_mlx_20260907_v1.json), [Common Compute Llama](../../benchmarks/results/apple_m1_llama32_3b_mlx_20260907_v1.json). Llama's measured model SHA-256 exactly matches Common Compute's maintained snapshot: `d75e1ee0ea653cc5b76191ec934c7c0d568e94d4e47846619f1f4bc715b7b265`.

## Current model-family scope

[The dated model matrix](../../benchmarks/targets/recovered_model_matrix_20260907.json) pins local canaries and larger compatibility targets. Selection combines Common Compute's local manifest with current publisher metadata and [HF trending](https://huggingface.co/models?sort=trending), not an undated claim of universal popularity.

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is a current high-interest target. Its publisher describes gated DeltaNet/attention hybrid layers; it must not be treated as ordinary dense GQA merely because it is dense rather than MoE.
- [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`glm5_next`) and [DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) (`deepseek_v4`) need new semantic/state/router audits and larger machines. Only metadata was fetched.
- [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) adds local/global attention and per-layer embeddings; effective parameter count understates total residency. It is the next compact-family artifact-selection target, not implemented native coverage.
- [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) remains a requested MoE/format compatibility target. No resident-fit assumption, weights download, or native support claim was made; 120B requires a separate larger-hardware plan.

This does not change Common Compute's catalog or prove live provider availability.

## Reproduce and continue

Run from the recovered checkout, using its approved local environment. These commands do not install packages or download models:

```sh
make -C native/ane -f Makefile.experiments test
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_recovery_experiment_contracts.py
.venv/bin/python benchmarks/compare_ane_swiglu.py --probe native/ane/build/experiments/swiglu-bench --activation tanh --layout-tile 16 --rounds 5 --output benchmarks/results/new-swiglu-receipt.json
.venv/bin/python benchmarks/recovered_mlx_baseline.py --model Llama-3.2-3B-Instruct-4bit --output benchmarks/results/new-llama-baseline.json
```

Final scoped regression validation: **64 tests plus 12 subtests passed**, covering the new fixture/hash/error gates, compiled native CLI bounds, existing ANE executor/segment behavior, and benchmark target contracts. **300 native layout cases passed with UBSan**; the final five-round hardware receipt matches the final native source/binary hashes. `git diff --check` and explicit whitespace checks over new active files passed.

Use fresh receipt names: existing evidence is never overwritten. Hardware runs require native Metal/ANE access, so the restricted Codex sandbox may abort MLX import. Native layout tests passed in debug, optimized, and UBSan builds. The combined ASan/UBSan test stalled; only its identified test processes were terminated. **ASan validation remains unresolved**, not passed.

Next: calibrate every candidate layer on varied/held-out prompts; extract a reusable native lifecycle; integrate a selectively qualified segment into real prefill with state parity; measure full ANE-on/off requests. Native Metal attention, whole-model correctness, energy/thermal soak, full-process residency, continuous batching, production IPC/wrappers, SSD paging, and matched Darkbloom execution remain unfinished. The 1.25x full-model MLX target has not been reached or claimed.
