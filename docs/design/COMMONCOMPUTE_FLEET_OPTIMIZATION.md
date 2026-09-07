# Strata: fleet-aware Apple-Silicon optimization, 16–512 GiB

Updated 2026-09-07. **Implemented here: artifact metadata audit, standalone native
capacity planning, and an experimental CPU SIMD copy path feeding direct ANE
execution. Not implemented here: a qualified full-model native engine, remote
fleet benchmark rollout, native model paging, or a production adaptive scheduler.**

The product objective remains matched end-to-end improvement over MLX-LM, not
maximum utilization counters. CPU, GPU and ANE share memory, power and thermal
budgets. Concurrent use is beneficial only when measured overlap exceeds copies,
dispatch overhead and contention. MLX-LM remains the oracle/fallback for models
it actually supports. If neither the native path nor the pinned baseline supports
a model, reject it; a fallback label must never conceal unsupported execution.

## Evidence and boundaries

- A read-only production inventory inspected every registered CommonCompute
  device. Detailed IDs, status, telemetry, catalog declarations and historical
  heartbeats are in Git-ignored `.local/fleet/`; do not publish them with Strata.
- Hardware profiles below are optimization targets, not advertised live capacity.
  Registry rows may be synthetic, stale, duplicated or incomplete. Host-side
  framework probes and exact-model dispatch eligibility remain mandatory.
- [Catalog artifact audit](../../benchmarks/targets/commoncompute_artifact_audit_20260907.json):
  33 pinned LLM catalog entries from the requested six families, plus 16 upstream
  release targets; every root safetensor shard has metadata size and digest.
- [Upstream releases](../../benchmarks/targets/family_metadata_20260907.json):
  dated IDs, revisions, formats and source links. These metadata reads downloaded
  no weights. Safetensor packed-element counts are not logical parameter counts.
- The original [local canary matrix](../../benchmarks/targets/recovered_model_matrix_20260907.json)
  still identifies the three downloaded, MLX-benchmarked artifacts. Larger
  targets are not tested merely because a small related model passed.

## Prioritized model families

| Family | Near-term catalog qualification | Current upstream expansion | Native work to qualify |
|---|---|---|---|
| Qwen | Qwen3 0.6B/4B/8B, Qwen3.5 4B/9B, Qwen3.8 27B; Qwen3 Coder 30B A3B and Next 80B | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Dense GQA and SwiGLU first; separate DeltaNet state/prefill scan; Flash-Next adds QSA sparse attention, gated residuals and large n-gram embeddings, not just a Qwen3.5 alias |
| Gemma | Gemma 4 E2B/E4B, 12B Unified, 26B A4B, 31B | [Gemma 4 family](https://huggingface.co/google/gemma-4-12B-it) | Sliding/global attention and exact state sharing; PLE lookup/cache for E models; GELU/norm semantics; Unified is a distinct architecture; multimodal correctness separate |
| GLM | GLM-4.7-Flash, then GLM-4.5-Air | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | Separate glm4_moe_lite, glm_moe_dsa and glm5_next contracts; Flash combines linear and sparse attention; fused expert gather/compute/scatter, precise routing/indexer semantics |
| DeepSeek | R1-Distill-Qwen-14B for catalog compatibility only | [V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731); [Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) tracked outside resident scope; Vision-Exp separate | Distill is qwen2, NOT DeepSeek MLA/MoE coverage. V4 needs its own compressed/sparse state, routing, quantization and token-format integration; V2/V3 assumptions are insufficient |
| GPT-OSS | Existing GPT-OSS-20B MXFP4-Q8 artifact | [GPT-OSS-20B/120B](https://huggingface.co/openai/gpt-oss-120b) | Preserve MXFP4 scale/packing semantics, attention sinks/window/global behavior, MoE routing and Harmony/tool output; compare identical artifacts |
| Llama | Llama 3.2 1B/3B, 3.1 8B, 3.3 70B | [Llama 4 Scout/Maverick](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | Mature dense GQA path first; Llama 4 requires separate MoE, attention/RoPE and multimodal contracts, not a relabeled dense model |

These are current published targets, not promises that every newest release is
the best serving choice. CommonCompute's exact advertised model IDs, actual
demand, end-to-end quality, latency and service cost determine production priority.

## Capacity tiers and artifacts

| Unified memory | Initial qualification workload | Capacity caution |
|---|---|---|
| 16–24 GiB | Existing tiny Qwen/Llama canaries; Gemma 4 E2B; 4–9B catalog models if admitted | Keep the compact lane first; reject unknown workspace/state sizes |
| 32–64 GiB | GPT-OSS-20B, Qwen3.8-27B, Gemma 12B/26B/31B, GLM-4.7-Flash, Qwen Coder 30B | Qualify quantized artifacts and context/concurrency independently |
| 96–128 GiB | Llama 3.3 70B, GPT-OSS-120B, GLM-4.5-Air, Qwen Next 80B | Memory capacity does not establish bandwidth or healthy runtime advertisement |
| 192 GiB | Above models with longer context or multiple sessions; careful DeepSeek V4 Flash investigation | 166.89 GB root weights alone leave little operating margin; local GPU working-set limits can reject the plan |
| 256–512 GiB | Resident Qwen Flash-Next, GLM-5.3-Flash, DeepSeek V4 Flash; Llama 4 Scout/Maverick with a verified feasible format | Exact quantization, all shards, caches and workspace decide fit; not every 256 GiB model plan fits |

At audited upstream revisions, root weight files total approximately:
Qwen Flash-Next BF16 **360.00 GB**, GLM-5.3-Flash mixed FP8/BF16 **328.34 GB**,
DeepSeek V4 Flash packed artifact **166.89 GB**, GPT-OSS-120B **65.25 GB**, and
Llama 4 Scout BF16 **217.28 GB**. These are disk artifact bytes, not execution
peak memory. Expanding packed weights can invalidate apparent fit. No such large
weights were downloaded or executed here.

GLM-5.3 upstream FP8 (**755.63 GB**), Llama 4 Maverick BF16 (**803.17 GB**),
DeepSeek V4 Pro (**892.74 GB**) and Qwen3.8 2.4T BF16 (**4,892.37 GB**) exceed
a 512 GiB resident budget before workspace. GLM/Maverick may merit a separately
audited low-bit conversion. Pro and the 2.4T model are out-of-scope resident
targets at ordinary 4-bit sizing too. Do not solve this by enabling SSD silently.

The local CommonCompute Qwen3.8-27B catalog entry says 4.67 GB; its pinned three
shards total **16,054,541,349 bytes**. The audit records the discrepancy; this
Strata change does not alter the CommonCompute checkout. Upstream/conversion
license metadata also differs for some Gemma entries; verify the actual artifact
license chain before release, never infer it from a family name.

## Native adaptive plan

The **target topology** is: trusted host probe + pinned semantic model manifest
→ native admission → bounded phase candidates → measured plan selection → native
worker execution → independently gated rollback. Python/TypeScript/Swift are
clients and experiment coordinators, not the hot tensor or scheduling path.

1. Objective-C++ probes actual `MTLDevice` features/working-set guidance, memory
   pressure, CPU topology, OS/compiler builds, SSD path/filesystem/free space.
   Zero GPU core counts or ANE TOPS in fleet metadata mean unknown. Run direct
   ANE compile/load/dispatch canaries per SoC/OS rather than infer support from
   a generation table. This integrated probe remains to be built.
2. C++ computes complete resident costs: all weight shards, packed format scale
   tensors, FP16 expansions, ANE programs/surfaces, state, bounded caches,
   scratch, staging, OS reserve and other reservations. Do not sum CPU/GPU/ANE
   memory as separate physical pools. Never use active MoE parameters for fit.
3. The standalone C ABI `strata_admit_v1` now performs checked 64-bit capacity
   arithmetic and bounds proposed sessions to 1..64. It uses the minimum of
   physical RAM minus max(4 GiB, 12.5%), the reported Metal working-set budget,
   and trusted currently available plan headroom. Unknown/incomplete inputs
   fail closed; state and scratch are per-session worst-phase bytes. This
   conservative policy can be tuned later only with device measurements.
4. Admission does not allocate/reserve memory, estimate model semantics, run
   inference or establish concurrency throughput. A production scheduler must
   atomically reserve capacity and recheck pressure before dispatch. All successful
   planner outputs still set `native_runtime_qualified=0`.
5. Prefill tests GPU matrix/attention fusion and coarse resident ANE FFNs;
   decode tests GPU quantized GEMV/expert kernels and state locality. CPU handles
   exact routing/tokenization, SIMD layout copies and preparation where measured
   beneficial. Avoid per-token small ANE calls that lose to dispatch/sync.
6. Phase plan keys include artifact/config/tokenizer digest, layer, shape,
   precision, batch/context, SoC/GPU family, OS/compiler and cache/residency state.
   Do not inherit M1 thresholds on M3 Ultra or infer speed from 512 GiB capacity.

### Caches and SSD

Account for immutable weight/program caches separately from request KV/MLA/
recurrent state. Bound warm program counts and avoid retaining an FP16 copy of
every quantized expert. Prefix-cache keys include tenant/security scope, complete
token prefix, tokenizer/template, model digest and state format; no cross-tenant
reuse by default. CPU cache tiles/SIMD vary by host. L1/L2/system caches are
hardware-managed, not additional weight capacity to add to unified RAM.

`ssd_offload=disabled` remains the resident default. The new native admission ABI
recognizes an explicit request but returns `STRATA_ADMIT_SSD_UNIMPLEMENTED`:
there is no native production pager to enable. Existing recovered Python paging
experiments are not promoted by this work. A future bounded pager must pin hot
weights/state, asynchronously prefetch cold experts or n-gram/PLE blocks, limit
queue depth/bytes and cache footprint, report hit/miss/stall distributions and
SSD writes, and preserve correctness under pressure/cancellation. Cache eviction
must not silently turn a resident claim into a paged run. Keep ≥40 GiB disk
reserve and sufficient staging/conversion space on each approved host.

## Measurement and rollout

Use exact CommonCompute artifact pins first; latest upstream IDs are a distinct
qualification lane. Locally installed MLX-LM 0.31.3 has no named qwen4_exp,
glm5_next or deepseek_v4 model module. Newest-family baseline support is a real
gate, not solved by adding an ID to JSON. Existing module presence for other
families still does not prove a complete artifact loads or generates correctly.

For each accepted artifact/host pair, measure input lengths 64/512/2048 first,
then 8K/32K and longer admitted contexts; output 64/256; concurrency 1/2/4/8
only after correctness and memory admission. Separate cold load/compile, warm
prefill, first-token API latency, decode/ITL, cache hit/miss and mixed-length
serving. Record model quality/logits/state/token parity, cancellation/recovery,
process footprint/allocator/VM compression/swap, sustained thermal behavior,
energy when measured, and per-backend copies/sync/dispatch. Missing metrics are
unknown, not zero.

Order-rotate native controls and the same pinned MLX-LM baseline on the same Mac.
Do not run simultaneous benchmarks on shared hardware. Repeated synthetic
fixtures are for development; held-out prompts, service-tail samples and
confidence bounds are required for promotion. Keep MLX on the Pareto frontier.
Native full-model promotion still requires ≥1.25x matched prefill AND decode,
no forbidden TTFT/p95/quality/memory/reliability regressions, and independently
recorded rollback. Darkbloom comparison needs a runnable pinned equivalent test;
published claims on unrelated hardware cannot satisfy the target.

Rollout order: local compact canary → one authorized 32–64 GiB worker → one
96–192 GiB worker → one 512 GiB worker → independent second-host replication.
Do not dispatch fleet jobs, install experimental private-ANE builds or download
hundreds of gigabytes onto a provider without explicit operator authorization.
Commercial/private-API distribution review remains separate from performance
validation. No provider, router or production setting was changed here.

## This iteration's native experiment

`STRATA_USE_NEON_TRANSPOSE` enables an 8x8 ARM64 integer-register transpose in
the existing Objective-C direct-private-ANE benchmark. Scalar and tile-16
controls remain; tensor bits and the ANE graph/precision gates are unchanged.
It is compile-time opt-in, not a new production default. The Python harness now
accepts `--control-layout-tile 16` for an interleaved native control alongside
both compiled MLX controls. See the [experiment report](../experiments/FLEET_NATIVE_OPTIMIZATION_20260907.md)
for all measured wins, variance and limits.

Next implementation gate: all-layer precision calibration and one actual
prefill integration with GPU↔ANE synchronization and state parity. The existing
layer-27 rejection remains in force. Do not substitute every model's FFN merely
because the layer-zero microbenchmark is fast.
