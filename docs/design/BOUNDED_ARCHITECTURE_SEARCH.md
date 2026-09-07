# Bounded native architecture search: internal-SSD recovery iteration

Status: **experimental implementation and measured components; no promoted native full-model runtime**. Updated 2026-09-07. The complete target topology is in [STRATA_NATIVE_RUNTIME_V2.md](STRATA_NATIVE_RUNTIME_V2.md). Current measurements and failures are in [the recovery report](../experiments/RECOVERY_NATIVE_PREFILL_20260907.md).

## Authority and evidence

Codex proposes small source-level variants from the last measured bottleneck. Native validation and independent MLX comparisons decide whether a variant survives. Codex cannot convert a fast component into a full-model promotion. Each candidate has explicit shape, architecture, precision, lifecycle, and residency bounds. Source/binary hashes and immutable receipts identify measurements; changed source is a new candidate.

This checkout starts from recovered remote commit `4ecda436ef2e5048c8272512d53c8d8fb1528df9`. On the recovery host, 600 files in Git-ignored `recovery/candidates/` are quarantined history-derived candidates, not a verified latest checkout. That local inventory and quarantine are deliberately absent from public clones. Do not import them wholesale or run their build scripts as trusted instructions. Reviewed work and non-sensitive receipts are published under the [build-in-public policy](../BUILD_IN_PUBLIC.md).

## Work order: compute and memory first

1. **Objective-C/C/C++ native execution:** repair and qualify direct private-ANE compilation, lifecycle, tensor layouts, and coarse subgraphs. Preserve failures as evidence.
2. **CPU and caches:** tune bounded out-of-place bit-preserving transpose tiles; reuse resident surfaces/programs; measure allocation, copying, synchronization, and compiler/load costs separately.
3. **GPU:** retain MLX as the working Metal implementation and correctness oracle. New native Metal attention, quantized projections, fused kernels, and command submission need their own model-state integration and gates. Do not count existing MLX GPU work as newly implemented native Strata Metal kernels.
4. **ANE/GPU integration:** use measured per-model, per-layer, per-shape qualification. A passing early-layer FFN does not qualify the last layer. Fixed-shape ANE kernels remain optional within a qualified plan; `ane=required` must reject unsupported plans rather than pretend ANE ran.
5. **SSD capacity track:** `ssd_offload=disabled` by default. Do not mix paging and resident speed claims. Before paging experiments, require an approved internal-SSD path, at least 40 GiB free-space reserve, bounded bytes outstanding, artifact identities, admission accounting, and measured page/staging stalls. SSD is storage, not execution memory. No new production pager is implemented by this iteration.
6. **Wrappers after native contracts:** C ABI over the native runtime, then Python, TypeScript, and Swift clients. The current Python scripts stage fixed fixtures and record measurements; the new ANE math and tensor copies execute in Objective-C/C. No production Swift/TypeScript FFN serving wrapper exists yet.

The dead external HDD is excluded from all source, package, model, cache, and benchmark I/O. Use the approved SSD checkout and model root only. Do not follow the preserved failed-HDD symlink.

## Candidate contract and limits

Every proposed change must state `parent_evidence`, `bottleneck`, `hypothesis`, `changed_sources`, `shapes`, `dtype`, `model_revision`, `baseline`, `correctness_gate`, `timing_boundary`, `resource_bounds`, and `rejection_action`.

For this recovery iteration:

| Dimension | Allowed values / limit |
|---|---|
| Proposals per bounded step | At most 3; no unbounded recursive process |
| Real-weight native operation | Qwen3 FFN, hidden 1024, intermediate 3072, batch 1 |
| ANE sequence buckets | 64, 256, 512 tokens |
| Real-activation smoke layers | 0, 14, 27; further layers require new evidence |
| SwiGLU activation lowering | Native `sigmoid`; algebraically equivalent `0.5*(tanh(0.5*x)+1)` |
| CPU copy tile | Scalar control, 8, 16; no global default change based on a single shape |
| Small synthetic control | Width 256, depths 1/2/4, split versus fused dense ReLU chain |
| Measurement repetitions | 5 warmups, 20 samples, at most 5 order-rotated rounds |
| Native subprocess deadline | 60 seconds for chain; 120 seconds for real FFN; one worker at a time |
| Compilation | Fresh bounded process; reuse compiled program within timing repetitions |
| SSD | Off for all resident speed comparisons |
| Runtime promotion | Always false for these component receipts |

The child process owns its generated compiler directory and IOSurfaces, unloads models, checks cleanup, refuses existing outputs, bounds tensor sizes, and rejects nonfinite tensors. The real FFN poisons its output surface before each dispatch, outside the measured interval, so unwritten output fails. This is a benchmark process, not a production untrusted-model sandbox.

## Gates, in order

1. Build with warnings as errors; run offline contract and layout tests. Only approved tensor shapes and source changes proceed.
2. Verify model revision and every tested root artifact's Git-blob/LFS digest against local pinned-download provenance; record independent SHA-256 digests. No runtime downloads or remote model code. Common Compute release integration must additionally use its maintained per-file snapshot manifest.
3. Prove direct ANE compile, load, dispatch, finite readback, cleanup, and expected invocation count. Core ML discovery alone is insufficient.
4. Compare the ANE result against MLX with equivalent FP16 expanded weights using frozen `atol=0.01`, `rtol=0.02`. Also report error against the quantized same-input control and original BF16 model block; require relative L2 at most 0.025 against the quantized FP16-input control. **These are component-development gates, not model-quality acceptance.** Do not relax a gate to save a failing layer. Keep reference-path conversion errors visible too.
5. Record serialized, order-rotated hardware samples for resident compute and resident input/output handling. Compare identical operations, not four ANE stages against one MLX stage or an ANE dispatch-only time against a full request.
6. Shortlist a component only within its tested layer/shape/precision envelope. Nominal thermal state is not joules/token evidence. Missing energy, full-process memory, cache-state, or reliability information cannot be interpreted as zero cost.
7. Before full-runtime promotion: integrate complete prefill, KV/recurrent state, decode, tokenization/sampling, cancellation, and fallback. Run paired full-model ANE-on/off and MLX baseline tests, including logits/state and deterministic tokens on varied prompts. Baseline fallback must restart from a valid state; do not splice incompatible state after a failed native phase.
8. Require predeclared matched full-model throughput target (at least 1.25x MLX-LM prefill **and** decode), TTFT and p95 limits, correctness, memory, reliability, and profile constraints. Independent held-out prompts, randomized order, sufficient repetitions/confidence bounds, and larger-device runs remain required. A component speed ratio cannot satisfy this step.
9. Darkbloom requires its own pinned runnable implementation, equivalent model/artifact/precision/hardware/context and service envelope; do not compare these M1 components with unrelated published hardware. The proposed additional target is at least 1.10x matched service goodput without SLO/quality regression. Not measured here.

## Adaptive selection and rollback

Selection keys must include model/artifact digest, semantic family and layer, shape bucket, precision, SoC, memory envelope, OS/compiler build, and workload profile. Proposed profiles: interactive TTFT/p95, sustained throughput, long-context capacity, and energy-constrained operation.

Within each profile, discard failures and constraint violations first. Compute the Pareto frontier over verified full-model latency, goodput, memory, and measured energy. MLX remains a candidate. A plan missing a required metric is ineligible for that profile. Use stable tie-breaking; retain prior plan digest and immutable gate receipts for rollback. The production selector and transactional route registry are **targets**, not implemented by these scripts.

Current example: tanh-based FFN layer 0 is a promising component; layer 14 has only a smoke pass; layer 27 failed. Larger buckets appear to benefit from tile 16 while the smallest does not. The correct next proposal is selective native integration and stronger calibration, not universal ANE routing.

## Next bounded integration milestone

The fleet expansion is specified in [COMMONCOMPUTE_FLEET_OPTIMIZATION.md](COMMONCOMPUTE_FLEET_OPTIMIZATION.md).
Its public target matrix covers 16..512 GiB without publishing private provider IDs.
All 33 catalog artifacts in the requested six families and 16 upstream release
targets have pinned metadata receipts; these do not establish support or local
download verification. Use complete shard sums, never the active-parameter count
or an unchecked catalog weight-size field, for memory admission.

One additional bounded native candidate is now permitted: ARM64 NEON 8x8
bit-preserving CPU transposition (`STRATA_USE_NEON_TRANSPOSE`), keeping scalar
and tile-16 controls. The same reverse-engineered ANE FFN graph, precision gates,
sequence buckets and layer limits apply. At most one candidate and two controls
(native tile-16 and equivalent MLX paths) participate; five order-rotated rounds
per run, no simultaneous model benchmarks, no remote provider jobs. Preserve
noisy measurements and do not select a new default from an outlier-inflated
speedup. The native C++ capacity planner in `native/planning/` is standalone
tested groundwork, not a production route selector or memory allocator. It
rejects SSD paging until a native pager is implemented and independently gated.

- Extract the benchmark's native lifecycle into a reusable C ABI with explicit ownership, error codes, and cancellation/deadline handling; keep private-API failure inside an isolated worker for serving.
- Add an offline all-layer activation/precision calibration pass before any model-wide substitution. Keep unqualified layers on MLX.
- Integrate one qualified FFN bucket into an actual Qwen3 prefill path, recording GPU-to-ANE and ANE-to-GPU synchronization and copies. Then measure full prompts and decode with state parity. Until this passes, runtime acceleration is unproven.
- Extend semantic tests separately to Qwen3.5 hybrid recurrence, the pinned Common Compute Llama artifact, and the current model matrix. GLM-5.3 and DeepSeek-V4 require new manifest audits; names alone do not establish old router/state compatibility.
- Implement native Metal attention/projection candidates once their input, output, and state boundaries are verified. Do not claim model coverage from a shared SwiGLU kernel alone.

Definition of done for this iteration is reproducible native component evidence, honest rejections, current pinned baselines, preserved recovery material, and a bounded integration handoff. Definition of done for the product remains the full-runtime gates above.
