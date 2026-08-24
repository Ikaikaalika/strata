# Strata Performance Contract

Status: benchmark and promotion policy for native Strata work. A faster isolated
kernel is useful evidence; it is not by itself a faster LLM runtime.

## Optimization vector

Every comparison records the same performance vector:

\[
Q=(T_{first}, L_{token}, R_{prompt}, R_{decode}, R_{server}, M_{peak}, E, C)
\]

| Metric | Unit and boundary |
|---|---|
| \(T_{first}\) | milliseconds from admitted request to first visible token |
| \(L_{token}\) | median and p95 inter-token milliseconds after the first token |
| \(R_{prompt}\) | prompt tokens per second over a declared prompt bucket |
| \(R_{decode}\) | generated tokens per second over a declared output length |
| \(R_{server}\) | aggregate tokens per second at a declared concurrency |
| \(M_{peak}\) | peak total runtime resident bytes, including caches and temporaries |
| \(E\) | joules per prompt and generated token when an accepted instrument is available |
| \(C\) | correctness outcome and maximum/mean error for the exact contract |

No plan wins merely because it activates more engines. It must improve at least
one declared objective without violating the request's correctness, memory,
tail-latency, reliability, or energy limits.

## Evidence ladder

| Level | Meaning | Permitted claim |
|---|---|---|
| L0 contract | shape, dtype, layout, semantics, timing boundary fixed | implementation target only |
| L1 correctness | deterministic independent-oracle parity | operator is eligible for timing |
| L2 kernel hardware | serialized warm hardware measurements | this kernel at this exact envelope |
| L3 phase-program hardware | complete prefill or decode segment with handoffs | this phase program at this envelope |
| L4 standalone model | real model, tokenizer, KV and sampling through public API | local model-runtime result |
| L5 Common Compute canary | adapter, cancellation, isolation, receipts, soak | bounded provider-runtime result |

Synthetic fixtures never support real-model throughput, energy, storage, or
capacity claims. A private ANE compile cache is not live weight paging, and ANE
use requires compile, dispatch, readback, and numerical verification.

## Required hardware record

Every hardware result includes:

- Strata revision and dirty status;
- chip, memory size, CPU/GPU identity, macOS version and build;
- model or generated-fixture digest, operation, exact shapes, dtype, layout,
  quantization, prompt/output lengths, batch/concurrency;
- warmup and measured iteration counts;
- median, p95, minimum, mean, units, and precise timing boundary;
- oracle, error tolerance, maximum/mean error, and pass/fail;
- thermal/power state when observable, backend versions, and known limitations.

## Baselines

Real-model release runs compare the same model revision, quantization, context,
sampling settings, and output tokens against:

- MLX as the Apple-Silicon correctness/compatibility baseline;
- llama.cpp Metal as the portable native baseline;
- Ollama for the complete local-service experience;
- Strata's supported fallback and each promoted native phase program.

Raw logs remain immutable. A summarized result must link to its exact raw
record and disclose any unavailable metric.

## Promotion gates

### Native kernel to phase-program candidate

- Exact contract correctness passes against an independent oracle.
- At least five warmups and thirty measured iterations are recorded.
- Complete segment latency or energy improves by at least 10 percent, with no
  correctness regression. An energy-only win may cost at most 2 percent latency.
- p95 latency, peak memory, conversion, synchronization, and engine-handoff
  time are reported.
- The complete enclosing phase retains a measurable win; otherwise the kernel
  stays experimental.

### Standalone runtime MVP

- One declared real-model matrix completes through the public API.
- Strata is within 5 percent of the better comparable MLX/llama.cpp result for
  both time-to-first-token and steady decode, and wins by at least 10 percent on
  one declared latency, throughput, memory, or energy objective.
- At least 100 streamed requests cover cancellation, bounded output, deterministic
  sampling, overload rejection, and model unload/reload without leaked state.
- Results distinguish cold start, warm model, warm prefix, and sustained load.

### Common Compute canary

- Feature flag, provider allowlist, automatic supported-backend fallback, and
  bounded memory/concurrency are active.
- Cancellation, process isolation, terminal receipts, and usage accounting pass.
- A 24-hour fault/soak run completes before broader provider rollout.

## Benchmark suites

1. **Primitive suite:** projection, normalization, RoPE, attention, quantized
   matrix-vector/matrix-matrix, sampling, and backend handoff.
2. **Phase suite:** short/long prefill and batch-1/batched decode with persistent
   weights and KV.
3. **Product suite:** public API, real models, continuous batching, cancellation,
   memory pressure, cold/warm load, and Common Compute adapter.

The first native primitive is the exact FP16 projection also used by the
bounded ANE probe:

```sh
./native/metal/build_linear_bench.sh
```

It uses a generated fixture and compares a native scalar CPU oracle with a
direct-Metal implementation. The result is L2 kernel hardware evidence only.

`/Volumes/Tyler HDD` is explicitly excluded from all SSD qualification, weight
placement, and offload benchmarks. Storage work waits for a user-approved SSD
destination.

