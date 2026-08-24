# Strata

Strata is an Apple Silicon runtime and learning laboratory for memory-tiered
large language model inference.

Strata is being engineered as the adaptive per-Mac LLM runtime beneath Common
Compute. Common Compute selects a provider Mac; Strata admits the request and
chooses a measured plan for that model, service objective, hardware identity,
and live memory/thermal/power state.

Its engineering objective is to run the largest useful model with the smallest
practical RAM footprint while preserving as much token throughput as possible.
The current baseline uses MLX, a runtime-neutral IR and adaptive planner, a
versioned SSD weight-pack path, native Metal probes, and an isolated private-ANE
qualification worker. Full-model heterogeneous execution and MoE expert paging
remain staged work.

> The installable distribution is named `strata-llm`. The Python import
> namespace remains `ollm` temporarily for compatibility.

## Current verified baseline

The offline test suite and serialized local M1 probes currently verify:

- Numerically stable chunked grouped-query attention against a NumPy reference
- Causal masking across query and key chunks
- Absolute query positions during cached decode
- RoPE against a deterministic NumPy reference
- Prompt prefill followed by one-token decode
- Cached decode logits against full causal forward passes in deterministic,
  two-layer Llama and DeepSeek fixtures
- KV-cache values, lengths, disk round trips, and replay rejection
- Tensor shape, dtype, byte size, phase timing, and MLX memory trace records
- Byte-budgeted layer residency, pinned-weight safety, exact next-layer
  prefetch, and warm-layer reuse across forwards
- Evidence-scoped prefill/decode planning with an MLX fallback
- Live-platform admission with explicit service objectives, safe memory budgets,
  full/paged residency decisions, and fail-closed SSD qualification
- Checksummed, aligned weight-pack exact-range and read-only mmap access
- Direct private-ANE compilation, IOSurface dispatch, and CPU numerical parity
  for one fixed fp16 projection on the local M1
- Exact StrataIR lowering into that fixed callable ANE prefill segment
- Weight-pack ranges loaded through the real residency and prefetch pipeline

No network access or model download is required for these tests. Real-model
throughput and memory claims are intentionally separate from this deterministic
correctness baseline.

> Storage boundary: `/Volumes/Tyler HDD` is the source checkout, not an SSD.
> Strata must not use it for SSD offload, SSD benchmarking, or model/cache
> placement intended to represent SSD behavior. SSD experiments require a
> separate destination approved by the user.

## Architecture direction

```mermaid
flowchart TD
    CLIENT["Common Compute client"] --> ROUTER["Fleet router"]
    ROUTER --> HOST["Swift provider host"]
    HOST -->|"fixed request + cancellation"| XPC["Persistent no-network Strata XPC"]
    XPC --> ADMIT["Live admission + memory budget"]
    ADMIT --> BATCH["Prefill/decode batch scheduler"]
    BATCH --> PLAN["Evidence-gated segment planner"]
    PLAN --> MLX["MLX compatibility baseline"]
    PLAN --> METAL["Custom Metal"]
    PLAN --> COREML["Supported Core ML"]
    PLAN --> ANE["Isolated ANE research"]
    BATCH --> KV["KV + prefix cache"]
    BATCH --> RES["Weight residency + prefetch"]
    RES --> RAM["Unified memory"]
    RES --> SSD["Approved measured SSD"]
    XPC -->|"bounded events + receipt"| HOST

    LAB["Python Strata oracle/lab"] -->|"golden fixtures + evidence"| PLAN
```

The boundary is deliberate: Common Compute owns networking, leases, routing,
billing, and provider lifecycle. Strata owns the persistent inference hot path:
admission, batching, KV state, residency, prefetching, eviction, backend
selection, and measurement. MLX and native adapters own tensor kernels.

See [docs/COMMON_COMPUTE_RUNTIME.md](docs/COMMON_COMPUTE_RUNTIME.md) for the
canonical ownership map, XPC protocol, adaptation loop, migration waves, and
acceptance gates.

## Strata Governor

The first rearchitected runtime slice is a persistent Apple Silicon memory
governor:

```mermaid
flowchart LR
    SSD["Cold weights on SSD"] --> LOAD["Reserved asynchronous load"]
    LOAD --> WARM["Byte-budgeted warm LRU"]
    WARM --> PIN["Pinned layer lease"]
    PIN --> MLX["MLX / Metal layer compute"]
    MLX --> RELEASE["Synchronize, clear references, unpin"]
    RELEASE --> WARM
    WARM -->|"budget pressure"| SSD
```

The governor prefetches layer `N+1` before synchronizing layer `N`, measures
total load time separately from demand stall time, and retains hot layers when
the configured budget permits it. See
[docs/STRATA_GOVERNOR.md](docs/STRATA_GOVERNOR.md) for the design, equations,
state machine, hardware profile, and explicit limitations.

## Repository map

| Path | Purpose |
|---|---|
| `src/ollm/backends/mlx_ops.py` | Chunked grouped-query attention and causal masking |
| `src/ollm/llama_mlx.py` | Custom Llama-family MLX adapter |
| `src/ollm/deepseek_mlx.py` | Custom DeepSeek-family MLX adapter scaffold |
| `src/ollm/mlx_kvcache.py` | In-memory and SSD-backed MLX KV cache |
| `src/ollm/generation.py` | Shared prefill and incremental decode loop |
| `src/ollm/tracing.py` | Runtime-neutral tensor and operation trace records |
| `src/ollm/core/` | Runtime-neutral tensor, group, capability, and plan contracts |
| `src/ollm/core/platform.py` | Common Compute service objective, live-state, storage, and admission contracts |
| `src/ollm/planning/adaptive_planner.py` | Evidence-gated prefill/decode target selection |
| `src/ollm/storage/` | Cold tensor-store and manifest adapters |
| `src/ollm/storage/weight_pack.py` | Versioned aligned pack, checksums, exact reads, and mmap |
| `src/ollm/storage/weight_pack_store.py` | Pack-backed group loading for residency and prefetch |
| `src/ollm/scheduling/` | Residency manager, prefetch scheduler, and dense pipeline |
| `src/ollm/backends/mlx_governor.py` | MLX hardware profile and governor builder |
| `docs/AGENTIC_ENGINEERING.md` | Agent roles, evidence ladder, and integration gates |
| `docs/COMMON_COMPUTE_RUNTIME.md` | Common Compute boundary, adaptive objective, and native-runtime roadmap |
| `docs/WAVE1_EVIDENCE.md` | Local M1 CPU, Metal, and ANE evidence and limitations |
| `docs/WAVE2_EVIDENCE.md` | Adaptive planner, weight pack, and direct-ANE projection proof |
| `docs/WAVE3_EVIDENCE.md` | Callable ANE segment and pack-backed residency evidence |
| `native/ane/` | Isolated private-runtime discovery and opt-in projection worker |
| `src/ollm/backends/ane_executor.py` | Bounded fixed-shape ANE request client |
| `src/ollm/backends/ane_segment.py` | Exact StrataIR-to-ANE linear lowering |
| `tests/` | Deterministic offline correctness suite |
| `animations/prefill_vs_decode.py` | First Manim learning lesson |
| `STRATA_ENGINEERING_CONTEXT.md` | Architecture, equations, roadmap, and handoff context |

## Local setup

Strata targets Apple Silicon and Python 3.10 or newer.

```bash
python3 -m venv strata_env
source strata_env/bin/activate
python -m pip install -e ".[dev]"
```

Run the deterministic suite from the repository root:

```bash
PYTHONPATH=src python -m pytest -q
```

The custom model adapters do not download weights during unit tests. Loading a
real model is a separate operation and may require Hugging Face authentication,
license acceptance, and substantial local storage.

## Tracing

Pass a `TensorTracer` to a custom MLX model adapter to collect metadata without
retaining the underlying tensors:

```python
from ollm import TensorTracer
from ollm.llama_mlx import MLXLlamaForCausalLM

tracer = TensorTracer()
model = MLXLlamaForCausalLM(config, tracer=tracer)

# After a model call:
for event in tracer.events:
    print(event)
```

When tracing is enabled, attention execution is synchronized so duration and
MLX active/peak memory readings describe completed device work. This adds
measurement overhead and should be disabled for throughput benchmarks.

## Roadmap

1. Freeze the versioned Common Compute request, live-state, admission, and
   receipt schema; mirror it in Swift with cross-language golden fixtures.
2. Prove fixed-request streaming, cancellation, path confinement, and restart
   through the no-network XPC service with a generated fixture.
3. Run one pinned model in a persistent native MLX XPC worker with exact
   `mlx_llm` compatibility and fallback.
4. Add bounded continuous batching with separate/chunked prefill and one-token
   decode scheduling, independent streams, fairness, and cancellation.
5. Connect adaptive KV/residency budgets and qualified SSD paging with explicit
   host/runtime copy and demand-stall accounting.
6. Grow Metal/Core ML/ANE operation envelopes into verified transformer
   segments and keep only alternatives that improve the full serving plan.

See [STRATA_ENGINEERING_CONTEXT.md](STRATA_ENGINEERING_CONTEXT.md) for the full
technical design and teaching context.

## License

See [LICENSE](LICENSE).
