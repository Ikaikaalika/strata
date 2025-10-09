# MLX-Only Port Build Plan

## Goals
- Run the fork on Apple Silicon using MLX end-to-end (no CUDA / torch GPU deps).
- Match or exceed existing long-context capabilities (100k tokens) for target models.
- Preserve public Python API while hiding backend changes behind an abstraction layer.
- Leave room to introduce Metal custom kernels where profiling shows MLX gaps.

## Scope & Assumptions
- Target hardware: M2/M3 class Apple Silicon with unified memory and fast local SSD.
- Target software: Python 3.11+, MLX latest stable, Hugging Face tokenizers/processors.
- Priority models: Llama 3.x (1B/3B/8B), GPT-OSS-20B, Gemma3-12B text, optional Gemma vision path.
- Disk offload and long-context KV cache must keep working within unified memory limits.
- We can add optional custom Metal kernels if MLX primitives underperform; default path is pure MLX.

## Phase 0 – Baseline Audit
**Tasks**
- Inventory Torch-only code (`src/ollm/inference.py`, loaders, `attention.py`, `kvcache.py`).
- Document current data flow: weight loading from SSD, KV cache behavior, Flash Attention usage.
- Record baseline correctness/perf numbers from example scripts on Apple Silicon (Torch CPU fallback).
- Identify minimum viable model/context combos to validate during the port.
**Deliverables**
- Architecture note describing torch dependencies, expected tensor shapes/dtypes, and I/O paths.
- Benchmark table for current behavior (latency, throughput, memory, disk usage).

## Phase 1 – Backend Abstraction Layer *(Completed)*
**Tasks**
- Define backend interface covering tensor ops, model init, KV cache, disk offload, statistics hooks.
- Implement Torch backend wrapper to maintain current functionality.
- Refactor inference entry points/examples to request backend via config (`device="torch:cuda"`, `device="mlx:0"`).
- Add unit tests that exercise backend selection and capability checks (text vs multimodal).
**Deliverables**
- Backend registry module with Torch backend defaulting to existing behavior.
- Passing tests ensuring inference path stays intact with backend indirection.

## Phase 2 – MLX Core Infrastructure
**Tasks**
- Environment tooling: script to create Apple venv, install MLX, compatible `transformers`, numpy.
- Weight conversion: CLI to map HF checkpoints / `gds_export` shards into MLX arrays (`.npz` with metadata, dtype checksums).
- KV cache implementation: port disk-backed cache to MLX tensors using memory-mapped chunks and unified memory heuristics.
- Attention kernels: implement grouped chunked attention + RoPE in MLX; verify numerical parity with Torch version.
  - MLX parity kernel drafted in `ollm/backends/mlx_ops.py`; needs integration into inference once backend matures.
  - Backend API exposes `attention_kernel()` so both Torch and MLX paths can supply compatible callables.
- Utilities: replicate chunked MLP, norms, rotary embeddings using MLX primitives.
- Backend plumbing: extend the MLX backend skeleton to support model placement and cache creation once kernels exist.
**Custom Kernel Considerations**
- Profile MLX attention and MLP ops; if perf < target, design Metal kernel overrides callable via MLX custom op hooks.
- Plan build system adjustments (Metal shader compilation, fallback to pure MLX when unavailable).
**Deliverables**
- `ollm/backends/mlx` module with attention, MLP, KV cache, and tensor helpers.
- Conversion tool with validation suite comparing Torch vs MLX weights on sample layers.
- Parity tests (unit + integration) asserting logits/token outputs match within tolerance for small prompts.

## Phase 3 – Model Integrations
**Tasks**
- Llama path: load converted weights, wire tokenizer outputs, ensure generation loop works with MLX backend.
- GPT-OSS path: port custom attention/MLP components, handle packed weights, confirm chunked loaders.
- Gemma3 text: adapt loader and special layers; optionally gate multimodal vision branch until MLX support exists.
- Update inference helpers (`offload_layers_to_cpu/gpu`) to make sense in MLX context (e.g., unified memory buckets).
**Deliverables**
- MLX backend producing valid generations for each target text model.
- Regression tests comparing Torch vs MLX logits for truncated sequences per model.
- Documented list of unsupported features (e.g., Gemma vision pending).

## Phase 4 – Performance, Custom Kernels & Reliability
**Tasks**
- Benchmark long-context runs (10k–100k tokens) for each model, capture latency, memory, SSD throughput.
- Profile hotspots using MLX tooling; tweak block sizes, precision (fp16/bf16/fp32), and offload strategies.
- When MLX primitives underperform, implement targeted custom Metal kernels (attention, fused MLP) using MLIR/Metal shading, integrate into backend behind feature flags.
- Stress-test disk cache recovery, repeated load/unload cycles, low-memory scenarios.
- Extend `Stats` instrumentation to log MLX timings/bandwidth for telemetry.
**Deliverables**
- Performance report vs goals, including when Metal custom kernels are required and how to enable them.
- Reliability test suite (long-context soak test, disk cache restart test) automated via scripts.
- Optional Metal kernel source and build steps packaged in repo with fallback path.

## Phase 5 – Tooling, CI & Documentation
**Tasks**
- Add macOS automation (local script or self-hosted runner) running lint, unit tests, weight conversion, and short generation smoke test.
- Update CLI/examples to auto-detect hardware and select MLX backend or advise conversion steps.
- Refresh README + new docs: Apple setup guide, backend selection, model support matrix, troubleshooting (Metal crashes, conversion errors).
- Prepare release notes, changelog, and pointer to pre-converted weights if licensing permits.
- Ensure backend selection tests run in CI (mock optional deps or provide light-weight stubs for cupy/kvikio).
**Deliverables**
- CI entries for macOS MLX smoke tests.
- Updated examples (`run_example.bash`, `example.py`) demonstrating `device="mlx:0"` usage.
- Comprehensive docs folder with setup, usage, troubleshooting, and custom kernel instructions.

## Validation Strategy
- Golden-output comparisons: store short context prompts with Torch logits and assert MLX stays within tolerance.
- Unit tests for converters, attention, KV cache, and backend selection.
- Integration tests executing full generation with disk cache enabled.
- Manual QA checklist for each release (convert weights, run long-context, verify stats logging).

## Risk Matrix
| Risk | Impact | Mitigation |
| --- | --- | --- |
| MLX lacks needed op/perf | Missed throughput goals | Implement custom Metal kernels, simplify attention (smaller blocks) while shipping MVP |
| Weight conversion drift | Incorrect outputs | Add shape/dtype checksums, Golden tests, conversion CI |
| Unified memory exhaustion | Crashes under long context | Implement adaptive offload (disk/CPU), expose config knobs |
| Vision model parity | Missing multimodal support | Ship text-only MLX first; keep torch backend as fallback for vision until ported |

## Kick-Off Checklist
- [x] Complete torch dependency + data flow audit (Phase 0 deliverables).
- [x] Lock backend interface design (review with team).
- [ ] Stand up Apple Silicon dev environment with MLX installed and sample model downloaded.
- [ ] Schedule profiling spike on MLX attention to estimate custom kernel need.
- [ ] Align on performance targets (latency/throughput) and acceptance tests for MVP release.
