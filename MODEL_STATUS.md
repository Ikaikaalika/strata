# Lōkahi implementation status

This file distinguishes locally verified behavior from adapter scaffolding and
future work. It is not a real-model benchmark report.

## Verified locally

The deterministic offline suite verifies the following on MLX:

| Area | Evidence |
|---|---|
| Grouped-query attention | MLX output matches an independent NumPy reference |
| Causal masking | Future values cannot affect earlier token outputs |
| RoPE | Query and key rotation matches a NumPy reference |
| Prefill/decode split | Generation submits the full prompt once, then one token per call |
| KV cache | Values and per-layer lengths append correctly; replayed positions fail closed |
| End-to-end tiny models | Cached decode logits match full causal forward passes in two-layer Llama and DeepSeek fixtures |
| Disk cache round trip | Serialized keys and values restore without numerical change |
| Tracing | Phase, layer, tensor shape/dtype/bytes, duration, and MLX memory are recorded |
| Lōkahi Governor | Hard byte budget, pinned-layer safety, exact prefetch ordering, LRU eviction, demand fallback, and warm reuse are deterministic tests |
| Governed Llama | Output matches the non-governed path and a full-budget second forward performs no layer reloads |

Run:

```bash
PYTHONPATH=src python -m pytest -q
```

No network access or model weights are required.

## Adapter status

| Model family | Current state |
|---|---|
| Llama | Custom MLX adapter has a deterministic two-layer correctness fixture; `mlx_lm` loading is also available in `Inference` |
| DeepSeek | Llama-like custom adapter has the same deterministic two-layer cached-versus-full correctness coverage; a current real-model validation run is still required |
| Qwen3-Next | Not implemented in the active MLX execution path |
| Gemma 3 | Not implemented in the active custom MLX execution path |
| GPT-OSS | Not implemented; packed MXFP4 support is required |

Model availability, license gates, and weight downloads are not treated as
proof of adapter correctness. Each real model still needs an explicit,
reproducible validation report.

## Memory-tiering status

- The custom Llama and DeepSeek adapters can use the persistent Lōkahi Governor
  when constructed with a weight loader and `memory_budget_bytes`.
- The governor reserves expected bytes before I/O, pins the active layer,
  prefetches the next exact dense layer, and evicts only unpinned LRU entries.
- MLX execution is materialized per layer before model references are cleared,
  preventing a lazy graph from retaining every prior layer's weights.
- `MLXKVCache` supports memory-only operation and explicit SSD serialization.
- Weight load time, effective bandwidth, cache status, demand stall, layer
  compute time, and MLX memory counters are traced separately.
- The current overlap benchmark is synthetic. Actual SSD files, model-token
  throughput, page-cache conditions, and memory-pressure behavior remain to be
  measured.
- MoE expert loading code is scaffolding, not a completed router-driven paging
  implementation.

## Next validation milestone

1. Benchmark real layer files on the target storage device.
2. Adapt standard `mlx_lm` modules to governed layer leases.
3. Make the residency budget respond to KV-cache growth and memory pressure.
4. Add router-driven MoE expert groups only after dense paging is measured.
