# Lōkahi native engine

C++20 inference engine with a Metal GPU backend, a portable CPU backend, a C
ABI ([`include/lokahi/lokahi.h`](include/lokahi/lokahi.h)), and the `lokahi`
CLI. Python is not on the token path.

```sh
cmake -S native/engine -B build/engine -DCMAKE_BUILD_TYPE=Release   # Metal is on by default on macOS
cmake --build build/engine -j
./build/engine/lokahi_engine_tests
./build/engine/lokahi info --model /path/to/mlx-snapshot
./build/engine/lokahi generate --model DIR --tokens 2,4521,603 --max-new 64
./build/engine/lokahi bench --model DIR --tokens-file prompt.txt --max-new 128
```

## Status

| Area | State | Evidence |
|---|---|---|
| Gemma 3 text decoder, CPU backend | Implemented | Correctness: matches the NumPy oracle within 2e-4 relative; greedy tokens exact (`tests/test_native_engine.py`) |
| Gemma 3 text decoder, Metal backend | Implemented, awaiting first hardware run | Builds on hosted macOS CI; execution and parity run on the self-hosted runner |
| MLX affine 2/4/8-bit weights, BF16/F16 scales, per-module bit overrides | Implemented | Unit tests; MLX cross-check on Apple Silicon |
| Tokenizer | Not yet native | Callers pass token ids; benchmarks tokenize with MLX-LM |

No throughput claim is made until the self-hosted runner records hardware
evidence against a fresh MLX-LM control.

## Design

**Loading.** Checkpoints are memory-mapped safetensors (single file or
shards) in MLX layout. The Metal backend copies every matrix once into a few
large shared buffers, aligned to 256 bytes and marked hazard-untracked
because they are read-only.

**Decode step.** One command buffer per token, with roughly 9 dispatches per layer:

1. Embedding gather.
2. Fused q/k/v matrix-vector product (`qmv_qkv`).
3. Fused QK-norm, RoPE and KV-cache write.
4. Split-K online-softmax attention, plus a reduce pass.
5. Output projection.
6. Fused residual and RMSNorm.
7. Fused gate/up/GELU (`qmv_geglu`).
8. Down projection.
9. Fused residual and RMSNorm.

The step ends with the last-row vocabulary projection and an argmax on the GPU.

**Pipelined greedy generation.** Each step reads the previous token from GPU
memory, so two steps stay in flight and the host never stalls the GPU between
tokens.

**Prefill.**

- Matrix products use a tiled quantized matmul (`qmm`) built on SIMD-group
  8×8 matrices.
- Chunks of up to `--prefill-chunk` tokens all go into one command buffer.
- The vocabulary projection runs only for the final prompt token.

**KV cache.** Global layers hold `max_context` slots. Sliding-window layers
hold `window + prefill_chunk` slots in a ring, so a whole chunk can be written
before its queries read the preceding window.

**Numerics.** Activations, norms, attention and the KV cache are float32.
Weights stay in their packed 4-bit or 8-bit form and are dequantized inside
the kernels.

## Checks on machines without Apple SDKs

`tests/syntax_check.sh` parses the Objective-C++ host and the Metal kernels
with clang against stub headers (`tests/syntax_stubs/`). It catches typos and
C++ type errors on Linux. It is not a Metal compiler.
