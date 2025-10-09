# Phase 0 Baseline Audit

## 1. Repository Architecture Snapshot

### 1.1 Public Entry Points
- `src/ollm/inference.py` – user-facing `Inference` class; orchestrates model downloads, loader selection, tokenizer init, and optional disk-backed KV cache. Hardcodes Torch tensors and assumes CUDA (`device="cuda:0"`).
- `example.py`, `run_example.bash` – demonstrate usage with CUDA GPUs; environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` baked in.

### 1.2 Core Execution Flow
1. **Model bootstrap**
   - `Inference.ini_model` downloads weights (`.zip` bundles from S3 or HuggingFace) and extracts manifest under `<models_dir>/<model_id>/`.
   - Model-specific modules (`llama.py`, `gpt_oss.py`, `qwen3_next.py`, `gemma3.py`) expose subclasses of HuggingFace `PreTrainedModel` that override forward passes for chunked loading/offloading.
2. **Weight streaming**
   - Llama / GPT-OSS loaders rely on `GDSWeights` (`gds_loader.py`).
   - Manifests reference `gds_export` files read through NVIDIA GPUDirect Stack (`kvikio.CuFile`, `cupy`); tensors are materialized directly on CUDA via DLPack.
3. **Forward pass**
   - Custom attention: `attention.online_chunked_grouped_attention_rope_no_mask` operates on 4D Torch CUDA tensors; uses float32 accumulators and streaming log-sum-exp.
   - Custom MLP: chunks activations on CUDA to fit 8 GB VRAM.
4. **KV-cache management**
   - `kvcache.KVCache` inherits HuggingFace `DynamicCache`, writes `.pt` tensors per layer, reloads from disk, and performs concatenations on CUDA tensors.
5. **Offload hooks**
   - Layers dynamically load/unload weights inside decoder layer forward methods; placeholders use Torch meta tensors.
6. **Statistics**
   - `utils.Stats` records timing buckets for `layer_load`, `kvsave`, `kvload`, etc.; assumes Torch CUDA sync is available when enabled.

## 2. Torch & CUDA Reliance Breakdown

| Area | File(s) | Torch/CUDA Usage | Notes |
| --- | --- | --- | --- |
| Tensor core | `attention.py` | Expects `torch.cuda` tensors; uses `.to(torch.float32)` conversions; no CPU code path. | Will need MLX equivalent of streaming attention kernel.
| CUDA device mgmt | `inference.py`, `llama.py`, `gpt_oss.py`, `gemma3.py`, `qwen3_next.py` | Instantiates `torch.device("cuda:0")`, `model.to(device)`; gating logic assumes CUDA availability. | Needs backend abstraction for device discovery.
| Weight streaming | `gds_loader.py` | Imports `cupy`, `kvikio`, `torch.utils.dlpack.from_dlpack`; loads into CUDA memory; relies on GPUDirect Storage. | Entire path incompatible with MLX/Metal.
| KV cache | `kvcache.py` | Saves/loads `.pt` tensors with CUDA storage; concatenates on CUDA; uses pinned memory for async copy. | Requires MLX-based cache and disk serialization format.
| Model internals | `llama.py` etc. | Custom modules call Torch linear layers, `torch.cat`, `torch.zeros`, etc.; rely on CUDA for weight placement. | Need MLX module rewrites or translation layer.
| Utility functions | `utils.py` | Includes `torch` operations within helper functions for stats and file patching. | Evaluate individually; some can stay backend-agnostic.

### External Dependencies
- `torch`, `torchvision` (via `mlp`?), `transformers` (HF) – expected to stay.
- `cupy`, `kvikio` – NVIDIA-only; replacable with NumPy/mmap or MLX-specific IO.
- `requests`, `huggingface_hub`, `safetensors` (optional) – remain usable on Apple.

## 3. Model Asset Layout
- **Local zip bundles** (`llama3-*.zip`, `gpt-oss-20B.zip`): unzip into `<models_dir>/<model_name>/gds_export/*` plus config/tokenizer files.
  - Manifest entries map parameter names to binary blocks (often row-major float16) and optionally `packed="mxfp4"` for GPT-OSS MoE weights.
- **HuggingFace models** (`qwen3-next-80B`, `gemma3-12B`): downloaded via `huggingface_hub.snapshot_download` with canonical HF directory structure.
- **KV cache**: stored under `<cache_dir>/kv_cache/layer_{idx}.pt`, containing Torch tuples `(key, value)`.

## 4. Data & Control Path Details
- Decoder layers load weights on-the-fly before each forward call and unload after to keep VRAM usage minimal.
- Streaming attention splits query length into 32k-token blocks; keys/values consumed in 1k-token chunks to match disk streaming.
- Disk-backed KV cache only writes the first time a layer is visited; subsequent tokens append to in-memory `key_cache2`/`value_cache2` lists.
- Multi-modal Gemma3 uses `AutoProcessor` and custom loader `Gemma3Loader`, but still depends on Torch for inference.

## 5. Baseline Benchmark Plan
Actual runs currently blocked on:
- Large weight bundles (13–160 GB) not present in repository.
- Torch defaults to CPU on Apple Silicon, causing extreme runtimes for 100k contexts.

### Proposed Measurement Procedure (to execute once assets are available)
1. Create CUDA-capable baseline (remote Linux w/ RTX 4090 or cloud) to capture reference numbers.
2. For Apple baseline, install PyTorch 2.4 Apple build, run `example.py` with smallest model (`llama3-1B-chat`) using CPU fallback.
3. Measure:
   - **Token throughput**: total new tokens / generation time.
   - **Latency**: time to first token, steady-state.
   - **Memory**: `psutil.Process().memory_info().rss`, `torch.cuda.memory_allocated` (CUDA baseline).
   - **Disk IO**: monitor with `iostat` or `sudo fs_usage` on macOS.
4. Record metrics for context lengths 4k and 32k using identical prompts.

### Benchmark Table Template
| Model | Context | Device | Throughput (tok/s) | Time-to-first (s) | Peak RAM (GB) | Peak Disk IO (MB/s) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-1B-chat | 4k | Torch CPU (M3 Max) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Baseline pending weights |
| llama3-1B-chat | 32k | Torch CPU (M3 Max) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |  | 
| gpt-oss-20B | 10k | Torch CUDA (RTX 4090) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Reference only |

_Action item_: populate table once download + run succeeds. For now, metrics remain placeholders.

## 6. Risks Identified in Baseline
- Weight loaders are tightly coupled to GPUDirect Storage; no CPU fallback implemented.
- Attention kernel assumes contiguous CUDA tensors; data movement to CPU will thrash performance.
- KV cache serialization uses Torch `.pt` (pickle); not directly consumable by MLX without conversion.
- Multimodal Gemma3 path may require significant rewrite (vision transformer layers not abstracted).

## 7. Recommendations Before Phase 1
- Freeze current branch with doc updates and confirm no functional regressions on CUDA path.
- Acquire smallest supported weight bundle and verify `example.py` still succeeds on existing GPUs.
- Outline acceptance thresholds for MLX parity (e.g., ±5% logits difference, throughput goal).

---
_Last updated: 2025-10-03 12:37_
