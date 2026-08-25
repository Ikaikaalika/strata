# Strata

Strata is an Apple-Silicon-first, MLX-based inference runtime for local and
large-context experiments. It is also a learning project: the code keeps the
important inference boundaries visible—weight loading, attention, KV-cache
growth, and token generation—so they can be studied close to the metal.

The repository is evolving from oLLM. The current Python import remains
`ollm` for compatibility; the supported runtime path in this checkout is
MLX on Apple Silicon.

## Current status

The implementation is experimental. The following table describes the state of
the code in this repository, not a future roadmap.

| Model or path | Status |
| --- | --- |
| `deepseek-coder-1.3b` | Verified end-to-end MLX smoke path |
| `deepseek-coder-6.7b` | Loader path exists; requires the model weights and local validation |
| `llama3-1B-chat`, `llama3-3B-chat`, `llama3-8B-chat` | Loader path uses `mlx-lm`; Hugging Face access is required and this path is not the verified smoke test |
| `qwen3-next-80B` | Not implemented in the current MLX loader |
| `gemma3-12B` | Not implemented in the current MLX loader |
| `gpt-oss-20B` | Not implemented; mxfp4 support is still required |

Implemented building blocks include:

- MLX backend selection with MLX as the default backend.
- MLX-native DeepSeek model execution.
- Safetensors loading with model-weight residency helpers.
- Chunked grouped-query attention with online softmax.
- An MLX KV cache with optional NumPy/NPZ disk offload.
- Simple greedy token generation for the MLX model path.

See [MODEL_STATUS.md](MODEL_STATUS.md) for the detailed model matrix and
known limitations.

The following remain target work rather than shipped capability: persistent
production XPC serving, multi-request continuous batching, complete native
Metal/ANE LLM execution, production SSD paging, and a separate `strata_llm`
distribution. Do not infer those capabilities from the presence of design or
integration artifacts.

## Requirements

- macOS on Apple Silicon
- Python 3.10 or newer
- MLX and `mlx-lm`
- Enough local storage for the selected Hugging Face model and optional cache

The dependencies are declared in [pyproject.toml](pyproject.toml). CUDA and
PyTorch are not required for the supported MLX path, although a few legacy
loader and test files still reference them.

## Install from source

Create an isolated environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Model downloads are performed through `huggingface_hub`. For gated models,
authenticate with Hugging Face before loading them:

```bash
huggingface-cli login
```

Use a model directory on a volume with enough free space. The default is
`./models`, or pass another location through `models_dir`.

## Quick start

The smallest current smoke test is [quick_demo.py](quick_demo.py):

```bash
python quick_demo.py
```

The equivalent API usage is:

```python
import mlx.core as mx

from ollm import Inference

inference = Inference("deepseek-coder-1.3b", device="mlx", logging=False)
inference.ini_model(models_dir="./models")

inputs = inference.tokenizer("def quicksort(arr):", return_tensors="np")
input_ids = mx.array(inputs["input_ids"])
output_ids = inference.model.generate(input_ids, max_new_tokens=50)
mx.eval(output_ids)

print(inference.tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

`ini_model` is the existing public method name and is retained for API
compatibility. `Inference(device=None)` also selects MLX automatically. To
select a specific MLX device, use values such as `mlx`, `mlx:gpu`, or
`mlx:cpu`.

## Local checks

Run the model smoke scripts from an activated environment:

```bash
python quick_demo.py
python test_models_automated.py
```

The automated script currently exercises `deepseek-coder-1.3b` and lists
larger, gated, and unimplemented models separately. The files under `tests/`
include older backend/parity coverage and should not be read as evidence that
the full test suite is currently green.

## Repository map

- `src/ollm/` — MLX backend, model implementations, loaders, attention, and KV cache.
- `quick_demo.py` — minimal end-to-end DeepSeek MLX example.
- `test_models_automated.py` — local model smoke-test runner.
- `test_single_model.py` — one-model smoke test; accepts a model ID as its first argument.
- `MODEL_STATUS.md` — detailed support matrix and current limitations.
- `samples/` — local text samples for experimentation.

## Design direction

Strata is intended to make the inference pipeline understandable and
measurable on Apple Silicon:

```text
prompt
  -> tokenizer
  -> MLX model
  -> attention + KV cache
  -> greedy decode
  -> generated tokens
```

Future runtime work should preserve the MLX path as a correctness oracle while
native Metal/ANE phases, residency management, batching, and bounded XPC
execution are validated independently.

## License

See [LICENSE](LICENSE).
