"""
MLX-compatible weight loader for Apple Silicon.
Provides disk-to-memory loading for MLX models without CUDA dependencies.
"""

import json
import os
import struct
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class MLXWeights:
    """
    MLX-compatible weight loader that mirrors GDSWeights API but uses
    standard file I/O instead of GPUDirect Storage.

    Works with the same manifest.json format and directory structure.
    """

    def __init__(self, path: str, device: Optional[Any] = None):
        """
        Initialize MLX weight loader.

        Args:
            path: Path to weight directory (e.g., models/llama3-1B-chat/gds_export/)
            device: MLX device (gpu/cpu), defaults to mx.default_device()
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not available. Install with: pip install mlx")

        self.path = path
        manifest_path = os.path.join(path, 'manifest.json')

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.device = device if device is not None else mx.default_device()
        self.offloaded_map = {}  # CPU-offloaded weights cache

        # MLX dtype mapping
        self.DTYPE_MAP = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
            "float64": mx.float32,  # MLX doesn't have float64, use float32
            "int8": mx.int8,
            "int32": mx.int32,
        }

        # NumPy dtype mapping for loading
        self.NP_DTYPE_MAP = {
            "float16": np.float16,
            "bfloat16": np.uint16,  # bfloat16 stored as uint16 in binary
            "float32": np.float32,
            "float64": np.float64,
            "int8": np.int8,
            "int32": np.int32,
        }

    def has(self, name: str) -> bool:
        """Check if a parameter exists in the manifest."""
        return name in self.manifest

    def tensor_metadata(self, name: str) -> Dict[str, Any]:
        """Return shape and dtype without reading tensor data."""
        if name not in self.manifest:
            raise KeyError(f"Parameter {name!r} is not present in the manifest")
        meta = self.manifest[name]
        if meta.get("packed") == "mxfp4":
            raise NotImplementedError("mxfp4 metadata is not supported yet")
        dtype = str(meta["dtype"]).replace("torch.", "")
        return {"shape": tuple(meta["shape"]), "dtype": dtype}

    def load_param_to_device(self, name: str) -> mx.array:
        """
        Load a parameter from disk to MLX device.

        Args:
            name: Parameter name from manifest

        Returns:
            MLX array on specified device
        """
        # Check CPU cache first
        cached = self.get_offloaded_from_cpu_to_device(name)
        if cached is not None:
            return cached

        meta = self.manifest[name]
        path = os.path.join(self.path, meta["path"])
        shape = meta["shape"]
        dtype = meta["dtype"]

        # Handle different storage formats
        if meta.get("packed") == "mxfp4":
            raise NotImplementedError(
                "mxfp4 packed weights not yet supported in MLX backend. "
                "This format is used for GPT-OSS models only."
            )
        elif dtype.startswith("torch"):
            # Torch pickle files - need torch to load
            return self.load_torch_pickle_to_mlx(path)
        else:
            # Raw binary format (kvikio/numpy compatible)
            return self.load_raw_binary_to_mlx(path, shape, dtype)

    def load_raw_binary_to_mlx(
        self,
        path: str,
        shape: List[int],
        dtype: str
    ) -> mx.array:
        """
        Load raw binary weight file to MLX array.

        Args:
            path: Path to binary weight file
            shape: Tensor shape
            dtype: Data type string

        Returns:
            MLX array
        """
        np_dtype = self.NP_DTYPE_MAP[dtype]
        mlx_dtype = self.DTYPE_MAP[dtype]

        # Calculate expected size
        n_elems = int(np.prod(shape))
        nbytes = n_elems * np.dtype(np_dtype).itemsize

        # Read binary file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file not found: {path}")

        with open(path, 'rb') as f:
            data = f.read()

        if len(data) != nbytes:
            raise IOError(
                f"Size mismatch: expected {nbytes} bytes, got {len(data)} "
                f"for {path}"
            )

        # Convert to numpy, then to MLX
        if dtype == "bfloat16":
            # bfloat16 needs special handling
            np_array = np.frombuffer(data, dtype=np.uint16).reshape(shape)
            # Convert through float32 (MLX will handle bfloat16 conversion)
            mlx_array = mx.array(np_array.astype(np.float32))
            mlx_array = mlx_array.astype(mx.bfloat16)
        else:
            np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
            mlx_array = mx.array(np_array, dtype=mlx_dtype)

        return mlx_array

    def load_torch_pickle_to_mlx(self, path: str) -> mx.array:
        """
        Load a torch pickle file and convert to MLX.
        Requires torch to be installed.

        Args:
            path: Path to .pt file

        Returns:
            MLX array
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required to load .pt files. "
                "Install with: pip install torch"
            )

        # Load on CPU
        tensor = torch.load(path, map_location='cpu', weights_only=True)

        # Convert to numpy, then to MLX
        if isinstance(tensor, dict):
            # Handle packed format (e.g., mxfp4)
            raise NotImplementedError(
                "Packed tensor dictionaries not yet supported in MLX backend"
            )

        np_array = tensor.numpy()
        mlx_array = mx.array(np_array)

        return mlx_array

    def offload_param_to_cpu(self, name: str):
        """
        Load parameter to CPU memory for caching.

        Args:
            name: Parameter name
        """
        meta = self.manifest[name]
        path = os.path.join(self.path, meta["path"])
        shape = meta["shape"]
        dtype = meta["dtype"]
        packed = meta.get("packed")

        if packed == "mxfp4":
            raise NotImplementedError("mxfp4 offloading not supported yet")
        elif dtype.startswith("torch"):
            # Keep as numpy array
            import torch
            tensor = torch.load(path, map_location='cpu', weights_only=True)
            np_array = tensor.numpy()
        else:
            # Load raw binary to numpy
            np_dtype = self.NP_DTYPE_MAP[dtype]
            with open(path, 'rb') as f:
                data = f.read()
            np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)

        self.offloaded_map[name] = {
            "shape": shape,
            "dtype": dtype,
            "packed": packed,
            "array": np_array
        }

    def get_offloaded_from_cpu_to_device(self, name: str) -> Optional[mx.array]:
        """
        Retrieve cached CPU parameter and move to device.

        Args:
            name: Parameter name

        Returns:
            MLX array or None if not cached
        """
        if name not in self.offloaded_map:
            return None

        meta = self.offloaded_map[name]
        np_array = meta["array"]
        dtype = meta["dtype"]

        # Convert numpy to MLX
        mlx_dtype = self.DTYPE_MAP[dtype]
        mlx_array = mx.array(np_array, dtype=mlx_dtype)

        return mlx_array


class SafeTensorMLXReader:
    """
    MLX-compatible safetensors reader.
    Memory-efficient loading without using safetensors library's mmap.
    """

    def __init__(self, path: str):
        """
        Initialize safetensors reader.

        Args:
            path: Path to .safetensors file
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required")

        self.path = path

        # Read header
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(header_len))
            self.data_offset = 8 + header_len

        self.DTYPE_MAP = {
            "F32": mx.float32,
            "F16": mx.float16,
            "BF16": mx.bfloat16,
            "I32": mx.int32,
            "I8": mx.int8,
        }

        self.NP_DTYPE_MAP = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.uint16,  # bfloat16 stored as uint16
            "I32": np.int32,
            "I8": np.int8,
        }

    def keys(self) -> List[str]:
        """Get list of tensor names in the file."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, name: str) -> mx.array:
        """
        Load a tensor from the safetensors file.

        Args:
            name: Tensor name

        Returns:
            MLX array
        """
        if name not in self.header:
            raise KeyError(f"Tensor '{name}' not found in {self.path}")

        info = self.header[name]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        off0, off1 = info["data_offsets"]

        mlx_dtype = self.DTYPE_MAP[dtype_str]
        np_dtype = self.NP_DTYPE_MAP[dtype_str]

        # Read tensor data
        with open(self.path, 'rb') as f:
            f.seek(self.data_offset + off0)
            data = f.read(off1 - off0)

        # Convert to MLX
        if dtype_str == "BF16":
            # Special handling for bfloat16
            np_array = np.frombuffer(data, dtype=np.uint16).reshape(shape)
            mlx_array = mx.array(np_array.astype(np.float32))
            mlx_array = mlx_array.astype(mx.bfloat16)
        else:
            np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
            mlx_array = mx.array(np_array, dtype=mlx_dtype)

        return mlx_array

    def close(self):
        """Close the reader (no-op for this implementation)."""
        pass


class MLXMoEWeightsLoader:
    """
    MLX-compatible MoE weights loader for Qwen3-Next and similar models.
    Loads from HuggingFace safetensors format.
    """

    def __init__(self, path: str, device: Optional[Any] = None):
        """
        Initialize MoE weights loader.

        Args:
            path: Path to model directory
            device: MLX device
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required")

        self.path = path
        self.device = device if device is not None else mx.default_device()

        # Check if we have sharded or single-file format
        index_path = os.path.join(path, 'model.safetensors.index.json')
        single_file_path = os.path.join(path, 'model.safetensors')

        self.manifest = {}
        self.safetensors = {}

        if os.path.exists(index_path):
            # Sharded format
            with open(index_path) as f:
                indexes = json.load(f)
            weight_map = indexes["weight_map"]
        elif os.path.exists(single_file_path):
            # Single file format - create weight map
            print(f"Loading from single safetensors file: {single_file_path}")
            from safetensors import safe_open
            with safe_open(single_file_path, framework="numpy") as f:
                weight_map = {key: "model.safetensors" for key in f.keys()}
        else:
            raise FileNotFoundError(
                f"Neither {index_path} nor {single_file_path} found"
            )

        # Parse weight map to group by layer/expert
        import re
        for manifest_name, filename in weight_map.items():
            match1 = re.search(r"(model\.layers\.\d+\.mlp\.experts\.\d+\.)", manifest_name)
            match2 = re.search(r"(model\.layers\.\d+\.)", manifest_name)

            if match1 or match2:
                base = match1.group(1) if match1 else match2.group(1)
                if base not in self.manifest:
                    self.manifest[base] = {}
                attr_path = manifest_name.replace(base, "")
                self.manifest[base][attr_path] = filename

        self.offloaded_map = {}

    def load_dict_to_device(self, base: str) -> Dict[str, mx.array]:
        """
        Load all weights for a layer/expert.

        Args:
            base: Base path (e.g., "model.layers.0.")

        Returns:
            Dictionary of attribute paths to MLX arrays
        """
        # Check cache
        cached = self.get_offloaded_dict_to_device(base)
        if cached:
            return cached

        return self.load_dict_from_disk(base)

    def load_dict_from_disk(self, base: str) -> Dict[str, mx.array]:
        """
        Load weights from disk.

        Args:
            base: Base path

        Returns:
            Dictionary of weights
        """
        if base not in self.manifest:
            raise KeyError(f"Base '{base}' not found in manifest")

        dbase = self.manifest[base]
        result = {}

        for attr_path, filename in dbase.items():
            # Lazy-load safetensors file
            if filename not in self.safetensors:
                filepath = os.path.join(self.path, filename)
                self.safetensors[filename] = SafeTensorMLXReader(filepath)

            reader = self.safetensors[filename]
            tensor_name = base + attr_path
            result[attr_path] = reader.get_tensor(tensor_name)

        return result

    def load_param_to_device(self, name: str) -> mx.array:
        """
        Load a single parameter from disk to device.

        Args:
            name: Full parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")

        Returns:
            MLX array
        """
        # Find which base this parameter belongs to
        import re
        match = re.search(r"(model\.layers\.\d+\.mlp\.experts\.\d+\.)", name)
        if not match:
            match = re.search(r"(model\.layers\.\d+\.)", name)

        if not match:
            raise KeyError(f"Parameter '{name}' does not match expected pattern")

        base = match.group(1)
        attr_path = name.replace(base, "")

        if base not in self.manifest:
            raise KeyError(f"Base '{base}' not found in manifest")

        if attr_path not in self.manifest[base]:
            raise KeyError(f"Parameter '{attr_path}' not found in base '{base}'")

        filename = self.manifest[base][attr_path]

        # Lazy-load safetensors file
        if filename not in self.safetensors:
            filepath = os.path.join(self.path, filename)
            self.safetensors[filename] = SafeTensorMLXReader(filepath)

        reader = self.safetensors[filename]
        return reader.get_tensor(name)

    def tensor_metadata(self, name: str) -> Dict[str, Any]:
        """Return safetensor metadata without reading the tensor payload."""
        import re

        match = re.search(r"(model\.layers\.\d+\.mlp\.experts\.\d+\.)", name)
        if not match:
            match = re.search(r"(model\.layers\.\d+\.)", name)
        if not match:
            raise KeyError(f"Parameter {name!r} does not match an indexed layer")

        base = match.group(1)
        attr_path = name.replace(base, "")
        if base not in self.manifest or attr_path not in self.manifest[base]:
            raise KeyError(f"Parameter {name!r} is not present in the manifest")

        filename = self.manifest[base][attr_path]
        if filename not in self.safetensors:
            filepath = os.path.join(self.path, filename)
            self.safetensors[filename] = SafeTensorMLXReader(filepath)
        info = self.safetensors[filename].header[name]
        dtype_map = {
            "F32": "float32",
            "F16": "float16",
            "BF16": "bfloat16",
            "I32": "int32",
            "I8": "int8",
        }
        start, end = info["data_offsets"]
        return {
            "shape": tuple(info["shape"]),
            "dtype": dtype_map[info["dtype"]],
            "nbytes": int(end - start),
        }

    def preload_layer_safetensors(self, base: str):
        """
        Preload safetensors files for a specific layer.

        Args:
            base: Base layer path (e.g., "model.layers.0.")
        """
        for base1 in self.manifest.keys():
            if base1.startswith(base):
                for attr_path, filename in self.manifest[base1].items():
                    if filename not in self.safetensors:
                        filepath = os.path.join(self.path, filename)
                        self.safetensors[filename] = SafeTensorMLXReader(filepath)

    def offload_dict_to_cpu(self, base: str):
        """
        Offload a layer's weights to CPU memory.

        Args:
            base: Base path
        """
        weights = self.load_dict_from_disk(base)
        # Convert to numpy for CPU storage
        cpu_dict = {}
        for k, v in weights.items():
            cpu_dict[k] = np.array(v)
        self.offloaded_map[base] = cpu_dict

    def get_offloaded_dict_to_device(self, base: str) -> Optional[Dict[str, mx.array]]:
        """
        Retrieve cached weights and move to device.

        Args:
            base: Base path

        Returns:
            Dictionary of MLX arrays or None
        """
        if base not in self.offloaded_map:
            return None

        cpu_dict = self.offloaded_map[base]
        result = {}
        for k, np_array in cpu_dict.items():
            result[k] = mx.array(np_array)

        return result


class MLXGemma3Loader(MLXMoEWeightsLoader):
    """
    MLX-compatible loader for Gemma3 models.
    Similar to MLXMoEWeightsLoader but with Gemma3-specific path patterns.
    """

    def __init__(self, path: str, device: Optional[Any] = None):
        """
        Initialize Gemma3 loader.

        Args:
            path: Path to model directory
            device: MLX device
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required")

        self.path = path
        self.device = device if device is not None else mx.default_device()

        index_path = os.path.join(path, 'model.safetensors.index.json')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")

        with open(index_path) as f:
            indexes = json.load(f)

        self.manifest = {}
        self.safetensors = {}

        # Parse weight map for Gemma3 (language model only, vision deferred)
        import re
        for manifest_name, filename in indexes["weight_map"].items():
            match1 = re.search(
                r"(vision_TEMP.model\.layers\.\d+\.mlp\.experts\.\d+\.)",
                manifest_name
            )
            match2 = re.search(
                r"(language_model.model\.layers\.\d+\.)",
                manifest_name
            )

            if match1 or match2:
                base = match1.group(1) if match1 else match2.group(1)
                if base not in self.manifest:
                    self.manifest[base] = {}
                attr_path = manifest_name.replace(base, "")
                self.manifest[base][attr_path] = filename

        self.offloaded_map = {}
