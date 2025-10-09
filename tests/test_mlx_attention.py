import unittest

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, os.fspath(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


attention_module = _load_module(BASE_DIR / "src" / "ollm" / "attention.py", "ollm_attention")
online_chunked_grouped_attention_rope_no_mask = (
    attention_module.online_chunked_grouped_attention_rope_no_mask
)

try:
    mlx_ops_module = _load_module(
        BASE_DIR / "src" / "ollm" / "backends" / "mlx_ops.py", "ollm_mlx_ops"
    )
    import mlx.core as mx

    online_chunked_grouped_attention_rope_no_mask_mx = (
        mlx_ops_module.online_chunked_grouped_attention_rope_no_mask_mx
    )
    MX_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    MX_AVAILABLE = False


@unittest.skipUnless(MX_AVAILABLE, "MLX not installed")
class MLXAttentionParityTest(unittest.TestCase):
    def _mx_to_numpy(self, arr):
        mx.eval(arr)
        if hasattr(arr, "numpy"):
            return arr.numpy()
        if hasattr(arr, "to_numpy"):
            return arr.to_numpy()
        return np.array(arr)

    def test_attention_matches_torch(self):
        torch.manual_seed(0)
        B, Hq, Hkv, Lq, Lk, D = 1, 4, 2, 8, 8, 6
        q = torch.randn(B, Hq, Lq, D, dtype=torch.float16)
        k = torch.randn(B, Hkv, Lk, D, dtype=torch.float16)
        v = torch.randn(B, Hkv, Lk, D, dtype=torch.float16)

        torch_out = online_chunked_grouped_attention_rope_no_mask(
            q, k, v, q_block_size=4, k_block_size=4
        )

        mx_q = mx.array(q.numpy())
        mx_k = mx.array(k.numpy())
        mx_v = mx.array(v.numpy())

        mx_out = online_chunked_grouped_attention_rope_no_mask_mx(
            mx_q, mx_k, mx_v, q_block_size=4, k_block_size=4
        )

        np_out = self._mx_to_numpy(mx_out)
        torch_out_np = torch_out.cpu().numpy()

        self.assertTrue(np.allclose(torch_out_np, np_out, atol=5e-3, rtol=5e-3))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
