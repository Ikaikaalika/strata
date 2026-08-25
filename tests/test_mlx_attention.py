import unittest

import numpy as np

try:
    import mlx.core as mx
    from ollm.backends.mlx_ops import (
        online_chunked_grouped_attention_mx,
        online_chunked_grouped_attention_rope_no_mask_mx,
    )

    MX_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    MX_AVAILABLE = False


def grouped_attention_reference(q, k, v, query_positions=None):
    _, query_heads, _, head_dim = q.shape
    kv_heads = k.shape[1]
    group_size = query_heads // kv_heads
    reference = np.zeros_like(q, dtype=np.float32)
    scale = 1.0 / np.sqrt(head_dim)

    for query_head in range(query_heads):
        kv_head = query_head // group_size
        scores = np.einsum(
            "bqd,bkd->bqk", q[:, query_head], k[:, kv_head]
        ) * scale
        if query_positions is not None:
            key_positions = np.arange(k.shape[2])
            allowed = key_positions[None, None, :] <= query_positions[:, :, None]
            scores = np.where(allowed, scores, -np.inf)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        reference[:, query_head] = np.einsum(
            "bqk,bkd->bqd", weights, v[:, kv_head]
        )

    return reference


@unittest.skipUnless(MX_AVAILABLE, "MLX is required")
class MLXAttentionParityTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    @staticmethod
    def _mx_to_numpy(arr):
        mx.eval(arr)
        return np.array(arr)

    def test_unmasked_compatibility_kernel_matches_numpy_reference(self):
        q = self.rng.standard_normal((1, 4, 7, 6), dtype=np.float32)
        k = self.rng.standard_normal((1, 2, 9, 6), dtype=np.float32)
        v = self.rng.standard_normal((1, 2, 9, 6), dtype=np.float32)

        output = online_chunked_grouped_attention_rope_no_mask_mx(
            mx.array(q),
            mx.array(k),
            mx.array(v),
            q_block_size=3,
            k_block_size=4,
        )

        reference = grouped_attention_reference(q, k, v)
        self.assertTrue(
            np.allclose(reference, self._mx_to_numpy(output), atol=1e-5, rtol=1e-5)
        )

    def test_causal_prefill_matches_numpy_reference_across_chunks(self):
        q = self.rng.standard_normal((2, 4, 7, 6), dtype=np.float32)
        k = self.rng.standard_normal((2, 2, 7, 6), dtype=np.float32)
        v = self.rng.standard_normal((2, 2, 7, 6), dtype=np.float32)
        positions = np.broadcast_to(np.arange(7), (2, 7))

        output = online_chunked_grouped_attention_mx(
            mx.array(q),
            mx.array(k),
            mx.array(v),
            query_positions=mx.array(positions),
            causal=True,
            q_block_size=3,
            k_block_size=2,
        )

        reference = grouped_attention_reference(q, k, v, positions)
        self.assertTrue(
            np.allclose(reference, self._mx_to_numpy(output), atol=1e-5, rtol=1e-5)
        )

    def test_cached_decode_query_uses_absolute_position(self):
        q = self.rng.standard_normal((1, 4, 1, 6), dtype=np.float32)
        k = self.rng.standard_normal((1, 2, 6, 6), dtype=np.float32)
        v = self.rng.standard_normal((1, 2, 6, 6), dtype=np.float32)
        positions = np.array([[5]], dtype=np.int32)

        output = online_chunked_grouped_attention_mx(
            mx.array(q),
            mx.array(k),
            mx.array(v),
            query_positions=mx.array(positions),
            causal=True,
            q_block_size=1,
            k_block_size=2,
        )

        reference = grouped_attention_reference(q, k, v, positions)
        self.assertTrue(
            np.allclose(reference, self._mx_to_numpy(output), atol=1e-5, rtol=1e-5)
        )

    def test_future_values_do_not_change_earlier_outputs(self):
        q = self.rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
        k = self.rng.standard_normal((1, 1, 5, 4), dtype=np.float32)
        v = self.rng.standard_normal((1, 1, 5, 4), dtype=np.float32)
        changed_v = v.copy()
        changed_v[:, :, 3:] += 1000.0

        original = online_chunked_grouped_attention_mx(
            mx.array(q), mx.array(k), mx.array(v), k_block_size=2
        )
        changed = online_chunked_grouped_attention_mx(
            mx.array(q), mx.array(k), mx.array(changed_v), k_block_size=2
        )

        original_np = self._mx_to_numpy(original)
        changed_np = self._mx_to_numpy(changed)
        self.assertTrue(np.allclose(original_np[:, :, :3], changed_np[:, :, :3]))
        self.assertFalse(np.allclose(original_np[:, :, 3:], changed_np[:, :, 3:]))

    def test_attention_rejects_invalid_grouped_head_shape(self):
        q = mx.ones((1, 3, 2, 4))
        k = mx.ones((1, 2, 2, 4))
        v = mx.ones((1, 2, 2, 4))
        with self.assertRaisesRegex(ValueError, "divisible"):
            online_chunked_grouped_attention_mx(q, k, v)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
