import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import mlx.core as mx
    from ollm.mlx_kvcache import MLXKVCache

    MX_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    MX_AVAILABLE = False


@unittest.skipUnless(MX_AVAILABLE, "MLX is required")
class MLXKVCacheTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(num_hidden_layers=2)

    @staticmethod
    def _states(start, length):
        values = np.arange(start, start + length * 8, dtype=np.float32)
        values = values.reshape(1, 2, length, 4)
        return mx.array(values), mx.array(values + 100)

    def test_update_appends_values_and_tracks_each_layer_length(self):
        cache = MLXKVCache(self.config, cache_dir=None)
        key_0, value_0 = self._states(0, 3)
        key_1, value_1 = self._states(24, 1)

        cache.update(
            key_0,
            value_0,
            0,
            {"cache_position": mx.arange(0, 3)},
        )
        keys, values = cache.update(
            key_1,
            value_1,
            0,
            {"cache_position": mx.array([3])},
        )

        self.assertEqual(cache.get_seq_length(0), 4)
        self.assertEqual(cache.get_seq_length(1), 0)
        self.assertEqual(keys.shape, (1, 2, 4, 4))
        self.assertTrue(np.array_equal(np.array(keys[:, :, :3]), np.array(key_0)))
        self.assertTrue(np.array_equal(np.array(values[:, :, 3:]), np.array(value_1)))

    def test_update_rejects_replayed_cache_positions(self):
        cache = MLXKVCache(self.config, cache_dir=None)
        keys, values = self._states(0, 2)
        cache.update(
            keys,
            values,
            0,
            {"cache_position": mx.array([0, 1])},
        )

        with self.assertRaisesRegex(ValueError, "append exactly"):
            cache.update(
                keys,
                values,
                0,
                {"cache_position": mx.array([0, 1])},
            )

    def test_disk_round_trip_preserves_values_and_length(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MLXKVCache(self.config, cache_dir=cache_dir)
            keys, values = self._states(0, 2)
            cache.update(keys, values, 1, {"cache_position": mx.array([0, 1])})
            cache.save_to_disk(1)
            cache_path = Path(cache_dir) / "kvcache_layer_1.npz"

            self.assertEqual(cache.get_seq_length(1), 2)
            self.assertNotIn(1, cache._cache)
            self.assertTrue(cache_path.exists())
            restored = cache[1]
            self.assertTrue(np.array_equal(np.array(restored["key"]), np.array(keys)))
            self.assertTrue(np.array_equal(np.array(restored["value"]), np.array(values)))
            self.assertEqual(cache.get_seq_length(1), 2)
            cache.clear()
            self.assertFalse(cache_path.exists())

    def test_memory_only_cache_requires_explicit_disk_destination(self):
        cache = MLXKVCache(self.config, cache_dir=None)
        keys, values = self._states(0, 1)
        cache.update(keys, values, 0)
        with self.assertRaisesRegex(ValueError, "filename is required"):
            cache.save_to_disk(0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
