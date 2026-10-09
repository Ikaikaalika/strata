import unittest
from collections import Counter
from threading import Lock
from types import SimpleNamespace

import numpy as np

try:
    import mlx.core as mx
    from lokahi import TensorTracer
    from lokahi.deepseek_mlx import MLXDeepSeekForCausalLM
    from lokahi.generation import greedy_generate_mx
    from lokahi.llama_mlx import (
        MLXLlamaForCausalLM,
        MLXLlamaDecoderLayer,
        MLXRotaryEmbedding,
        _linear,
        apply_rotary_pos_emb_mlx,
    )
    from lokahi.mlx_kvcache import MLXKVCache
    import lokahi.deepseek_mlx as deepseek_mlx
    import lokahi.llama_mlx as llama_mlx

    MX_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    MX_AVAILABLE = False


class DictLoader:
    def __init__(self, weights):
        self.weights = weights
        self.calls = Counter()
        self._lock = Lock()

    def load_param_to_device(self, name):
        with self._lock:
            self.calls[name] += 1
        return self.weights[name]

    def tensor_metadata(self, name):
        tensor = self.weights[name]
        return {
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "nbytes": tensor.nbytes,
        }


def tiny_config():
    return SimpleNamespace(
        pad_token_id=0,
        vocab_size=17,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        rope_theta=10000.0,
    )


def tiny_weights(config):
    rng = np.random.default_rng(7)

    def matrix(shape):
        return mx.array(rng.normal(0.0, 0.15, shape).astype(np.float32))

    weights = {}
    for layer_index in range(config.num_hidden_layers):
        base = f"model.layers.{layer_index}"
        weights.update(
            {
                f"{base}.self_attn.q_proj.weight": matrix((8, 8)),
                f"{base}.self_attn.k_proj.weight": matrix((4, 8)),
                f"{base}.self_attn.v_proj.weight": matrix((4, 8)),
                f"{base}.self_attn.o_proj.weight": matrix((8, 8)),
                f"{base}.mlp.gate_proj.weight": matrix((12, 8)),
                f"{base}.mlp.up_proj.weight": matrix((12, 8)),
                f"{base}.mlp.down_proj.weight": matrix((8, 12)),
                f"{base}.input_layernorm.weight": mx.ones((8,)),
                f"{base}.post_attention_layernorm.weight": mx.ones((8,)),
            }
        )
    embedding = matrix((config.vocab_size, config.hidden_size))
    lm_head = matrix((config.vocab_size, config.hidden_size))
    return weights, embedding, lm_head


@unittest.skipUnless(MX_AVAILABLE, "MLX is required")
class MLXGenerationCorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.previous_loader = llama_mlx.loader
        self.previous_deepseek_loader = deepseek_mlx.loader

    def tearDown(self):
        llama_mlx.loader = self.previous_loader
        deepseek_mlx.loader = self.previous_deepseek_loader

    def _model(self, tracer=None, governed=False):
        config = tiny_config()
        weights, embedding, lm_head = tiny_weights(config)
        weight_loader = DictLoader(weights)
        self.last_loader = weight_loader
        llama_mlx.loader = weight_loader
        memory_budget_bytes = (
            sum(tensor.nbytes for tensor in weights.values()) if governed else None
        )
        model = MLXLlamaForCausalLM(
            config,
            tracer=tracer,
            weight_loader=weight_loader,
            memory_budget_bytes=memory_budget_bytes,
        )
        model.model.embed_tokens_weight = embedding
        model.model.lm_head_weight = lm_head
        model.model.norm.weight = mx.ones((config.hidden_size,))
        return model

    def _deepseek_model(self, tracer=None, governed=False):
        config = tiny_config()
        weights, embedding, lm_head = tiny_weights(config)
        weight_loader = DictLoader(weights)
        deepseek_mlx.loader = weight_loader
        memory_budget_bytes = (
            sum(tensor.nbytes for tensor in weights.values()) if governed else None
        )
        model = MLXDeepSeekForCausalLM(
            config,
            tracer=tracer,
            weight_loader=weight_loader,
            memory_budget_bytes=memory_budget_bytes,
        )
        model.model.embed_tokens_weight = embedding
        model.model.lm_head_weight = lm_head
        model.model.norm.weight = mx.ones((config.hidden_size,))
        return model

    def assert_cached_decode_matches_full_forward(self, model):
        token_ids = mx.array([[2, 5, 3, 11]], dtype=mx.int32)
        full_logits = model(token_ids, use_cache=False)
        cache = MLXKVCache(model.config, cache_dir=None)
        model(token_ids[:, :3], past_key_values=cache, use_cache=True)
        decode_logits = model(token_ids[:, 3:], past_key_values=cache, use_cache=True)
        mx.eval(full_logits, decode_logits)

        self.assertEqual(cache.get_seq_length(), 4)
        self.assertTrue(
            np.allclose(
                np.array(full_logits[:, -1, :]),
                np.array(decode_logits[:, -1, :]),
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_cached_decode_matches_full_causal_forward(self):
        tracer = TensorTracer()
        model = self._model(tracer=tracer)
        self.assert_cached_decode_matches_full_forward(model)
        tracer.clear()

        token_ids = mx.array([[2, 5, 3, 11]], dtype=mx.int32)
        cache = MLXKVCache(model.config, cache_dir=None)
        model(token_ids[:, :3], past_key_values=cache, use_cache=True)
        decode_logits = model(token_ids[:, 3:], past_key_values=cache, use_cache=True)
        mx.eval(decode_logits)

        phases = {
            event.phase for event in tracer.events if event.name == "attention"
        }
        self.assertEqual(phases, {"prefill", "decode"})
        tensor_events = [event for event in tracer.events if event.kind == "tensor"]
        self.assertTrue(tensor_events)
        self.assertTrue(all(event.byte_size > 0 for event in tensor_events))

    def test_quantized_linear_matches_mlx_primitive_and_expands_layer_manifest(self):
        source = mx.arange(1024, dtype=mx.float32).reshape(32, 32) / 1024.0
        packed, scales, biases = mx.quantize(source, group_size=32, bits=4)
        inputs = mx.arange(64, dtype=mx.float32).reshape(2, 32) / 64.0

        actual = _linear(
            inputs,
            packed,
            scales,
            biases,
            group_size=32,
            bits=4,
            mode="affine",
        )
        expected = mx.quantized_matmul(
            inputs,
            packed,
            scales,
            biases,
            group_size=32,
            bits=4,
            mode="affine",
        )
        mx.eval(actual, expected)

        self.assertTrue(np.array_equal(np.array(actual), np.array(expected)))
        config = tiny_config()
        config.quantization = {"group_size": 32, "bits": 4}
        manifest = MLXLlamaDecoderLayer(config, 0)._layer_param_manifest_names()
        self.assertIn("self_attn.q_proj.scales", manifest)
        self.assertIn("self_attn.q_proj.biases", manifest)
        self.assertIn("mlp.down_proj.scales", manifest)

    def test_deepseek_cached_decode_matches_full_causal_forward(self):
        self.assert_cached_decode_matches_full_forward(self._deepseek_model())

    def test_governed_deepseek_cached_decode_matches_full_forward(self):
        model = self._deepseek_model(governed=True)
        try:
            self.assert_cached_decode_matches_full_forward(model)
        finally:
            model.close()

    def test_memory_governor_retains_hot_layers_across_forwards(self):
        token_ids = mx.array([[2, 5, 3, 11]], dtype=mx.int32)
        reference = self._model()(token_ids, use_cache=False)
        mx.eval(reference)

        tracer = TensorTracer()
        governed = self._model(tracer=tracer, governed=True)
        loader = self.last_loader
        try:
            first = governed(token_ids, use_cache=False)
            second = governed(token_ids, use_cache=False)
            mx.eval(first, second)
        finally:
            governed.close()

        self.assertTrue(np.allclose(np.array(reference), np.array(first), atol=1e-5))
        self.assertTrue(np.array_equal(np.array(first), np.array(second)))
        self.assertEqual(set(loader.calls.values()), {1})
        scheduler = governed.model.layer_pipeline.scheduler.snapshot()
        self.assertEqual(scheduler.bytes_loaded, sum(t.nbytes for t in loader.weights.values()))
        self.assertGreaterEqual(scheduler.resident_hits, len(governed.model.layers))
        statuses = {
            event.cache_status
            for event in tracer.events
            if event.name == "weight_acquire"
        }
        self.assertIn("resident_hit", statuses)

    def test_cache_is_not_mutated_when_use_cache_is_false(self):
        model = self._model()
        cache = MLXKVCache(model.config, cache_dir=None)
        model(
            mx.array([[2, 5, 3]], dtype=mx.int32),
            past_key_values=cache,
            use_cache=False,
        )
        self.assertEqual(cache.get_seq_length(), 0)

    def test_rope_matches_numpy_half_rotation_reference(self):
        rng = np.random.default_rng(11)
        q = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
        k = rng.normal(size=(1, 1, 3, 4)).astype(np.float32)
        positions = mx.array([[0, 2, 5]], dtype=mx.int32)
        rotary = MLXRotaryEmbedding(dim=4, max_position_embeddings=16)
        cos, sin = rotary(mx.array(q), positions)

        rotated_q, rotated_k = apply_rotary_pos_emb_mlx(
            mx.array(q), mx.array(k), cos, sin
        )
        cos_np = np.array(cos)[:, None, :, :]
        sin_np = np.array(sin)[:, None, :, :]

        def rotate_half(values):
            first, second = np.split(values, 2, axis=-1)
            return np.concatenate([-second, first], axis=-1)

        q_reference = q * cos_np + rotate_half(q) * sin_np
        k_reference = k * cos_np + rotate_half(k) * sin_np
        self.assertTrue(np.allclose(np.array(rotated_q), q_reference, atol=1e-6))
        self.assertTrue(np.allclose(np.array(rotated_k), k_reference, atol=1e-6))

    def test_generation_prefills_once_then_decodes_one_token_at_a_time(self):
        class RecordingModel:
            def __init__(self):
                self.config = SimpleNamespace(num_hidden_layers=1)
                self.input_lengths = []
                self.cache_lengths = []

            def __call__(self, input_ids, past_key_values=None, use_cache=False):
                self.input_lengths.append(input_ids.shape[1])
                self.cache_lengths.append(past_key_values.get_seq_length())
                batch_size, sequence_length = input_ids.shape
                states = mx.broadcast_to(
                    input_ids[:, None, :, None].astype(mx.float32),
                    (batch_size, 1, sequence_length, 2),
                )
                start = past_key_values.get_seq_length()
                positions = mx.arange(start, start + sequence_length)
                past_key_values.update(
                    states,
                    states,
                    0,
                    {"cache_position": positions},
                )
                token_scores = mx.array([0.0, 0.0, 0.0, 1.0, 0.0])
                return mx.broadcast_to(
                    token_scores,
                    (batch_size, sequence_length, token_scores.shape[0]),
                )

        model = RecordingModel()
        prompt = mx.array([[1, 2, 4, 2]], dtype=mx.int32)
        generated = greedy_generate_mx(
            model,
            prompt,
            max_new_tokens=3,
            temperature=1.0,
        )

        self.assertEqual(model.input_lengths, [4, 1, 1])
        self.assertEqual(model.cache_lengths, [0, 4, 5])
        self.assertEqual(generated.shape, (1, 7))
        self.assertTrue(np.array_equal(np.array(generated[:, -3:]), [[3, 3, 3]]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
