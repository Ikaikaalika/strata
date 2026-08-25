"""Shared generation helpers for Strata's custom MLX model adapters."""
from __future__ import annotations

from typing import Any, Optional


def greedy_generate_mx(
    model: Any,
    input_ids: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    past_key_values: Optional[Any] = None,
):
    """Run one prompt prefill followed by one-token cached decode calls."""
    import mlx.core as mx

    from .mlx_kvcache import MLXKVCache

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [batch, sequence] with sequence > 0")
    if max_new_tokens == 0:
        return input_ids

    cache = past_key_values
    if cache is None:
        cache = MLXKVCache(model.config, cache_dir=None)

    generated_ids = input_ids
    model_input = input_ids

    for _ in range(max_new_tokens):
        logits = model(
            model_input,
            past_key_values=cache,
            use_cache=True,
        )
        next_token_logits = logits[:, -1, :] / temperature
        next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)
        generated_ids = mx.concatenate([generated_ids, next_token], axis=1)

        # Only the sampled token is new after prompt prefill. Replaying the
        # growing sequence would duplicate keys and values in every cache layer.
        model_input = next_token

    return generated_ids
