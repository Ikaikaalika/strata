"""
MLX-native Llama implementation for Apple Silicon.
Provides efficient layer-by-layer loading and chunked operations for long-context inference.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple, Dict, Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

from .backends.mlx_ops import online_chunked_grouped_attention_mx
from .generation import greedy_generate_mx

# Import Metal optimizations
try:
    from .metal_optimizations import (
        optimized_rope,
        optimized_rmsnorm,
        optimized_silu,
        optimized_mlp_chunked,
        use_optimizations
    )
    METAL_OPTIMIZED = use_optimizations()
except ImportError:
    METAL_OPTIMIZED = False

# Global loader and stats (set by inference.py)
loader = None
stats = None


def _quantization_parameters(config: Any) -> Tuple[Optional[int], Optional[int], str]:
    quantization = getattr(config, "quantization", None) or getattr(
        config, "quantization_config", None
    )
    if not isinstance(quantization, dict):
        return None, None, "affine"
    return (
        int(quantization["group_size"]),
        int(quantization["bits"]),
        str(quantization.get("mode", "affine")),
    )


def _linear(
    x: Any,
    weight: Any,
    scales: Any,
    biases: Any,
    *,
    group_size: Optional[int],
    bits: Optional[int],
    mode: str,
) -> Any:
    if scales is None:
        return mx.matmul(x, weight.T)
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


class MLXRMSNorm:
    """RMS Normalization for MLX (Metal optimized)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Use optimized version if available
        if METAL_OPTIMIZED:
            return optimized_rmsnorm(x, self.weight, self.eps)
        else:
            # Fallback to standard implementation
            variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
            x = x * mx.rsqrt(variance + self.eps)
            return self.weight * x


class MLXRotaryEmbedding:
    """
    Rotary Position Embedding (RoPE) for MLX.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        self.inv_freq = inv_freq

    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute cos and sin for rotary embeddings.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            position_ids: Position indices [batch, seq_len]

        Returns:
            Tuple of (cos, sin) tensors
        """
        # Expand position_ids to match inv_freq
        freqs = mx.outer(
            position_ids.reshape(-1).astype(mx.float32),
            self.inv_freq
        )
        # Duplicate for complex components
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)

        # Reshape to [batch, seq_len, dim]
        batch_size, seq_len = position_ids.shape
        cos = cos.reshape(batch_size, seq_len, -1)
        sin = sin.reshape(batch_size, seq_len, -1)

        return cos, sin


def apply_rotary_pos_emb_mlx(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """
    Apply rotary position embeddings to query and key tensors (Metal optimized).

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine embeddings
        sin: Sine embeddings

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Use optimized version if available
    if METAL_OPTIMIZED:
        return optimized_rope(q, k, cos, sin)
    else:
        # Fallback to standard implementation
        cos = mx.expand_dims(cos, axis=1)
        sin = mx.expand_dims(sin, axis=1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input."""
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


class MLXLlamaAttention:
    """MLX-native Llama attention with chunked computation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        tracer: Optional[Any] = None,
        quantization: Tuple[Optional[int], Optional[int], str] = (None, None, "affine"),
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.tracer = tracer
        self.group_size, self.bits, self.quantization_mode = quantization

        # Weights will be loaded dynamically
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.o_proj_weight = None
        self.q_proj_scales = None
        self.q_proj_biases = None
        self.k_proj_scales = None
        self.k_proj_biases = None
        self.v_proj_scales = None
        self.v_proj_biases = None
        self.o_proj_scales = None
        self.o_proj_biases = None

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        position_ids: mx.array,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass for attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            cos: Cosine embeddings
            sin: Sine embeddings
            position_ids: Position IDs
            past_key_value: KV cache
            cache_position: Cache position

        Returns:
            Attention output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        past_length = (
            past_key_value.get_seq_length(self.layer_idx)
            if past_key_value is not None
            else 0
        )
        phase = (
            "decode"
            if past_length > 0
            else ("prefill" if past_key_value is not None else "forward")
        )
        started_at = time.perf_counter()

        # Project to Q, K, V
        query_states = _linear(
            hidden_states, self.q_proj_weight, self.q_proj_scales, self.q_proj_biases,
            group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
        )
        key_states = _linear(
            hidden_states, self.k_proj_weight, self.k_proj_scales, self.k_proj_biases,
            group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
        )
        value_states = _linear(
            hidden_states, self.v_proj_weight, self.v_proj_scales, self.v_proj_biases,
            group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
        )

        # Reshape for multi-head attention
        query_states = query_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        key_states = key_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        value_states = value_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Apply rotary embeddings
        query_states, key_states = apply_rotary_pos_emb_mlx(
            query_states, key_states, cos, sin
        )

        # Update KV cache
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Chunked attention
        attn_output = online_chunked_grouped_attention_mx(
            query_states,
            key_states,
            value_states,
            query_positions=position_ids,
            causal=True,
            q_block_size=32768,
            k_block_size=(1024 if seq_len > 128 else 1000000)
        )

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = _linear(
            attn_output, self.o_proj_weight, self.o_proj_scales, self.o_proj_biases,
            group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
        )

        if self.tracer is not None:
            mx.eval(attn_output)
            mx.synchronize()
            self.tracer.record_tensor(
                "query", query_states, phase=phase, layer=self.layer_idx
            )
            self.tracer.record_tensor(
                "key", key_states, phase=phase, layer=self.layer_idx
            )
            self.tracer.record_tensor(
                "value", value_states, phase=phase, layer=self.layer_idx
            )
            self.tracer.record_tensor(
                "attention_output", attn_output, phase=phase, layer=self.layer_idx
            )
            self.tracer.record_operation(
                "attention",
                (time.perf_counter() - started_at) * 1000.0,
                phase=phase,
                layer=self.layer_idx,
                active_memory_bytes=mx.get_active_memory(),
                peak_memory_bytes=mx.get_peak_memory(),
            )

        return attn_output


class MLXLlamaMLP:
    """MLX-native Llama MLP with chunked computation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quantization: Tuple[Optional[int], Optional[int], str] = (None, None, "affine"),
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Weights will be loaded dynamically
        self.gate_proj_weight = None
        self.up_proj_weight = None
        self.down_proj_weight = None
        self.gate_proj_scales = None
        self.gate_proj_biases = None
        self.up_proj_scales = None
        self.up_proj_biases = None
        self.down_proj_scales = None
        self.down_proj_biases = None
        self.group_size, self.bits, self.quantization_mode = quantization

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for MLP with chunking (Metal optimized).

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Use optimized version if available
        if METAL_OPTIMIZED and self.gate_proj_scales is None:
            return optimized_mlp_chunked(
                x,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                chunk_size=16384
            )

        # Fallback to standard implementation
        chunk_size = 16384
        batch_size, seq_len, hidden_size = x.shape

        if seq_len <= chunk_size:
            # Small enough to process at once
            gate = _linear(
                x, self.gate_proj_weight, self.gate_proj_scales, self.gate_proj_biases,
                group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
            )
            gate = optimized_silu(gate) if METAL_OPTIMIZED else gate * mx.sigmoid(gate)
            up = _linear(
                x, self.up_proj_weight, self.up_proj_scales, self.up_proj_biases,
                group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
            )
            out = _linear(
                gate * up, self.down_proj_weight, self.down_proj_scales, self.down_proj_biases,
                group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
            )
            return out

        # Process in chunks
        chunks = []
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            x_chunk = x[:, i:end, :]

            gate = _linear(
                x_chunk, self.gate_proj_weight, self.gate_proj_scales, self.gate_proj_biases,
                group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
            )
            gate = optimized_silu(gate) if METAL_OPTIMIZED else gate * mx.sigmoid(gate)
            up = _linear(
                x_chunk, self.up_proj_weight, self.up_proj_scales, self.up_proj_biases,
                group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
            )
            out_chunk = _linear(
                gate * up, self.down_proj_weight, self.down_proj_scales, self.down_proj_biases,
                group_size=self.group_size, bits=self.bits, mode=self.quantization_mode,
            )

            chunks.append(out_chunk)

        return mx.concatenate(chunks, axis=1)


class MLXLlamaDecoderLayer:
    """MLX-native Llama decoder layer with dynamic weight loading."""

    def __init__(self, config: Any, layer_idx: int, tracer: Optional[Any] = None):
        self.layer_idx = layer_idx
        self.config = config

        # Initialize components
        quantization = _quantization_parameters(config)
        self.self_attn = MLXLlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            layer_idx=layer_idx,
            tracer=tracer,
            quantization=quantization,
        )

        self.mlp = MLXLlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quantization=quantization,
        )

        self.input_layernorm = MLXRMSNorm(
            config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-6)
        )

        self.post_attention_layernorm = MLXRMSNorm(
            config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-6)
        )

    def _layer_param_manifest_names(self) -> Dict[str, str]:
        """Get manifest names for this layer's parameters."""
        base = f"model.layers.{self.layer_idx}"
        manifest = {
            "self_attn.q_proj.weight": f"{base}.self_attn.q_proj.weight",
            "self_attn.k_proj.weight": f"{base}.self_attn.k_proj.weight",
            "self_attn.v_proj.weight": f"{base}.self_attn.v_proj.weight",
            "self_attn.o_proj.weight": f"{base}.self_attn.o_proj.weight",
            "mlp.gate_proj.weight": f"{base}.mlp.gate_proj.weight",
            "mlp.up_proj.weight": f"{base}.mlp.up_proj.weight",
            "mlp.down_proj.weight": f"{base}.mlp.down_proj.weight",
            "input_layernorm.weight": f"{base}.input_layernorm.weight",
            "post_attention_layernorm.weight": f"{base}.post_attention_layernorm.weight",
        }
        if self.self_attn.group_size is not None:
            for prefix in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ):
                manifest[f"{prefix}.scales"] = f"{base}.{prefix}.scales"
                manifest[f"{prefix}.biases"] = f"{base}.{prefix}.biases"
        return manifest

    def _assign_layer_weight(self, attr_path: str, tensor: Any) -> None:
        if attr_path.startswith("self_attn."):
            param_name = attr_path.replace("self_attn.", "")
            if param_name == "q_proj.weight":
                self.self_attn.q_proj_weight = tensor
            elif param_name == "q_proj.scales":
                self.self_attn.q_proj_scales = tensor
            elif param_name == "q_proj.biases":
                self.self_attn.q_proj_biases = tensor
            elif param_name == "k_proj.weight":
                self.self_attn.k_proj_weight = tensor
            elif param_name == "k_proj.scales":
                self.self_attn.k_proj_scales = tensor
            elif param_name == "k_proj.biases":
                self.self_attn.k_proj_biases = tensor
            elif param_name == "v_proj.weight":
                self.self_attn.v_proj_weight = tensor
            elif param_name == "v_proj.scales":
                self.self_attn.v_proj_scales = tensor
            elif param_name == "v_proj.biases":
                self.self_attn.v_proj_biases = tensor
            elif param_name == "o_proj.weight":
                self.self_attn.o_proj_weight = tensor
            elif param_name == "o_proj.scales":
                self.self_attn.o_proj_scales = tensor
            elif param_name == "o_proj.biases":
                self.self_attn.o_proj_biases = tensor
        elif attr_path.startswith("mlp."):
            param_name = attr_path.replace("mlp.", "")
            if param_name == "gate_proj.weight":
                self.mlp.gate_proj_weight = tensor
            elif param_name == "gate_proj.scales":
                self.mlp.gate_proj_scales = tensor
            elif param_name == "gate_proj.biases":
                self.mlp.gate_proj_biases = tensor
            elif param_name == "up_proj.weight":
                self.mlp.up_proj_weight = tensor
            elif param_name == "up_proj.scales":
                self.mlp.up_proj_scales = tensor
            elif param_name == "up_proj.biases":
                self.mlp.up_proj_biases = tensor
            elif param_name == "down_proj.weight":
                self.mlp.down_proj_weight = tensor
            elif param_name == "down_proj.scales":
                self.mlp.down_proj_scales = tensor
            elif param_name == "down_proj.biases":
                self.mlp.down_proj_biases = tensor
        elif attr_path == "input_layernorm.weight":
            self.input_layernorm.weight = tensor
        elif attr_path == "post_attention_layernorm.weight":
            self.post_attention_layernorm.weight = tensor

    def _load_layer_weights(self, weight_bundle: Optional[Dict[str, Any]] = None):
        """Load weights from disk for this layer."""
        global loader, stats

        manifest_map = self._layer_param_manifest_names()
        if weight_bundle is not None:
            missing = set(manifest_map) - set(weight_bundle)
            if missing:
                raise KeyError(
                    f"layer {self.layer_idx} bundle is missing weights: {sorted(missing)}"
                )
            for attr_path in manifest_map:
                self._assign_layer_weight(attr_path, weight_bundle[attr_path])
            return

        if loader is None:
            raise RuntimeError("Loader not initialized")

        for attr_path, manifest_name in manifest_map.items():
            try:
                t1 = time.perf_counter()
                tensor = loader.load_param_to_device(manifest_name)
                self._assign_layer_weight(attr_path, tensor)

                if stats:
                    stats.set("layer_load", t1)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load {manifest_name} into {attr_path}: {e}"
                )

    def _unload_layer_weights(self):
        """Unload weights to free memory (MLX unified memory handles this automatically)."""
        # In MLX with unified memory, we can set weights to None
        # The memory will be reclaimed when no longer referenced
        self.self_attn.q_proj_weight = None
        self.self_attn.k_proj_weight = None
        self.self_attn.v_proj_weight = None
        self.self_attn.o_proj_weight = None
        self.self_attn.q_proj_scales = None
        self.self_attn.q_proj_biases = None
        self.self_attn.k_proj_scales = None
        self.self_attn.k_proj_biases = None
        self.self_attn.v_proj_scales = None
        self.self_attn.v_proj_biases = None
        self.self_attn.o_proj_scales = None
        self.self_attn.o_proj_biases = None
        self.mlp.gate_proj_weight = None
        self.mlp.up_proj_weight = None
        self.mlp.down_proj_weight = None
        self.mlp.gate_proj_scales = None
        self.mlp.gate_proj_biases = None
        self.mlp.up_proj_scales = None
        self.mlp.up_proj_biases = None
        self.mlp.down_proj_scales = None
        self.mlp.down_proj_biases = None

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        position_ids: mx.array,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[mx.array] = None,
        weight_bundle: Optional[Dict[str, Any]] = None,
        manage_weights: bool = True,
    ) -> mx.array:
        """
        Forward pass with dynamic weight loading/unloading.

        Args:
            hidden_states: Input tensor
            cos: Cosine embeddings
            sin: Sine embeddings
            position_ids: Position IDs
            past_key_value: KV cache
            cache_position: Cache position

        Returns:
            Output hidden states
        """
        # Load weights for this layer
        self._load_layer_weights(weight_bundle)

        # Attention block with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position
        )
        hidden_states = residual + hidden_states

        # MLP block with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if manage_weights:
            self._unload_layer_weights()

        return hidden_states


class MLXLlamaModel:
    """MLX-native Llama model."""

    def __init__(
        self,
        config: Any,
        tracer: Optional[Any] = None,
        weight_loader: Optional[Any] = None,
        memory_budget_bytes: Optional[int] = None,
        prefetch_distance: int = 1,
        prefetch_workers: int = 1,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding (loaded from HF model)
        self.embed_tokens_weight = None  # Set externally
        self.embed_tokens_scales = None
        self.embed_tokens_biases = None
        self.group_size, self.bits, self.quantization_mode = _quantization_parameters(
            config
        )

        # Decoder layers
        self.layers = [
            MLXLlamaDecoderLayer(config, i, tracer=tracer)
            for i in range(config.num_hidden_layers)
        ]

        # Final norm
        self.norm = MLXRMSNorm(
            config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-6)
        )

        # RoPE
        self.rotary_emb = MLXRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=getattr(config, 'max_position_embeddings', 2048),
            base=getattr(config, 'rope_theta', 10000.0)
        )

        # LM head (set externally)
        self.lm_head_weight = None
        self.lm_head_scales = None
        self.lm_head_biases = None
        self.layer_pipeline = None
        if memory_budget_bytes is not None:
            if weight_loader is None:
                raise ValueError("weight_loader is required with memory_budget_bytes")
            from .backends.mlx_governor import build_mlx_dense_pipeline

            self.layer_pipeline = build_mlx_dense_pipeline(
                model_id=str(getattr(config, "name_or_path", "strata-llama")),
                layers=self.layers,
                loader=weight_loader,
                budget_bytes=memory_budget_bytes,
                tracer=tracer,
                prefetch_distance=prefetch_distance,
                prefetch_workers=prefetch_workers,
            )

    def embed_tokens(self, input_ids: Any) -> Any:
        if self.embed_tokens_scales is None:
            return self.embed_tokens_weight[input_ids]
        return mx.dequantize(
            self.embed_tokens_weight[input_ids],
            self.embed_tokens_scales[input_ids],
            (
                None
                if self.embed_tokens_biases is None
                else self.embed_tokens_biases[input_ids]
            ),
            group_size=self.group_size,
            bits=self.bits,
            mode=self.quantization_mode,
        )

    def __call__(
        self,
        input_ids: mx.array,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Any]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            past_key_values: KV cache
            use_cache: Whether to use KV caching

        Returns:
            Tuple of (hidden_states, past_key_values)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        active_cache = past_key_values if use_cache else None

        # Prepare cache position
        if active_cache is not None:
            past_seen_tokens = active_cache.get_seq_length()
        else:
            past_seen_tokens = 0

        cache_position = mx.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len,
            dtype=mx.int32
        )

        # Position IDs
        position_ids = mx.expand_dims(cache_position, axis=0)

        # RoPE embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Process through decoder layers
        phase = (
            "decode"
            if past_seen_tokens > 0
            else ("prefill" if active_cache is not None else "forward")
        )
        if self.layer_pipeline is not None:
            def execute_layer(state, group, weights):
                return self.layers[group.order](
                    state,
                    cos=cos,
                    sin=sin,
                    position_ids=position_ids,
                    past_key_value=active_cache,
                    cache_position=cache_position,
                    weight_bundle=weights,
                    manage_weights=False,
                )

            def release_weights(group, weights):
                del weights
                self.layers[group.order]._unload_layer_weights()

            hidden_states = self.layer_pipeline.run(
                hidden_states,
                execute_layer,
                phase=phase,
                release_weights=release_weights,
            )
        else:
            for layer in self.layers:
                try:
                    hidden_states = layer(
                        hidden_states,
                        cos=cos,
                        sin=sin,
                        position_ids=position_ids,
                        past_key_value=active_cache,
                        cache_position=cache_position,
                        manage_weights=False,
                    )
                    # MLX is lazy. Materializing each layer is required before
                    # dropping its weights or the graph retains the whole model.
                    mx.eval(hidden_states)
                    mx.synchronize()
                finally:
                    layer._unload_layer_weights()

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, active_cache

    def close(self) -> None:
        if self.layer_pipeline is not None:
            self.layer_pipeline.close()


class MLXLlamaForCausalLM:
    """MLX-native Llama for causal language modeling."""

    def __init__(
        self,
        config: Any,
        tracer: Optional[Any] = None,
        weight_loader: Optional[Any] = None,
        memory_budget_bytes: Optional[int] = None,
        prefetch_distance: int = 1,
        prefetch_workers: int = 1,
    ):
        self.config = config
        self.model = MLXLlamaModel(
            config,
            tracer=tracer,
            weight_loader=weight_loader,
            memory_budget_bytes=memory_budget_bytes,
            prefetch_distance=prefetch_distance,
            prefetch_workers=prefetch_workers,
        )
        self.vocab_size = config.vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> mx.array:
        """
        Forward pass for language modeling.

        Args:
            input_ids: Input token IDs
            past_key_values: KV cache
            use_cache: Whether to use caching

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        hidden_states, past_key_values = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        # Project to vocabulary
        head_weight = (
            self.model.embed_tokens_weight
            if self.model.lm_head_weight is None
            else self.model.lm_head_weight
        )
        head_scales = (
            self.model.embed_tokens_scales
            if self.model.lm_head_scales is None
            else self.model.lm_head_scales
        )
        head_biases = (
            self.model.embed_tokens_biases
            if self.model.lm_head_biases is None
            else self.model.lm_head_biases
        )
        logits = _linear(
            hidden_states,
            head_weight,
            head_scales,
            head_biases,
            group_size=self.model.group_size,
            bits=self.model.bits,
            mode=self.model.quantization_mode,
        )

        return logits

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        past_key_values: Optional[Any] = None,
    ) -> mx.array:
        """
        Simple greedy generation (can be extended with sampling).

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            past_key_values: KV cache

        Returns:
            Generated token IDs
        """
        del top_p  # Sampling is intentionally greedy until nucleus sampling lands.
        return greedy_generate_mx(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            past_key_values=past_key_values,
        )

    def close(self) -> None:
        self.model.close()
