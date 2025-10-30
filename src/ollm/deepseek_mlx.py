"""
MLX-native DeepSeek implementation for Apple Silicon.
DeepSeek uses Llama-like architecture with minor differences.
"""

from typing import Optional, Tuple, Any
import time

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

# Import Llama components (DeepSeek is architecturally similar)
from .llama_mlx import (
    MLXRMSNorm,
    MLXRotaryEmbedding,
    apply_rotary_pos_emb_mlx,
    rotate_half,
    METAL_OPTIMIZED,
)
from .backends.mlx_ops import online_chunked_grouped_attention_rope_no_mask_mx

# Import Metal optimizations (DeepSeek shares Llama architecture)
try:
    from .metal_optimizations import (
        optimized_silu,
        optimized_mlp_chunked,
    )
except ImportError:
    pass

# Global loader and stats
loader = None
stats = None


class MLXDeepSeekAttention:
    """
    DeepSeek attention module (similar to Llama).
    Uses the same architecture as Llama with potential config differences.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        # Weights loaded dynamically
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.o_proj_weight = None

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        position_ids: mx.array,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass for DeepSeek attention."""
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = mx.matmul(hidden_states, self.q_proj_weight.T)
        key_states = mx.matmul(hidden_states, self.k_proj_weight.T)
        value_states = mx.matmul(hidden_states, self.v_proj_weight.T)

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
        attn_output = online_chunked_grouped_attention_rope_no_mask_mx(
            query_states,
            key_states,
            value_states,
            position_ids=position_ids,
            q_block_size=32768,
            k_block_size=(1024 if seq_len > 128 else 1000000)
        )

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = mx.matmul(attn_output, self.o_proj_weight.T)

        return attn_output


class MLXDeepSeekMLP:
    """DeepSeek MLP (same as Llama - SwiGLU)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Weights loaded dynamically
        self.gate_proj_weight = None
        self.up_proj_weight = None
        self.down_proj_weight = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with chunking for memory efficiency (Metal optimized)."""
        # Use optimized version if available
        if METAL_OPTIMIZED:
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
            # Process at once
            gate = mx.matmul(x, self.gate_proj_weight.T)
            gate = optimized_silu(gate) if METAL_OPTIMIZED else gate * mx.sigmoid(gate)
            up = mx.matmul(x, self.up_proj_weight.T)
            out = mx.matmul(gate * up, self.down_proj_weight.T)
            return out

        # Process in chunks
        chunks = []
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            x_chunk = x[:, i:end, :]

            gate = mx.matmul(x_chunk, self.gate_proj_weight.T)
            gate = optimized_silu(gate) if METAL_OPTIMIZED else gate * mx.sigmoid(gate)
            up = mx.matmul(x_chunk, self.up_proj_weight.T)
            out_chunk = mx.matmul(gate * up, self.down_proj_weight.T)

            chunks.append(out_chunk)

        return mx.concatenate(chunks, axis=1)


class MLXDeepSeekDecoderLayer:
    """DeepSeek decoder layer with dynamic weight loading."""

    def __init__(self, config: Any, layer_idx: int):
        self.layer_idx = layer_idx
        self.config = config

        # Initialize components
        self.self_attn = MLXDeepSeekAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            head_dim=config.hidden_size // config.num_attention_heads,
            layer_idx=layer_idx
        )

        self.mlp = MLXDeepSeekMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )

        self.input_layernorm = MLXRMSNorm(
            config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-6)
        )

        self.post_attention_layernorm = MLXRMSNorm(
            config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-6)
        )

    def _layer_param_manifest_names(self):
        """Get manifest names for DeepSeek layer parameters."""
        base = f"model.layers.{self.layer_idx}"
        return {
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

    def _load_layer_weights(self):
        """Load weights from disk for this layer."""
        global loader, stats

        if loader is None:
            raise RuntimeError("Loader not initialized")

        manifest_map = self._layer_param_manifest_names()

        for attr_path, manifest_name in manifest_map.items():
            try:
                t1 = time.perf_counter()
                tensor = loader.load_param_to_device(manifest_name)

                # Assign weights
                if attr_path.startswith("self_attn."):
                    param_name = attr_path.replace("self_attn.", "")
                    if param_name == "q_proj.weight":
                        self.self_attn.q_proj_weight = tensor
                    elif param_name == "k_proj.weight":
                        self.self_attn.k_proj_weight = tensor
                    elif param_name == "v_proj.weight":
                        self.self_attn.v_proj_weight = tensor
                    elif param_name == "o_proj.weight":
                        self.self_attn.o_proj_weight = tensor
                elif attr_path.startswith("mlp."):
                    param_name = attr_path.replace("mlp.", "")
                    if param_name == "gate_proj.weight":
                        self.mlp.gate_proj_weight = tensor
                    elif param_name == "up_proj.weight":
                        self.mlp.up_proj_weight = tensor
                    elif param_name == "down_proj.weight":
                        self.mlp.down_proj_weight = tensor
                elif attr_path == "input_layernorm.weight":
                    self.input_layernorm.weight = tensor
                elif attr_path == "post_attention_layernorm.weight":
                    self.post_attention_layernorm.weight = tensor

                if stats:
                    stats.set("layer_load", t1)

            except Exception as e:
                raise RuntimeError(f"Failed to load {manifest_name}: {e}")

    def _unload_layer_weights(self):
        """Unload weights to free memory."""
        self.self_attn.q_proj_weight = None
        self.self_attn.k_proj_weight = None
        self.self_attn.v_proj_weight = None
        self.self_attn.o_proj_weight = None
        self.mlp.gate_proj_weight = None
        self.mlp.up_proj_weight = None
        self.mlp.down_proj_weight = None

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        position_ids: mx.array,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass with dynamic weight loading."""
        # Load weights
        self._load_layer_weights()

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

        # Unload weights
        self._unload_layer_weights()

        return hidden_states


class MLXDeepSeekModel:
    """MLX-native DeepSeek model."""

    def __init__(self, config: Any):
        self.config = config
        self.vocab_size = config.vocab_size

        # Embeddings (set externally)
        self.embed_tokens_weight = None

        # Decoder layers
        self.layers = [
            MLXDeepSeekDecoderLayer(config, i)
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
            max_position_embeddings=getattr(config, 'max_position_embeddings', 4096),
            base=getattr(config, 'rope_theta', 10000.0)
        )

        # LM head (set externally)
        self.lm_head_weight = None

    def __call__(
        self,
        input_ids: mx.array,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Any]]:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens_weight[input_ids]

        # Prepare cache position
        if past_key_values is not None:
            past_seen_tokens = past_key_values.get_seq_length()
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

        # Process through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values


class MLXDeepSeekForCausalLM:
    """MLX-native DeepSeek for causal language modeling."""

    def __init__(self, config: Any):
        self.config = config
        self.model = MLXDeepSeekModel(config)
        self.vocab_size = config.vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> mx.array:
        """Forward pass for language modeling."""
        hidden_states, past_key_values = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        # Project to vocabulary
        logits = mx.matmul(hidden_states, self.model.lm_head_weight.T)

        return logits

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        past_key_values: Optional[Any] = None,
    ) -> mx.array:
        """Simple greedy generation."""
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

            # Get next token (greedy)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)

            # Append to sequence
            input_ids = mx.concatenate([input_ids, next_token], axis=1)

        return input_ids
