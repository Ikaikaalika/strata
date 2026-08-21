"""MLX implementations of core kernels."""
from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency
    import mlx.core as mx
except ModuleNotFoundError:  # pragma: no cover
    mx = None


def require_mx():  # pragma: no cover - simple guard
    if mx is None:
        raise RuntimeError("MLX is not available; install mlx to use MLX kernels")


def _normalize_query_positions(query_positions, batch_size: int, query_length: int):
    """Return query positions as ``[batch, query_length]`` int32 values."""
    positions = mx.array(query_positions, dtype=mx.int32)
    if positions.ndim == 1:
        positions = mx.expand_dims(positions, axis=0)
    if positions.shape == (1, query_length) and batch_size > 1:
        positions = mx.broadcast_to(positions, (batch_size, query_length))
    if positions.shape != (batch_size, query_length):
        raise ValueError(
            "query_positions must have shape [query_length] or "
            f"[batch, query_length], got {positions.shape}"
        )
    return positions


def online_chunked_grouped_attention_mx(
    q,
    k,
    v,
    query_positions: Optional[object] = None,
    causal: bool = True,
    q_block_size: int = 32768,
    k_block_size: int = 1024,
    eps: float = 1e-12,
):
    """Compute numerically stable, chunked grouped-query attention.

    Query positions are absolute positions in the key/value sequence. When they
    are omitted, queries are assumed to be the newest ``Lq`` entries in a key
    sequence of length ``Lk``. That bottom-right alignment makes the same
    kernel correct for both full-sequence prefill and cached decode.
    """
    require_mx()

    if q_block_size <= 0 or k_block_size <= 0:
        raise ValueError("attention block sizes must be positive")

    B, Hq, Lq, D = q.shape
    _, Hkv, Lk, Dk = k.shape
    if D != Dk:
        raise ValueError("q and k must share last-dim size")
    if k.shape[0] != B or v.shape[0] != B:
        raise ValueError("q, k, and v must share batch size")
    if v.shape[1] != Hkv or v.shape[2] != Lk:
        raise ValueError("k and v must share head and sequence dimensions")
    if v.shape[3] != D:
        raise ValueError("q, k, and v must share last-dim size")
    if Hkv <= 0 or Hq % Hkv != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if causal and Lk < Lq:
        raise ValueError("causal attention requires key length >= query length")

    if causal:
        if query_positions is None:
            query_positions = mx.arange(Lk - Lq, Lk, dtype=mx.int32)
        query_positions = _normalize_query_positions(query_positions, B, Lq)

    # Determine mapping of query heads to key-value heads
    group_size = Hq // Hkv

    dtype = q.dtype
    scale = 1.0 / (D ** 0.5)
    group_outputs = []

    for hkv_idx in range(Hkv):
        k_h = k[:, hkv_idx].astype(mx.float32)
        v_h = v[:, hkv_idx].astype(mx.float32)
        head_start = hkv_idx * group_size
        head_end = head_start + group_size
        q_sub = q[:, head_start:head_end].astype(mx.float32)
        Hq_g = q_sub.shape[1]
        query_outputs = []

        for q_start in range(0, Lq, q_block_size):
            q_end = min(Lq, q_start + q_block_size)
            Bq = q_end - q_start

            q_block = q_sub[:, :, q_start:q_end, :]

            neg_inf = float("-inf")
            m = mx.full((B, Hq_g, Bq), neg_inf, dtype=mx.float32)
            s = mx.zeros((B, Hq_g, Bq), dtype=mx.float32)
            wv = mx.zeros((B, Hq_g, Bq, D), dtype=mx.float32)

            for k_start in range(0, Lk, k_block_size):
                k_end = min(Lk, k_start + k_block_size)
                k_block = k_h[:, k_start:k_end, :]
                v_block = v_h[:, k_start:k_end, :]

                scores = mx.einsum("bhqd,bkd->bhqk", q_block, k_block) * scale
                if causal:
                    q_positions = query_positions[:, q_start:q_end]
                    k_positions = mx.arange(k_start, k_end, dtype=mx.int32)
                    allowed = k_positions[None, None, None, :] <= q_positions[:, None, :, None]
                    scores = mx.where(allowed, scores, neg_inf)
                    local_valid = mx.any(allowed, axis=-1)
                else:
                    allowed = None
                    local_valid = mx.ones((B, 1, Bq), dtype=mx.bool_)

                raw_local_max = mx.max(scores, axis=-1)
                local_max = mx.where(local_valid, raw_local_max, mx.zeros_like(raw_local_max))
                exp_scores = mx.exp(scores - local_max[..., None])
                if allowed is not None:
                    exp_scores = mx.where(allowed, exp_scores, mx.zeros_like(exp_scores))
                sum_exp = mx.sum(exp_scores, axis=-1)
                weighted_v_chunk = mx.einsum("bhqk,bkd->bhqd", exp_scores, v_block)

                prev_m = m
                prev_s = s
                prev_wv = wv
                prev_valid = prev_s > 0
                local_valid = mx.broadcast_to(local_valid, prev_s.shape)
                both_valid = prev_valid & local_valid
                new_m = mx.where(
                    both_valid,
                    mx.maximum(prev_m, local_max),
                    mx.where(prev_valid, prev_m, local_max),
                )
                alpha = mx.where(prev_valid, mx.exp(prev_m - new_m), mx.zeros_like(prev_m))
                beta = mx.where(local_valid, mx.exp(local_max - new_m), mx.zeros_like(local_max))

                s = (alpha * prev_s + beta * sum_exp).astype(mx.float32)
                wv = (
                    alpha[..., None] * prev_wv
                    + beta[..., None] * weighted_v_chunk
                ).astype(mx.float32)
                m = new_m

            denom = mx.maximum(s[..., None], eps)
            out_block = (wv / denom).astype(dtype)
            query_outputs.append(out_block)

        group_outputs.append(mx.concatenate(query_outputs, axis=2))

    return mx.concatenate(group_outputs, axis=1)


def online_chunked_grouped_attention_rope_no_mask_mx(
    q,
    k,
    v,
    position_ids: Optional[object] = None,
    q_block_size: int = 32768,
    k_block_size: int = 1024,
    eps: float = 1e-12,
):
    """Backward-compatible unmasked attention entry point.

    ``position_ids`` remains accepted because older callers supplied it even
    though this function never used it.
    """
    del position_ids
    return online_chunked_grouped_attention_mx(
        q,
        k,
        v,
        causal=False,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        eps=eps,
    )
