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


def online_chunked_grouped_attention_rope_no_mask_mx(
    q,
    k,
    v,
    position_ids: Optional[object] = None,
    q_block_size: int = 32768,
    k_block_size: int = 1024,
    eps: float = 1e-12,
):
    """MLX variant of grouped attention kernel used during inference."""
    require_mx()

    if position_ids is not None:
        pass  # kept for API parity; masking handled upstream

    B, Hq, Lq, D = q.shape
    _, Hkv, Lk, Dk = k.shape
    if D != Dk:
        raise ValueError("q and k must share last-dim size")

    # Determine mapping of query heads to key-value heads
    group_size = (Hq + Hkv - 1) // Hkv
    head_mapping = (mx.arange(Hq, dtype=mx.int32) // group_size).tolist()
    groups = []
    for hkv_idx in range(Hkv):
        q_head_idxs = [i for i, h in enumerate(head_mapping) if h == hkv_idx]
        groups.append(q_head_idxs or None)

    dtype = q.dtype
    out = mx.zeros((B, Hq, Lq, D), dtype=dtype)
    scale = 1.0 / (D ** 0.5)

    for hkv_idx, q_head_idxs in enumerate(groups):
        if not q_head_idxs:
            continue

        k_h = k[:, hkv_idx].astype(mx.float32)
        v_h = v[:, hkv_idx].astype(mx.float32)
        q_sub = q[:, q_head_idxs].astype(mx.float32)
        Hq_g = q_sub.shape[1]

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
                local_max = mx.max(scores, axis=-1)
                exp_scores = mx.exp(scores - local_max[..., None])
                sum_exp = mx.sum(exp_scores, axis=-1)
                weighted_v_chunk = mx.einsum("bhqk,bkd->bhqd", exp_scores, v_block)

                prev_m = m
                first_mask = mx.equal(prev_m, neg_inf)
                new_m = mx.where(first_mask, local_max, mx.maximum(prev_m, local_max))
                alpha = mx.where(first_mask, mx.zeros_like(prev_m), mx.exp(prev_m - new_m))
                beta = mx.exp(local_max - new_m)

                prev_s = s
                prev_wv = wv

                s = mx.where(first_mask, sum_exp, alpha * prev_s + beta * sum_exp)
                s = s.astype(mx.float32)

                wv = mx.where(
                    first_mask[..., None],
                    weighted_v_chunk,
                    alpha[..., None] * prev_wv + beta[..., None] * weighted_v_chunk,
                )
                wv = wv.astype(mx.float32)
                m = new_m

            denom = s[..., None] + eps
            out_block = (wv / denom).astype(dtype)

            # Update output for this group of query heads using scatter
            # Create indices for scatter operation
            for local_head_idx, global_head_idx in enumerate(q_head_idxs):
                # Extract the slice we want to update
                before = out[:, :global_head_idx, :, :]
                after = out[:, global_head_idx+1:, :, :]

                # Build the middle part (the updated head)
                middle_before = out[:, global_head_idx:global_head_idx+1, :q_start, :]
                middle_updated = mx.expand_dims(out_block[:, local_head_idx, :, :], axis=1)
                middle_after = out[:, global_head_idx:global_head_idx+1, q_end:, :]
                middle = mx.concatenate([middle_before, middle_updated, middle_after], axis=2)

                # Reconstruct the full output
                parts = []
                if global_head_idx > 0:
                    parts.append(before)
                parts.append(middle)
                if global_head_idx < Hq - 1:
                    parts.append(after)
                out = mx.concatenate(parts, axis=1)

    return out
