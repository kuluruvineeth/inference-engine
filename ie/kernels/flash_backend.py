from __future__ import annotations

import torch

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    AVAILABLE = True
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    AVAILABLE = False


def paged_cache_view(cache: torch.Tensor, block_size: int) -> torch.Tensor:
    total_slots, num_kv_heads, head_dim = cache.shape
    return cache.view(total_slots // block_size, block_size, num_kv_heads, head_dim)


def decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
           block_tables: torch.Tensor, context_lens: torch.Tensor,
           block_size: int, scale: float) -> torch.Tensor:
    out = flash_attn_with_kvcache(
        q.unsqueeze(1),
        paged_cache_view(k_cache, block_size),
        paged_cache_view(v_cache, block_size),
        cache_seqlens=context_lens,
        block_table=block_tables.to(torch.int32),
        softmax_scale=scale,
        causal=True,
    )
    return out.squeeze(1)


def prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            k_cache: torch.Tensor, v_cache: torch.Tensor,
            cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int, max_seqlen_k: int, block_tables: torch.Tensor | None,
            block_size: int, scale: float) -> torch.Tensor:
    if block_tables is None:
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=scale, causal=True,
        )
    return flash_attn_varlen_func(
        q,
        paged_cache_view(k_cache, block_size),
        paged_cache_view(v_cache, block_size),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_scale=scale, causal=True,
        block_table=block_tables,
    )
