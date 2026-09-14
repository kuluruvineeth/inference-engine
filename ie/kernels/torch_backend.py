from __future__ import annotations

import torch


def allocate_kv_cache(num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int,
                      device: torch.device, dtype: torch.dtype = torch.float16):
    shape = (num_blocks * block_size, num_kv_heads, head_dim)
    return (torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype))


def write_kv(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
             slots: torch.Tensor) -> None:
    k_cache.index_copy_(0, slots, k.to(k_cache.dtype))
    v_cache.index_copy_(0, slots, v.to(v_cache.dtype))


def gather_context(cache: torch.Tensor, block_table: torch.Tensor, context_len: int,
                   block_size: int) -> torch.Tensor:
    needed = (context_len + block_size - 1) // block_size
    offsets = torch.arange(block_size, device=cache.device)
    slots = (block_table[:needed, None] * block_size + offsets[None, :]).reshape(-1)
    return cache.index_select(0, slots[:context_len])


def repeat_kv_heads(x: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_kv_heads = x.shape[1]
    if num_query_heads == num_kv_heads:
        return x
    return x.repeat_interleave(num_query_heads // num_kv_heads, dim=1)


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    block_tables: torch.Tensor, context_lens: list[int],
                    query_lens: list[int], block_size: int) -> torch.Tensor:
    num_heads = q.shape[1]
    outputs = []
    start = 0

    for index, (context_len, query_len) in enumerate(zip(context_lens, query_lens)):
        q_slice = q[start : start + query_len]
        k = repeat_kv_heads(gather_context(k_cache, block_tables[index], context_len, block_size),
                            num_heads)
        v = repeat_kv_heads(gather_context(v_cache, block_tables[index], context_len, block_size),
                            num_heads)

        attended = torch.nn.functional.scaled_dot_product_attention(
            q_slice.transpose(0, 1).unsqueeze(0).float(),
            k.transpose(0, 1).unsqueeze(0).float(),
            v.transpose(0, 1).unsqueeze(0).float(),
            attn_mask=_causal_mask(query_len, context_len, q.device),
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
        start += query_len

    return torch.cat(outputs, dim=0)


def _causal_mask(query_len: int, context_len: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(query_len, device=device)[:, None] + (context_len - query_len)
    cols = torch.arange(context_len, device=device)[None, :]
    return (cols <= rows).unsqueeze(0)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * scale).to(x.dtype) * weight


def apply_rope(x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor, halved: bool = False) -> torch.Tensor:
    c = cos.index_select(0, positions).unsqueeze(1)
    s = sin.index_select(0, positions).unsqueeze(1)
    out = torch.empty_like(x)
    if halved:
        half = x.shape[-1] // 2
        left, right = x[..., :half], x[..., half:]
        out[..., :half] = left * c - right * s
        out[..., half:] = right * c + left * s
    else:
        even, odd = x[..., 0::2], x[..., 1::2]
        out[..., 0::2] = even * c - odd * s
        out[..., 1::2] = even * s + odd * c
    return out


def swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
           down: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down
