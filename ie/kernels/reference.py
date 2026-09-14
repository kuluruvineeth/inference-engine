from __future__ import annotations

import numpy as np


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = scores - scores.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def repeat_kv_heads(x: np.ndarray, num_query_heads: int) -> np.ndarray:
    num_kv_heads = x.shape[1]
    if num_query_heads == num_kv_heads:
        return x
    if num_query_heads % num_kv_heads:
        raise ValueError(f"{num_query_heads} query heads do not divide into {num_kv_heads} kv heads")
    return np.repeat(x, num_query_heads // num_kv_heads, axis=1)


def dense_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    causal: bool = True, query_offset: int = 0) -> np.ndarray:
    query_len, num_heads, head_dim = q.shape
    k = repeat_kv_heads(k, num_heads)
    v = repeat_kv_heads(v, num_heads)
    context_len = k.shape[0]

    scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(head_dim)

    if causal:
        rows = np.arange(query_len)[:, None] + query_offset
        cols = np.arange(context_len)[None, :]
        scores = np.where(cols <= rows, scores, -np.inf)

    weights = softmax(scores, axis=-1)
    return np.einsum("hqk,khd->qhd", weights, v)


def allocate_kv_cache(num_blocks: int, block_size: int, num_kv_heads: int,
                      head_dim: int, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    shape = (num_blocks * block_size, num_kv_heads, head_dim)
    return np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype)


def write_kv(k: np.ndarray, v: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
             slot_mapping: list[int]) -> None:
    if len(slot_mapping) != k.shape[0]:
        raise ValueError(f"{len(slot_mapping)} slots for {k.shape[0]} tokens")
    slots = np.asarray(slot_mapping, dtype=np.int64)
    k_cache[slots] = k
    v_cache[slots] = v


def gather_context(cache: np.ndarray, block_table: list[int], context_len: int,
                   block_size: int) -> np.ndarray:
    needed = (context_len + block_size - 1) // block_size
    if needed > len(block_table):
        raise IndexError(f"context of {context_len} needs {needed} blocks, table has {len(block_table)}")
    slots = np.concatenate([
        np.arange(block_table[i] * block_size, (block_table[i] + 1) * block_size)
        for i in range(needed)
    ])
    return cache[slots[:context_len]]


def paged_attention(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
                    block_tables: list[list[int]], context_lens: list[int],
                    query_lens: list[int], block_size: int) -> np.ndarray:
    outputs = []
    start = 0

    for block_table, context_len, query_len in zip(block_tables, context_lens, query_lens):
        q_slice = q[start : start + query_len]
        k = gather_context(k_cache, block_table, context_len, block_size)
        v = gather_context(v_cache, block_table, context_len, block_size)
        outputs.append(dense_attention(q_slice, k, v, causal=True,
                                       query_offset=context_len - query_len))
        start += query_len

    return np.concatenate(outputs, axis=0)
