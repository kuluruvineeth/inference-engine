import numpy as np
import pytest

from ie.core.block_manager import BlockManager
from ie.core.layout import build_layout
from ie.core.scheduler import Scheduler
from ie.core.sequence import Sequence
from ie.kernels.reference import (
    allocate_kv_cache,
    dense_attention,
    gather_context,
    paged_attention,
    softmax,
    write_kv,
)

BS = 4
HEADS = 4
KV_HEADS = 2
DIM = 8


def rng(seed=0):
    return np.random.default_rng(seed)


def random_qkv(num_tokens, generator, heads=HEADS, kv_heads=KV_HEADS, dim=DIM):
    q = generator.standard_normal((num_tokens, heads, dim)).astype(np.float32)
    k = generator.standard_normal((num_tokens, kv_heads, dim)).astype(np.float32)
    v = generator.standard_normal((num_tokens, kv_heads, dim)).astype(np.float32)
    return q, k, v


def test_softmax_rows_sum_to_one():
    out = softmax(rng().standard_normal((3, 5)))
    assert np.allclose(out.sum(axis=-1), 1.0)


def test_causal_mask_hides_the_future():
    generator = rng(1)
    q, k, v = random_qkv(6, generator, kv_heads=HEADS)
    full = dense_attention(q, k, v, causal=True)
    truncated = dense_attention(q[:3], k[:3], v[:3], causal=True)
    assert np.allclose(full[:3], truncated, atol=1e-5)


def test_grouped_query_heads_broadcast_over_kv_heads():
    generator = rng(2)
    q, k, v = random_qkv(5, generator)
    out = dense_attention(q, k, v)
    assert out.shape == (5, HEADS, DIM)


def test_gather_context_follows_the_block_table():
    k_cache, _ = allocate_kv_cache(num_blocks=8, block_size=BS, num_kv_heads=1, head_dim=2)
    for slot in range(k_cache.shape[0]):
        k_cache[slot, 0, 0] = slot

    gathered = gather_context(k_cache, block_table=[3, 1], context_len=6, block_size=BS)
    assert gathered.shape[0] == 6
    assert list(gathered[:, 0, 0]) == [12, 13, 14, 15, 4, 5]


def test_write_kv_scatters_into_the_slots():
    k_cache, v_cache = allocate_kv_cache(num_blocks=4, block_size=BS, num_kv_heads=1, head_dim=2)
    k = np.arange(6, dtype=np.float32).reshape(3, 1, 2)
    write_kv(k, k, k_cache, v_cache, slot_mapping=[9, 0, 5])
    assert np.allclose(k_cache[9], k[0])
    assert np.allclose(k_cache[0], k[1])
    assert np.allclose(k_cache[5], k[2])


def test_write_kv_rejects_a_length_mismatch():
    k_cache, v_cache = allocate_kv_cache(num_blocks=2, block_size=BS, num_kv_heads=1, head_dim=2)
    k = np.zeros((3, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        write_kv(k, k, k_cache, v_cache, slot_mapping=[0, 1])


def test_paged_prefill_matches_dense_attention():
    generator = rng(3)
    prompt_len = 10
    q, k, v = random_qkv(prompt_len, generator)

    manager = BlockManager(num_blocks=16, block_size=BS)
    scheduler = Scheduler(manager, max_batch_tokens=64)
    seq = Sequence(list(range(prompt_len)), block_size=BS)
    scheduler.add(seq)
    batch = scheduler.schedule()
    layout = build_layout(batch, BS)

    k_cache, v_cache = allocate_kv_cache(16, BS, KV_HEADS, DIM)
    write_kv(k, v, k_cache, v_cache, layout.slot_mapping)

    paged = paged_attention(q, k_cache, v_cache, layout.block_tables,
                            layout.context_lens, layout.query_lens, BS)
    expected = dense_attention(q, k, v, causal=True)
    assert np.allclose(paged, expected, atol=1e-5)


def test_paged_decode_matches_dense_attention():
    generator = rng(4)
    prompt_len = 9
    q_all, k_all, v_all = random_qkv(prompt_len + 1, generator)

    manager = BlockManager(num_blocks=16, block_size=BS)
    scheduler = Scheduler(manager, max_batch_tokens=64)
    seq = Sequence(list(range(prompt_len)), block_size=BS, max_new_tokens=8)
    scheduler.add(seq)

    k_cache, v_cache = allocate_kv_cache(16, BS, KV_HEADS, DIM)

    prefill = scheduler.schedule()
    prefill_layout = build_layout(prefill, BS)
    write_kv(k_all[:prompt_len], v_all[:prompt_len], k_cache, v_cache, prefill_layout.slot_mapping)
    scheduler.finish_step(prefill, [123])

    decode = scheduler.schedule()
    decode_layout = build_layout(decode, BS)
    assert decode_layout.num_tokens == 1
    assert decode_layout.context_lens == [prompt_len + 1]

    write_kv(k_all[prompt_len:], v_all[prompt_len:], k_cache, v_cache, decode_layout.slot_mapping)

    paged = paged_attention(q_all[prompt_len:], k_cache, v_cache, decode_layout.block_tables,
                            decode_layout.context_lens, decode_layout.query_lens, BS)
    expected = dense_attention(q_all, k_all, v_all, causal=True)[prompt_len:]
    assert np.allclose(paged, expected, atol=1e-5)


def test_chunked_prefill_matches_dense_attention():
    generator = rng(5)
    prompt_len = 14
    q, k, v = random_qkv(prompt_len, generator)

    manager = BlockManager(num_blocks=32, block_size=BS)
    scheduler = Scheduler(manager, max_batch_tokens=6)
    seq = Sequence(list(range(prompt_len)), block_size=BS)
    scheduler.add(seq)

    k_cache, v_cache = allocate_kv_cache(32, BS, KV_HEADS, DIM)
    chunks = []

    while not seq.prompt_is_fully_computed:
        batch = scheduler.schedule()
        layout = build_layout(batch, BS)
        start = layout.positions[0]
        stop = layout.positions[-1] + 1

        write_kv(k[start:stop], v[start:stop], k_cache, v_cache, layout.slot_mapping)
        chunks.append(paged_attention(q[start:stop], k_cache, v_cache, layout.block_tables,
                                      layout.context_lens, layout.query_lens, BS))
        scheduler.finish_step(batch, [0] * len(batch))

    assert len(chunks) > 1
    stitched = np.concatenate(chunks, axis=0)
    expected = dense_attention(q, k, v, causal=True)
    assert np.allclose(stitched, expected, atol=1e-5)


def test_two_sequences_sharing_a_prefix_attend_identically():
    generator = rng(6)
    prompt = list(range(8))
    q, k, v = random_qkv(len(prompt), generator)

    manager = BlockManager(num_blocks=32, block_size=BS)
    scheduler = Scheduler(manager, max_batch_tokens=64)
    k_cache, v_cache = allocate_kv_cache(32, BS, KV_HEADS, DIM)

    first = Sequence(prompt, block_size=BS, max_new_tokens=4)
    scheduler.add(first)
    batch = scheduler.schedule()
    layout = build_layout(batch, BS)
    write_kv(k, v, k_cache, v_cache, layout.slot_mapping)
    baseline = paged_attention(q, k_cache, v_cache, layout.block_tables,
                               layout.context_lens, layout.query_lens, BS)
    scheduler.finish_step(batch, [1])

    second = Sequence(prompt, block_size=BS, max_new_tokens=4)
    scheduler.add(second)
    warm = scheduler.schedule()
    warm_layout = build_layout(warm, BS)
    assert second.num_computed > 0

    start = warm_layout.positions[0]
    write_kv(k[start:], v[start:], k_cache, v_cache, warm_layout.slot_mapping)
    reused = paged_attention(q[start:], k_cache, v_cache, warm_layout.block_tables,
                             warm_layout.context_lens, warm_layout.query_lens, BS)

    assert np.allclose(reused, baseline[start:], atol=1e-5)
