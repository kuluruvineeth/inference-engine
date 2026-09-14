import numpy as np
import pytest

from ie.core.block_manager import BlockManager
from ie.core.layout import build_layout
from ie.core.scheduler import Scheduler
from ie.core.sequence import Sequence
from ie.kernels.reference import allocate_kv_cache
from ie.model.transformer import ModelConfig, TransformerBlock
from ie.parallel.sharding import (
    Shard,
    all_gather,
    all_reduce,
    split_heads,
    split_input_dim,
    split_output_dim,
)
from ie.parallel.tensor_parallel import TensorParallelBlock, kv_heads_per_rank

CONFIG = ModelConfig(vocab_size=64, hidden_size=32, num_layers=1, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)
BS = 8


def test_shard_rejects_an_out_of_range_rank():
    with pytest.raises(ValueError):
        Shard(rank=2, world_size=2)
    with pytest.raises(ValueError):
        Shard(rank=0, world_size=0)


def test_shard_rejects_an_uneven_split():
    with pytest.raises(ValueError):
        Shard(rank=0, world_size=3).slice_of(8)


def test_column_shards_concatenate_back_to_the_whole_weight():
    weight = np.arange(24, dtype=np.float32).reshape(4, 6)
    pieces = [split_output_dim(weight, Shard(r, 3)) for r in range(3)]
    assert np.array_equal(all_gather(pieces, axis=-1), weight)


def test_row_shards_stack_back_to_the_whole_weight():
    weight = np.arange(24, dtype=np.float32).reshape(6, 4)
    pieces = [split_input_dim(weight, Shard(r, 3)) for r in range(3)]
    assert np.array_equal(np.concatenate(pieces, axis=0), weight)


def test_head_sharding_keeps_each_head_contiguous():
    num_heads, head_dim = 4, 3
    weight = np.arange(2 * num_heads * head_dim, dtype=np.float32).reshape(2, num_heads * head_dim)
    left = split_heads(weight, Shard(0, 2), num_heads, head_dim)
    right = split_heads(weight, Shard(1, 2), num_heads, head_dim)

    assert left.shape == (2, 2 * head_dim)
    assert np.array_equal(all_gather([left, right], axis=-1), weight)


def test_column_then_row_reproduces_the_unsharded_matmul():
    generator = np.random.default_rng(0)
    x = generator.standard_normal((5, 8)).astype(np.float32)
    first = generator.standard_normal((8, 12)).astype(np.float32)
    second = generator.standard_normal((12, 8)).astype(np.float32)

    whole = (x @ first) @ second
    partials = []
    for rank in range(4):
        shard = Shard(rank, 4)
        hidden = x @ split_output_dim(first, shard)
        partials.append(hidden @ split_input_dim(second, shard))

    assert np.allclose(all_reduce(partials), whole, atol=1e-4)


def run_block(block_or_parallel, world_size, num_kv_heads):
    manager = BlockManager(num_blocks=64, block_size=BS)
    scheduler = Scheduler(manager, max_batch_tokens=128)
    scheduler.add(Sequence(list(range(12)), block_size=BS))
    batch = scheduler.schedule()
    layout = build_layout(batch, BS)

    caches = [allocate_kv_cache(64, BS, num_kv_heads, CONFIG.head_dim)
              for _ in range(world_size)]
    k_caches = [pair[0] for pair in caches]
    v_caches = [pair[1] for pair in caches]

    generator = np.random.default_rng(7)
    x = generator.standard_normal((layout.num_tokens, CONFIG.hidden_size)).astype(np.float32)
    cos, sin = np.cos(np.zeros((64, CONFIG.head_dim // 2))), np.sin(np.zeros((64, CONFIG.head_dim // 2)))
    return x, layout, k_caches, v_caches, cos.astype(np.float32), sin.astype(np.float32)


@pytest.mark.parametrize("world_size", [1, 2])
def test_sharded_block_matches_the_unsharded_block(world_size):
    block = TransformerBlock(CONFIG, np.random.default_rng(3))

    x, layout, k_caches, v_caches, cos, sin = run_block(block, 1, CONFIG.num_kv_heads)
    expected = block.forward(x, layout, k_caches[0], v_caches[0], cos, sin, BS)

    parallel = TensorParallelBlock(block, world_size)
    _, layout2, k_shards, v_shards, _, _ = run_block(
        block, world_size, kv_heads_per_rank(CONFIG, world_size))
    actual = parallel.forward(x, layout2, k_shards, v_shards, cos, sin, BS)

    assert np.allclose(actual, expected, atol=1e-4)


def test_each_rank_holds_its_fraction_of_the_weights():
    block = TransformerBlock(CONFIG, np.random.default_rng(4))
    parallel = TensorParallelBlock(block, world_size=2)

    whole = block.q_proj.size + block.o_proj.size + block.gate_proj.size + block.down_proj.size
    per_rank = (parallel.attention[0].q_proj.size + parallel.attention[0].o_proj.size
                + parallel.mlp[0].gate_proj.size + parallel.mlp[0].down_proj.size)
    assert per_rank * 2 == whole


def test_kv_cache_shrinks_proportionally_with_world_size():
    assert kv_heads_per_rank(CONFIG, 1) == 2
    assert kv_heads_per_rank(CONFIG, 2) == 1
    with pytest.raises(ValueError):
        kv_heads_per_rank(CONFIG, 4)
