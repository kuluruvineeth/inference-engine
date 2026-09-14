from __future__ import annotations

import numpy as np

from ..core.layout import BatchLayout
from ..kernels.reference import paged_attention, write_kv
from ..layers.functional import apply_rope, rms_norm, silu
from ..model.transformer import ModelConfig, TransformerBlock
from .sharding import Shard, all_reduce, split_heads, split_input_dim, split_output_dim


class ShardedAttention:
    def __init__(self, block: TransformerBlock, shard: Shard) -> None:
        config = block.config
        self.config = config
        self.shard = shard
        self.num_heads = config.num_heads // shard.world_size
        self.num_kv_heads = config.num_kv_heads // shard.world_size

        self.norm = block.attn_norm
        self.q_proj = split_heads(block.q_proj, shard, config.num_heads, config.head_dim)
        self.k_proj = split_heads(block.k_proj, shard, config.num_kv_heads, config.head_dim)
        self.v_proj = split_heads(block.v_proj, shard, config.num_kv_heads, config.head_dim)
        self.o_proj = split_input_dim(block.o_proj, shard)

    def forward(self, x: np.ndarray, layout: BatchLayout, k_cache: np.ndarray,
                v_cache: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                block_size: int) -> np.ndarray:
        normed = rms_norm(x, self.norm, self.config.norm_eps)
        head_dim = self.config.head_dim

        q = (normed @ self.q_proj).reshape(-1, self.num_heads, head_dim)
        k = (normed @ self.k_proj).reshape(-1, self.num_kv_heads, head_dim)
        v = (normed @ self.v_proj).reshape(-1, self.num_kv_heads, head_dim)

        positions = np.asarray(layout.positions)
        q = apply_rope(q, positions, cos, sin)
        k = apply_rope(k, positions, cos, sin)

        write_kv(k, v, k_cache, v_cache, layout.slot_mapping)
        attended = paged_attention(q, k_cache, v_cache, layout.block_tables,
                                   layout.context_lens, layout.query_lens, block_size)
        return attended.reshape(-1, self.num_heads * head_dim) @ self.o_proj


class ShardedMLP:
    def __init__(self, block: TransformerBlock, shard: Shard) -> None:
        self.config = block.config
        self.norm = block.mlp_norm
        self.gate_proj = split_output_dim(block.gate_proj, shard)
        self.up_proj = split_output_dim(block.up_proj, shard)
        self.down_proj = split_input_dim(block.down_proj, shard)

    def forward(self, x: np.ndarray) -> np.ndarray:
        normed = rms_norm(x, self.norm, self.config.norm_eps)
        return (silu(normed @ self.gate_proj) * (normed @ self.up_proj)) @ self.down_proj


class TensorParallelBlock:
    def __init__(self, block: TransformerBlock, world_size: int) -> None:
        shards = [Shard(rank, world_size) for rank in range(world_size)]
        self.attention = [ShardedAttention(block, shard) for shard in shards]
        self.mlp = [ShardedMLP(block, shard) for shard in shards]

    def forward(self, x: np.ndarray, layout: BatchLayout, k_caches: list[np.ndarray],
                v_caches: list[np.ndarray], cos: np.ndarray, sin: np.ndarray,
                block_size: int) -> np.ndarray:
        partials = [rank.forward(x, layout, k_caches[i], v_caches[i], cos, sin, block_size)
                    for i, rank in enumerate(self.attention)]
        x = x + all_reduce(partials)

        partials = [rank.forward(x) for rank in self.mlp]
        return x + all_reduce(partials)


def kv_heads_per_rank(config: ModelConfig, world_size: int) -> int:
    if config.num_kv_heads % world_size:
        raise ValueError(f"{config.num_kv_heads} kv heads do not divide across {world_size} ranks")
    return config.num_kv_heads // world_size
