from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.layout import BatchLayout
from ..kernels.reference import paged_attention, write_kv
from ..layers.functional import apply_rope, rms_norm, rope_tables, swiglu


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 16
    intermediate_size: int = 128
    max_position: int = 2048
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    rope_halved: bool = False

    def __post_init__(self):
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 4


class TransformerBlock:
    def __init__(self, config: ModelConfig, generator: np.random.Generator) -> None:
        self.config = config
        scale = 1.0 / np.sqrt(config.hidden_size)
        q_out = config.num_heads * config.head_dim
        kv_out = config.num_kv_heads * config.head_dim

        def normal(*shape):
            return (generator.standard_normal(shape) * scale).astype(np.float32)

        self.attn_norm = np.ones(config.hidden_size, dtype=np.float32)
        self.q_proj = normal(config.hidden_size, q_out)
        self.k_proj = normal(config.hidden_size, kv_out)
        self.v_proj = normal(config.hidden_size, kv_out)
        self.o_proj = normal(q_out, config.hidden_size)
        self.q_bias = np.zeros(q_out, dtype=np.float32)
        self.k_bias = np.zeros(kv_out, dtype=np.float32)
        self.v_bias = np.zeros(kv_out, dtype=np.float32)

        self.mlp_norm = np.ones(config.hidden_size, dtype=np.float32)
        self.gate_proj = normal(config.hidden_size, config.intermediate_size)
        self.up_proj = normal(config.hidden_size, config.intermediate_size)
        self.down_proj = normal(config.intermediate_size, config.hidden_size)

    def forward(self, x: np.ndarray, layout: BatchLayout, k_cache: np.ndarray,
                v_cache: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                block_size: int) -> np.ndarray:
        config = self.config
        normed = rms_norm(x, self.attn_norm, config.norm_eps)

        q = (normed @ self.q_proj + self.q_bias).reshape(-1, config.num_heads, config.head_dim)
        k = (normed @ self.k_proj + self.k_bias).reshape(-1, config.num_kv_heads, config.head_dim)
        v = (normed @ self.v_proj + self.v_bias).reshape(-1, config.num_kv_heads, config.head_dim)

        positions = np.asarray(layout.positions)
        q = apply_rope(q, positions, cos, sin, config.rope_halved)
        k = apply_rope(k, positions, cos, sin, config.rope_halved)

        write_kv(k, v, k_cache, v_cache, layout.slot_mapping)

        attended = paged_attention(q, k_cache, v_cache, layout.block_tables,
                                   layout.context_lens, layout.query_lens, block_size)
        x = x + attended.reshape(-1, config.num_heads * config.head_dim) @ self.o_proj

        normed = rms_norm(x, self.mlp_norm, config.norm_eps)
        return x + swiglu(normed, self.gate_proj, self.up_proj, self.down_proj)


class Transformer:
    def __init__(self, config: ModelConfig, seed: int = 0) -> None:
        self.config = config
        generator = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(config.hidden_size)

        self.embedding = (generator.standard_normal(
            (config.vocab_size, config.hidden_size)) * scale).astype(np.float32)
        self.blocks = [TransformerBlock(config, generator) for _ in range(config.num_layers)]
        self.final_norm = np.ones(config.hidden_size, dtype=np.float32)
        self.lm_head = (generator.standard_normal(
            (config.hidden_size, config.vocab_size)) * scale).astype(np.float32)

        self.cos, self.sin = rope_tables(config.max_position, config.head_dim, config.rope_base)

    def forward(self, layout: BatchLayout, k_caches: list[np.ndarray],
                v_caches: list[np.ndarray], block_size: int) -> np.ndarray:
        x = self.embedding[np.asarray(layout.token_ids)]
        for block, k_cache, v_cache in zip(self.blocks, k_caches, v_caches):
            x = block.forward(x, layout, k_cache, v_cache, self.cos, self.sin, block_size)
        return rms_norm(x, self.final_norm, self.config.norm_eps)

    def logits_for_last_token_of_each(self, hidden: np.ndarray,
                                      query_lens: list[int]) -> np.ndarray:
        ends = np.cumsum(query_lens) - 1
        return hidden[ends] @ self.lm_head
