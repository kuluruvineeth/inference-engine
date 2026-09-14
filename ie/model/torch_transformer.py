from __future__ import annotations

import numpy as np
import torch

from ..core.layout import BatchLayout
from ..kernels import torch_backend as tb
from .transformer import ModelConfig, Transformer


class TorchBlock:
    def __init__(self, source, device: torch.device, dtype: torch.dtype) -> None:
        self.config = source.config
        take = lambda w: torch.from_numpy(np.ascontiguousarray(w)).to(device=device, dtype=dtype)
        self.attn_norm = take(source.attn_norm)
        self.q_proj = take(source.q_proj)
        self.k_proj = take(source.k_proj)
        self.v_proj = take(source.v_proj)
        self.o_proj = take(source.o_proj)
        self.q_bias = take(source.q_bias)
        self.k_bias = take(source.k_bias)
        self.v_bias = take(source.v_bias)
        self.mlp_norm = take(source.mlp_norm)
        self.gate_proj = take(source.gate_proj)
        self.up_proj = take(source.up_proj)
        self.down_proj = take(source.down_proj)

    def forward(self, x, layout, k_cache, v_cache, cos, sin, block_size, positions,
                block_tables):
        config = self.config
        normed = tb.rms_norm(x, self.attn_norm, config.norm_eps)

        q = (normed @ self.q_proj + self.q_bias).view(-1, config.num_heads, config.head_dim)
        k = (normed @ self.k_proj + self.k_bias).view(-1, config.num_kv_heads, config.head_dim)
        v = (normed @ self.v_proj + self.v_bias).view(-1, config.num_kv_heads, config.head_dim)

        q = tb.apply_rope(q, positions, cos, sin, config.rope_halved)
        k = tb.apply_rope(k, positions, cos, sin, config.rope_halved)

        slots = torch.as_tensor(layout.slot_mapping, device=x.device, dtype=torch.long)
        tb.write_kv(k, v, k_cache, v_cache, slots)

        attended = tb.paged_attention(q, k_cache, v_cache, block_tables,
                                      layout.context_lens, layout.query_lens, block_size)
        x = x + attended.reshape(-1, config.num_heads * config.head_dim).to(x.dtype) @ self.o_proj

        normed = tb.rms_norm(x, self.mlp_norm, config.norm_eps)
        return x + tb.swiglu(normed, self.gate_proj, self.up_proj, self.down_proj)


class TorchTransformer:
    def __init__(self, source: Transformer, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float16) -> None:
        self.config = source.config
        self.device = torch.device(device)
        self.dtype = dtype
        take = lambda w: torch.from_numpy(np.ascontiguousarray(w)).to(device=self.device,
                                                                      dtype=dtype)
        self.embedding = take(source.embedding)
        self.blocks = [TorchBlock(block, self.device, dtype) for block in source.blocks]
        self.final_norm = take(source.final_norm)
        self.lm_head = take(source.lm_head)
        self.cos = take(source.cos)
        self.sin = take(source.sin)

    def allocate_caches(self, num_blocks: int, block_size: int):
        pairs = [tb.allocate_kv_cache(num_blocks, block_size, self.config.num_kv_heads,
                                      self.config.head_dim, self.device, self.dtype)
                 for _ in range(self.config.num_layers)]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def forward(self, layout: BatchLayout, k_caches, v_caches, block_size: int):
        tokens = torch.as_tensor(layout.token_ids, device=self.device, dtype=torch.long)
        positions = torch.as_tensor(layout.positions, device=self.device, dtype=torch.long)
        block_tables = torch.as_tensor(layout.block_tables, device=self.device, dtype=torch.long)

        x = self.embedding.index_select(0, tokens)
        for block, k_cache, v_cache in zip(self.blocks, k_caches, v_caches):
            x = block.forward(x, layout, k_cache, v_cache, self.cos, self.sin,
                              block_size, positions, block_tables)
        return tb.rms_norm(x, self.final_norm, self.config.norm_eps)

    def logits_for_last_token_of_each(self, hidden, query_lens: list[int]) -> np.ndarray:
        ends = torch.as_tensor(np.cumsum(query_lens) - 1, device=self.device, dtype=torch.long)
        logits = hidden.index_select(0, ends) @ self.lm_head
        return logits.float().cpu().numpy()
