from __future__ import annotations

import numpy as np
import torch

from ..core.layout import BatchLayout
from ..kernels import flash_backend as fb
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
                block_tables, plan):
        config = self.config
        normed = tb.rms_norm(x, self.attn_norm, config.norm_eps)

        q = (normed @ self.q_proj + self.q_bias).view(-1, config.num_heads, config.head_dim)
        k = (normed @ self.k_proj + self.k_bias).view(-1, config.num_kv_heads, config.head_dim)
        v = (normed @ self.v_proj + self.v_bias).view(-1, config.num_kv_heads, config.head_dim)

        q = tb.apply_rope(q, positions, cos, sin, config.rope_halved)
        k = tb.apply_rope(k, positions, cos, sin, config.rope_halved)

        tb.write_kv(k, v, k_cache, v_cache, plan["slots"])

        if plan["flash"]:
            if layout.is_prefill:
                attended = fb.prefill(q, k, v, k_cache, v_cache,
                                      plan["cu_q"], plan["cu_k"],
                                      layout.max_query_len, layout.max_context_len,
                                      plan["paged_tables"], block_size, plan["scale"])
            else:
                attended = fb.decode(q, k_cache, v_cache, block_tables,
                                     plan["context_lens"], block_size, plan["scale"])
        else:
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
        self.graphs = None

    def allocate_caches(self, num_blocks: int, block_size: int):
        pairs = [tb.allocate_kv_cache(num_blocks, block_size, self.config.num_kv_heads,
                                      self.config.head_dim, self.device, self.dtype)
                 for _ in range(self.config.num_layers)]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    @property
    def uses_flash(self) -> bool:
        return fb.AVAILABLE and self.dtype in (torch.float16, torch.bfloat16)

    def run(self, tokens, positions, block_tables, plan, layout, k_caches, v_caches,
            block_size: int):
        x = self.embedding.index_select(0, tokens)
        for block, k_cache, v_cache in zip(self.blocks, k_caches, v_caches):
            x = block.forward(x, layout, k_cache, v_cache, self.cos, self.sin,
                              block_size, positions, block_tables, plan)
        return tb.rms_norm(x, self.final_norm, self.config.norm_eps)

    def forward(self, layout: BatchLayout, k_caches, v_caches, block_size: int):
        if self.graphs is not None and self.graphs.can_replay(layout):
            return self.graphs.replay(layout, k_caches, v_caches)

        as_tensor = lambda values, dtype=torch.long: torch.as_tensor(
            values, device=self.device, dtype=dtype)
        tokens = as_tensor(layout.token_ids)
        positions = as_tensor(layout.positions)
        block_tables = as_tensor(layout.block_tables)
        use_flash = self.uses_flash
        plan = {
            "flash": use_flash,
            "scale": self.config.head_dim ** -0.5,
            "slots": as_tensor(layout.slot_mapping),
            "cu_q": as_tensor(layout.cu_seqlens_q, torch.int32) if use_flash else None,
            "cu_k": as_tensor(layout.cu_seqlens_k, torch.int32) if use_flash else None,
            "context_lens": as_tensor(layout.context_lens, torch.int32) if use_flash else None,
            "paged_tables": (block_tables.to(torch.int32)
                             if use_flash and layout.max_context_len > layout.max_query_len
                             else None),
        }
        return self.run(tokens, positions, block_tables, plan, layout,
                        k_caches, v_caches, block_size)

    def logits_for_last_token_of_each(self, hidden, query_lens: list[int]) -> np.ndarray:
        return self.logits_on_device(hidden, query_lens).float().cpu().numpy()

    def logits_on_device(self, hidden, query_lens: list[int]) -> torch.Tensor:
        ends = torch.as_tensor(np.cumsum(query_lens) - 1, device=self.device, dtype=torch.long)
        return hidden.index_select(0, ends) @ self.lm_head

    def sample(self, hidden, query_lens: list[int], temperatures) -> list[int]:
        logits = self.logits_on_device(hidden, query_lens).float()
        temps = torch.as_tensor(temperatures, device=self.device, dtype=torch.float32)
        greedy = temps <= 0.0
        safe = torch.where(greedy, torch.ones_like(temps), temps).unsqueeze(1)
        probs = torch.softmax(logits / safe, dim=-1)
        race = torch.empty_like(probs).exponential_(1.0).clamp_min(1e-10)
        sampled = (probs / race).argmax(dim=-1)
        chosen = torch.where(greedy, logits.argmax(dim=-1), sampled)
        return chosen.tolist()
