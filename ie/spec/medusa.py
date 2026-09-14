from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kernels.reference import gather_context, tree_attention
from ..layers.functional import apply_rope, rms_norm, swiglu
from ..model.transformer import ModelConfig, Transformer
from .tree import CandidateTree, build_dense_tree


@dataclass(slots=True)
class TreeAcceptance:
    accepted_nodes: list[int]
    tokens: list[int]

    @property
    def num_accepted(self) -> int:
        return len(self.accepted_nodes) - 1

    @property
    def num_emitted(self) -> int:
        return len(self.tokens)


class TreeVerifier:
    def __init__(self, model: Transformer, block_size: int) -> None:
        self.model = model
        self.block_size = block_size

    def logits_for_tree(self, tree: CandidateTree, base_position: int,
                        block_table: list[int], context_len: int,
                        k_caches: list[np.ndarray], v_caches: list[np.ndarray]) -> np.ndarray:
        config = self.model.config
        positions = np.asarray(tree.positions_from(base_position))
        visibility = tree.visibility_mask()
        x = self.model.embedding[np.asarray(tree.tokens)]

        for block, k_cache, v_cache in zip(self.model.blocks, k_caches, v_caches):
            normed = rms_norm(x, block.attn_norm, config.norm_eps)
            q = (normed @ block.q_proj).reshape(-1, config.num_heads, config.head_dim)
            k = (normed @ block.k_proj).reshape(-1, config.num_kv_heads, config.head_dim)
            v = (normed @ block.v_proj).reshape(-1, config.num_kv_heads, config.head_dim)

            q = apply_rope(q, positions, self.model.cos, self.model.sin)
            k = apply_rope(k, positions, self.model.cos, self.model.sin)

            context_k = gather_context(k_cache, block_table, context_len, self.block_size)
            context_v = gather_context(v_cache, block_table, context_len, self.block_size)

            attended = tree_attention(q, context_k, context_v, k, v, visibility)
            x = x + attended.reshape(-1, config.num_heads * config.head_dim) @ block.o_proj

            normed = rms_norm(x, block.mlp_norm, config.norm_eps)
            x = x + swiglu(normed, block.gate_proj, block.up_proj, block.down_proj)

        hidden = rms_norm(x, self.model.final_norm, config.norm_eps)
        return hidden @ self.model.lm_head


class MedusaHeads:
    def __init__(self, config: ModelConfig, num_heads: int, seed: int = 0) -> None:
        generator = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(config.hidden_size)
        self.num_heads = num_heads
        self.weights = [(generator.standard_normal((config.hidden_size, config.vocab_size))
                         * scale).astype(np.float32) for _ in range(num_heads)]

    def candidates(self, hidden_state: np.ndarray, widths: list[int]) -> list[list[int]]:
        if len(widths) > self.num_heads:
            raise ValueError(f"{len(widths)} widths for {self.num_heads} heads")
        picks: list[list[int]] = []
        for head, width in zip(self.weights, widths):
            logits = hidden_state @ head
            picks.append(np.argsort(logits)[-width:][::-1].tolist())
        return picks


def align_with_target(model: Transformer, heads: MedusaHeads, strength: float) -> None:
    for index, head in enumerate(heads.weights):
        heads.weights[index] = (strength * model.lm_head
                                + (1.0 - strength) * head).astype(np.float32)


def accept_greedily(tree: CandidateTree, target_logits: np.ndarray) -> TreeAcceptance:
    node = tree.root
    accepted = [node]
    tokens: list[int] = []

    while True:
        wanted = int(np.argmax(target_logits[node]))
        match = next((child for child in tree.children_of(node)
                      if tree.tokens[child] == wanted), None)
        tokens.append(wanted)
        if match is None:
            return TreeAcceptance(accepted_nodes=accepted, tokens=tokens)
        accepted.append(match)
        node = match


def propose_tree(root_token: int, hidden_state: np.ndarray, heads: MedusaHeads,
                 widths: list[int]) -> CandidateTree:
    return build_dense_tree(root_token, heads.candidates(hidden_state, widths))
