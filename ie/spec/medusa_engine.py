from __future__ import annotations

import numpy as np

from ..engine.engine import Engine, EngineConfig
from ..model.transformer import Transformer
from .medusa import MedusaHeads, TreeVerifier, accept_greedily, propose_tree
from .speculator import SpeculativeStats


class MedusaEngine:
    def __init__(self, model: Transformer, heads: MedusaHeads,
                 config: EngineConfig | None = None,
                 widths: list[int] | None = None) -> None:
        self.engine = Engine(model, config or EngineConfig())
        self.heads = heads
        self.widths = widths or [2, 2]
        self.verifier = TreeVerifier(model, self.engine.config.block_size)
        self.stats = SpeculativeStats()

    @property
    def blocks(self):
        return self.engine.blocks

    def _commit_pending_and_get_hidden(self, seq) -> np.ndarray:
        from ..core.layout import build_layout
        from ..core.scheduler import Batch

        block_size = self.engine.config.block_size
        needed = (len(seq) + block_size - 1) // block_size
        self.blocks.reserve(seq, needed)

        seq.num_scheduled = len(seq) - seq.num_computed
        layout = build_layout(Batch(sequences=[seq], is_prefill=True), block_size)
        hidden = self.engine.model.forward(layout, self.engine.k_caches,
                                           self.engine.v_caches, block_size)
        seq.commit_scheduled()
        return hidden[-1]

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 32) -> list[int]:
        seq = self.engine.submit(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.0)
        while not seq.prompt_is_fully_computed:
            self.engine.step()

        while seq.num_generated < max_new_tokens:
            budget = max_new_tokens - seq.num_generated
            depth = min(len(self.widths), budget)
            if depth == 0:
                break

            hidden = self._commit_pending_and_get_hidden(seq)
            tree = propose_tree(seq.last_token, hidden, self.heads, self.widths[:depth])

            needed = (len(seq) + tree.size + self.engine.config.block_size - 1) // self.engine.config.block_size
            if not self.blocks.reserve(seq, needed):
                self.engine.step()
                continue

            logits = self.verifier.logits_for_tree(
                tree, base_position=len(seq) - 1, block_table=seq.block_table,
                context_len=len(seq) - 1, k_caches=self.engine.k_caches,
                v_caches=self.engine.v_caches)

            result = accept_greedily(tree, logits)
            emitted = result.tokens[:budget]
            for token in emitted:
                seq.append(token)

            self.stats.steps += 1
            self.stats.drafted += tree.size - 1
            self.stats.accepted += result.num_accepted
            self.stats.emitted += len(emitted)
            self.stats.target_passes += 1

        self.blocks.free(seq)
        return seq.token_ids[seq.num_prompt:]
