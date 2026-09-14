from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.layout import build_layout
from ..core.scheduler import Batch
from ..engine.engine import Engine, EngineConfig
from ..model.transformer import Transformer
from .speculator import SpeculativeStats
from .tree import build_chain

Proposer = Callable[[list[int], np.ndarray, int], list[int]]


class ChainSpeculativeEngine:
    def __init__(self, model: Transformer, propose: Proposer,
                 config: EngineConfig | None = None, lookahead: int = 4) -> None:
        self.engine = Engine(model, config or EngineConfig())
        self.propose = propose
        self.lookahead = lookahead
        self.block_size = self.engine.config.block_size
        self.stats = SpeculativeStats()

    @property
    def blocks(self):
        return self.engine.blocks

    def _commit_pending_and_get_hidden(self, seq) -> np.ndarray:
        needed = (len(seq) + self.block_size - 1) // self.block_size
        self.blocks.reserve(seq, needed)
        seq.num_scheduled = len(seq) - seq.num_computed
        layout = build_layout(Batch(sequences=[seq], is_prefill=True), self.block_size)
        hidden = self.engine.model.forward(layout, self.engine.k_caches,
                                           self.engine.v_caches, self.block_size)
        seq.commit_scheduled()
        return hidden[-1]

    def _verify(self, seq, drafted: list[int]) -> np.ndarray:
        original_len = len(seq)
        seq.token_ids.extend(drafted)
        needed = (len(seq) + self.block_size - 1) // self.block_size
        self.blocks.reserve(seq, needed)

        seq.num_computed = original_len - 1
        seq.num_scheduled = len(drafted) + 1
        layout = build_layout(Batch(sequences=[seq], is_prefill=True), self.block_size)
        hidden = self.engine.model.forward(layout, self.engine.k_caches,
                                           self.engine.v_caches, self.block_size)
        logits = hidden @ self.engine.model.lm_head

        del seq.token_ids[original_len:]
        seq.num_scheduled = 0
        seq.num_computed = original_len - 1
        return logits

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 32) -> list[int]:
        seq = self.engine.submit(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.0)
        while not seq.prompt_is_fully_computed:
            self.engine.step()

        while seq.num_generated < max_new_tokens:
            budget = max_new_tokens - seq.num_generated
            hidden = self._commit_pending_and_get_hidden(seq)
            drafted = self.propose(seq.token_ids, hidden, min(self.lookahead, budget))

            if not drafted:
                logits = self.engine.model.logits_for_last_token_of_each(
                    hidden[None, :], [1])
                seq.append(int(np.argmax(logits[0])))
                self.stats.steps += 1
                self.stats.emitted += 1
                self.stats.target_passes += 1
                continue

            logits = self._verify(seq, drafted)
            chain = build_chain(seq.last_token, drafted)

            accepted = 0
            emitted: list[int] = []
            for depth, token in enumerate(drafted):
                wanted = int(np.argmax(logits[depth]))
                emitted.append(wanted)
                if wanted != token:
                    break
                accepted += 1
            else:
                emitted.append(int(np.argmax(logits[len(drafted)])))

            for token in emitted[:budget]:
                seq.append(token)

            self.stats.steps += 1
            self.stats.drafted += len(drafted)
            self.stats.accepted += accepted
            self.stats.emitted += len(emitted[:budget])
            self.stats.target_passes += 1

        self.blocks.free(seq)
        return seq.token_ids[seq.num_prompt:]
