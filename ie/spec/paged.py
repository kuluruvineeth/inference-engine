from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.layout import build_layout
from ..core.scheduler import Batch
from ..core.sequence import Sequence
from ..engine.engine import Engine, EngineConfig
from ..model.transformer import Transformer
from .rejection import verify_draft
from .speculator import SpeculativeStats, softmax_rows


@dataclass(slots=True)
class DraftProposal:
    tokens: list[int]
    probs: np.ndarray


class PagedSpeculativeEngine:
    def __init__(self, target_model: Transformer, draft_model: Transformer,
                 config: EngineConfig | None = None, lookahead: int = 4) -> None:
        config = config or EngineConfig()
        self.target = Engine(target_model, config)
        self.draft = Engine(draft_model, config)
        self.lookahead = lookahead
        self.block_size = config.block_size
        self.stats = SpeculativeStats()

    def _propose(self, prompt_ids: list[int], history: list[int], num_tokens: int,
                 temperature: float) -> DraftProposal:
        draft_seq = self.draft.scheduler
        tokens: list[int] = []
        rows: list[np.ndarray] = []

        pending = self.draft.submit(prompt_ids + history, max_new_tokens=num_tokens,
                                    temperature=temperature)
        while len(tokens) < num_tokens and draft_seq.has_work:
            sequences, logits = self.draft.step_returning_logits()
            if logits is None:
                break
            if pending.prompt_is_fully_computed and pending in sequences:
                index = sequences.index(pending)
                rows.append(softmax_rows(logits[index][None, :], temperature)[0])
                tokens.append(pending.token_ids[-1])

        if pending in draft_seq.running:
            draft_seq.running.remove(pending)
        if pending in draft_seq.waiting:
            draft_seq.waiting.remove(pending)
        self.draft.blocks.free(pending)

        return DraftProposal(tokens=tokens, probs=np.stack(rows) if rows else np.empty((0, 0)))

    def _reserve_blocks(self, seq: Sequence, extra: int) -> bool:
        needed = (len(seq) + extra + self.block_size - 1) // self.block_size
        return self.target.blocks.reserve(seq, needed)

    def _verify(self, seq: Sequence, proposal: DraftProposal,
                temperature: float) -> np.ndarray:
        k = len(proposal.tokens)
        original_len = len(seq)
        seq.token_ids.extend(proposal.tokens)

        seq.num_computed = original_len - 1
        seq.num_scheduled = k + 1
        layout = build_layout(Batch(sequences=[seq], is_prefill=True), self.block_size)

        hidden = self.target.model.forward(layout, self.target.k_caches,
                                           self.target.v_caches, self.block_size)
        logits = hidden @ self.target.model.lm_head

        del seq.token_ids[original_len:]
        seq.num_scheduled = 0
        return softmax_rows(logits, temperature)

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 32,
                 temperature: float = 1.0, seed: int = 0) -> list[int]:
        generator = np.random.default_rng(seed)
        seq = self.target.submit(prompt_ids, max_new_tokens=max_new_tokens,
                                 temperature=temperature)

        while not seq.prompt_is_fully_computed:
            self.target.step()

        while seq.num_generated < max_new_tokens:
            budget = max_new_tokens - seq.num_generated
            k = min(self.lookahead, budget)

            if not self._reserve_blocks(seq, k + 1):
                self.target.step()
                continue

            proposal = self._propose(prompt_ids, seq.token_ids[seq.num_prompt:], k, temperature)
            if not proposal.tokens:
                self.target.step()
                continue

            target_probs = self._verify(seq, proposal, temperature)
            result = verify_draft(proposal.tokens, proposal.probs, target_probs, generator)

            accepted = result.tokens[:budget]
            for token in accepted:
                seq.append(token)
            seq.num_computed = len(seq) - 1

            self.stats.steps += 1
            self.stats.drafted += len(proposal.tokens)
            self.stats.accepted += result.num_accepted
            self.stats.emitted += len(accepted)
            self.stats.target_passes += 1

        self.target.blocks.free(seq)
        return seq.token_ids[seq.num_prompt:]
