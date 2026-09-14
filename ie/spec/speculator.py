from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.layout import build_layout
from ..core.scheduler import Batch
from ..core.sequence import Sequence
from ..engine.engine import Engine
from ..kernels.reference import allocate_kv_cache
from ..model.transformer import Transformer
from .rejection import verify_draft


def softmax_rows(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float32) / max(temperature, 1e-6)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


@dataclass(slots=True)
class SpeculativeStats:
    steps: int = 0
    drafted: int = 0
    accepted: int = 0
    emitted: int = 0
    target_passes: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_target_pass(self) -> float:
        return self.emitted / self.target_passes if self.target_passes else 0.0


class DraftModel:
    def __init__(self, model: Transformer, block_size: int, num_blocks: int) -> None:
        self.model = model
        self.block_size = block_size
        caches = [allocate_kv_cache(num_blocks, block_size, model.config.num_kv_heads,
                                    model.config.head_dim)
                  for _ in range(model.config.num_layers)]
        self.k_caches = [pair[0] for pair in caches]
        self.v_caches = [pair[1] for pair in caches]

    def propose(self, token_ids: list[int], num_tokens: int, temperature: float,
                generator: np.random.Generator) -> tuple[list[int], np.ndarray]:
        history = list(token_ids)
        proposed: list[int] = []
        distributions: list[np.ndarray] = []

        for _ in range(num_tokens):
            logits = self._logits_for(history)
            probs = softmax_rows(logits[None, :], temperature)[0]
            token = int(np.searchsorted(np.cumsum(probs), generator.random()))
            token = min(token, probs.size - 1)
            proposed.append(token)
            distributions.append(probs)
            history.append(token)

        return proposed, np.stack(distributions)

    def _logits_for(self, token_ids: list[int]) -> np.ndarray:
        scratch = Sequence(token_ids, block_size=self.block_size)
        scratch.block_table = list(range(scratch.num_blocks))
        scratch.num_scheduled = len(token_ids)
        layout = build_layout(Batch(sequences=[scratch], is_prefill=True), self.block_size)
        hidden = self.model.forward(layout, self.k_caches, self.v_caches, self.block_size)
        return self.model.logits_for_last_token_of_each(hidden, layout.query_lens)[0]


class SpeculativeEngine:
    def __init__(self, target: Engine, draft: DraftModel, lookahead: int = 4) -> None:
        self.target = target
        self.draft = draft
        self.lookahead = lookahead
        self.stats = SpeculativeStats()

    def _target_probs(self, seq: Sequence, extra_tokens: list[int],
                      temperature: float) -> np.ndarray:
        scratch = Sequence(seq.token_ids + extra_tokens, block_size=self.draft.block_size)
        scratch.block_table = list(range(scratch.num_blocks))
        scratch.num_scheduled = len(scratch)
        layout = build_layout(Batch(sequences=[scratch], is_prefill=True), self.draft.block_size)

        hidden = self.target.model.forward(layout, self.draft.k_caches, self.draft.v_caches,
                                           self.draft.block_size)
        logits = hidden @ self.target.model.lm_head
        scored = logits[len(seq) - 1 :]
        return softmax_rows(scored, temperature)

    def generate(self, prompt_ids: list[int], max_new_tokens: int,
                 temperature: float = 1.0, seed: int = 0) -> list[int]:
        generator = np.random.default_rng(seed)
        seq = Sequence(prompt_ids, block_size=self.draft.block_size,
                       max_new_tokens=max_new_tokens, temperature=temperature)

        while seq.num_generated < max_new_tokens:
            budget = max_new_tokens - seq.num_generated
            k = min(self.lookahead, budget)

            drafted, draft_probs = self.draft.propose(seq.token_ids, k, temperature, generator)
            target_probs = self._target_probs(seq, drafted, temperature)
            result = verify_draft(drafted, draft_probs, target_probs, generator)

            for token in result.tokens[:budget]:
                seq.append(token)

            self.stats.steps += 1
            self.stats.drafted += k
            self.stats.accepted += result.num_accepted
            self.stats.emitted += min(result.num_emitted, budget)
            self.stats.target_passes += 1

        return seq.token_ids[seq.num_prompt :]
