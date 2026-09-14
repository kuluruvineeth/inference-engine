from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.block_manager import BlockManager
from ..core.layout import build_layout
from ..core.scheduler import Scheduler
from ..core.sequence import Sequence
from ..kernels.reference import allocate_kv_cache
from ..layers.functional import sample_from_logits
from ..model.transformer import ModelConfig, Transformer


@dataclass(slots=True)
class EngineConfig:
    block_size: int = 16
    num_blocks: int = 1024
    max_batch_sequences: int = 64
    max_batch_tokens: int = 2048
    seed: int = 0


class Engine:
    def __init__(self, model: Transformer, config: EngineConfig | None = None) -> None:
        self.model = model
        self.config = config or EngineConfig()
        self.blocks = BlockManager(self.config.num_blocks, self.config.block_size)
        self.scheduler = Scheduler(self.blocks,
                                   max_batch_sequences=self.config.max_batch_sequences,
                                   max_batch_tokens=self.config.max_batch_tokens)
        self.generator = np.random.default_rng(self.config.seed)

        caches = [allocate_kv_cache(self.config.num_blocks, self.config.block_size,
                                    model.config.num_kv_heads, model.config.head_dim)
                  for _ in range(model.config.num_layers)]
        self.k_caches = [pair[0] for pair in caches]
        self.v_caches = [pair[1] for pair in caches]

    def submit(self, prompt_ids: list[int], max_new_tokens: int = 32,
               temperature: float = 1.0, eos_id: int | None = None) -> Sequence:
        seq = Sequence(prompt_ids, block_size=self.config.block_size,
                       max_new_tokens=max_new_tokens, temperature=temperature,
                       eos_id=eos_id)
        self.scheduler.add(seq)
        return seq

    def forward(self, batch):
        layout = build_layout(batch, self.config.block_size)
        hidden = self.model.forward(layout, self.k_caches, self.v_caches,
                                    self.config.block_size)
        return layout, self.model.logits_for_last_token_of_each(hidden, layout.query_lens)

    def step(self) -> list[Sequence]:
        batch = self.scheduler.schedule()
        if not batch:
            return []

        _, logits = self.forward(batch)
        temperatures = np.array([seq.temperature for seq in batch.sequences], dtype=np.float32)
        token_ids = sample_from_logits(logits, temperatures, self.generator).tolist()
        return self.scheduler.finish_step(batch, token_ids)

    def step_returning_logits(self) -> tuple[list[Sequence], np.ndarray | None]:
        batch = self.scheduler.schedule()
        if not batch:
            return [], None

        _, logits = self.forward(batch)
        temperatures = np.array([seq.temperature for seq in batch.sequences], dtype=np.float32)
        token_ids = sample_from_logits(logits, temperatures, self.generator).tolist()
        self.scheduler.finish_step(batch, token_ids)
        return batch.sequences, logits

    def run(self, max_steps: int = 100_000) -> list[Sequence]:
        completed: list[Sequence] = []
        for _ in range(max_steps):
            if not self.scheduler.has_work:
                break
            completed.extend(self.step())
        return completed

    def generate(self, prompts: list[list[int]], max_new_tokens: int = 32,
                 temperature: float = 1.0) -> list[list[int]]:
        requested = [self.submit(p, max_new_tokens=max_new_tokens, temperature=temperature)
                     for p in prompts]
        self.run()
        return [seq.token_ids[seq.num_prompt:] for seq in requested]


def build_engine(model_config: ModelConfig | None = None,
                 engine_config: EngineConfig | None = None,
                 model_seed: int = 0) -> Engine:
    return Engine(Transformer(model_config or ModelConfig(), seed=model_seed), engine_config)
