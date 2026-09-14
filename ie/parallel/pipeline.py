from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.layout import BatchLayout
from ..model.transformer import Transformer


@dataclass(slots=True)
class PipelineStats:
    activations_sent: int = 0
    bytes_sent: int = 0
    stage_invocations: int = 0

    @property
    def hops(self) -> int:
        return self.activations_sent


class PipelineStage:
    def __init__(self, model: Transformer, stage_index: int, num_stages: int) -> None:
        total = len(model.blocks)
        if num_stages > total:
            raise ValueError(f"{num_stages} stages for only {total} layers")
        boundaries = np.linspace(0, total, num_stages + 1).astype(int)
        self.model = model
        self.index = stage_index
        self.num_stages = num_stages
        self.layer_slice = slice(int(boundaries[stage_index]), int(boundaries[stage_index + 1]))
        self.blocks = model.blocks[self.layer_slice]

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.index == self.num_stages - 1

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def forward(self, x: np.ndarray, layout: BatchLayout, k_caches, v_caches,
                block_size: int) -> np.ndarray:
        for offset, block in enumerate(self.blocks):
            layer = self.layer_slice.start + offset
            x = block.forward(x, layout, k_caches[layer], v_caches[layer],
                              self.model.cos, self.model.sin, block_size)
        return x


class PipelineParallelModel:
    def __init__(self, model: Transformer, num_stages: int) -> None:
        self.model = model
        self.config = model.config
        self.stages = [PipelineStage(model, index, num_stages) for index in range(num_stages)]
        self.stats = PipelineStats()

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def layers_per_stage(self) -> list[int]:
        return [stage.num_layers for stage in self.stages]

    def forward(self, layout: BatchLayout, k_caches, v_caches, block_size: int) -> np.ndarray:
        from ..layers.functional import rms_norm

        x = self.model.embedding[np.asarray(layout.token_ids)]
        for stage in self.stages:
            x = stage.forward(x, layout, k_caches, v_caches, block_size)
            self.stats.stage_invocations += 1
            if not stage.is_last:
                self.stats.activations_sent += 1
                self.stats.bytes_sent += int(x.nbytes)
        return rms_norm(x, self.model.final_norm, self.config.norm_eps)

    def logits_for_last_token_of_each(self, hidden, query_lens):
        return self.model.logits_for_last_token_of_each(hidden, query_lens)
