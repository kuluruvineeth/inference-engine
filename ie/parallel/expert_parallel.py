from __future__ import annotations

import numpy as np

from ..model.moe import MixtureOfExperts, RoutingStats
from .sharding import Shard, all_reduce


class ExpertShard:
    def __init__(self, layer: MixtureOfExperts, shard: Shard) -> None:
        self.shard = shard
        self.layer = layer
        span = shard.slice_of(layer.num_experts)
        self.owned = list(range(span.start, span.stop))

    def owns(self, expert_id: int) -> bool:
        return expert_id in self.owned

    def forward(self, x: np.ndarray, chosen: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for expert_id in self.owned:
            rows, slots = np.nonzero(chosen == expert_id)
            if rows.size == 0:
                continue
            contribution = self.layer.experts[expert_id].forward(x[rows])
            out[rows] += contribution * weights[rows, slots][:, None]
        return out


class ExpertParallelMoE:
    def __init__(self, layer: MixtureOfExperts, world_size: int) -> None:
        if layer.num_experts % world_size:
            raise ValueError(f"{layer.num_experts} experts do not divide across "
                             f"{world_size} ranks")
        self.layer = layer
        self.world_size = world_size
        self.shards = [ExpertShard(layer, Shard(rank, world_size))
                       for rank in range(world_size)]

    @property
    def experts_per_rank(self) -> int:
        return self.layer.num_experts // self.world_size

    def forward(self, x: np.ndarray, stats: RoutingStats | None = None) -> np.ndarray:
        chosen, weights = self.layer.route(x)
        partials = [shard.forward(x, chosen, weights) for shard in self.shards]

        if stats is not None:
            hits = [0] * self.layer.num_experts
            for expert_id in range(self.layer.num_experts):
                hits[expert_id] = int(np.count_nonzero(chosen == expert_id))
            stats.tokens_routed += x.shape[0]
            stats.expert_hits = [a + b for a, b in zip(
                stats.expert_hits or [0] * self.layer.num_experts, hits)]

        return all_reduce(partials)

    def tokens_per_rank(self, x: np.ndarray) -> list[int]:
        chosen, _ = self.layer.route(x)
        return [int(sum(np.count_nonzero(chosen == e) for e in shard.owned))
                for shard in self.shards]
