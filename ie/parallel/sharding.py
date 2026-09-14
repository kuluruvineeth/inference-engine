from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Shard:
    rank: int
    world_size: int

    def __post_init__(self):
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} outside world of {self.world_size}")

    def slice_of(self, length: int) -> slice:
        if length % self.world_size:
            raise ValueError(f"{length} does not divide evenly across {self.world_size} ranks")
        width = length // self.world_size
        return slice(self.rank * width, (self.rank + 1) * width)


def split_output_dim(weight: np.ndarray, shard: Shard) -> np.ndarray:
    return weight[:, shard.slice_of(weight.shape[1])]


def split_input_dim(weight: np.ndarray, shard: Shard) -> np.ndarray:
    return weight[shard.slice_of(weight.shape[0]), :]


def split_heads(weight: np.ndarray, shard: Shard, num_heads: int, head_dim: int) -> np.ndarray:
    per_rank = shard.slice_of(num_heads)
    reshaped = weight.reshape(weight.shape[0], num_heads, head_dim)
    return reshaped[:, per_rank, :].reshape(weight.shape[0], -1)


def all_reduce(partials: list[np.ndarray]) -> np.ndarray:
    total = partials[0].copy()
    for part in partials[1:]:
        total += part
    return total


def all_gather(partials: list[np.ndarray], axis: int = -1) -> np.ndarray:
    return np.concatenate(partials, axis=axis)


class ColumnParallelLinear:
    def __init__(self, weight: np.ndarray, shard: Shard) -> None:
        self.shard = shard
        self.weight = split_output_dim(weight, shard)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight


class RowParallelLinear:
    def __init__(self, weight: np.ndarray, shard: Shard) -> None:
        self.shard = shard
        self.weight = split_input_dim(weight, shard)

    def forward(self, x_shard: np.ndarray) -> np.ndarray:
        return x_shard @ self.weight
