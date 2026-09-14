from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..layers.functional import silu
from .transformer import ModelConfig


@dataclass(slots=True)
class RoutingStats:
    tokens_routed: int = 0
    expert_hits: list[int] = field(default_factory=list)

    @property
    def num_experts(self) -> int:
        return len(self.expert_hits)

    @property
    def load_fractions(self) -> list[float]:
        total = sum(self.expert_hits)
        return [hits / total for hits in self.expert_hits] if total else []

    @property
    def imbalance(self) -> float:
        fractions = self.load_fractions
        if not fractions:
            return 0.0
        ideal = 1.0 / len(fractions)
        return max(fractions) / ideal

    @property
    def idle_experts(self) -> int:
        return sum(1 for hits in self.expert_hits if hits == 0)


class Expert:
    def __init__(self, hidden_size: int, intermediate_size: int,
                 generator: np.random.Generator) -> None:
        scale = 1.0 / np.sqrt(hidden_size)
        normal = lambda *shape: (generator.standard_normal(shape) * scale).astype(np.float32)
        self.gate_proj = normal(hidden_size, intermediate_size)
        self.up_proj = normal(hidden_size, intermediate_size)
        self.down_proj = normal(intermediate_size, hidden_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (silu(x @ self.gate_proj) * (x @ self.up_proj)) @ self.down_proj


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def top_k_routing(gate_logits: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if top_k < 1 or top_k > gate_logits.shape[-1]:
        raise ValueError(f"top_k {top_k} outside 1..{gate_logits.shape[-1]}")
    chosen = np.argsort(gate_logits, axis=-1)[:, -top_k:][:, ::-1]
    picked = np.take_along_axis(gate_logits, chosen, axis=-1)
    weights = softmax_rows(picked)
    return chosen, weights


class MixtureOfExperts:
    def __init__(self, config: ModelConfig, num_experts: int, top_k: int = 2,
                 seed: int = 0) -> None:
        if num_experts < 1:
            raise ValueError("need at least one expert")
        generator = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(config.hidden_size)
        self.config = config
        self.top_k = top_k
        self.gate = (generator.standard_normal((config.hidden_size, num_experts))
                     * scale).astype(np.float32)
        self.experts = [Expert(config.hidden_size, config.intermediate_size, generator)
                        for _ in range(num_experts)]

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def active_fraction(self) -> float:
        return self.top_k / self.num_experts

    def route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return top_k_routing(x @ self.gate, self.top_k)

    def forward(self, x: np.ndarray, stats: RoutingStats | None = None) -> np.ndarray:
        chosen, weights = self.route(x)
        out = np.zeros_like(x)
        hits = [0] * self.num_experts

        for expert_id, expert in enumerate(self.experts):
            rows, slots = np.nonzero(chosen == expert_id)
            if rows.size == 0:
                continue
            hits[expert_id] = int(rows.size)
            contribution = expert.forward(x[rows])
            out[rows] += contribution * weights[rows, slots][:, None]

        if stats is not None:
            stats.tokens_routed += x.shape[0]
            stats.expert_hits = [a + b for a, b in zip(
                stats.expert_hits or [0] * self.num_experts, hits)]
        return out


def make_experts_identical(layer: MixtureOfExperts) -> None:
    first = layer.experts[0]
    for expert in layer.experts[1:]:
        expert.gate_proj = first.gate_proj.copy()
        expert.up_proj = first.up_proj.copy()
        expert.down_proj = first.down_proj.copy()
