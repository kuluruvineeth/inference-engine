from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Verification:
    tokens: list[int]
    num_accepted: int

    @property
    def num_emitted(self) -> int:
        return len(self.tokens)


def normalize(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0:
        return np.full_like(weights, 1.0 / weights.size)
    return weights / total


def acceptance_probability(target_prob: float, draft_prob: float) -> float:
    if draft_prob <= 0.0:
        return 1.0 if target_prob > 0.0 else 0.0
    return min(1.0, target_prob / draft_prob)


def residual_distribution(target: np.ndarray, draft: np.ndarray) -> np.ndarray:
    return normalize(np.maximum(target - draft, 0.0))


def draw(distribution: np.ndarray, generator: np.random.Generator) -> int:
    thresholds = np.cumsum(normalize(distribution))
    return int(np.searchsorted(thresholds, generator.random() * thresholds[-1]))


def verify_draft(draft_tokens: list[int], draft_probs: np.ndarray, target_probs: np.ndarray,
                 generator: np.random.Generator) -> Verification:
    num_draft = len(draft_tokens)
    if draft_probs.shape[0] != num_draft:
        raise ValueError(f"{draft_probs.shape[0]} draft distributions for {num_draft} tokens")
    if target_probs.shape[0] != num_draft + 1:
        raise ValueError(f"target must score {num_draft + 1} positions, got {target_probs.shape[0]}")

    emitted: list[int] = []

    for step, token in enumerate(draft_tokens):
        target = target_probs[step]
        proposal = draft_probs[step]
        if generator.random() < acceptance_probability(target[token], proposal[token]):
            emitted.append(token)
            continue
        emitted.append(draw(residual_distribution(target, proposal), generator))
        return Verification(tokens=emitted, num_accepted=step)

    emitted.append(draw(target_probs[num_draft], generator))
    return Verification(tokens=emitted, num_accepted=num_draft)
