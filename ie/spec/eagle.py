from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..layers.functional import rms_norm, silu
from ..model.transformer import ModelConfig, Transformer


@dataclass(slots=True)
class EagleDraft:
    tokens: list[int]
    features: np.ndarray


class EagleHead:
    def __init__(self, config: ModelConfig, seed: int = 0) -> None:
        generator = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(2 * config.hidden_size)
        self.config = config
        self.combine = (generator.standard_normal((2 * config.hidden_size, config.hidden_size))
                        * scale).astype(np.float32)
        self.norm = np.ones(config.hidden_size, dtype=np.float32)

    def next_feature(self, feature: np.ndarray, token_embedding: np.ndarray) -> np.ndarray:
        joined = np.concatenate([feature, token_embedding], axis=-1)
        return feature + silu(joined @ self.combine)


class EagleDrafter:
    def __init__(self, model: Transformer, head: EagleHead) -> None:
        self.model = model
        self.head = head

    def draft(self, feature: np.ndarray, last_token: int, num_tokens: int) -> EagleDraft:
        tokens: list[int] = []
        features: list[np.ndarray] = []

        current_feature = feature
        current_token = last_token

        for _ in range(num_tokens):
            embedding = self.model.embedding[current_token]
            current_feature = self.head.next_feature(current_feature, embedding)
            normed = rms_norm(current_feature, self.model.final_norm, self.model.config.norm_eps)
            current_token = int(np.argmax(normed @ self.model.lm_head))
            tokens.append(current_token)
            features.append(current_feature)

        return EagleDraft(tokens=tokens, features=np.stack(features))


def teach_head_to_copy(head: EagleHead, strength: float) -> None:
    head.combine = (head.combine * (1.0 - strength)).astype(np.float32)
