from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ..kernels.reference import dense_attention
from ..layers.functional import rms_norm, swiglu
from .transformer import ModelConfig, Transformer


class Pooling(Enum):
    MEAN = auto()
    CLS = auto()
    LAST = auto()


@dataclass(slots=True)
class EmbeddingStats:
    documents: int = 0
    tokens: int = 0
    forward_passes: int = 0

    @property
    def tokens_per_pass(self) -> float:
        return self.tokens / self.forward_passes if self.forward_passes else 0.0


def pool(hidden: np.ndarray, strategy: Pooling, lengths: list[int]) -> np.ndarray:
    vectors = []
    start = 0
    for length in lengths:
        span = hidden[start : start + length]
        if strategy is Pooling.MEAN:
            vectors.append(span.mean(axis=0))
        elif strategy is Pooling.CLS:
            vectors.append(span[0])
        else:
            vectors.append(span[-1])
        start += length
    return np.stack(vectors)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


class EmbeddingModel:
    def __init__(self, source: Transformer, pooling: Pooling = Pooling.MEAN,
                 bidirectional: bool = True) -> None:
        self.config = source.config
        self.source = source
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.stats = EmbeddingStats()

    def encode(self, documents: list[list[int]]) -> np.ndarray:
        lengths = [len(document) for document in documents]
        tokens = [token for document in documents for token in document]
        positions = [index for length in lengths for index in range(length)]

        x = self.source.embedding[np.asarray(tokens)]
        boundaries = np.cumsum([0] + lengths)

        for block in self.source.blocks:
            normed = rms_norm(x, block.attn_norm, self.config.norm_eps)
            q = (normed @ block.q_proj).reshape(-1, self.config.num_heads, self.config.head_dim)
            k = (normed @ block.k_proj).reshape(-1, self.config.num_kv_heads, self.config.head_dim)
            v = (normed @ block.v_proj).reshape(-1, self.config.num_kv_heads, self.config.head_dim)

            from ..layers.functional import apply_rope
            positions_array = np.asarray(positions)
            q = apply_rope(q, positions_array, self.source.cos, self.source.sin)
            k = apply_rope(k, positions_array, self.source.cos, self.source.sin)

            pieces = []
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                pieces.append(dense_attention(q[start:stop], k[start:stop], v[start:stop],
                                              causal=not self.bidirectional))
            attended = np.concatenate(pieces, axis=0)

            x = x + attended.reshape(-1, self.config.num_heads * self.config.head_dim) @ block.o_proj
            normed = rms_norm(x, block.mlp_norm, self.config.norm_eps)
            x = x + swiglu(normed, block.gate_proj, block.up_proj, block.down_proj)

        hidden = rms_norm(x, self.source.final_norm, self.config.norm_eps)

        self.stats.documents += len(documents)
        self.stats.tokens += len(tokens)
        self.stats.forward_passes += 1
        return l2_normalize(pool(hidden, self.pooling, lengths))
