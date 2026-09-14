from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True) + eps)
    return (x / scale) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray, gate_weight: np.ndarray, up_weight: np.ndarray,
           down_weight: np.ndarray) -> np.ndarray:
    return (silu(x @ gate_weight) * (x @ up_weight)) @ down_weight


def rope_frequencies(head_dim: int, base: float = 10000.0) -> np.ndarray:
    if head_dim % 2:
        raise ValueError("rotary embeddings need an even head dimension")
    exponents = np.arange(0, head_dim, 2, dtype=np.float32) / head_dim
    return np.power(base, -exponents)


def rope_tables(max_position: int, head_dim: int, base: float = 10000.0
                ) -> tuple[np.ndarray, np.ndarray]:
    angles = np.outer(np.arange(max_position, dtype=np.float32),
                      rope_frequencies(head_dim, base))
    return np.cos(angles), np.sin(angles)


def apply_rope_interleaved(x: np.ndarray, positions: list[int] | np.ndarray,
                           cos_table: np.ndarray, sin_table: np.ndarray) -> np.ndarray:
    cos = cos_table[positions][:, None, :]
    sin = sin_table[positions][:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = np.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


def apply_rope_halved(x: np.ndarray, positions: list[int] | np.ndarray,
                      cos_table: np.ndarray, sin_table: np.ndarray) -> np.ndarray:
    cos = cos_table[positions][:, None, :]
    sin = sin_table[positions][:, None, :]
    half = x.shape[-1] // 2
    left, right = x[..., :half], x[..., half:]
    rotated = np.empty_like(x)
    rotated[..., :half] = left * cos - right * sin
    rotated[..., half:] = right * cos + left * sin
    return rotated


def apply_rope(x: np.ndarray, positions: list[int] | np.ndarray,
               cos_table: np.ndarray, sin_table: np.ndarray,
               halved: bool = False) -> np.ndarray:
    kernel = apply_rope_halved if halved else apply_rope_interleaved
    return kernel(x, positions, cos_table, sin_table)


def sample_from_logits(logits: np.ndarray, temperatures: np.ndarray,
                       generator: np.random.Generator) -> np.ndarray:
    greedy = temperatures <= 0.0
    safe = np.where(greedy, 1.0, temperatures)[:, None]
    scaled = logits.astype(np.float32) / safe
    probs = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    race = generator.exponential(1.0, size=probs.shape).clip(min=1e-10)
    sampled = np.argmax(probs / race, axis=-1)
    return np.where(greedy, np.argmax(logits, axis=-1), sampled)
