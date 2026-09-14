from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formats import FloatFormat, IntegerFormat, round_to_float_format

PER_TENSOR = None
PER_OUTPUT_CHANNEL = 0


@dataclass(slots=True)
class QuantizedWeight:
    codes: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray | None
    format_name: str
    bits: int
    original_shape: tuple[int, ...]

    def dequantized(self) -> np.ndarray:
        codes = self.codes.astype(np.float32)
        if self.zero_point is not None:
            codes = codes - self.zero_point
        return (codes * self.scale).astype(np.float32)

    @property
    def stored_bytes(self) -> float:
        payload = self.codes.size * self.bits / 8
        overhead = self.scale.size * 4
        if self.zero_point is not None:
            overhead += self.zero_point.size * 4
        return payload + overhead

    @property
    def dense_bytes(self) -> int:
        return int(np.prod(self.original_shape)) * 4

    @property
    def compression(self) -> float:
        return self.dense_bytes / self.stored_bytes


def _reduce_axes(weight: np.ndarray, axis: int | None) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(weight.ndim))
    return tuple(d for d in range(weight.ndim) if d != axis)


def quantize_integer(weight: np.ndarray, fmt: IntegerFormat,
                     axis: int | None = PER_OUTPUT_CHANNEL) -> QuantizedWeight:
    axes = _reduce_axes(weight, axis)

    if fmt.symmetric:
        peak = np.max(np.abs(weight), axis=axes, keepdims=True)
        scale = np.where(peak == 0, 1.0, peak / fmt.signed_max).astype(np.float32)
        codes = np.clip(np.round(weight / scale), fmt.signed_min, fmt.signed_max)
        zero_point = None
    else:
        low = np.min(weight, axis=axes, keepdims=True)
        high = np.max(weight, axis=axes, keepdims=True)
        span = np.where(high - low == 0, 1.0, high - low)
        scale = (span / fmt.unsigned_max).astype(np.float32)
        zero_point = np.round(-low / scale).astype(np.float32)
        codes = np.clip(np.round(weight / scale + zero_point), 0, fmt.unsigned_max)

    return QuantizedWeight(codes=codes.astype(np.int32), scale=scale, zero_point=zero_point,
                           format_name=f"{fmt.name}/{'tensor' if axis is None else 'channel'}",
                           bits=fmt.bits, original_shape=weight.shape)


def quantize_float(weight: np.ndarray, fmt: FloatFormat,
                   axis: int | None = PER_OUTPUT_CHANNEL) -> QuantizedWeight:
    axes = _reduce_axes(weight, axis)
    peak = np.max(np.abs(weight), axis=axes, keepdims=True)
    scale = np.where(peak == 0, 1.0, peak / fmt.max_value).astype(np.float32)
    codes = round_to_float_format(weight / scale, fmt)
    return QuantizedWeight(codes=codes.astype(np.float32), scale=scale, zero_point=None,
                           format_name=f"{fmt.name}/{'tensor' if axis is None else 'channel'}",
                           bits=fmt.bits, original_shape=weight.shape)


def relative_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    denominator = np.linalg.norm(original)
    if denominator == 0:
        return 0.0
    return float(np.linalg.norm(original - reconstructed) / denominator)


def max_absolute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.max(np.abs(original - reconstructed)))
