from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model.transformer import Transformer
from .formats import FP8_E4M3, INT4, INT8, FloatFormat, IntegerFormat
from .weights import PER_OUTPUT_CHANNEL, quantize_float, quantize_integer, relative_error

LINEAR_ATTRIBUTES = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")


@dataclass(slots=True)
class QuantizationReport:
    scheme: str
    dense_bytes: int = 0
    stored_bytes: float = 0.0
    errors: list[float] = field(default_factory=list)

    @property
    def compression(self) -> float:
        return self.dense_bytes / self.stored_bytes if self.stored_bytes else 0.0

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors else 0.0

    @property
    def worst_error(self) -> float:
        return float(np.max(self.errors)) if self.errors else 0.0


def quantize_tensor(weight: np.ndarray, fmt: IntegerFormat | FloatFormat,
                    axis: int | None = PER_OUTPUT_CHANNEL):
    if isinstance(fmt, IntegerFormat):
        return quantize_integer(weight, fmt, axis)
    return quantize_float(weight, fmt, axis)


def quantize_transformer(model: Transformer, fmt: IntegerFormat | FloatFormat,
                         axis: int | None = PER_OUTPUT_CHANNEL,
                         include_lm_head: bool = False) -> QuantizationReport:
    report = QuantizationReport(scheme=f"{fmt.name}/{'tensor' if axis is None else 'channel'}")

    targets = [(block, name) for block in model.blocks for name in LINEAR_ATTRIBUTES]
    for owner, name in targets:
        original = getattr(owner, name)
        packed = quantize_tensor(original, fmt, axis)
        restored = packed.dequantized()
        setattr(owner, name, restored)

        report.dense_bytes += packed.dense_bytes
        report.stored_bytes += packed.stored_bytes
        report.errors.append(relative_error(original, restored))

    if include_lm_head:
        packed = quantize_tensor(model.lm_head, fmt, axis)
        report.errors.append(relative_error(model.lm_head, packed.dequantized()))
        report.dense_bytes += packed.dense_bytes
        report.stored_bytes += packed.stored_bytes
        model.lm_head = packed.dequantized()

    return report


def named_format(name: str) -> IntegerFormat | FloatFormat:
    table = {"int8": INT8, "int4": INT4, "fp8": FP8_E4M3}
    if name not in table:
        raise ValueError(f"unknown format {name!r}; expected one of {sorted(table)}")
    return table[name]
