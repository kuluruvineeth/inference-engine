from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class IntegerFormat:
    bits: int
    symmetric: bool = True

    @property
    def name(self) -> str:
        return f"int{self.bits}{'' if self.symmetric else '-asym'}"

    @property
    def num_levels(self) -> int:
        return 1 << self.bits

    @property
    def signed_max(self) -> int:
        return (self.num_levels // 2) - 1

    @property
    def signed_min(self) -> int:
        return -self.signed_max

    @property
    def unsigned_max(self) -> int:
        return self.num_levels - 1


@dataclass(frozen=True, slots=True)
class FloatFormat:
    exponent_bits: int
    mantissa_bits: int
    reserves_infinity: bool = True

    @property
    def name(self) -> str:
        return f"fp{1 + self.exponent_bits + self.mantissa_bits}-e{self.exponent_bits}m{self.mantissa_bits}"

    @property
    def bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def exponent_bias(self) -> int:
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def max_value(self) -> float:
        if self.reserves_infinity:
            largest_exponent = (1 << self.exponent_bits) - 2 - self.exponent_bias
            mantissa_scale = 2.0 - 2.0 ** (-self.mantissa_bits)
        else:
            largest_exponent = (1 << self.exponent_bits) - 1 - self.exponent_bias
            mantissa_scale = 2.0 - 2.0 ** (1 - self.mantissa_bits)
        return float(2.0**largest_exponent * mantissa_scale)

    @property
    def min_normal(self) -> float:
        return float(2.0 ** (1 - self.exponent_bias))


INT8 = IntegerFormat(bits=8)
INT4 = IntegerFormat(bits=4)
INT8_ASYM = IntegerFormat(bits=8, symmetric=False)
FP8_E4M3 = FloatFormat(exponent_bits=4, mantissa_bits=3, reserves_infinity=False)
FP8_E5M2 = FloatFormat(exponent_bits=5, mantissa_bits=2)


def round_to_float_format(x: np.ndarray, fmt: FloatFormat) -> np.ndarray:
    mantissa, exponent = np.frexp(np.clip(x, -fmt.max_value, fmt.max_value))
    quantum = 1 << fmt.mantissa_bits
    out = np.ldexp(np.round(mantissa * quantum) / quantum, exponent)
    out = np.clip(out, -fmt.max_value, fmt.max_value)
    return np.where(np.abs(out) < fmt.min_normal * 2.0 ** -fmt.mantissa_bits, 0.0, out)
