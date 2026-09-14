import numpy as np
import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig, Transformer
from ie.quant.formats import (
    FP8_E4M3,
    FP8_E5M2,
    INT4,
    INT8,
    INT8_ASYM,
    round_to_float_format,
)
from ie.quant.model import named_format, quantize_transformer
from ie.quant.weights import (
    PER_OUTPUT_CHANNEL,
    PER_TENSOR,
    max_absolute_error,
    quantize_float,
    quantize_integer,
    relative_error,
)

CONFIG = ModelConfig(vocab_size=128, hidden_size=64, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=16, intermediate_size=128)


def weight(shape=(64, 128), seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_integer_format_ranges():
    assert INT8.signed_max == 127
    assert INT8.signed_min == -127
    assert INT4.signed_max == 7
    assert INT8.unsigned_max == 255


def test_float_format_limits_match_the_spec():
    assert FP8_E4M3.bits == 8
    assert FP8_E4M3.max_value == pytest.approx(448.0)
    assert FP8_E5M2.max_value == pytest.approx(57344.0)


def test_rounding_to_fp8_keeps_values_representable():
    x = np.array([1.0, 1.0625, 1.125, 2.5, -3.75], dtype=np.float32)
    once = round_to_float_format(x, FP8_E4M3)
    twice = round_to_float_format(once, FP8_E4M3)
    assert np.allclose(once, twice)


def test_fp8_clamps_beyond_its_maximum():
    out = round_to_float_format(np.array([1e6, -1e6], dtype=np.float32), FP8_E4M3)
    assert np.all(np.abs(out) <= FP8_E4M3.max_value)


def test_quantize_dequantize_stays_close_to_the_original():
    w = weight()
    restored = quantize_integer(w, INT8).dequantized()
    assert relative_error(w, restored) < 0.01


def test_int8_beats_int4():
    w = weight()
    int8_error = relative_error(w, quantize_integer(w, INT8).dequantized())
    int4_error = relative_error(w, quantize_integer(w, INT4).dequantized())
    assert int8_error < int4_error


def test_per_channel_beats_per_tensor_on_uneven_columns():
    w = weight()
    w[:, 0] *= 500.0
    per_tensor = relative_error(w, quantize_integer(w, INT8, PER_TENSOR).dequantized())
    per_channel = relative_error(w, quantize_integer(w, INT8, PER_OUTPUT_CHANNEL).dequantized())
    assert per_channel < per_tensor


def test_asymmetric_helps_a_one_sided_distribution():
    w = np.abs(weight()) + 1.0
    symmetric = relative_error(w, quantize_integer(w, INT8, PER_TENSOR).dequantized())
    asymmetric = relative_error(w, quantize_integer(w, INT8_ASYM, PER_TENSOR).dequantized())
    assert asymmetric < symmetric


def test_quantization_error_is_bounded_by_half_a_step():
    w = weight(shape=(8, 4))
    packed = quantize_integer(w, INT8, PER_TENSOR)
    step = float(packed.scale.reshape(-1)[0])
    assert max_absolute_error(w, packed.dequantized()) <= step * 0.5 + 1e-6


def test_codes_stay_inside_the_representable_range():
    packed = quantize_integer(weight(), INT4)
    assert packed.codes.min() >= INT4.signed_min
    assert packed.codes.max() <= INT4.signed_max


def test_compression_tracks_the_bit_width():
    w = weight(shape=(256, 256))
    assert quantize_integer(w, INT8).compression == pytest.approx(4.0, rel=0.05)
    assert quantize_integer(w, INT4).compression == pytest.approx(8.0, rel=0.05)


def test_fp8_round_trips_better_than_int4():
    w = weight()
    fp8_error = relative_error(w, quantize_float(w, FP8_E4M3).dequantized())
    int4_error = relative_error(w, quantize_integer(w, INT4).dequantized())
    assert fp8_error < int4_error


def test_a_constant_weight_survives_quantization():
    w = np.full((4, 4), 0.25, dtype=np.float32)
    assert relative_error(w, quantize_integer(w, INT8).dequantized()) < 1e-3


def test_an_all_zero_weight_does_not_divide_by_zero():
    w = np.zeros((4, 4), dtype=np.float32)
    assert np.allclose(quantize_integer(w, INT8).dequantized(), 0.0)
    assert np.allclose(quantize_float(w, FP8_E4M3).dequantized(), 0.0)


def test_named_format_rejects_an_unknown_name():
    assert named_format("int8") is INT8
    with pytest.raises(ValueError):
        named_format("int3")


def test_quantizing_the_model_reports_compression_and_error():
    model = Transformer(CONFIG, seed=1)
    report = quantize_transformer(model, INT8)
    assert report.compression > 3.5
    assert report.mean_error < 0.02
    assert report.worst_error < 0.05


def test_int8_generation_stays_close_to_full_precision():
    prompt = [[3, 1, 4, 1, 5]]
    engine_config = EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=256)

    dense = build_engine(CONFIG, engine_config, model_seed=1)
    expected = dense.generate(prompt, max_new_tokens=8, temperature=0.0)[0]

    quantized_model = Transformer(CONFIG, seed=1)
    quantize_transformer(quantized_model, INT8)
    from ie.engine.engine import Engine
    got = Engine(quantized_model, engine_config).generate(prompt, max_new_tokens=8,
                                                          temperature=0.0)[0]

    assert sum(a == b for a, b in zip(expected, got)) >= 6


def test_lower_precision_drifts_further_from_full_precision():
    prompt = [[7, 7, 7, 7]]
    engine_config = EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=256)
    from ie.engine.engine import Engine

    reference = build_engine(CONFIG, engine_config, model_seed=2)
    baseline = reference.model.forward.__self__

    def logit_drift(fmt):
        model = Transformer(CONFIG, seed=2)
        quantize_transformer(model, fmt)
        engine = Engine(model, engine_config)
        seq = engine.submit(prompt[0], max_new_tokens=1, temperature=0.0)
        batch = engine.scheduler.schedule()
        _, logits = engine.forward(batch)
        return logits

    dense_engine = build_engine(CONFIG, engine_config, model_seed=2)
    seq = dense_engine.submit(prompt[0], max_new_tokens=1, temperature=0.0)
    batch = dense_engine.scheduler.schedule()
    _, dense_logits = dense_engine.forward(batch)

    int8_drift = relative_error(dense_logits, logit_drift(INT8))
    int4_drift = relative_error(dense_logits, logit_drift(INT4))
    assert int8_drift < int4_drift


def kl_of_first_token(reference_logits, other_logits):
    def distribution(logits):
        shifted = logits - logits.max()
        weights = np.exp(shifted)
        return weights / weights.sum()

    p = distribution(reference_logits.astype(np.float64))
    q = distribution(other_logits.astype(np.float64))
    return float(np.sum(p * np.log(p / np.clip(q, 1e-12, None))))


def first_token_logits(model):
    from ie.engine.engine import Engine
    engine = Engine(model, EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=256))
    engine.submit([3, 1, 4, 1, 5], max_new_tokens=1, temperature=0.0)
    batch = engine.scheduler.schedule()
    _, logits = engine.forward(batch)
    return logits[0]


def test_kl_divergence_ranks_precision_correctly():
    dense = first_token_logits(Transformer(CONFIG, seed=3))

    def drift(fmt):
        model = Transformer(CONFIG, seed=3)
        quantize_transformer(model, fmt)
        return kl_of_first_token(dense, first_token_logits(model))

    assert drift(INT8) < drift(FP8_E4M3) < drift(INT4)


def test_kl_of_an_unquantized_model_is_zero():
    dense = first_token_logits(Transformer(CONFIG, seed=4))
    assert kl_of_first_token(dense, dense) == pytest.approx(0.0, abs=1e-12)
