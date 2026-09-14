import numpy as np

from ie.engine.engine import Engine, EngineConfig, build_engine
from ie.model.transformer import ModelConfig, Transformer
from ie.quant.formats import FP8_E4M3, FP8_E5M2, INT4, INT8, INT8_ASYM
from ie.quant.model import quantize_transformer
from ie.quant.weights import PER_OUTPUT_CHANNEL, PER_TENSOR, quantize_integer, relative_error

CONFIG = ModelConfig(vocab_size=512, hidden_size=128, num_layers=4, num_heads=8,
                     num_kv_heads=2, head_dim=16, intermediate_size=256)
ENGINE = EngineConfig(block_size=16, num_blocks=512, max_batch_tokens=512)
PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]
NEW_TOKENS = 16

def first_token_logits(model):
    engine = Engine(model, ENGINE)
    engine.submit(PROMPT, max_new_tokens=1, temperature=0.0)
    batch = engine.scheduler.schedule()
    _, logits = engine.forward(batch)
    return logits[0].astype(np.float64)


def distribution(logits):
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def kl_divergence(reference_logits, other_logits):
    p = distribution(reference_logits)
    q = distribution(other_logits)
    return float(np.sum(p * np.log(p / np.clip(q, 1e-12, None))))


dense = build_engine(CONFIG, ENGINE, model_seed=1)
reference = dense.generate([PROMPT], max_new_tokens=NEW_TOKENS, temperature=0.0)[0]
dense_logits = first_token_logits(Transformer(CONFIG, seed=1))

print(f"{'scheme':<22}{'bits':>5}{'compress':>10}{'weight err':>12}"
      f"{'logit drift':>13}{'KL':>10}{'tokens kept':>13}")
print("-" * 85)

for label, fmt, axis in [
    ("int8 per-channel", INT8, PER_OUTPUT_CHANNEL),
    ("int8 per-tensor", INT8, PER_TENSOR),
    ("int8 asymmetric", INT8_ASYM, PER_TENSOR),
    ("fp8 e4m3", FP8_E4M3, PER_OUTPUT_CHANNEL),
    ("fp8 e5m2", FP8_E5M2, PER_OUTPUT_CHANNEL),
    ("int4 per-channel", INT4, PER_OUTPUT_CHANNEL),
]:
    model = Transformer(CONFIG, seed=1)
    report = quantize_transformer(model, fmt, axis)
    logits = first_token_logits(model)
    got = Engine(model, ENGINE).generate([PROMPT], max_new_tokens=NEW_TOKENS,
                                         temperature=0.0)[0]
    kept = sum(a == b for a, b in zip(reference, got))
    print(f"{label:<22}{fmt.bits:>5}{report.compression:>9.2f}x{report.mean_error:>12.4f}"
          f"{relative_error(dense_logits, logits):>13.4f}"
          f"{kl_divergence(dense_logits, logits):>10.4f}{kept:>9}/{NEW_TOKENS}")

print("\ntokens-kept cascades: one flipped token diverges the rest, so logit drift\n"
      "and KL are the metrics that actually rank the schemes")

print("\nper-channel vs per-tensor on a weight with one outlier column")
w = np.random.default_rng(0).standard_normal((128, 256)).astype(np.float32)
w[:, 0] *= 400.0
for label, axis in [("per-tensor", PER_TENSOR), ("per-channel", PER_OUTPUT_CHANNEL)]:
    err = relative_error(w, quantize_integer(w, INT8, axis).dequantized())
    print(f"  int8 {label:<14}{err:.5f}")
