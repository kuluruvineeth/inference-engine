import time

import numpy as np

from ie.engine.engine import EngineConfig, build_engine
from ie.layers.functional import sample_from_logits
from ie.model.transformer import ModelConfig, Transformer
from ie.serve.disaggregated import DisaggregatedEngine

VOCAB = 4096
CONFIG = ModelConfig(vocab_size=VOCAB, hidden_size=128, num_layers=4, num_heads=8,
                     num_kv_heads=2, head_dim=16, intermediate_size=256)
ENGINE = EngineConfig(block_size=16, num_blocks=4096, max_batch_tokens=4096)

CHATTERS = [list(range(100 + i * 40, 100 + i * 40 + 32)) for i in range(3)]
LATE_ARRIVAL = list(range(3000, 3000 + 1024))
DECODE_STEPS = 14
ARRIVES_AT = 5


def step(engine):
    batch = engine.scheduler.schedule()
    if not batch:
        return None
    _, logits = engine.forward(batch)
    temps = np.array([s.temperature for s in batch.sequences], dtype=np.float32)
    engine.scheduler.finish_step(batch, sample_from_logits(logits, temps, engine.generator).tolist())
    return batch


def colocated_gaps():
    engine = build_engine(CONFIG, ENGINE, model_seed=1)
    for prompt in CHATTERS:
        engine.submit(prompt, max_new_tokens=64, temperature=0.0)

    gaps, decodes, last = [], 0, None
    for index in range(DECODE_STEPS + 4):
        if index == ARRIVES_AT:
            engine.submit(LATE_ARRIVAL, max_new_tokens=2, temperature=0.0)
        started = time.perf_counter()
        batch = step(engine)
        if batch is None:
            break
        if not batch.is_prefill:
            if last is not None:
                gaps.append(time.perf_counter() - last)
            decodes += 1
        last = time.perf_counter()
        if decodes >= DECODE_STEPS:
            break
    return gaps


def disaggregated_gaps():
    engine = DisaggregatedEngine(Transformer(CONFIG, seed=1), ENGINE)
    for prompt in CHATTERS:
        engine.decode.admit(engine.prefill.prefill(prompt), max_new_tokens=64)

    gaps, last = [], None
    for index in range(DECODE_STEPS):
        if index == ARRIVES_AT:
            transfer = engine.prefill.prefill(LATE_ARRIVAL)
            engine.stats.blocks_transferred += transfer.num_blocks
            engine.stats.bytes_transferred += transfer.bytes_moved
            last = time.perf_counter()
        started = time.perf_counter()
        if step(engine.decode.engine) is None:
            break
        if last is not None:
            gaps.append(time.perf_counter() - last)
        last = time.perf_counter()
    return gaps, engine.stats


def summarize(name, gaps):
    ordered = sorted(gaps)
    print(f"{name:<20}{len(gaps):>9}{ordered[len(ordered)//2]*1000:>13.1f}"
          f"{max(gaps)*1000:>13.1f}{sum(gaps)*1000:>13.1f}")


print(f"a {len(LATE_ARRIVAL)}-token prompt arrives at decode step {ARRIVES_AT}")
print("the disaggregated prefill is treated as off-box, so its cost is not charged")
print("to the decode worker's gaps -- that is the whole point of separating them\n")
print(f"{'setup':<20}{'gaps':>9}{'itl p50 ms':>13}{'worst ms':>13}{'total ms':>13}")
print("-" * 68)
summarize("colocated", colocated_gaps())
gaps, stats = disaggregated_gaps()
summarize("disaggregated", gaps)
print(f"\ntransferred {stats.blocks_transferred} blocks = {stats.bytes_transferred/1024:.0f} KiB")
