import time

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig

CONFIG = ModelConfig(vocab_size=512, hidden_size=128, num_layers=4, num_heads=8,
                     num_kv_heads=2, head_dim=16, intermediate_size=256)
SHARED = list(range(100, 164))
PROMPTS = [SHARED + [i] for i in range(8)]
NEW_TOKENS = 12


def run(label, **engine_kw):
    eng = build_engine(CONFIG, EngineConfig(**engine_kw))
    started = time.perf_counter()
    out = eng.generate(PROMPTS, max_new_tokens=NEW_TOKENS, temperature=0.0)
    elapsed = time.perf_counter() - started
    stats = eng.scheduler.stats
    print(f"{label:<34} steps {stats.steps:>4}  computed {stats.tokens_prefilled + stats.tokens_decoded:>5}"
          f"  reused {stats.tokens_reused_from_cache:>5}  preempt {stats.preemptions:>3}"
          f"  {elapsed * 1000:>6.0f} ms")
    return out


baseline = run("roomy, whole prefill", block_size=16, num_blocks=2048, max_batch_tokens=4096)
chunked = run("prefill chunked to 24 tokens", block_size=16, num_blocks=2048, max_batch_tokens=24)
tight = run("pool squeezed to 40 blocks", block_size=16, num_blocks=40, max_batch_tokens=4096)
small_blocks = run("block size 8", block_size=8, num_blocks=4096, max_batch_tokens=4096)

print()
print(f"chunked prefill  == baseline : {chunked == baseline}")
print(f"under preemption == baseline : {tight == baseline}")
print(f"different block size == baseline : {small_blocks == baseline}")
