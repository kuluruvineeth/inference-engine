from ie.bench.harness import (
    distinct_prompts,
    offline_workload,
    poisson_workload,
    run_benchmark,
    shared_prefix_prompts,
)
from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig

VOCAB = 512
CONFIG = ModelConfig(vocab_size=VOCAB, hidden_size=128, num_layers=4, num_heads=8,
                     num_kv_heads=2, head_dim=16, intermediate_size=256)
COUNT, NEW_TOKENS = 12, 16


def engine(num_blocks=1024, max_batch_tokens=2048):
    return build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=num_blocks,
                                             max_batch_tokens=max_batch_tokens))


def show(label, result):
    s = result.summary()
    print(f"{label:<30}{s['ttft_p50']:>9.3f}{s['ttft_p99']:>9.3f}{s['itl_p50']:>9.3f}"
          f"{s['output_tok_per_s']:>11.1f}{s['engine_steps']:>8}{s['tokens_reused']:>9}"
          f"{s['preemptions']:>8}")


print(f"{'scenario':<30}{'ttft p50':>9}{'ttft p99':>9}{'itl p50':>9}"
      f"{'tok/s':>11}{'steps':>8}{'reused':>9}{'preempt':>8}")
print("-" * 84)

distinct = distinct_prompts(COUNT, prompt_len=96, vocab_size=VOCAB, seed=1)
shared = shared_prefix_prompts(COUNT, prefix_len=80, suffix_len=16, vocab_size=VOCAB, seed=1)

baseline = engine()
run_benchmark(baseline, offline_workload(distinct[:1], NEW_TOKENS))
show("distinct prompts, cache cold", run_benchmark(baseline, offline_workload(distinct[1:], NEW_TOKENS)))
cold = engine()
run_benchmark(cold, offline_workload(shared[:1], NEW_TOKENS))
show("shared prefix, cache warm", run_benchmark(cold, offline_workload(shared[1:], NEW_TOKENS)))
show("offline, chunked prefill 64",
     run_benchmark(engine(max_batch_tokens=64), offline_workload(distinct, NEW_TOKENS)))
show("offline, pool squeezed",
     run_benchmark(engine(num_blocks=90), offline_workload(distinct, NEW_TOKENS)))
show("online, 20 req/s",
     run_benchmark(engine(), poisson_workload(distinct, 20.0, NEW_TOKENS, seed=1)))
show("online, 200 req/s",
     run_benchmark(engine(), poisson_workload(distinct, 200.0, NEW_TOKENS, seed=1)))

print("\nwall-clock numbers are CPU numpy and noisy; steps, reused and preempt are exact")
