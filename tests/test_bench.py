import pytest

from ie.bench.harness import (
    distinct_prompts,
    offline_workload,
    poisson_workload,
    run_benchmark,
    shared_prefix_prompts,
)
from ie.bench.metrics import BenchmarkResult, RequestTrace, percentile
from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig

VOCAB = 256
CONFIG = ModelConfig(vocab_size=VOCAB, hidden_size=32, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)


def engine(**kw):
    return build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=512,
                                             max_batch_tokens=512, **kw))


def test_prompt_generators_stay_inside_the_vocabulary():
    for prompt in distinct_prompts(count=5, prompt_len=20, vocab_size=VOCAB, seed=9):
        assert all(0 <= token < VOCAB for token in prompt)
    for prompt in shared_prefix_prompts(count=3, prefix_len=16, suffix_len=8,
                                        vocab_size=VOCAB, seed=9):
        assert all(0 <= token < VOCAB for token in prompt)
    with pytest.raises(ValueError):
        shared_prefix_prompts(count=1, prefix_len=200, suffix_len=100,
                              vocab_size=VOCAB, seed=9)


def test_percentile_interpolates_between_samples():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([5.0], 0.99) == 5.0
    assert percentile([], 0.5) == 0.0


def test_percentiles_are_ordered():
    samples = [float(i) for i in range(100)]
    assert (percentile(samples, 0.5) <= percentile(samples, 0.9)
            <= percentile(samples, 0.99))


def test_trace_reports_ttft_and_inter_token_gaps():
    trace = RequestTrace(seq_id=0, prompt_len=4, submitted_at=0.0, arrived_at=1.0)
    trace.token_times = [2.0, 2.5, 3.5]
    trace.finished_at = 3.5

    assert trace.time_to_first_token == pytest.approx(1.0)
    assert trace.inter_token_latencies == pytest.approx([0.5, 1.0])
    assert trace.total_latency == pytest.approx(2.5)
    assert trace.num_generated == 3


def test_trace_without_tokens_has_no_ttft():
    trace = RequestTrace(seq_id=0, prompt_len=4, submitted_at=0.0, arrived_at=0.0)
    assert trace.time_to_first_token is None
    assert trace.total_latency is None


def test_offline_benchmark_finishes_every_request():
    prompts = distinct_prompts(count=6, prompt_len=24, vocab_size=VOCAB, seed=1)
    result = run_benchmark(engine(), offline_workload(prompts, max_new_tokens=8))

    assert result.num_requests == 6
    assert len(result.completed) == 6
    assert result.generated_tokens == 6 * 8
    assert result.prompt_tokens == 6 * 24


def test_every_request_reports_a_first_token_and_a_finish():
    prompts = distinct_prompts(count=4, prompt_len=16, vocab_size=VOCAB, seed=2)
    result = run_benchmark(engine(), offline_workload(prompts, max_new_tokens=5))

    for trace in result.traces:
        assert trace.time_to_first_token is not None
        assert trace.time_to_first_token >= 0.0
        assert trace.total_latency >= trace.time_to_first_token
        assert trace.num_generated == 5


def test_summary_exposes_the_serving_metrics():
    prompts = distinct_prompts(count=4, prompt_len=16, vocab_size=VOCAB, seed=3)
    summary = run_benchmark(engine(), offline_workload(prompts, max_new_tokens=6)).summary()

    for key in ("ttft_p50", "ttft_p99", "itl_p50", "latency_p50",
                "output_tok_per_s", "requests_per_s", "tokens_per_step"):
        assert key in summary
    assert summary["output_tok_per_s"] > 0
    assert summary["generated_tokens"] == 24


def test_a_shared_prefix_reduces_prefill_work():
    shared = shared_prefix_prompts(count=8, prefix_len=64, suffix_len=8, vocab_size=VOCAB, seed=4)
    distinct = distinct_prompts(count=8, prompt_len=72, vocab_size=VOCAB, seed=4)

    warm = run_benchmark(engine(), offline_workload(shared, max_new_tokens=4))
    cold = run_benchmark(engine(), offline_workload(distinct, max_new_tokens=4))

    assert warm.tokens_reused > 0
    assert cold.tokens_reused == 0


def test_poisson_arrivals_are_increasing():
    prompts = distinct_prompts(count=10, prompt_len=8, vocab_size=VOCAB, seed=5)
    workload = poisson_workload(prompts, requests_per_second=50.0, seed=5)

    arrivals = [request.arrival for request in workload.requests]
    assert arrivals == sorted(arrivals)
    assert arrivals[0] > 0.0
    assert not workload.is_offline


def test_offline_workload_has_no_arrival_delay():
    workload = offline_workload(distinct_prompts(3, 8, vocab_size=VOCAB, seed=6))
    assert workload.is_offline
    assert all(request.arrival == 0.0 for request in workload.requests)


def test_online_benchmark_still_completes_every_request():
    prompts = distinct_prompts(count=5, prompt_len=12, vocab_size=VOCAB, seed=7)
    workload = poisson_workload(prompts, requests_per_second=500.0,
                                max_new_tokens=4, seed=7)
    result = run_benchmark(engine(), workload)

    assert len(result.completed) == 5
    assert result.generated_tokens == 20


def test_a_squeezed_pool_reports_preemptions():
    prompts = distinct_prompts(count=6, prompt_len=32, vocab_size=VOCAB, seed=8)
    tight = build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=14,
                                              max_batch_tokens=512))
    result = run_benchmark(tight, offline_workload(prompts, max_new_tokens=16))

    assert len(result.completed) == 6
    assert result.preemptions > 0
