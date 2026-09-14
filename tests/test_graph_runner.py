import pytest

torch = pytest.importorskip("torch")

from ie.core.layout import BatchLayout
from ie.model.graph_runner import CudaGraphRunner, GraphStats, bucket_sizes


def test_buckets_start_dense_then_step_by_sixteen():
    assert bucket_sizes(8) == [1, 2, 4, 8]
    assert bucket_sizes(64)[:5] == [1, 2, 4, 8, 16]
    assert bucket_sizes(64)[-1] == 64


def test_buckets_always_include_the_ceiling():
    for ceiling in (3, 20, 100):
        assert bucket_sizes(ceiling)[-1] == ceiling


def test_buckets_never_exceed_the_ceiling():
    assert all(size <= 40 for size in bucket_sizes(40))


def test_replay_rate_counts_fallbacks():
    stats = GraphStats(replays=3, eager_fallbacks=1)
    assert stats.replay_rate == pytest.approx(0.75)
    assert GraphStats().replay_rate == 0.0


def decode_layout(batch, width=4):
    return BatchLayout(token_ids=[1] * batch, positions=[5] * batch,
                       slot_mapping=[0] * batch, context_lens=[6] * batch,
                       query_lens=[1] * batch,
                       block_tables=[[0] * width for _ in range(batch)],
                       is_prefill=False)


class FakeModel:
    def __init__(self):
        from ie.model.transformer import ModelConfig
        self.config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1,
                                  num_heads=2, num_kv_heads=1, head_dim=8,
                                  intermediate_size=32)
        self.device = torch.device("cpu")
        self.dtype = torch.float32


def test_an_uncaptured_runner_never_claims_it_can_replay():
    runner = CudaGraphRunner(FakeModel(), block_size=16, max_batch=8, max_blocks=4)
    assert not runner.is_captured
    assert not runner.can_replay(decode_layout(2))


def test_prefill_layouts_are_never_replayed():
    runner = CudaGraphRunner(FakeModel(), block_size=16, max_batch=8, max_blocks=4)
    runner.graphs = {1: None, 2: None}
    layout = decode_layout(2)
    layout.is_prefill = True
    assert not runner.can_replay(layout)


def test_multi_token_queries_are_never_replayed():
    runner = CudaGraphRunner(FakeModel(), block_size=16, max_batch=8, max_blocks=4)
    runner.graphs = {4: None}
    layout = decode_layout(2)
    layout.query_lens = [2, 1]
    assert not runner.can_replay(layout)


def test_a_batch_rounds_up_to_the_next_bucket():
    runner = CudaGraphRunner(FakeModel(), block_size=16, max_batch=32, max_blocks=4)
    runner.graphs = {size: None for size in bucket_sizes(32)}
    assert runner.bucket_for(1) == 1
    assert runner.bucket_for(3) == 4
    assert runner.bucket_for(17) == 32


def test_a_batch_beyond_every_bucket_has_no_graph():
    runner = CudaGraphRunner(FakeModel(), block_size=16, max_batch=8, max_blocks=4)
    runner.graphs = {1: None, 2: None}
    assert runner.bucket_for(64) is None
    assert not runner.can_replay(decode_layout(64))
