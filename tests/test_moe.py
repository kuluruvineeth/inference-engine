import numpy as np
import pytest

from ie.model.moe import (
    MixtureOfExperts,
    RoutingStats,
    make_experts_identical,
    top_k_routing,
)
from ie.model.transformer import ModelConfig
from ie.parallel.expert_parallel import ExpertParallelMoE

CONFIG = ModelConfig(vocab_size=64, hidden_size=32, num_layers=1, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)


def tokens(count=24, seed=0):
    return np.random.default_rng(seed).standard_normal(
        (count, CONFIG.hidden_size)).astype(np.float32)


def test_routing_picks_the_highest_scoring_experts():
    logits = np.array([[0.1, 5.0, 0.2, 3.0]], dtype=np.float32)
    chosen, weights = top_k_routing(logits, top_k=2)
    assert list(chosen[0]) == [1, 3]
    assert np.isclose(weights.sum(), 1.0)


def test_routing_weights_sum_to_one_per_token():
    logits = np.random.default_rng(1).standard_normal((10, 8)).astype(np.float32)
    _, weights = top_k_routing(logits, top_k=3)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_routing_rejects_an_impossible_k():
    logits = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        top_k_routing(logits, top_k=0)
    with pytest.raises(ValueError):
        top_k_routing(logits, top_k=5)


def test_identical_experts_make_the_layer_behave_like_one_dense_mlp():
    layer = MixtureOfExperts(CONFIG, num_experts=4, top_k=2, seed=0)
    make_experts_identical(layer)
    x = tokens()

    out = layer.forward(x)
    dense = layer.experts[0].forward(x)
    assert np.allclose(out, dense, atol=1e-5)


def test_only_the_selected_experts_contribute():
    layer = MixtureOfExperts(CONFIG, num_experts=4, top_k=1, seed=1)
    x = tokens(count=1)
    chosen, _ = layer.route(x)
    winner = int(chosen[0, 0])

    out = layer.forward(x)
    assert np.allclose(out, layer.experts[winner].forward(x), atol=1e-5)


def test_every_token_reaches_exactly_top_k_experts():
    layer = MixtureOfExperts(CONFIG, num_experts=8, top_k=2, seed=2)
    stats = RoutingStats()
    x = tokens(count=40)
    layer.forward(x, stats)
    assert sum(stats.expert_hits) == 40 * 2
    assert stats.tokens_routed == 40


def test_active_fraction_reports_the_sparsity():
    layer = MixtureOfExperts(CONFIG, num_experts=8, top_k=2, seed=3)
    assert layer.active_fraction == pytest.approx(0.25)


def test_imbalance_is_one_when_load_is_even():
    stats = RoutingStats(tokens_routed=8, expert_hits=[4, 4, 4, 4])
    assert stats.imbalance == pytest.approx(1.0)
    assert stats.idle_experts == 0


def test_imbalance_grows_when_one_expert_dominates():
    stats = RoutingStats(tokens_routed=8, expert_hits=[16, 0, 0, 0])
    assert stats.imbalance == pytest.approx(4.0)
    assert stats.idle_experts == 3


def test_a_layer_needs_at_least_one_expert():
    with pytest.raises(ValueError):
        MixtureOfExperts(CONFIG, num_experts=0)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_expert_parallel_matches_the_single_device_layer(world_size):
    layer = MixtureOfExperts(CONFIG, num_experts=8, top_k=2, seed=4)
    x = tokens(count=32)

    expected = layer.forward(x)
    actual = ExpertParallelMoE(layer, world_size).forward(x)
    assert np.allclose(actual, expected, atol=1e-5)


def test_each_rank_owns_a_disjoint_slice_of_experts():
    layer = MixtureOfExperts(CONFIG, num_experts=8, top_k=2, seed=5)
    parallel = ExpertParallelMoE(layer, world_size=4)

    owned = [set(shard.owned) for shard in parallel.shards]
    assert parallel.experts_per_rank == 2
    assert set.union(*owned) == set(range(8))
    for a in range(4):
        for b in range(a + 1, 4):
            assert owned[a].isdisjoint(owned[b])


def test_expert_parallel_rejects_an_uneven_split():
    layer = MixtureOfExperts(CONFIG, num_experts=6, top_k=2, seed=6)
    with pytest.raises(ValueError):
        ExpertParallelMoE(layer, world_size=4)


def test_token_load_is_reported_per_rank():
    layer = MixtureOfExperts(CONFIG, num_experts=4, top_k=2, seed=7)
    parallel = ExpertParallelMoE(layer, world_size=2)
    x = tokens(count=30)
    per_rank = parallel.tokens_per_rank(x)
    assert sum(per_rank) == 30 * 2
    assert len(per_rank) == 2
