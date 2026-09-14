import numpy as np
import pytest

from ie.spec.rejection import (
    acceptance_probability,
    residual_distribution,
    verify_draft,
)

VOCAB = 6


def rng(seed=0):
    return np.random.default_rng(seed)


def random_distribution(generator, size=VOCAB):
    weights = generator.random(size) + 0.05
    return weights / weights.sum()


def test_acceptance_is_certain_when_target_likes_it_more():
    assert acceptance_probability(target_prob=0.6, draft_prob=0.3) == 1.0


def test_acceptance_is_the_ratio_when_target_likes_it_less():
    assert acceptance_probability(target_prob=0.1, draft_prob=0.4) == pytest.approx(0.25)


def test_a_token_the_draft_never_proposes_is_always_accepted():
    assert acceptance_probability(target_prob=0.2, draft_prob=0.0) == 1.0


def test_residual_is_a_distribution_concentrated_where_target_exceeds_draft():
    target = np.array([0.5, 0.3, 0.2])
    draft = np.array([0.1, 0.8, 0.1])
    residual = residual_distribution(target, draft)
    assert np.isclose(residual.sum(), 1.0)
    assert residual[1] == 0.0
    assert residual[0] > residual[2]


def test_identical_models_accept_everything():
    generator = rng(1)
    probs = random_distribution(generator)
    draft_tokens = [2, 4, 1]
    draft_probs = np.tile(probs, (3, 1))
    target_probs = np.tile(probs, (4, 1))

    result = verify_draft(draft_tokens, draft_probs, target_probs, generator)
    assert result.num_accepted == 3
    assert result.tokens[:3] == draft_tokens
    assert result.num_emitted == 4


def test_a_draft_the_target_rejects_outright_stops_at_the_first_token():
    generator = rng(2)
    target = np.zeros(VOCAB)
    target[0] = 1.0
    draft = np.zeros(VOCAB)
    draft[5] = 1.0

    result = verify_draft([5, 5], np.tile(draft, (2, 1)), np.tile(target, (3, 1)), generator)
    assert result.num_accepted == 0
    assert result.tokens == [0]


def test_emitted_length_is_always_between_one_and_k_plus_one():
    generator = rng(3)
    for _ in range(200):
        k = int(generator.integers(1, 5))
        draft_probs = np.stack([random_distribution(generator) for _ in range(k)])
        target_probs = np.stack([random_distribution(generator) for _ in range(k + 1)])
        tokens = [int(generator.integers(0, VOCAB)) for _ in range(k)]
        result = verify_draft(tokens, draft_probs, target_probs, generator)
        assert 1 <= result.num_emitted <= k + 1
        assert result.num_emitted == result.num_accepted + 1


def test_output_distribution_matches_the_target_exactly():
    generator = rng(4)
    target = np.array([0.45, 0.25, 0.15, 0.10, 0.03, 0.02])
    draft = np.array([0.05, 0.10, 0.20, 0.25, 0.20, 0.20])

    trials = 20_000
    proposals = generator.choice(VOCAB, size=trials, p=draft)
    stacked_target = np.tile(target, (2, 1))
    counts = np.zeros(VOCAB)
    for token in proposals:
        result = verify_draft([int(token)], draft[None, :], stacked_target, generator)
        counts[result.tokens[0]] += 1

    empirical = counts / trials
    assert np.max(np.abs(empirical - target)) < 0.01


def test_output_distribution_holds_even_for_a_hostile_draft():
    generator = rng(5)
    target = np.array([0.7, 0.2, 0.05, 0.03, 0.01, 0.01])
    draft = np.array([0.01, 0.01, 0.03, 0.05, 0.2, 0.7])

    trials = 20_000
    proposals = generator.choice(VOCAB, size=trials, p=draft)
    stacked_target = np.tile(target, (2, 1))
    counts = np.zeros(VOCAB)
    for token in proposals:
        result = verify_draft([int(token)], draft[None, :], stacked_target, generator)
        counts[result.tokens[0]] += 1

    assert np.max(np.abs(counts / trials - target)) < 0.01


def test_acceptance_rate_rises_as_the_draft_approaches_the_target():
    target = np.array([0.5, 0.2, 0.15, 0.1, 0.03, 0.02])

    def measured_rate(draft, seed):
        generator = rng(seed)
        trials = 4000
        proposals = generator.choice(VOCAB, size=trials, p=draft)
        stacked_target = np.tile(target, (2, 1))
        accepted = sum(verify_draft([int(t)], draft[None, :], stacked_target, generator).num_accepted
                       for t in proposals)
        return accepted / trials

    aligned = measured_rate(target.copy(), seed=6)
    blurred = measured_rate(np.full(VOCAB, 1.0 / VOCAB), seed=7)
    opposed = measured_rate(target[::-1].copy(), seed=8)

    assert aligned == 1.0
    assert opposed < blurred < aligned


def test_verify_rejects_mismatched_shapes():
    generator = rng(9)
    probs = np.tile(random_distribution(generator), (2, 1))
    with pytest.raises(ValueError):
        verify_draft([1, 2], probs, probs, generator)
    with pytest.raises(ValueError):
        verify_draft([1, 2, 3], probs, np.tile(probs[0], (4, 1)), generator)


def test_speculation_with_the_target_as_its_own_draft_accepts_everything():
    from ie.engine.engine import EngineConfig, build_engine
    from ie.model.transformer import ModelConfig, Transformer
    from ie.spec.speculator import DraftModel, SpeculativeEngine

    config = ModelConfig(vocab_size=32, hidden_size=32, num_layers=2, num_heads=4,
                         num_kv_heads=2, head_dim=8, intermediate_size=64)
    target = build_engine(config, EngineConfig(block_size=8, num_blocks=128), model_seed=1)
    draft = DraftModel(Transformer(config, seed=1), block_size=8, num_blocks=128)
    spec = SpeculativeEngine(target, draft, lookahead=4)

    out = spec.generate([1, 2, 3, 4], max_new_tokens=12, temperature=1.0, seed=3)
    assert len(out) == 12
    assert spec.stats.acceptance_rate > 0.95
    assert spec.stats.tokens_per_target_pass >= 4.0


def test_a_mismatched_draft_still_produces_the_right_number_of_tokens():
    from ie.engine.engine import EngineConfig, build_engine
    from ie.model.transformer import ModelConfig, Transformer
    from ie.spec.speculator import DraftModel, SpeculativeEngine

    config = ModelConfig(vocab_size=32, hidden_size=32, num_layers=2, num_heads=4,
                         num_kv_heads=2, head_dim=8, intermediate_size=64)
    target = build_engine(config, EngineConfig(block_size=8, num_blocks=128), model_seed=1)
    draft = DraftModel(Transformer(config, seed=99), block_size=8, num_blocks=128)
    spec = SpeculativeEngine(target, draft, lookahead=4)

    out = spec.generate([1, 2, 3, 4], max_new_tokens=10, temperature=1.0, seed=4)
    assert len(out) == 10
    assert spec.stats.acceptance_rate < 0.95
    assert spec.stats.tokens_per_target_pass >= 1.0


def test_lookahead_never_overshoots_the_token_budget():
    from ie.engine.engine import EngineConfig, build_engine
    from ie.model.transformer import ModelConfig, Transformer
    from ie.spec.speculator import DraftModel, SpeculativeEngine

    config = ModelConfig(vocab_size=32, hidden_size=32, num_layers=1, num_heads=2,
                         num_kv_heads=1, head_dim=8, intermediate_size=32)
    target = build_engine(config, EngineConfig(block_size=8, num_blocks=128), model_seed=2)
    draft = DraftModel(Transformer(config, seed=2), block_size=8, num_blocks=128)
    spec = SpeculativeEngine(target, draft, lookahead=8)

    assert len(spec.generate([5, 6], max_new_tokens=3, temperature=1.0, seed=5)) == 3
