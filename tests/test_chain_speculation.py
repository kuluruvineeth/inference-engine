import numpy as np
import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig, Transformer
from ie.spec.chain_engine import ChainSpeculativeEngine
from ie.spec.eagle import EagleDrafter, EagleHead, teach_head_to_copy
from ie.spec.ngram import NgramProposer

CONFIG = ModelConfig(vocab_size=128, hidden_size=64, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=16, intermediate_size=128)
ENGINE = EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=256)
PROMPT = [3, 1, 4, 1, 5]


def baseline(tokens=14, prompt=None):
    return build_engine(CONFIG, ENGINE, model_seed=1).generate(
        [prompt or PROMPT], max_new_tokens=tokens, temperature=0.0)[0]


def ngram_engine(lookahead=4, max_context=3):
    proposer = NgramProposer(max_context=max_context, lookahead=lookahead)
    return ChainSpeculativeEngine(Transformer(CONFIG, seed=1),
                                  lambda toks, hidden, k: proposer.propose(toks)[:k],
                                  ENGINE, lookahead=lookahead)


def eagle_engine(strength, lookahead=4, head_seed=2):
    model = Transformer(CONFIG, seed=1)
    head = EagleHead(CONFIG, seed=head_seed)
    teach_head_to_copy(head, strength)
    drafter = EagleDrafter(model, head)
    return ChainSpeculativeEngine(model,
                                  lambda toks, hidden, k: drafter.draft(hidden, toks[-1], k).tokens,
                                  ENGINE, lookahead=lookahead)


def test_ngram_repeats_what_followed_the_last_match():
    proposer = NgramProposer(max_context=2, lookahead=3)
    assert proposer.propose([1, 2, 3, 4, 1, 2]) == [3, 4, 1]


def test_ngram_prefers_the_longest_context():
    proposer = NgramProposer(max_context=3, lookahead=2)
    assert proposer.propose([9, 1, 2, 3, 7, 1, 2, 3]) == [7, 1]


def test_ngram_returns_nothing_without_a_match():
    proposer = NgramProposer(max_context=3, lookahead=4)
    assert proposer.propose([1, 2, 3, 4, 5]) == []


def test_ngram_needs_a_sane_configuration():
    with pytest.raises(ValueError):
        NgramProposer(min_context=0)
    with pytest.raises(ValueError):
        NgramProposer(max_context=1, min_context=2)
    with pytest.raises(ValueError):
        NgramProposer(lookahead=0)


def test_ngram_shines_on_repetitive_text():
    proposer = NgramProposer(max_context=4, lookahead=6)
    assert proposer.propose([7, 8, 9, 10] * 5) == [7, 8, 9, 10]


def test_ngram_continuation_stops_at_the_end_of_what_was_written():
    proposer = NgramProposer(max_context=2, lookahead=8)
    assert proposer.propose([1, 2, 3, 4, 5, 1, 2]) == [3, 4, 5, 1, 2]


def test_eagle_head_returns_a_feature_of_the_right_shape():
    head = EagleHead(CONFIG, seed=0)
    feature = np.zeros(CONFIG.hidden_size, dtype=np.float32)
    embedding = np.zeros(CONFIG.hidden_size, dtype=np.float32)
    assert head.next_feature(feature, embedding).shape == (CONFIG.hidden_size,)


def test_eagle_drafts_the_requested_number_of_tokens():
    model = Transformer(CONFIG, seed=1)
    drafter = EagleDrafter(model, EagleHead(CONFIG, seed=0))
    feature = np.random.default_rng(0).standard_normal(CONFIG.hidden_size).astype(np.float32)
    draft = drafter.draft(feature, last_token=5, num_tokens=4)
    assert len(draft.tokens) == 4
    assert draft.features.shape == (4, CONFIG.hidden_size)
    assert all(0 <= t < CONFIG.vocab_size for t in draft.tokens)


@pytest.mark.parametrize("lookahead", [1, 2, 4, 6])
def test_ngram_speculation_reproduces_greedy_decoding(lookahead):
    assert ngram_engine(lookahead).generate(PROMPT, max_new_tokens=14) == baseline()


@pytest.mark.parametrize("strength", [0.0, 0.5, 0.9])
def test_eagle_speculation_reproduces_greedy_decoding(strength):
    assert eagle_engine(strength).generate(PROMPT, max_new_tokens=14) == baseline()


def test_a_better_eagle_head_needs_fewer_target_passes():
    weak = eagle_engine(0.0)
    weak.generate(PROMPT, max_new_tokens=16)
    strong = eagle_engine(0.9)
    strong.generate(PROMPT, max_new_tokens=16)

    assert strong.stats.target_passes < weak.stats.target_passes
    assert strong.stats.acceptance_rate > weak.stats.acceptance_rate


def test_chain_speculation_returns_its_blocks():
    engine = ngram_engine()
    engine.generate(PROMPT, max_new_tokens=12)
    assert engine.blocks.num_free == engine.blocks.num_total


def test_emitted_always_exceeds_accepted_by_one():
    engine = eagle_engine(0.5)
    engine.generate(PROMPT, max_new_tokens=16)
    assert engine.stats.emitted >= engine.stats.accepted + engine.stats.steps - 1
