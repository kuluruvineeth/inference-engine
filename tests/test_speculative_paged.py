import numpy as np
import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig, Transformer
from ie.spec.paged import PagedSpeculativeEngine

CONFIG = ModelConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)
ENGINE = EngineConfig(block_size=8, num_blocks=256, max_batch_tokens=256)
PROMPT = [1, 2, 3, 4, 5]


def speculative(lookahead, draft_seed=1):
    return PagedSpeculativeEngine(Transformer(CONFIG, seed=1),
                                  Transformer(CONFIG, seed=draft_seed),
                                  ENGINE, lookahead=lookahead)


def baseline(max_new_tokens=12):
    return build_engine(CONFIG, ENGINE, model_seed=1).generate(
        [PROMPT], max_new_tokens=max_new_tokens, temperature=0.0)[0]


@pytest.mark.parametrize("lookahead", [1, 2, 4, 6])
def test_greedy_speculation_reproduces_the_plain_engine_exactly(lookahead):
    got = speculative(lookahead).generate(PROMPT, max_new_tokens=12,
                                          temperature=0.0, seed=0)
    assert got == baseline()


def test_a_perfect_draft_accepts_everything_and_cuts_target_passes():
    spec = speculative(lookahead=6)
    spec.generate(PROMPT, max_new_tokens=12, temperature=0.0, seed=0)
    assert spec.stats.acceptance_rate == 1.0
    assert spec.stats.target_passes < 12
    assert spec.stats.tokens_per_target_pass > 3.0


def test_a_mismatched_draft_still_emits_the_requested_tokens():
    spec = speculative(lookahead=4, draft_seed=99)
    out = spec.generate(PROMPT, max_new_tokens=10, temperature=1.0, seed=2)
    assert len(out) == 10
    assert spec.stats.acceptance_rate < 1.0


def test_speculation_returns_every_block_it_borrowed():
    spec = speculative(lookahead=4)
    spec.generate(PROMPT, max_new_tokens=10, temperature=1.0, seed=1)
    assert spec.target.blocks.num_free == spec.target.blocks.num_total
    assert spec.draft.blocks.num_free == spec.draft.blocks.num_total


def test_lookahead_larger_than_the_budget_is_clamped():
    spec = speculative(lookahead=16)
    assert len(spec.generate(PROMPT, max_new_tokens=3, temperature=0.0, seed=0)) == 3


def test_draft_quality_drives_tokens_per_pass():
    good = speculative(lookahead=4, draft_seed=1)
    good.generate(PROMPT, max_new_tokens=16, temperature=1.0, seed=5)

    bad = speculative(lookahead=4, draft_seed=99)
    bad.generate(PROMPT, max_new_tokens=16, temperature=1.0, seed=5)

    assert good.stats.tokens_per_target_pass > bad.stats.tokens_per_target_pass
