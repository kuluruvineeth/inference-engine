import numpy as np
import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig, Transformer
from ie.spec.medusa import MedusaHeads, accept_greedily, align_with_target, propose_tree
from ie.spec.medusa_engine import MedusaEngine
from ie.spec.tree import build_chain, build_dense_tree

CONFIG = ModelConfig(vocab_size=128, hidden_size=64, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=16, intermediate_size=128)
ENGINE = EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=256)
PROMPT = [3, 1, 4, 1, 5]


def baseline(max_new_tokens=12):
    return build_engine(CONFIG, ENGINE, model_seed=1).generate(
        [PROMPT], max_new_tokens=max_new_tokens, temperature=0.0)[0]


def medusa(strength, widths=None, head_seed=5):
    model = Transformer(CONFIG, seed=1)
    heads = MedusaHeads(CONFIG, num_heads=3, seed=head_seed)
    align_with_target(model, heads, strength)
    return MedusaEngine(model, heads, ENGINE, widths=widths or [2, 2])


def test_heads_propose_the_requested_number_of_candidates():
    heads = MedusaHeads(CONFIG, num_heads=3, seed=0)
    hidden = np.random.default_rng(0).standard_normal(CONFIG.hidden_size).astype(np.float32)
    picks = heads.candidates(hidden, widths=[3, 2])
    assert [len(p) for p in picks] == [3, 2]
    assert all(0 <= token < CONFIG.vocab_size for level in picks for token in level)


def test_heads_reject_more_widths_than_heads():
    heads = MedusaHeads(CONFIG, num_heads=2, seed=0)
    hidden = np.zeros(CONFIG.hidden_size, dtype=np.float32)
    with pytest.raises(ValueError):
        heads.candidates(hidden, widths=[2, 2, 2])


def test_proposed_tree_has_one_node_per_candidate_path():
    heads = MedusaHeads(CONFIG, num_heads=3, seed=1)
    hidden = np.random.default_rng(1).standard_normal(CONFIG.hidden_size).astype(np.float32)
    tree = propose_tree(root_token=7, hidden_state=hidden, heads=heads, widths=[2, 3])
    assert tree.size == 1 + 2 + 6
    assert len(tree.paths()) == 6


def test_greedy_acceptance_walks_the_matching_branch():
    tree = build_dense_tree(root_token=0, level_candidates=[[5, 9], [11, 12]])
    logits = np.zeros((tree.size, 32), dtype=np.float32)
    logits[tree.root, 9] = 10.0
    branch = next(c for c in tree.children_of(tree.root) if tree.tokens[c] == 9)
    logits[branch, 12] = 10.0
    leaf = next(c for c in tree.children_of(branch) if tree.tokens[c] == 12)
    logits[leaf, 3] = 10.0

    result = accept_greedily(tree, logits)
    assert result.tokens == [9, 12, 3]
    assert result.num_accepted == 2


def test_greedy_acceptance_stops_at_the_first_mismatch():
    tree = build_chain(root_token=0, tokens=[5, 6])
    logits = np.zeros((tree.size, 32), dtype=np.float32)
    logits[tree.root, 31] = 10.0

    result = accept_greedily(tree, logits)
    assert result.num_accepted == 0
    assert result.tokens == [31]


def test_acceptance_always_emits_one_more_than_it_accepts():
    tree = build_dense_tree(0, [[1, 2], [3, 4]])
    generator = np.random.default_rng(2)
    for _ in range(20):
        logits = generator.standard_normal((tree.size, 16)).astype(np.float32)
        result = accept_greedily(tree, logits)
        assert result.num_emitted == result.num_accepted + 1


@pytest.mark.parametrize("strength", [0.0, 0.3, 0.7, 1.0])
def test_medusa_reproduces_greedy_decoding_exactly(strength):
    assert medusa(strength).generate(PROMPT, max_new_tokens=12) == baseline()


@pytest.mark.parametrize("widths", [[2], [2, 2], [3, 2], [2, 2, 2]])
def test_tree_shape_does_not_change_the_output(widths):
    assert medusa(1.0, widths=widths).generate(PROMPT, max_new_tokens=10) == baseline(10)


def test_better_heads_need_fewer_target_passes():
    strong = medusa(1.0)
    strong.generate(PROMPT, max_new_tokens=16)
    weak = medusa(0.0)
    weak.generate(PROMPT, max_new_tokens=16)

    assert strong.stats.target_passes < weak.stats.target_passes
    assert strong.stats.tokens_per_target_pass > weak.stats.tokens_per_target_pass


def test_random_heads_degrade_to_one_token_per_pass():
    engine = medusa(0.0)
    engine.generate(PROMPT, max_new_tokens=8)
    assert engine.stats.tokens_per_target_pass == pytest.approx(1.0)


def test_medusa_returns_its_blocks():
    engine = medusa(1.0)
    engine.generate(PROMPT, max_new_tokens=10)
    assert engine.blocks.num_free == engine.blocks.num_total
