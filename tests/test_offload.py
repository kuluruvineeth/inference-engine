import numpy as np
import pytest

from ie.core.block_manager import BlockManager, chain_hash
from ie.core.offload import KVTier
from ie.core.sequence import Sequence
from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig

BS = 4
VOCAB = 2048
CONFIG = ModelConfig(vocab_size=VOCAB, hidden_size=32, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)


class RecordingCopier:
    def __init__(self):
        self.contents: dict[int, str] = {}
        self.reads = 0
        self.writes = 0

    def read_block(self, block_id):
        self.reads += 1
        return self.contents.get(block_id, f"data-{block_id}")

    def write_block(self, block_id, payload):
        self.writes += 1
        self.contents[block_id] = payload


def seq(tokens, block_size=BS):
    return Sequence(tokens, block_size=block_size)


def test_tier_holds_and_returns_a_payload():
    tier = KVTier(capacity_blocks=4)
    assert tier.admit("k", [1, 2], "payload")
    assert "k" in tier
    assert tier.holds("k", [1, 2])
    assert not tier.holds("k", [9, 9])
    assert tier.take("k") == ([1, 2], "payload")
    assert "k" not in tier


def test_tier_evicts_the_least_recently_admitted():
    tier = KVTier(capacity_blocks=2)
    tier.admit("a", [1], "A")
    tier.admit("b", [2], "B")
    tier.admit("c", [3], "C")

    assert "a" not in tier
    assert "b" in tier and "c" in tier
    assert tier.stats.evicted == 1


def test_a_zero_capacity_tier_rejects_everything():
    tier = KVTier(capacity_blocks=0)
    assert not tier.admit("a", [1], "A")
    assert tier.stats.rejected == 1
    assert len(tier) == 0


def test_tier_rejects_a_negative_capacity():
    with pytest.raises(ValueError):
        KVTier(capacity_blocks=-1)


def test_an_evicted_block_lands_in_the_tier():
    tier = KVTier(capacity_blocks=8)
    copier = RecordingCopier()
    manager = BlockManager(num_blocks=2, block_size=BS, tier=tier, copier=copier)

    first = seq([1, 2, 3, 4, 5])
    manager.allocate(first)
    first.num_computed = len(first)
    manager.share_computed_blocks(first)
    manager.free(first)

    crowd = seq(list(range(50, 58)))
    manager.allocate(crowd)

    assert len(tier) >= 1
    assert copier.reads >= 1


def test_a_block_restored_from_the_tier_carries_its_payload_back():
    tier = KVTier(capacity_blocks=8)
    copier = RecordingCopier()
    manager = BlockManager(num_blocks=2, block_size=BS, tier=tier, copier=copier)

    original = seq([1, 2, 3, 4, 5])
    manager.allocate(original)
    original.num_computed = len(original)
    copier.contents[original.block_table[0]] = "the-first-block"
    manager.share_computed_blocks(original)
    manager.free(original)

    manager.allocate(seq(list(range(50, 58))))
    stored = chain_hash([1, 2, 3, 4])
    assert tier.holds(stored, [1, 2, 3, 4])


def test_offload_does_not_change_generated_tokens():
    prompts = [list(range(100 + i * 64, 100 + i * 64 + 64)) for i in range(6)]

    def run(offload_blocks):
        engine = build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=12,
                                                   max_batch_tokens=512,
                                                   offload_blocks=offload_blocks))
        for prompt in prompts:
            engine.generate([prompt], max_new_tokens=2, temperature=0.0)
        revisit = engine.generate([prompts[0]], max_new_tokens=2, temperature=0.0)[0]
        return revisit, engine.blocks.stats()

    without, stats_without = run(0)
    with_tier, stats_with = run(64)

    assert without == with_tier
    assert stats_without["restored"] == 0
    assert stats_with["restored"] > 0
    assert stats_with["computed"] < stats_without["computed"]


def test_offload_is_off_by_default():
    engine = build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=32))
    assert not engine.blocks.offload_enabled
    assert engine.blocks.stats()["tier_held"] == 0
