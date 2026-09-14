"""Tests for paged allocation and prefix sharing.

These run on any machine — no GPU, no model, no torch. The control plane is
pure bookkeeping, and bookkeeping is exactly where paged attention lives.
"""

import pytest

from ie.core.block_manager import BlockManager, chain_hash
from ie.core.sequence import Sequence

BS = 4  # small block size keeps the arithmetic readable in tests


def seq(tokens, block_size=BS, **kw):
    return Sequence(tokens, block_size=block_size, **kw)


# ---- geometry -------------------------------------------------------------

def test_block_count_rounds_up():
    assert seq([1] * 4).num_blocks == 1
    assert seq([1] * 5).num_blocks == 2
    assert seq([1] * 8).num_blocks == 2


def test_only_full_blocks_are_shareable():
    s = seq([1, 2, 3, 4, 5])
    assert s.is_block_full(0)
    assert not s.is_block_full(1)


# ---- allocation -----------------------------------------------------------

def test_allocate_gives_one_block_per_logical_block():
    bm = BlockManager(num_blocks=8, block_size=BS)
    s = seq([1, 2, 3, 4, 5, 6])          # 2 blocks
    assert bm.can_allocate(s)
    bm.allocate(s)
    assert len(s.block_table) == 2
    assert bm.num_free == 6


def test_free_returns_blocks_to_the_pool():
    bm = BlockManager(num_blocks=4, block_size=BS)
    s = seq([1, 2, 3, 4, 5])
    bm.allocate(s)
    assert bm.num_free == 2
    bm.free(s)
    assert bm.num_free == 4
    assert s.block_table == []


def test_cannot_allocate_beyond_capacity():
    bm = BlockManager(num_blocks=2, block_size=BS)
    s = seq(list(range(12)))              # needs 3 blocks, only 2 exist
    assert not bm.can_allocate(s)
    with pytest.raises(RuntimeError):
        bm.allocate(s)


def test_fragmentation_is_bounded_by_one_partial_block():
    """The headline claim of paged attention: waste per sequence is at most
    block_size - 1 tokens, no matter how long the sequence is."""
    bm = BlockManager(num_blocks=64, block_size=BS)
    for length in range(1, 40):
        s = seq(list(range(length)))
        bm.allocate(s)
        capacity = len(s.block_table) * BS
        assert 0 <= capacity - len(s) < BS
        bm.free(s)


# ---- growth ---------------------------------------------------------------

def test_append_only_takes_a_block_at_a_boundary():
    bm = BlockManager(num_blocks=8, block_size=BS)
    s = seq([1, 2, 3])                    # 1 block, 1 slot spare
    bm.allocate(s)
    before = bm.num_free

    s.append(9)                           # fills the block exactly
    bm.append_slot(s)
    assert bm.num_free == before          # no new block needed

    s.append(10)                          # spills into a second block
    bm.append_slot(s)
    assert bm.num_free == before - 1


def test_can_append_is_false_when_pool_is_empty_at_a_boundary():
    bm = BlockManager(num_blocks=1, block_size=BS)
    s = seq([1, 2, 3, 4])                 # exactly one full block
    bm.allocate(s)
    assert bm.num_free == 0
    s.append(5)
    assert not bm.can_append(s)


# ---- the chained hash -----------------------------------------------------

def test_hash_chain_depends_on_the_prefix():
    """Same block contents, different history => different hash. This is the
    property that makes sharing safe."""
    a = chain_hash([5, 6, 7, 8], parent=chain_hash([1, 2, 3, 4]))
    b = chain_hash([5, 6, 7, 8], parent=chain_hash([9, 9, 9, 9]))
    assert a != b


def test_hash_is_stable():
    assert chain_hash([1, 2, 3]) == chain_hash([1, 2, 3])


# ---- prefix sharing -------------------------------------------------------

def test_identical_prefix_is_stored_once():
    bm = BlockManager(num_blocks=16, block_size=BS)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]     # 2 full blocks

    a = seq(prompt + [9])
    bm.allocate(a)
    a.num_computed = len(a)               # pretend the forward pass ran
    bm.publish(a)

    free_before = bm.num_free
    b = seq(prompt + [42])                # same prefix, different tail
    cached = bm.allocate(b)

    assert cached == 8                            # both prefix blocks reused
    assert b.block_table[:2] == a.block_table[:2] # literally the same blocks
    assert b.block_table[2] != a.block_table[2]   # tails diverge
    assert bm.num_free == free_before - 1         # only the tail block is new


def test_shared_blocks_are_refcounted_and_survive_one_owner_leaving():
    bm = BlockManager(num_blocks=16, block_size=BS)
    prompt = [1, 2, 3, 4]

    a = seq(prompt + [7])
    bm.allocate(a)
    a.num_computed = len(a)
    bm.publish(a)

    b = seq(prompt + [8])
    bm.allocate(b)
    shared = b.block_table[0]
    assert bm.blocks[shared].ref_count == 2

    bm.free(a)
    assert bm.blocks[shared].ref_count == 1
    assert shared in b.block_table          # still owned by b
    assert shared not in bm.free_ids        # and not handed out again

    bm.free(b)
    assert bm.blocks[shared].ref_count == 0


def test_a_diverging_prefix_shares_nothing_past_the_divergence():
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = seq([1, 2, 3, 4, 5, 6, 7, 8])
    bm.allocate(a)
    a.num_computed = len(a)
    bm.publish(a)

    # differs in the FIRST block, so the chain breaks immediately
    b = seq([1, 2, 3, 99, 5, 6, 7, 8])
    cached = bm.allocate(b)
    assert cached == 0
    assert set(a.block_table).isdisjoint(b.block_table)


def test_partial_blocks_are_never_published():
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = seq([1, 2, 3])                    # one partial block
    bm.allocate(a)
    a.num_computed = len(a)
    bm.publish(a)
    assert bm.by_hash == {}               # nothing shareable yet

    b = seq([1, 2, 3])
    assert bm.allocate(b) == 0            # so no hit


def test_uncomputed_blocks_are_never_published():
    """A block can be allocated but not yet filled by a forward pass. Sharing
    it would hand out uninitialised K,V."""
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = seq([1, 2, 3, 4, 5, 6, 7, 8])
    bm.allocate(a)
    a.num_computed = 4                    # only the first block is real
    bm.publish(a)
    assert len(bm.by_hash) == 1


def test_eviction_removes_the_stale_index_entry():
    """A freed block stays cached until it is reallocated. Once reused, its
    old hash must not resolve to it any more."""
    bm = BlockManager(num_blocks=2, block_size=BS)
    a = seq([1, 2, 3, 4])
    bm.allocate(a)
    a.num_computed = 4
    bm.publish(a)
    stale = chain_hash([1, 2, 3, 4])
    assert stale in bm.by_hash

    bm.free(a)
    # drain the pool so the cached block must be recycled
    big = seq(list(range(100, 108)))
    bm.allocate(big)

    assert bm.by_hash.get(stale) is None
    probe = seq([1, 2, 3, 4])
    assert not bm.can_allocate(probe) or bm.allocate(probe) == 0


def test_hit_rate_is_tracked():
    bm = BlockManager(num_blocks=32, block_size=BS)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    a = seq(prompt)
    bm.allocate(a)
    a.num_computed = len(a)
    bm.publish(a)

    for _ in range(3):
        b = seq(prompt)
        bm.allocate(b)

    st = bm.stats()
    assert st["hits"] == 6        # 3 sequences x 2 shared blocks
    assert st["misses"] == 2      # only the first sequence paid
    assert st["hit_rate"] > 0.7
