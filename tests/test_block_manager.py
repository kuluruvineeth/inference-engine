import pytest

from ie.core.block_manager import BlockManager, chain_hash
from ie.core.sequence import Sequence

BS = 4


def seq(tokens, block_size=BS, **kw):
    return Sequence(tokens, block_size=block_size, **kw)


def test_block_count_rounds_up():
    assert seq([1] * 4).num_blocks == 1
    assert seq([1] * 5).num_blocks == 2
    assert seq([1] * 8).num_blocks == 2


def test_only_full_blocks_are_shareable():
    s = seq([1, 2, 3, 4, 5])
    assert s.block_is_full(0)
    assert not s.block_is_full(1)


def test_allocate_gives_one_block_per_logical_block():
    bm = BlockManager(num_blocks=8, block_size=BS)
    s = seq([1, 2, 3, 4, 5, 6])
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
    s = seq(list(range(12)))
    assert not bm.can_allocate(s)
    with pytest.raises(RuntimeError):
        bm.allocate(s)


def test_fragmentation_is_bounded_by_one_partial_block():
    bm = BlockManager(num_blocks=64, block_size=BS)
    for length in range(1, 40):
        s = seq(list(range(length)))
        bm.allocate(s)
        capacity = len(s.block_table) * BS
        assert 0 <= capacity - len(s) < BS
        bm.free(s)


def test_append_only_takes_a_block_at_a_boundary():
    bm = BlockManager(num_blocks=8, block_size=BS)
    s = seq([1, 2, 3])
    bm.allocate(s)
    before = bm.num_free

    s.append(9)
    bm.append_slot(s)
    assert bm.num_free == before

    s.append(10)
    bm.append_slot(s)
    assert bm.num_free == before - 1


def test_can_append_is_false_when_pool_is_empty_at_a_boundary():
    bm = BlockManager(num_blocks=1, block_size=BS)
    s = seq([1, 2, 3, 4])
    bm.allocate(s)
    assert bm.num_free == 0
    s.append(5)
    assert not bm.can_append(s)


def test_hash_chain_depends_on_the_prefix():
    a = chain_hash([5, 6, 7, 8], parent=chain_hash([1, 2, 3, 4]))
    b = chain_hash([5, 6, 7, 8], parent=chain_hash([9, 9, 9, 9]))
    assert a != b


def test_hash_is_stable():
    assert chain_hash([1, 2, 3]) == chain_hash([1, 2, 3])


def test_identical_prefix_is_stored_once():
    bm = BlockManager(num_blocks=16, block_size=BS)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]

    a = seq(prompt + [9])
    bm.allocate(a)
    a.num_computed = len(a)
    bm.share_computed_blocks(a)

    free_before = bm.num_free
    b = seq(prompt + [42])
    cached = bm.allocate(b)

    assert cached == 8
    assert b.block_table[:2] == a.block_table[:2]
    assert b.block_table[2] != a.block_table[2]
    assert bm.num_free == free_before - 1


def test_shared_blocks_are_refcounted_and_survive_one_owner_leaving():
    bm = BlockManager(num_blocks=16, block_size=BS)
    prompt = [1, 2, 3, 4]

    a = seq(prompt + [7])
    bm.allocate(a)
    a.num_computed = len(a)
    bm.share_computed_blocks(a)

    b = seq(prompt + [8])
    bm.allocate(b)
    shared = b.block_table[0]
    assert bm.blocks[shared].ref_count == 2

    bm.free(a)
    assert bm.blocks[shared].ref_count == 1
    assert shared in b.block_table
    assert shared not in bm.free_ids

    bm.free(b)
    assert bm.blocks[shared].ref_count == 0


def test_a_diverging_prefix_shares_nothing_past_the_divergence():
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = seq([1, 2, 3, 4, 5, 6, 7, 8])
    bm.allocate(a)
    a.num_computed = len(a)
    bm.share_computed_blocks(a)

    b = seq([1, 2, 3, 99, 5, 6, 7, 8])
    cached = bm.allocate(b)
    assert cached == 0
    assert set(a.block_table).isdisjoint(b.block_table)


def test_partial_blocks_are_never_shared():
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = seq([1, 2, 3])
    bm.allocate(a)
    a.num_computed = len(a)
    bm.share_computed_blocks(a)
    assert bm.block_id_by_hash == {}

    b = seq([1, 2, 3])
    assert bm.allocate(b) == 0


def test_uncomputed_blocks_are_never_shared():
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = seq([1, 2, 3, 4, 5, 6, 7, 8])
    bm.allocate(a)
    a.num_computed = 4
    bm.share_computed_blocks(a)
    assert len(bm.block_id_by_hash) == 1


def test_eviction_removes_the_stale_index_entry():
    bm = BlockManager(num_blocks=2, block_size=BS)
    a = seq([1, 2, 3, 4])
    bm.allocate(a)
    a.num_computed = 4
    bm.share_computed_blocks(a)
    stale = chain_hash([1, 2, 3, 4])
    assert stale in bm.block_id_by_hash

    bm.free(a)
    big = seq(list(range(100, 108)))
    bm.allocate(big)

    assert bm.block_id_by_hash.get(stale) is None
    probe = seq([1, 2, 3, 4])
    assert not bm.can_allocate(probe) or bm.allocate(probe) == 0


def test_reuse_rate_is_tracked():
    bm = BlockManager(num_blocks=32, block_size=BS)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    a = seq(prompt)
    bm.allocate(a)
    a.num_computed = len(a)
    bm.share_computed_blocks(a)

    for _ in range(3):
        b = seq(prompt)
        bm.allocate(b)

    st = bm.stats()
    assert st["reused"] == 6
    assert st["computed"] == 2
    assert st["reuse_rate"] > 0.7
