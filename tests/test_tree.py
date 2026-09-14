import numpy as np
import pytest

from ie.kernels.reference import dense_attention, tree_attention
from ie.spec.tree import CandidateTree, build_chain, build_dense_tree

HEADS, KV_HEADS, DIM = 4, 2, 8


def rng(seed=0):
    return np.random.default_rng(seed)


def test_a_chain_has_one_path():
    tree = build_chain(root_token=5, tokens=[1, 2, 3])
    assert tree.size == 4
    assert tree.paths() == [[0, 1, 2, 3]]
    assert tree.depth_of(3) == 3


def test_a_dense_tree_branches_at_every_level():
    tree = build_dense_tree(root_token=0, level_candidates=[[1, 2], [3, 4]])
    assert tree.size == 1 + 2 + 4
    assert len(tree.paths()) == 4
    assert all(len(path) == 3 for path in tree.paths())


def test_ancestors_walk_back_to_the_root():
    tree = build_dense_tree(0, [[1, 2], [3, 4]])
    deepest = tree.leaves()[-1]
    assert tree.ancestors(deepest)[0] == tree.root
    assert tree.path_to(deepest)[0] == tree.root


def test_a_node_sees_itself_and_its_ancestors_only():
    tree = build_dense_tree(0, [[1, 2], [3, 4]])
    mask = tree.visibility_mask()

    for node in range(tree.size):
        assert mask[node, node]
        for ancestor in tree.ancestors(node):
            assert mask[node, ancestor]
        visible = set(tree.ancestors(node)) | {node}
        for other in range(tree.size):
            assert mask[node, other] == (other in visible)


def test_siblings_cannot_see_each_other():
    tree = build_dense_tree(0, [[1, 2]])
    mask = tree.visibility_mask()
    left, right = tree.children_of(tree.root)
    assert not mask[left, right]
    assert not mask[right, left]


def test_positions_follow_depth_not_index():
    tree = build_dense_tree(0, [[1, 2], [3, 4]])
    positions = tree.positions_from(base_position=100)
    for node in range(tree.size):
        assert positions[node] == 100 + tree.depth_of(node)


def test_add_rejects_a_bad_parent():
    tree = CandidateTree()
    with pytest.raises(ValueError):
        tree.add(1, parent=0)
    tree.add(1, parent=-1)
    with pytest.raises(ValueError):
        tree.add(2, parent=5)


def test_tree_attention_matches_running_each_path_alone():
    generator = rng(1)
    context_len = 6
    tree = build_dense_tree(root_token=0, level_candidates=[[11, 12], [21, 22]])

    context_k = generator.standard_normal((context_len, KV_HEADS, DIM)).astype(np.float32)
    context_v = generator.standard_normal((context_len, KV_HEADS, DIM)).astype(np.float32)
    tree_q = generator.standard_normal((tree.size, HEADS, DIM)).astype(np.float32)
    tree_k = generator.standard_normal((tree.size, KV_HEADS, DIM)).astype(np.float32)
    tree_v = generator.standard_normal((tree.size, KV_HEADS, DIM)).astype(np.float32)

    together = tree_attention(tree_q, context_k, context_v, tree_k, tree_v,
                              tree.visibility_mask())

    for leaf in tree.leaves():
        path = tree.path_to(leaf)
        k = np.concatenate([context_k, tree_k[path]], axis=0)
        v = np.concatenate([context_v, tree_v[path]], axis=0)
        alone = dense_attention(tree_q[path], k, v, causal=True,
                                query_offset=context_len)
        assert np.allclose(together[path], alone, atol=1e-5)


def test_tree_attention_on_a_chain_equals_plain_causal_attention():
    generator = rng(2)
    context_len = 4
    tree = build_chain(root_token=0, tokens=[1, 2, 3])

    context_k = generator.standard_normal((context_len, KV_HEADS, DIM)).astype(np.float32)
    context_v = generator.standard_normal((context_len, KV_HEADS, DIM)).astype(np.float32)
    q = generator.standard_normal((tree.size, HEADS, DIM)).astype(np.float32)
    k = generator.standard_normal((tree.size, KV_HEADS, DIM)).astype(np.float32)
    v = generator.standard_normal((tree.size, KV_HEADS, DIM)).astype(np.float32)

    via_tree = tree_attention(q, context_k, context_v, k, v, tree.visibility_mask())
    via_dense = dense_attention(q, np.concatenate([context_k, k]),
                                np.concatenate([context_v, v]),
                                causal=True, query_offset=context_len)
    assert np.allclose(via_tree, via_dense, atol=1e-5)
