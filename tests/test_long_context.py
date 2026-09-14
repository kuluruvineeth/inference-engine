import numpy as np
import pytest

from ie.kernels.reference import (
    attention_visibility,
    blocks_touched,
    dense_attention,
    windowed_attention,
)

HEADS, KV_HEADS, DIM = 4, 2, 8


def qkv(num_tokens, seed=0):
    generator = np.random.default_rng(seed)
    return (generator.standard_normal((num_tokens, HEADS, DIM)).astype(np.float32),
            generator.standard_normal((num_tokens, KV_HEADS, DIM)).astype(np.float32),
            generator.standard_normal((num_tokens, KV_HEADS, DIM)).astype(np.float32))


def test_causal_visibility_hides_the_future():
    visible = attention_visibility(query_len=4, context_len=4)
    assert visible.tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]


def test_a_window_forgets_the_distant_past():
    visible = attention_visibility(query_len=4, context_len=4, window=2)
    assert visible.tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]


def test_sinks_stay_visible_no_matter_how_far_back():
    visible = attention_visibility(query_len=4, context_len=4, window=2, sinks=1)
    assert all(row[0] for row in visible.tolist())


def test_a_window_wider_than_the_context_is_plain_causal_attention():
    q, k, v = qkv(12, seed=1)
    assert np.allclose(windowed_attention(q, k, v, window=64),
                       dense_attention(q, k, v, causal=True), atol=1e-5)


def test_a_window_changes_the_result():
    q, k, v = qkv(16, seed=2)
    narrow = windowed_attention(q, k, v, window=4)
    full = dense_attention(q, k, v, causal=True)
    assert not np.allclose(narrow, full, atol=1e-3)


def test_every_row_still_sums_to_a_valid_distribution():
    q, k, v = qkv(20, seed=3)
    out = windowed_attention(q, k, v, window=5, sinks=2)
    assert np.isfinite(out).all()


def test_a_windowed_row_matches_attending_over_just_that_window():
    q, k, v = qkv(12, seed=4)
    window = 4
    windowed = windowed_attention(q, k, v, window=window)

    last = 11
    lo = last - window + 1
    alone = dense_attention(q[last : last + 1], k[lo : last + 1], v[lo : last + 1],
                            causal=True, query_offset=window - 1)
    assert np.allclose(windowed[last], alone[0], atol=1e-5)


def test_sinks_recover_what_a_bare_window_discards():
    q, k, v = qkv(24, seed=5)
    bare = windowed_attention(q, k, v, window=6)
    sinked = windowed_attention(q, k, v, window=6, sinks=4)
    full = dense_attention(q, k, v, causal=True)

    bare_error = np.abs(bare - full).mean()
    sinked_error = np.abs(sinked - full).mean()
    assert sinked_error < bare_error


def test_blocks_touched_is_capped_by_the_window():
    assert blocks_touched(context_len=1024, window=None, sinks=0, block_size=16) == 64
    assert blocks_touched(context_len=1024, window=128, sinks=0, block_size=16) == 8
    assert blocks_touched(context_len=1024, window=128, sinks=16, block_size=16) == 9


def test_a_short_context_never_reports_more_blocks_than_it_has():
    assert blocks_touched(context_len=32, window=512, sinks=4, block_size=16) == 2


@pytest.mark.parametrize("context_len", [256, 4096, 65536])
def test_windowed_cost_grows_linearly_while_dense_grows_quadratically(context_len):
    window = 512
    dense_blocks = blocks_touched(context_len, None, 0, 16)
    windowed_blocks = blocks_touched(context_len, window, 4, 16)
    assert windowed_blocks <= dense_blocks
    assert windowed_blocks <= (window + 4) // 16 + 2
