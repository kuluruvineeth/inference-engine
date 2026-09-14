import numpy as np
import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.layers.functional import (
    apply_rope,
    rms_norm,
    rope_tables,
    sample_from_logits,
    silu,
    swiglu,
)
from ie.model.transformer import ModelConfig

SMALL = ModelConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
                    num_kv_heads=2, head_dim=8, intermediate_size=64)


def engine(**kw):
    cfg = EngineConfig(block_size=8, num_blocks=256, max_batch_tokens=128, **kw)
    return build_engine(SMALL, cfg)


def test_rms_norm_gives_unit_root_mean_square():
    x = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    out = rms_norm(x, np.ones(16, dtype=np.float32), eps=0.0)
    assert np.allclose(np.sqrt(np.mean(out**2, axis=-1)), 1.0, atol=1e-5)


def test_rms_norm_is_scale_invariant():
    x = np.random.default_rng(1).standard_normal((3, 8)).astype(np.float32)
    weight = np.ones(8, dtype=np.float32)
    assert np.allclose(rms_norm(x, weight), rms_norm(x * 7.5, weight), atol=1e-4)


def test_silu_is_smooth_and_negative_below_zero():
    assert silu(np.array([0.0])) == 0.0
    assert silu(np.array([-1.0]))[0] < 0.0
    assert silu(np.array([-8.0]))[0] > silu(np.array([-1.0]))[0]


def test_swiglu_shapes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 16)).astype(np.float32)
    gate = rng.standard_normal((16, 32)).astype(np.float32)
    up = rng.standard_normal((16, 32)).astype(np.float32)
    down = rng.standard_normal((32, 16)).astype(np.float32)
    assert swiglu(x, gate, up, down).shape == (5, 16)


def test_rope_preserves_vector_norms():
    cos, sin = rope_tables(64, 8)
    x = np.random.default_rng(3).standard_normal((5, 2, 8)).astype(np.float32)
    rotated = apply_rope(x, list(range(5)), cos, sin)
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(rotated, axis=-1), atol=1e-5)


def test_rope_at_position_zero_is_the_identity():
    cos, sin = rope_tables(8, 8)
    x = np.random.default_rng(4).standard_normal((1, 1, 8)).astype(np.float32)
    assert np.allclose(apply_rope(x, [0], cos, sin), x, atol=1e-6)


def test_rope_dot_product_depends_only_on_relative_position():
    cos, sin = rope_tables(128, 8)
    rng = np.random.default_rng(5)
    q = rng.standard_normal((1, 1, 8)).astype(np.float32)
    k = rng.standard_normal((1, 1, 8)).astype(np.float32)

    near = np.sum(apply_rope(q, [5], cos, sin) * apply_rope(k, [3], cos, sin))
    far = np.sum(apply_rope(q, [105], cos, sin) * apply_rope(k, [103], cos, sin))
    assert np.isclose(near, far, atol=1e-4)


def test_greedy_sampling_picks_the_argmax():
    logits = np.array([[0.1, 5.0, 0.2], [9.0, 0.0, 0.0]], dtype=np.float32)
    out = sample_from_logits(logits, np.zeros(2, dtype=np.float32), np.random.default_rng(0))
    assert list(out) == [1, 0]


def test_sampling_follows_the_distribution():
    logits = np.log(np.array([[0.7, 0.2, 0.1]], dtype=np.float32))
    generator = np.random.default_rng(0)
    draws = [int(sample_from_logits(logits, np.ones(1, dtype=np.float32), generator)[0])
             for _ in range(4000)]
    assert 0.65 < draws.count(0) / len(draws) < 0.75
    assert 0.15 < draws.count(1) / len(draws) < 0.25


def test_engine_generates_the_requested_number_of_tokens():
    eng = engine()
    out = eng.generate([[1, 2, 3, 4, 5]], max_new_tokens=6)
    assert len(out) == 1
    assert len(out[0]) == 6
    assert all(0 <= t < SMALL.vocab_size for t in out[0])


def test_greedy_generation_is_deterministic():
    a = engine().generate([[3, 1, 4, 1, 5]], max_new_tokens=8, temperature=0.0)
    b = engine().generate([[3, 1, 4, 1, 5]], max_new_tokens=8, temperature=0.0)
    assert a == b


def test_batched_generation_matches_one_at_a_time():
    prompts = [[1, 2, 3], [9, 8, 7, 6], [4, 4, 4, 4, 4, 4]]

    together = engine().generate(prompts, max_new_tokens=5, temperature=0.0)
    apart = [engine().generate([p], max_new_tokens=5, temperature=0.0)[0] for p in prompts]

    assert together == apart


def test_chunked_prefill_does_not_change_the_output():
    prompt = [list(range(40))]
    whole = build_engine(SMALL, EngineConfig(block_size=8, num_blocks=256,
                                             max_batch_tokens=512))
    chunked = build_engine(SMALL, EngineConfig(block_size=8, num_blocks=256,
                                               max_batch_tokens=12))

    assert whole.generate(prompt, max_new_tokens=6, temperature=0.0) == \
           chunked.generate(prompt, max_new_tokens=6, temperature=0.0)


def test_prefix_cache_does_not_change_the_output():
    shared = list(range(24))
    cold = engine().generate([shared + [7]], max_new_tokens=6, temperature=0.0)

    warm = engine()
    warm.generate([shared + [1]], max_new_tokens=6, temperature=0.0)
    reused = warm.generate([shared + [7]], max_new_tokens=6, temperature=0.0)

    assert warm.blocks.blocks_reused > 0
    assert reused == cold


def test_preemption_does_not_change_the_output():
    prompts = [list(range(i, i + 12)) for i in range(4)]
    roomy = build_engine(SMALL, EngineConfig(block_size=8, num_blocks=256,
                                             max_batch_tokens=256))
    tight = build_engine(SMALL, EngineConfig(block_size=8, num_blocks=8,
                                             max_batch_tokens=256))

    expected = roomy.generate(prompts, max_new_tokens=10, temperature=0.0)
    under_pressure = tight.generate(prompts, max_new_tokens=10, temperature=0.0)

    assert tight.scheduler.stats.preemptions > 0
    assert under_pressure == expected


def test_eos_halts_generation_early():
    eng = engine()
    seq = eng.submit([1, 2, 3], max_new_tokens=50, temperature=0.0)
    first = eng.step()
    assert not first

    seq.eos_id = seq.token_ids[-1]
    eng.run(max_steps=200)
    assert seq.is_finished
    assert seq.num_generated < 50


def test_a_block_is_only_taken_when_a_token_will_be_computed_in_it():
    eng = build_engine(SMALL, EngineConfig(block_size=8, num_blocks=9,
                                           max_batch_tokens=256))
    prompts = [list(range(i, i + 12)) for i in range(4)]
    eng.generate(prompts, max_new_tokens=5, temperature=0.0)
    assert eng.scheduler.stats.preemptions == 0


def test_all_blocks_are_returned_after_every_request_finishes():
    eng = engine()
    eng.generate([[1, 2, 3], [4, 5, 6, 7], [8, 9]], max_new_tokens=4)
    assert eng.blocks.num_free == eng.blocks.num_total


def test_model_config_rejects_incompatible_head_counts():
    with pytest.raises(ValueError):
        ModelConfig(num_heads=4, num_kv_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(head_dim=7)
