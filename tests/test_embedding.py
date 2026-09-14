import numpy as np
import pytest

from ie.model.embedding import (
    EmbeddingModel,
    Pooling,
    cosine_similarity,
    l2_normalize,
    pool,
)
from ie.model.transformer import ModelConfig, Transformer

CONFIG = ModelConfig(vocab_size=128, hidden_size=32, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)


def model(**kw):
    return EmbeddingModel(Transformer(CONFIG, seed=1), **kw)


def test_mean_pooling_averages_the_span():
    hidden = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=np.float32)
    assert np.allclose(pool(hidden, Pooling.MEAN, [3]), [[3.0, 3.0]])


def test_cls_and_last_pick_the_ends():
    hidden = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    assert pool(hidden, Pooling.CLS, [3])[0, 0] == 1.0
    assert pool(hidden, Pooling.LAST, [3])[0, 0] == 3.0


def test_pooling_respects_document_boundaries():
    hidden = np.array([[1.0], [3.0], [10.0]], dtype=np.float32)
    pooled = pool(hidden, Pooling.MEAN, [2, 1])
    assert pooled.shape == (2, 1)
    assert pooled[0, 0] == 2.0 and pooled[1, 0] == 10.0


def test_normalized_vectors_have_unit_length():
    x = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
    assert np.allclose(np.linalg.norm(l2_normalize(x), axis=-1), 1.0, atol=1e-6)


def test_normalizing_a_zero_vector_does_not_divide_by_zero():
    assert np.allclose(l2_normalize(np.zeros((1, 4), dtype=np.float32)), 0.0)


def test_cosine_similarity_of_a_vector_with_itself_is_one():
    x = np.random.default_rng(1).standard_normal((3, 8)).astype(np.float32)
    assert np.allclose(np.diag(cosine_similarity(x, x)), 1.0, atol=1e-5)


def test_encoding_returns_one_unit_vector_per_document():
    vectors = model().encode([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    assert vectors.shape == (3, CONFIG.hidden_size)
    assert np.allclose(np.linalg.norm(vectors, axis=-1), 1.0, atol=1e-5)


def test_documents_do_not_attend_across_each_other():
    encoder = model()
    together = encoder.encode([[1, 2, 3], [40, 41, 42]])
    apart = np.concatenate([encoder.encode([[1, 2, 3]]), encoder.encode([[40, 41, 42]])])
    assert np.allclose(together, apart, atol=1e-5)


def test_identical_documents_embed_identically():
    vectors = model().encode([[5, 6, 7], [5, 6, 7]])
    assert np.allclose(vectors[0], vectors[1], atol=1e-6)
    assert cosine_similarity(vectors[:1], vectors[1:])[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_bidirectional_attention_differs_from_causal():
    bidirectional = model(bidirectional=True).encode([[1, 2, 3, 4, 5]])
    causal = model(bidirectional=False).encode([[1, 2, 3, 4, 5]])
    assert not np.allclose(bidirectional, causal, atol=1e-3)


def test_pooling_strategy_changes_the_embedding():
    document = [[1, 2, 3, 4]]
    mean = model(pooling=Pooling.MEAN).encode(document)
    last = model(pooling=Pooling.LAST).encode(document)
    assert not np.allclose(mean, last, atol=1e-3)


def test_a_whole_corpus_is_encoded_in_one_pass():
    encoder = model()
    encoder.encode([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert encoder.stats.forward_passes == 1
    assert encoder.stats.documents == 4
    assert encoder.stats.tokens == 8
    assert encoder.stats.tokens_per_pass == 8.0
