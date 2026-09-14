import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ie.engine.engine import Engine, EngineConfig, build_engine
from ie.model.torch_transformer import TorchTransformer
from ie.model.transformer import ModelConfig, Transformer

CONFIG = ModelConfig(vocab_size=128, hidden_size=64, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=16, intermediate_size=128)
ENGINE = EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=512)
PROMPTS = [[3, 1, 4, 1, 5], [7, 7, 7, 7, 7, 7]]


def devices():
    available = ["cpu"]
    if torch.cuda.is_available():
        available.append("cuda")
    return available


@pytest.mark.parametrize("device", devices())
def test_torch_generation_matches_numpy(device):
    expected = build_engine(CONFIG, ENGINE, model_seed=1).generate(
        PROMPTS, max_new_tokens=10, temperature=0.0)

    source = Transformer(CONFIG, seed=1)
    ported = Engine(TorchTransformer(source, device, torch.float32), ENGINE)
    assert ported.generate(PROMPTS, max_new_tokens=10, temperature=0.0) == expected


@pytest.mark.parametrize("device", devices())
def test_torch_engine_returns_its_blocks(device):
    source = Transformer(CONFIG, seed=1)
    engine = Engine(TorchTransformer(source, device, torch.float32), ENGINE)
    engine.generate(PROMPTS, max_new_tokens=6, temperature=0.0)
    assert engine.blocks.num_free == engine.blocks.num_total


@pytest.mark.parametrize("device", devices())
def test_chunked_prefill_is_unchanged_on_torch(device):
    source = Transformer(CONFIG, seed=1)
    whole = Engine(TorchTransformer(source, device, torch.float32), ENGINE)
    chunked = Engine(TorchTransformer(source, device, torch.float32),
                     EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=8))

    prompt = [list(range(40))]
    assert (whole.generate(prompt, max_new_tokens=6, temperature=0.0)
            == chunked.generate(prompt, max_new_tokens=6, temperature=0.0))


@pytest.mark.parametrize("device", devices())
def test_batched_decode_matches_the_per_sequence_path(device):
    from ie.kernels.torch_backend import (
        allocate_kv_cache,
        batched_decode_attention,
        paged_attention,
    )

    torch.manual_seed(0)
    block_size, heads, kv_heads, dim = 16, 4, 2, 8
    k_cache, v_cache = allocate_kv_cache(64, block_size, kv_heads, dim,
                                         torch.device(device), torch.float32)
    k_cache.normal_()
    v_cache.normal_()

    context_lens = [9, 33, 17]
    tables = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          device=device, dtype=torch.long)
    q = torch.randn(3, heads, dim, device=device, dtype=torch.float32)

    one_by_one = torch.cat([
        paged_attention(q[i : i + 1], k_cache, v_cache, tables[i : i + 1],
                        [context_lens[i]], [1], block_size)
        for i in range(3)
    ])
    together = batched_decode_attention(
        q, k_cache, v_cache, tables,
        torch.tensor(context_lens, device=device, dtype=torch.long), block_size)

    assert together.shape == (3, heads, dim)
    assert together.shape == one_by_one.shape
    assert torch.allclose(one_by_one, together, atol=1e-4)


@pytest.mark.parametrize("device", devices())
def test_the_batched_path_is_used_only_for_decode(device):
    source = Transformer(CONFIG, seed=1)
    engine = Engine(TorchTransformer(source, device, torch.float32), ENGINE)
    expected = build_engine(CONFIG, ENGINE, model_seed=1).generate(
        PROMPTS, max_new_tokens=8, temperature=0.0)
    assert engine.generate(PROMPTS, max_new_tokens=8, temperature=0.0) == expected
