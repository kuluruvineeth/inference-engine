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
