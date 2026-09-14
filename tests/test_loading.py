import json

import numpy as np
import pytest

from ie.engine.engine import Engine, EngineConfig
from ie.load.huggingface import config_from_hf, layer_names, load_transformer, read_config
from ie.load.safetensors import (
    SafetensorsFile,
    ShardedCheckpoint,
    bfloat16_to_float32,
    write_safetensors,
)
from ie.model.transformer import ModelConfig

HF_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "intermediate_size": 64, "max_position_embeddings": 512,
    "rope_theta": 1000000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
}


def synthetic_checkpoint(directory, config=None, tied=True, shards=1):
    raw = dict(config or HF_CONFIG)
    raw["tie_word_embeddings"] = tied
    (directory / "config.json").write_text(json.dumps(raw))

    spec = config_from_hf(raw)
    generator = np.random.default_rng(0)
    normal = lambda *shape: generator.standard_normal(shape).astype(np.float32) * 0.02

    tensors = {"model.embed_tokens.weight": normal(spec.vocab_size, spec.hidden_size),
               "model.norm.weight": np.ones(spec.hidden_size, dtype=np.float32)}
    if not tied:
        tensors["lm_head.weight"] = normal(spec.vocab_size, spec.hidden_size)

    for index in range(spec.num_layers):
        names = layer_names(index)
        q_out = spec.num_heads * spec.head_dim
        kv_out = spec.num_kv_heads * spec.head_dim
        tensors[names["attn_norm"]] = np.ones(spec.hidden_size, dtype=np.float32)
        tensors[names["mlp_norm"]] = np.ones(spec.hidden_size, dtype=np.float32)
        tensors[names["q_proj"]] = normal(q_out, spec.hidden_size)
        tensors[names["k_proj"]] = normal(kv_out, spec.hidden_size)
        tensors[names["v_proj"]] = normal(kv_out, spec.hidden_size)
        tensors[names["o_proj"]] = normal(spec.hidden_size, q_out)
        tensors[names["gate_proj"]] = normal(spec.intermediate_size, spec.hidden_size)
        tensors[names["up_proj"]] = normal(spec.intermediate_size, spec.hidden_size)
        tensors[names["down_proj"]] = normal(spec.hidden_size, spec.intermediate_size)

    if shards == 1:
        write_safetensors(directory / "model.safetensors", tensors)
    else:
        names = list(tensors)
        per_shard = -(-len(names) // shards)
        for index in range(shards):
            chunk = names[index * per_shard : (index + 1) * per_shard]
            write_safetensors(directory / f"model-{index:05d}-of-{shards:05d}.safetensors",
                              {name: tensors[name] for name in chunk})
    return tensors


def test_roundtrip_preserves_values_and_shapes(tmp_path):
    original = {"a": np.arange(6, dtype=np.float32).reshape(2, 3),
                "b": np.array([1.5], dtype=np.float32)}
    path = tmp_path / "m.safetensors"
    write_safetensors(path, original)

    loaded = SafetensorsFile(path)
    assert loaded.names == ["a", "b"]
    assert len(loaded) == 2
    assert np.array_equal(loaded.get("a"), original["a"])
    assert loaded.get("a").shape == (2, 3)


def test_reading_an_absent_tensor_raises(tmp_path):
    path = tmp_path / "m.safetensors"
    write_safetensors(path, {"a": np.zeros(2, dtype=np.float32)})
    with pytest.raises(KeyError):
        SafetensorsFile(path).get("nope")


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SafetensorsFile(tmp_path / "absent.safetensors")


def test_a_truncated_file_is_rejected(tmp_path):
    path = tmp_path / "short.safetensors"
    path.write_bytes(b"\x01\x02")
    with pytest.raises(ValueError):
        SafetensorsFile(path)


def test_parameter_count_matches_the_tensors(tmp_path):
    path = tmp_path / "m.safetensors"
    write_safetensors(path, {"a": np.zeros((3, 4), dtype=np.float32),
                             "b": np.zeros(5, dtype=np.float32)})
    assert SafetensorsFile(path).total_parameters == 17


def test_bfloat16_widens_to_the_same_value():
    values = np.array([1.0, -2.5, 0.25], dtype=np.float32)
    as_bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)
    assert np.allclose(bfloat16_to_float32(as_bf16), values, atol=1e-2)


def test_sharded_checkpoints_present_one_namespace(tmp_path):
    synthetic_checkpoint(tmp_path, shards=3)
    checkpoint = ShardedCheckpoint(tmp_path)
    assert len(checkpoint.shards) == 3
    assert "model.embed_tokens.weight" in checkpoint
    assert checkpoint.get("model.norm.weight").shape == (32,)


def test_a_directory_without_shards_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ShardedCheckpoint(tmp_path)


def test_hf_config_maps_onto_our_config():
    spec = config_from_hf(HF_CONFIG)
    assert isinstance(spec, ModelConfig)
    assert spec.num_kv_heads == 2
    assert spec.rope_base == 1000000.0
    assert spec.norm_eps == 1e-5


def test_head_dim_is_derived_when_absent():
    raw = dict(HF_CONFIG)
    del raw["head_dim"]
    assert config_from_hf(raw).head_dim == raw["hidden_size"] // raw["num_attention_heads"]


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_config(tmp_path)


def test_loading_a_checkpoint_fills_every_weight(tmp_path):
    tensors = synthetic_checkpoint(tmp_path)
    model, report = load_transformer(tmp_path)

    assert report.is_complete
    assert report.missing == []
    assert report.architecture == "Qwen3ForCausalLM"
    assert report.layers_loaded == 2
    assert np.array_equal(model.embedding, tensors["model.embed_tokens.weight"])


def test_projections_are_transposed_into_our_layout(tmp_path):
    tensors = synthetic_checkpoint(tmp_path)
    model, _ = load_transformer(tmp_path)
    stored = tensors[layer_names(0)["q_proj"]]

    assert model.blocks[0].q_proj.shape == (stored.shape[1], stored.shape[0])
    assert np.allclose(model.blocks[0].q_proj, stored.T)


def test_tied_embeddings_reuse_the_embedding_matrix(tmp_path):
    synthetic_checkpoint(tmp_path, tied=True)
    model, report = load_transformer(tmp_path)
    assert report.tied_embeddings
    assert np.allclose(model.lm_head, model.embedding.T)


def test_an_untied_head_is_loaded_separately(tmp_path):
    tensors = synthetic_checkpoint(tmp_path, tied=False)
    model, report = load_transformer(tmp_path)
    assert not report.tied_embeddings
    assert np.allclose(model.lm_head, tensors["lm_head.weight"].T)


def test_a_loaded_model_generates(tmp_path):
    synthetic_checkpoint(tmp_path)
    model, _ = load_transformer(tmp_path)
    engine = Engine(model, EngineConfig(block_size=16, num_blocks=64, max_batch_tokens=128))
    out = engine.generate([[1, 2, 3]], max_new_tokens=5, temperature=0.0)[0]
    assert len(out) == 5
    assert all(0 <= token < model.config.vocab_size for token in out)


def test_a_sharded_checkpoint_loads_identically(tmp_path):
    single = tmp_path / "single"
    sharded = tmp_path / "sharded"
    single.mkdir()
    sharded.mkdir()
    synthetic_checkpoint(single, shards=1)
    synthetic_checkpoint(sharded, shards=4)

    from_single, _ = load_transformer(single)
    from_shards, _ = load_transformer(sharded)
    assert np.allclose(from_single.blocks[1].down_proj, from_shards.blocks[1].down_proj)


def test_missing_weights_are_reported_not_silently_ignored(tmp_path):
    synthetic_checkpoint(tmp_path)
    import json as _json

    raw = _json.loads((tmp_path / "config.json").read_text())
    raw["num_hidden_layers"] = 3
    (tmp_path / "config.json").write_text(_json.dumps(raw))

    _, report = load_transformer(tmp_path)
    assert not report.is_complete
    assert any("layers.2" in name for name in report.missing)
