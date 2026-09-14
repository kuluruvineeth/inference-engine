from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..model.transformer import ModelConfig, Transformer, TransformerBlock
from .safetensors import SafetensorsFile, ShardedCheckpoint

SUPPORTED_ARCHITECTURES = {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
                           "MistralForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM"}


@dataclass(slots=True)
class LoadReport:
    architecture: str
    parameters: int
    layers_loaded: int
    tied_embeddings: bool
    missing: list[str]
    unexpected: list[str]

    @property
    def is_complete(self) -> bool:
        return not self.missing


def config_from_hf(raw: dict) -> ModelConfig:
    num_heads = raw["num_attention_heads"]
    hidden = raw["hidden_size"]
    head_dim = raw.get("head_dim", hidden // num_heads)
    return ModelConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=hidden,
        num_layers=raw["num_hidden_layers"],
        num_heads=num_heads,
        num_kv_heads=raw.get("num_key_value_heads", num_heads),
        head_dim=head_dim,
        intermediate_size=raw["intermediate_size"],
        max_position=raw.get("max_position_embeddings", 4096),
        rope_base=float(raw.get("rope_theta", 10000.0)),
        norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        rope_halved=True,
    )


def read_config(directory: str | Path) -> dict:
    path = Path(directory) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"no config.json in {directory}")
    return json.loads(path.read_text())


def layer_names(index: int) -> dict[str, str]:
    prefix = f"model.layers.{index}"
    return {
        "attn_norm": f"{prefix}.input_layernorm.weight",
        "q_proj": f"{prefix}.self_attn.q_proj.weight",
        "k_proj": f"{prefix}.self_attn.k_proj.weight",
        "v_proj": f"{prefix}.self_attn.v_proj.weight",
        "o_proj": f"{prefix}.self_attn.o_proj.weight",
        "q_bias": f"{prefix}.self_attn.q_proj.bias",
        "k_bias": f"{prefix}.self_attn.k_proj.bias",
        "v_bias": f"{prefix}.self_attn.v_proj.bias",
        "mlp_norm": f"{prefix}.post_attention_layernorm.weight",
        "gate_proj": f"{prefix}.mlp.gate_proj.weight",
        "up_proj": f"{prefix}.mlp.up_proj.weight",
        "down_proj": f"{prefix}.mlp.down_proj.weight",
    }


TRANSPOSED = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
OPTIONAL = {"q_bias", "k_bias", "v_bias"}


def open_checkpoint(directory: str | Path):
    directory = Path(directory)
    single = directory / "model.safetensors"
    return SafetensorsFile(single) if single.exists() else ShardedCheckpoint(directory)


def load_transformer(directory: str | Path) -> tuple[Transformer, LoadReport]:
    raw = read_config(directory)
    architectures = raw.get("architectures", [])
    architecture = architectures[0] if architectures else "unknown"
    config = config_from_hf(raw)

    checkpoint = open_checkpoint(directory)
    model = Transformer(config, seed=0)
    missing: list[str] = []
    consumed: set[str] = set()

    def take(name: str, optional: bool = False) -> np.ndarray | None:
        if name not in checkpoint:
            if not optional:
                missing.append(name)
            return None
        consumed.add(name)
        return checkpoint.get(name).astype(np.float32)

    embedding = take("model.embed_tokens.weight")
    if embedding is not None:
        model.embedding = embedding

    layers_loaded = 0
    for index in range(config.num_layers):
        block: TransformerBlock = model.blocks[index]
        for attribute, name in layer_names(index).items():
            weight = take(name, optional=attribute in OPTIONAL)
            if weight is None:
                continue
            setattr(block, attribute, weight.T.copy() if attribute in TRANSPOSED else weight)
        layers_loaded += 1

    final_norm = take("model.norm.weight")
    if final_norm is not None:
        model.final_norm = final_norm

    tied = raw.get("tie_word_embeddings", False) or "lm_head.weight" not in checkpoint
    if tied and embedding is not None:
        model.lm_head = embedding.T.copy()
        if "lm_head.weight" in missing:
            missing.remove("lm_head.weight")
    else:
        head = take("lm_head.weight")
        if head is not None:
            model.lm_head = head.T.copy()

    unexpected = sorted(name for name in checkpoint.names
                        if name not in consumed and not name.endswith(".inv_freq"))

    report = LoadReport(architecture=architecture, parameters=checkpoint.total_parameters,
                        layers_loaded=layers_loaded, tied_embeddings=bool(tied),
                        missing=missing, unexpected=unexpected)
    return model, report
