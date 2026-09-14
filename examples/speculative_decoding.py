import numpy as np

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig, Transformer
from ie.spec.speculator import DraftModel, SpeculativeEngine

CONFIG = ModelConfig(vocab_size=128, hidden_size=64, num_layers=3, num_heads=4,
                     num_kv_heads=2, head_dim=16, intermediate_size=128)
PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]
NEW_TOKENS = 32


def measure(draft_seed, lookahead):
    target = build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=256), model_seed=1)
    draft = DraftModel(Transformer(CONFIG, seed=draft_seed), block_size=16, num_blocks=256)
    spec = SpeculativeEngine(target, draft, lookahead=lookahead)
    spec.generate(PROMPT, max_new_tokens=NEW_TOKENS, temperature=1.0, seed=7)
    return spec.stats


print(f"{'draft quality':<22}{'k':>3}{'accept':>9}{'tok/pass':>10}{'passes':>8}")
print("-" * 52)

for label, seed in [("perfect (same model)", 1), ("close", 4), ("unrelated", 99)]:
    for k in (2, 4, 8):
        st = measure(seed, k)
        print(f"{label:<22}{k:>3}{st.acceptance_rate:>8.0%}"
              f"{st.tokens_per_target_pass:>10.2f}{st.target_passes:>8}")

print(f"\nwithout speculation: 1.00 tok/pass, {NEW_TOKENS} target passes")
