import numpy as np

from ie.model.moe import MixtureOfExperts, RoutingStats
from ie.model.transformer import ModelConfig
from ie.parallel.expert_parallel import ExpertParallelMoE

CONFIG = ModelConfig(vocab_size=512, hidden_size=256, num_layers=1, num_heads=8,
                     num_kv_heads=2, head_dim=32, intermediate_size=512)
TOKENS = np.random.default_rng(0).standard_normal((512, CONFIG.hidden_size)).astype(np.float32)


def parameter_count(layer):
    per_expert = (layer.experts[0].gate_proj.size + layer.experts[0].up_proj.size
                  + layer.experts[0].down_proj.size)
    return per_expert * layer.num_experts, per_expert * layer.top_k


print(f"{'experts':>8}{'top-k':>7}{'active':>9}{'total params':>15}{'active params':>15}"
      f"{'imbalance':>11}{'idle':>6}")
print("-" * 72)

for num_experts, top_k in [(1, 1), (4, 1), (8, 2), (16, 2), (64, 2), (64, 8)]:
    layer = MixtureOfExperts(CONFIG, num_experts=num_experts, top_k=top_k, seed=1)
    stats = RoutingStats()
    layer.forward(TOKENS, stats)
    total, active = parameter_count(layer)
    print(f"{num_experts:>8}{top_k:>7}{layer.active_fraction:>8.0%}"
          f"{total/1e6:>14.1f}M{active/1e6:>14.1f}M"
          f"{stats.imbalance:>11.2f}{stats.idle_experts:>6}")

print("\nexpert parallelism: same maths, experts split across ranks")
layer = MixtureOfExperts(CONFIG, num_experts=8, top_k=2, seed=1)
single = layer.forward(TOKENS)
for world_size in (1, 2, 4, 8):
    parallel = ExpertParallelMoE(layer, world_size)
    out = parallel.forward(TOKENS)
    load = parallel.tokens_per_rank(TOKENS)
    print(f"  {world_size} ranks  {parallel.experts_per_rank} experts each  "
          f"matches single device: {np.allclose(out, single, atol=1e-5)}  "
          f"token load per rank: {load}")
