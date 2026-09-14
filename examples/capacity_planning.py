from ie.model.transformer import ModelConfig
from ie.serve.autoscale import AdmissionController, Autoscaler, CapacityPlan, ScalingPolicy

MODELS = [
    ("7B-ish   32L 4096H 32/8 heads", ModelConfig(vocab_size=32000, hidden_size=4096,
                                                  num_layers=32, num_heads=32,
                                                  num_kv_heads=8, head_dim=128,
                                                  intermediate_size=11008)),
    ("7B MHA   32L 4096H 32/32 heads", ModelConfig(vocab_size=32000, hidden_size=4096,
                                                   num_layers=32, num_heads=32,
                                                   num_kv_heads=32, head_dim=128,
                                                   intermediate_size=11008)),
]
GPU_BYTES_FOR_CACHE = 20 * 2**30
BLOCK_SIZE = 16

print("how many concurrent requests fit in 20 GiB of KV cache\n")
print(f"{'model':<34}{'B/token':>10}{'blocks':>9}{'2k ctx':>9}{'8k ctx':>9}{'32k ctx':>9}")
print("-" * 80)

for label, config in MODELS:
    per_token = config.kv_bytes_per_token
    num_blocks = int(GPU_BYTES_FOR_CACHE // (per_token * BLOCK_SIZE))
    plan = CapacityPlan(num_blocks=num_blocks, block_size=BLOCK_SIZE, bytes_per_token=per_token)
    fits = [plan.concurrent_sequences(prompt_tokens=n - 256, generated_tokens=256)
            for n in (2048, 8192, 32768)]
    print(f"{label:<34}{per_token:>10,}{num_blocks:>9,}{fits[0]:>9}{fits[1]:>9}{fits[2]:>9}")

print("\ngrouped-query attention is a serving decision, not only a quality one\n")

config = MODELS[0][1]
per_token = config.kv_bytes_per_token
plan = CapacityPlan(num_blocks=int(GPU_BYTES_FOR_CACHE // (per_token * BLOCK_SIZE)),
                    block_size=BLOCK_SIZE, bytes_per_token=per_token)
print(f"{'shared system prompt':<24}{'concurrent requests':>22}")
for shared in (0, 512, 1024, 1792):
    fits = plan.concurrent_sequences(prompt_tokens=2048, generated_tokens=256,
                                     shared_prefix_tokens=shared)
    print(f"{shared:>16} tokens{fits:>22}")

print("\nautoscaling a burst\n")
policy = ScalingPolicy(target_per_replica=8, min_replicas=1, max_replicas=12,
                       scale_to_zero_after=60.0, cold_start_seconds=25.0)
scaler = Autoscaler(policy)
gate = AdmissionController(max_queue_depth=64)

print(f"{'t':>4}{'in flight':>11}{'replicas':>10}{'load':>8}  action")
for tick, in_flight in enumerate([2, 6, 20, 48, 90, 90, 40, 12, 3, 0, 0, 0]):
    decision = scaler.observe(in_flight, seconds_since_last=30.0)
    print(f"{tick:>4}{in_flight:>11}{decision.replicas:>10}{decision.load_factor:>8.2f}"
          f"  {decision.action.name.lower()} · {decision.reason}")
