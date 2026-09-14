from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig
from ie.serve.openai import completion_chunk, done_event, server_sent_event
from ie.serve.server import InferenceServer

VOCAB = 2048
CONFIG = ModelConfig(vocab_size=VOCAB, hidden_size=64, num_layers=3, num_heads=4,
                     num_kv_heads=2, head_dim=16, intermediate_size=128)
ENGINE = EngineConfig(block_size=16, num_blocks=512, max_batch_tokens=512)

api = InferenceServer(build_engine(CONFIG, ENGINE, model_seed=1))
api.submit(list(range(100, 116)), max_new_tokens=6, request_id="chat-a")
api.submit(list(range(200, 224)), max_new_tokens=6, request_id="chat-b")
api.submit(list(range(300, 308)), max_new_tokens=6, request_id="chat-c")

print("streaming three concurrent requests as server-sent events\n")
for chunk in api.stream():
    payload = completion_chunk(chunk.request_id, "ie-small", f"<{chunk.token_id}>",
                               index=chunk.index,
                               finish_reason=chunk.stop_reason, created=1700000000)
    line = server_sent_event(payload).strip()
    print(f"  {line[:96]}")
print(f"  {done_event().strip()}")

api.submit(list(range(400, 420)), max_new_tokens=64, request_id="chat-d")
api.step()
before = api.engine.blocks.num_free
api.cancel("chat-d")
print(f"\ncancelled chat-d mid-flight: blocks {before} -> {api.engine.blocks.num_free}")

print("\nmetrics")
for key, value in api.metrics.snapshot().items():
    print(f"  {key:<22}{value}")
