# Inference Engine

An LLM serving engine built from the papers up. Paged KV cache, continuous
batching, prefix caching, speculative decoding, quantization, parallelism.

Runs real checkpoints on GPU. Runs entirely on CPU with just NumPy for testing.

## Quick start

```python
import torch
from ie.engine.engine import Engine, EngineConfig
from ie.engine.text import TextEngine
from ie.load.huggingface import load_transformer
from ie.load.tokenizer import Tokenizer
from ie.model.torch_transformer import TorchTransformer

model, _ = load_transformer("Qwen2.5-0.5B-Instruct")
tokenizer = Tokenizer.from_directory("Qwen2.5-0.5B-Instruct")

engine = Engine(TorchTransformer(model, "cuda", torch.float16),
                EngineConfig(block_size=256, num_blocks=2048, capture_graphs=True))
text = TextEngine(engine, tokenizer)

print(text.complete(["The capital of France is"], max_new_tokens=20)[0].text)
# ' Paris. It is the largest city in Europe and the third largest city in the world.'

for piece in text.stream("def fibonacci(n):", max_new_tokens=40):
    print(piece, end="", flush=True)
```

## Benchmark

Qwen2.5-0.5B-Instruct, one A10G, 64 sequences, 13,797 prompt tokens,
8,107 output tokens, greedy, same workload and seed.

| engine | time | throughput |
|---|---|---|
| vLLM 0.10.1.1 | 2.29 s | 3,547 tok/s |
| **this engine** | **4.18 s** | **1,940 tok/s** |
| nano-vllm | 4.50 s | 1,803 tok/s |

How it got there:

| | tok/s |
|---|---|
| per-sequence decode loop, sampling on host | 64 |
| batched decode attention | 219 |
| sampling on device | 1,120 |
| FlashAttention paged kernels | 1,365 |
| CUDA graphs for decode | 1,940 |

Output is byte-identical with graphs on and off.

Latency on a T4, 12-layer model, 16 requests:

| | TTFT p50 | ITL p50 |
|---|---|---|
| cold cache | 864 ms | 109 ms |
| warm prefix cache | 105 ms | 103 ms |

## Install

```bash
pip install numpy                 # CPU only, all tests pass
pip install torch tokenizers      # GPU
pip install flash-attn            # fastest attention path, optional
```

```bash
pytest -q        # 325 tests, ~20s, no GPU needed
```

## What it implements

Paged KV cache with prefix sharing and CPU offload · continuous batching with
chunked prefill and preemption · speculative decoding (draft-target, Medusa,
EAGLE, n-gram) · quantization (int4, int8, fp8) · tensor, expert and pipeline
parallelism · mixture of experts · disaggregated prefill and decode · sliding
window attention with sinks · CUDA graph capture for decode · streaming server
with metrics, autoscaling and cache-aware routing.

Every feature has a test proving it does not change the model's output:
paged == dense, chunked == whole, sharded == unsharded, GPU == CPU,
speculative == greedy.

Examples in `examples/`. GPU runs via `modal run modal_app.py::<function>`.
