# An LLM inference engine, built from the paper up

A serving engine for transformer models, written to be read. Paged KV cache,
continuous batching, speculative decoding, quantization, tensor and expert and
pipeline parallelism, disaggregated prefill and decode — each one traced back to
the paper it comes from and pinned by a test proving it does not change the
model's output.

It runs real checkpoints. It also runs entirely on CPU with nothing but NumPy,
so the whole control plane is testable on a laptop.

```
pip install numpy
python -m pytest -q          # 325 tests, ~20s, no GPU, no torch
```

## It generates

Qwen2.5-0.5B-Instruct, greedy, on a T4:

```
'The capital of France is'  -> ' Paris. It is the largest city in Europe and the
                                third largest city in the world.'
'def fibonacci(n):'         -> '\n    if n <= 1:\n        return n\n    else:
                                \n        return(fibonacci(n-'
'Water boils at'            -> ' 212 °F or 100 °C and ice melts at 32 °F'
```

## Throughput

Qwen2.5-0.5B-Instruct on one A10G, 64 sequences, 13,797 prompt tokens,
8,107 generated tokens, greedy, identical workload and seed.

| engine | time | throughput |
|---|---|---|
| vLLM 0.10.1.1 | 2.29 s | 3,547 tok/s |
| this engine | 7.24 s | 1,120 tok/s |

vLLM is 3.2x faster and that is the honest number. The gap is one thing:
vLLM calls a fused paged-attention CUDA kernel, this engine gathers blocks and
calls `scaled_dot_product_attention`. Everything else — paging, scheduling,
batching, the cache — is the same design.

Getting to 1,120 took two fixes worth knowing about:

| change | tok/s |
|---|---|
| per-sequence decode loop, sampling on CPU | 64 |
| batched decode attention | 219 |
| sampling moved onto the device | 1,120 |

The second one mattered most. Sampling on the host shipped a
`[64, 151665]` logits tensor to CPU every step and synchronised the GPU
190 times. Moving `argmax` to the device removed both.

## What is in here

| area | module |
|---|---|
| Paged KV cache, prefix sharing, offload tier | `ie/core/block_manager.py`, `ie/core/offload.py` |
| Continuous batching, chunked prefill, preemption | `ie/core/scheduler.py` |
| Paged, windowed, tree and sink attention | `ie/kernels/` |
| Speculative decoding: draft-target, Medusa, EAGLE, n-gram | `ie/spec/` |
| Quantization: int4, int8, fp8 e4m3 and e5m2 | `ie/quant/` |
| Tensor, expert and pipeline parallelism | `ie/parallel/` |
| Mixture of experts | `ie/model/moe.py` |
| Disaggregated prefill and decode | `ie/serve/disaggregated.py` |
| Streaming server, metrics, autoscaling, routing | `ie/serve/` |
| Safetensors reader and checkpoint loader | `ie/load/` |
| Benchmark harness: TTFT, ITL, throughput | `ie/bench/` |

## How correctness is established

Every feature ships with a test showing the output does not change:

- paged attention == dense attention
- chunked prefill == whole prefill
- preemption == no preemption
- tensor-sharded == unsharded
- pipelined == single stage
- torch on GPU == numpy on CPU, in fp32 and fp16
- greedy speculation == plain greedy, for draft-target, Medusa, EAGLE and n-gram
- a mixture of identical experts == one dense MLP
- speculative sampling == the target distribution, over 20,000 trials

Each of those was then mutation-checked: break the feature, confirm the test fails.

## Bugs those tests caught

- a full prefix-cache hit reused every block, leaving nothing to compute
- `can_allocate` double-counted capacity when a reusable block sat on the free list
- blocks were claimed before the admission decision, losing cache hits
- FP8 E4M3's maximum was 240 instead of 448, and the clamp ran before rounding
- Medusa never wrote accepted tokens' K,V — invisible until acceptance worked
- a disaggregated sequence arriving already complete was decoded one extra time

Two more only appeared once a real checkpoint was loaded, and no property test
could have found them:

- **QKV biases.** Qwen2 has them, this engine did not. The loader reported zero
  missing weights and listed the biases as unexpected. Output was noise.
- **Rotary convention.** HF checkpoints rotate halves of the head dimension;
  this engine rotated interleaved pairs. Both preserve norms, both give relative
  position invariance, and every property test passed for both. Only the trained
  weights tell them apart. Output started coherent and collapsed into repetition.

Property tests prove a mechanism is self-consistent. They cannot prove it matches
the convention the weights were trained under.

## Measured

| claim | number | where |
|---|---|---|
| decode is memory bound | step time -4% while throughput rose 133x across batch 1..128 | T4 |
| prefix caching cuts TTFT | 864 ms -> 105 ms | T4 |
| batching trades latency for throughput | ITL 18 -> 113 ms, 56 -> 129 tok/s | T4 |
| fused attention beats materialised scores | 6.79 -> 0.56 ms, 396 -> 136 MiB | T4 |
| host transfer is the expensive hop | 5.8 GB/s vs 110 GB/s on device | T4 |
| GQA is a serving decision | 4x concurrency at equal memory | `examples/capacity_planning.py` |
| a shared prefix buys concurrency | 35 -> 156 concurrent requests | `examples/capacity_planning.py` |
| MoE decouples capacity from cost | 64x parameters, 2x compute | `examples/mixture_of_experts.py` |
| cache-aware routing | 73% -> 93% prefix hit rate | `examples/serving.py` |

## Running it

```python
from ie.engine.engine import Engine, EngineConfig
from ie.engine.text import TextEngine
from ie.load.huggingface import load_transformer
from ie.load.tokenizer import Tokenizer
from ie.model.torch_transformer import TorchTransformer
import torch

model, report = load_transformer("Qwen2.5-0.5B-Instruct")
tokenizer = Tokenizer.from_directory("Qwen2.5-0.5B-Instruct")

engine = Engine(TorchTransformer(model, "cuda", torch.float16),
                EngineConfig(block_size=16, num_blocks=4096))

for piece in TextEngine(engine, tokenizer).stream("The capital of France is"):
    print(piece, end="", flush=True)
```

Eleven runnable examples are in `examples/`. GPU work runs on Modal via
`modal run modal_app.py::<function>`.

## Not here

Vision, speech and audio models. CUDA graph capture is implemented and tested
(`ie/model/graph_runner.py`) but not wired into the forward pass, which would
need the model to read from static buffers. A fused paged-attention kernel,
which is the whole remaining throughput gap.
