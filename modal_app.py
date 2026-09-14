import modal

REPO = "/root/engine"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=2.0", "pytest>=8.0", "torch>=2.4")
    .add_local_dir("ie", f"{REPO}/ie")
    .add_local_dir("tests", f"{REPO}/tests")
    .add_local_dir("examples", f"{REPO}/examples")
)

app = modal.App("inference-engine")


@app.function(image=image, timeout=900)
def run_tests() -> str:
    import subprocess

    result = subprocess.run(["python", "-m", "pytest", "-q", "tests"],
                            cwd=REPO, capture_output=True, text=True)
    report = result.stdout[-4000:] + result.stderr[-2000:]
    print(report)
    return report


@app.function(image=image, gpu="T4", timeout=900)
def gpu_report() -> str:
    import torch

    lines = [
        f"torch {torch.__version__}",
        f"cuda available {torch.cuda.is_available()}",
        f"device {torch.cuda.get_device_name(0)}",
        f"capability {torch.cuda.get_device_capability(0)}",
        f"memory {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
    ]
    report = "\n".join(lines)
    print(report)
    return report


@app.function(image=image, gpu="T4", timeout=1800)
def paged_attention_matches_torch(num_tokens: int = 128, heads: int = 8,
                                  kv_heads: int = 2, dim: int = 64) -> str:
    import sys

    sys.path.insert(0, REPO)
    import numpy as np
    import torch

    from ie.kernels.reference import dense_attention

    generator = np.random.default_rng(0)
    q = generator.standard_normal((num_tokens, heads, dim)).astype(np.float32)
    k = generator.standard_normal((num_tokens, kv_heads, dim)).astype(np.float32)
    v = generator.standard_normal((num_tokens, kv_heads, dim)).astype(np.float32)

    expected = dense_attention(q, k, v, causal=True)

    device = torch.device("cuda")
    tq = torch.from_numpy(q).to(device).permute(1, 0, 2)
    tk = torch.from_numpy(k).to(device).permute(1, 0, 2)
    tv = torch.from_numpy(v).to(device).permute(1, 0, 2)
    repeats = heads // kv_heads
    tk = tk.repeat_interleave(repeats, dim=0)
    tv = tv.repeat_interleave(repeats, dim=0)

    out = torch.nn.functional.scaled_dot_product_attention(
        tq.unsqueeze(0), tk.unsqueeze(0), tv.unsqueeze(0), is_causal=True)
    actual = out.squeeze(0).permute(1, 0, 2).cpu().numpy()

    drift = float(np.max(np.abs(actual - expected)))
    report = (f"tokens={num_tokens} heads={heads} kv_heads={kv_heads} dim={dim}\n"
              f"max abs difference vs cpu reference: {drift:.3e}\n"
              f"match within 1e-3: {drift < 1e-3}")
    print(report)
    return report


@app.function(image=image, gpu="T4", timeout=1800)
def decode_is_memory_bound(hidden: int = 4096, layers: int = 32) -> str:
    import time

    import torch

    device = torch.device("cuda")
    weights = [torch.randn(hidden, hidden, device=device, dtype=torch.float16)
               for _ in range(layers)]
    properties = torch.cuda.get_device_properties(0)
    weight_bytes = sum(w.numel() * w.element_size() for w in weights)

    rows = [f"device {properties.name}",
            f"weights {weight_bytes / 2**30:.2f} GiB across {layers} layers",
            "",
            f"{'batch':>6}{'ms/step':>10}{'tok/s':>10}{'GB/s':>9}{'ms/token':>10}"]

    for batch in (1, 2, 4, 8, 16, 32, 64, 128):
        x = torch.randn(batch, hidden, device=device, dtype=torch.float16)
        for _ in range(3):
            h = x
            for w in weights:
                h = h @ w
        torch.cuda.synchronize()

        started = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            h = x
            for w in weights:
                h = h @ w
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / iterations

        achieved = weight_bytes / elapsed / 1e9
        rows.append(f"{batch:>6}{elapsed * 1000:>10.2f}{batch / elapsed:>10.0f}"
                    f"{achieved:>9.0f}{elapsed * 1000 / batch:>10.3f}")

    rows.append("")
    rows.append("weight bytes are identical at every batch size; if ms/step barely moves")
    rows.append("as batch grows, the step is bandwidth bound, not compute bound")
    report = "\n".join(rows)
    print(report)
    return report


@app.function(image=image, gpu="T4", timeout=1800)
def torch_matches_numpy() -> str:
    import sys

    sys.path.insert(0, REPO)
    import numpy as np
    import torch

    from ie.engine.engine import Engine, EngineConfig, build_engine
    from ie.model.torch_transformer import TorchTransformer
    from ie.model.transformer import ModelConfig, Transformer

    config = ModelConfig(vocab_size=256, hidden_size=128, num_layers=4, num_heads=8,
                         num_kv_heads=2, head_dim=16, intermediate_size=256)
    engine_config = EngineConfig(block_size=16, num_blocks=512, max_batch_tokens=1024)
    prompts = [[3, 1, 4, 1, 5, 9, 2, 6], [7, 7, 7, 7], list(range(20, 44))]

    cpu = build_engine(config, engine_config, model_seed=1)
    expected = cpu.generate(prompts, max_new_tokens=12, temperature=0.0)

    source = Transformer(config, seed=1)
    for dtype, tolerance in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        gpu = Engine(TorchTransformer(source, "cuda", dtype), engine_config)
        actual = gpu.generate(prompts, max_new_tokens=12, temperature=0.0)
        agree = sum(a == b for row_a, row_b in zip(expected, actual)
                    for a, b in zip(row_a, row_b))
        total = sum(len(row) for row in expected)
        print(f"{tolerance}: exact rows {sum(a == b for a, b in zip(expected, actual))}"
              f"/{len(prompts)}  token agreement {agree}/{total}")

    return "done"


@app.function(image=image, gpu="T4", timeout=3600)
def gpu_benchmark() -> str:
    import sys

    sys.path.insert(0, REPO)
    import torch

    from ie.bench.harness import distinct_prompts, offline_workload, run_benchmark, shared_prefix_prompts
    from ie.engine.engine import Engine, EngineConfig
    from ie.model.torch_transformer import TorchTransformer
    from ie.model.transformer import ModelConfig, Transformer

    vocab = 4096
    config = ModelConfig(vocab_size=vocab, hidden_size=1024, num_layers=12, num_heads=16,
                         num_kv_heads=4, head_dim=64, intermediate_size=2816)
    source = Transformer(config, seed=1)
    rows = [f"model: {config.num_layers}L {config.hidden_size}H "
            f"{config.num_heads}/{config.num_kv_heads} heads, vocab {vocab}",
            f"{'scenario':<32}{'ttft p50':>10}{'itl p50':>9}{'tok/s':>10}{'reused':>9}"]

    def engine():
        return Engine(TorchTransformer(source, "cuda", torch.float16),
                      EngineConfig(block_size=16, num_blocks=4096, max_batch_tokens=4096))

    def show(label, result):
        s = result.summary()
        rows.append(f"{label:<32}{s['ttft_p50']:>10.3f}{s['itl_p50']:>9.4f}"
                    f"{s['output_tok_per_s']:>10.1f}{s['tokens_reused']:>9}")

    distinct = distinct_prompts(16, prompt_len=256, vocab_size=vocab, seed=1)
    shared = shared_prefix_prompts(16, prefix_len=224, suffix_len=32, vocab_size=vocab, seed=1)

    show("16 distinct prompts", run_benchmark(engine(), offline_workload(distinct, 32)))

    warm = engine()
    run_benchmark(warm, offline_workload(shared[:1], 32))
    show("16 shared prefix (warm)", run_benchmark(warm, offline_workload(shared[1:], 32)))

    for batch_cap in (1, 4, 16):
        eng = Engine(TorchTransformer(source, "cuda", torch.float16),
                     EngineConfig(block_size=16, num_blocks=4096, max_batch_tokens=4096,
                                  max_batch_sequences=batch_cap))
        show(f"max {batch_cap} sequences in flight",
             run_benchmark(eng, offline_workload(distinct, 32)))

    report = "\n".join(rows)
    print(report)
    return report
