import modal

REPO = "/root/engine"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=2.0", "pytest>=8.0", "torch>=2.4", "tokenizers>=0.20", "huggingface_hub>=0.25")
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


@app.function(image=image, gpu="T4", timeout=1800)
def kernel_fusion() -> str:
    import time

    import torch

    device = torch.device("cuda")
    rows, hidden, inner = 4096, 2048, 8192
    x = torch.randn(rows, hidden, device=device, dtype=torch.float16)
    gate = torch.randn(hidden, inner, device=device, dtype=torch.float16)
    up = torch.randn(hidden, inner, device=device, dtype=torch.float16)
    down = torch.randn(inner, hidden, device=device, dtype=torch.float16)

    def unfused(x):
        a = x @ gate
        b = torch.sigmoid(a)
        c = a * b
        d = x @ up
        e = c * d
        return e @ down

    def fused(x):
        return (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down

    compiled = torch.compile(fused)

    def measure(fn, label):
        for _ in range(3):
            fn(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(20):
            fn(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / 20
        peak = torch.cuda.max_memory_allocated() / 2**20
        return f"{label:<26}{elapsed*1000:>10.2f}{peak:>14.0f}"

    rows_out = [f"swiglu over {rows}x{hidden} -> {inner}",
                "",
                f"{'variant':<26}{'ms/call':>10}{'peak MiB':>14}",
                measure(unfused, "separate ops"),
                measure(fused, "silu fused"),
                measure(compiled, "torch.compile")]

    attn_q = torch.randn(1, 16, 2048, 64, device=device, dtype=torch.float16)
    attn_k = torch.randn(1, 16, 2048, 64, device=device, dtype=torch.float16)
    attn_v = torch.randn(1, 16, 2048, 64, device=device, dtype=torch.float16)

    def naive_attention():
        scores = (attn_q @ attn_k.transpose(-1, -2)) / 8.0
        mask = torch.ones(2048, 2048, device=device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ attn_v

    def flash_attention():
        return torch.nn.functional.scaled_dot_product_attention(
            attn_q, attn_k, attn_v, is_causal=True)

    rows_out += ["", "attention over 2048 tokens, 16 heads, dim 64", "",
                 f"{'variant':<26}{'ms/call':>10}{'peak MiB':>14}",
                 measure(lambda _: naive_attention(), "materialised scores"),
                 measure(lambda _: flash_attention(), "fused sdpa")]

    report = "\n".join(rows_out)
    print(report)
    return report


@app.function(image=image, gpu="T4", timeout=1800)
def hardware_profile() -> str:
    import time

    import torch

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)

    rows = [f"{properties.name}",
            f"  streaming multiprocessors {properties.multi_processor_count}",
            f"  total memory              {properties.total_memory / 2**30:.1f} GiB",
            f"  compute capability        {properties.major}.{properties.minor}",
            ""]

    sizes_mb = [1, 4, 16, 64, 256]
    rows.append(f"{'transfer MiB':>13}{'HtoD GB/s':>12}{'DtoH GB/s':>12}{'DtoD GB/s':>12}")
    for size_mb in sizes_mb:
        count = size_mb * 2**20 // 4
        host = torch.randn(count, pin_memory=True)
        dev = torch.empty(count, device=device)
        other = torch.empty(count, device=device)

        def timed(fn):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            return size_mb / 2**10 / ((time.perf_counter() - started) / 5)

        h2d = timed(lambda: dev.copy_(host, non_blocking=True))
        d2h = timed(lambda: host.copy_(dev, non_blocking=True))
        d2d = timed(lambda: other.copy_(dev))
        rows.append(f"{size_mb:>13}{h2d:>12.1f}{d2h:>12.1f}{d2d:>12.1f}")

    rows += ["", "device-to-device is the bandwidth the KV cache actually runs at;",
             "host transfers are what offloading a block costs"]
    report = "\n".join(rows)
    print(report)
    return report


MODEL_CACHE = modal.Volume.from_name("ie-models", create_if_missing=True)


@app.function(image=image, volumes={"/models": MODEL_CACHE}, timeout=3600)
def fetch_model(repo_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id, local_dir=f"/models/{repo_id}",
                             allow_patterns=["*.safetensors", "*.json", "*.txt"])
    MODEL_CACHE.commit()
    import os
    files = sorted(os.listdir(path))
    report = f"{repo_id} -> {path}\n" + "\n".join(f"  {f}" for f in files)
    print(report)
    return report


@app.function(image=image, gpu="T4", volumes={"/models": MODEL_CACHE}, timeout=3600)
def real_model_generate(repo_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
    import sys

    sys.path.insert(0, REPO)
    import torch

    from ie.engine.engine import Engine, EngineConfig
    from ie.engine.text import TextEngine
    from ie.load.huggingface import load_transformer
    from ie.load.tokenizer import Tokenizer
    from ie.model.torch_transformer import TorchTransformer

    directory = f"/models/{repo_id}"
    model, report = load_transformer(directory)
    tokenizer = Tokenizer.from_directory(directory)

    lines = [f"loaded {report.architecture}",
             f"  parameters      {report.parameters/1e6:.1f}M",
             f"  layers          {report.layers_loaded}",
             f"  tied embeddings {report.tied_embeddings}",
             f"  missing         {len(report.missing)}",
             f"  unexpected      {report.unexpected[:3]}",
             f"  vocab           {tokenizer.vocab_size}",
             f"  eos id          {tokenizer.eos_id}",
             ""]

    engine = Engine(TorchTransformer(model, "cuda", torch.float16),
                    EngineConfig(block_size=16, num_blocks=2048, max_batch_tokens=2048))
    text = TextEngine(engine, tokenizer)

    prompts = ["The capital of France is",
               "def fibonacci(n):",
               "Water boils at"]
    for completion in text.complete(prompts, max_new_tokens=24, temperature=0.0):
        lines.append(f"  {completion.prompt!r}")
        lines.append(f"    -> {completion.text!r}")

    report_text = "\n".join(lines)
    print(report_text)
    return report_text


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.2", "huggingface_hub>=0.25")
    .add_local_dir("ie", f"{REPO}/ie")
)

BENCH_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _workload(tokenizer, count, seed=0):
    import random

    generator = random.Random(seed)
    vocab = tokenizer.vocab_size
    prompts, lengths = [], []
    for _ in range(count):
        prompt_len = generator.randint(96, 320)
        prompts.append([generator.randrange(1000, min(vocab, 60000)) for _ in range(prompt_len)])
        lengths.append(generator.randint(64, 192))
    return prompts, lengths


@app.function(image=vllm_image, gpu="A10G", volumes={"/models": MODEL_CACHE},
              timeout=3600)
def vllm_throughput(count: int = 64) -> str:
    import time

    from tokenizers import Tokenizer
    from vllm import LLM, SamplingParams

    directory = f"/models/{BENCH_MODEL}"
    tokenizer = Tokenizer.from_file(f"{directory}/tokenizer.json")

    class Shim:
        vocab_size = tokenizer.get_vocab_size()

    prompts, lengths = _workload(Shim(), count)
    params = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in lengths]

    engine = LLM(model=directory, enforce_eager=False, max_model_len=1024,
                 gpu_memory_utilization=0.85, disable_log_stats=True)

    started = time.perf_counter()
    engine.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
    elapsed = time.perf_counter() - started

    total_out = sum(lengths)
    report = (f"vllm 0.10.1.1  {count} seqs  {sum(len(p) for p in prompts)} in / "
              f"{total_out} out  {elapsed:.2f}s  {total_out/elapsed:.0f} tok/s")
    print(report)
    return report


@app.function(image=image, gpu="A10G", volumes={"/models": MODEL_CACHE}, timeout=3600)
def our_throughput(count: int = 64) -> str:
    import sys
    import time

    sys.path.insert(0, REPO)
    import torch

    from ie.engine.engine import Engine, EngineConfig
    from ie.load.huggingface import load_transformer
    from ie.load.tokenizer import Tokenizer
    from ie.model.torch_transformer import TorchTransformer

    directory = f"/models/{BENCH_MODEL}"
    model, _ = load_transformer(directory)
    tokenizer = Tokenizer.from_directory(directory)
    prompts, lengths = _workload(tokenizer, count)

    engine = Engine(TorchTransformer(model, "cuda", torch.float16),
                    EngineConfig(block_size=16, num_blocks=8192,
                                 max_batch_tokens=8192, max_batch_sequences=256))
    for prompt, length in zip(prompts, lengths):
        engine.submit(prompt, max_new_tokens=length, temperature=0.0)

    started = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - started

    total_out = sum(lengths)
    report = (f"ours           {count} seqs  {sum(len(p) for p in prompts)} in / "
              f"{total_out} out  {elapsed:.2f}s  {total_out/elapsed:.0f} tok/s")
    print(report)
    return report


FLASH_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

flash_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("numpy>=2.0", "tokenizers>=0.20", "huggingface_hub>=0.25")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(FLASH_WHEEL)
    .add_local_dir("ie", f"{REPO}/ie")
)


@app.function(image=flash_image, gpu="A10G", volumes={"/models": MODEL_CACHE}, timeout=3600)
def flash_throughput(count: int = 64) -> str:
    import sys
    import time

    sys.path.insert(0, REPO)
    import torch

    from ie.engine.engine import Engine, EngineConfig
    from ie.kernels import flash_backend
    from ie.load.huggingface import load_transformer
    from ie.load.tokenizer import Tokenizer
    from ie.model.torch_transformer import TorchTransformer

    directory = f"/models/{BENCH_MODEL}"
    model, _ = load_transformer(directory)
    tokenizer = Tokenizer.from_directory(directory)
    prompts, lengths = _workload(tokenizer, count)

    engine = Engine(TorchTransformer(model, "cuda", torch.float16),
                    EngineConfig(block_size=256, num_blocks=2048,
                                 max_batch_tokens=16384, max_batch_sequences=512))
    for prompt, length in zip(prompts, lengths):
        engine.submit(prompt, max_new_tokens=length, temperature=0.0)

    started = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - started

    total_out = sum(lengths)
    report = (f"flash-attn available: {flash_backend.AVAILABLE}\n"
              f"ours+flash     {count} seqs  {sum(len(p) for p in prompts)} in / "
              f"{total_out} out  {elapsed:.2f}s  {total_out/elapsed:.0f} tok/s")
    print(report)
    return report


@app.function(image=flash_image, gpu="A10G", timeout=900)
def flash_import_check() -> str:
    import subprocess
    import sys

    lines = []
    try:
        import torch
        lines.append(f"torch {torch.__version__}")
    except Exception as error:
        lines.append(f"torch import failed: {error}")

    try:
        import flash_attn
        lines.append(f"flash_attn {flash_attn.__version__}")
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        lines.append("both functions imported")
    except Exception as error:
        lines.append(f"flash_attn import failed: {type(error).__name__}: {error}")

    listing = subprocess.run([sys.executable, "-m", "pip", "list"],
                            capture_output=True, text=True).stdout
    lines += [line for line in listing.splitlines() if "flash" in line.lower() or "torch" in line.lower()]
    report = "\n".join(lines)
    print(report)
    return report
