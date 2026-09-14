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
