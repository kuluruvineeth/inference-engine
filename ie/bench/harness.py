from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..engine.engine import Engine
from ..layers.functional import sample_from_logits
from .metrics import BenchmarkResult, RequestTrace


@dataclass(slots=True)
class Request:
    prompt_ids: list[int]
    max_new_tokens: int = 32
    temperature: float = 0.0
    arrival: float = 0.0


@dataclass(slots=True)
class Workload:
    requests: list[Request] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def is_offline(self) -> bool:
        return all(request.arrival == 0.0 for request in self.requests)


def offline_workload(prompts: list[list[int]], max_new_tokens: int = 32,
                     temperature: float = 0.0) -> Workload:
    return Workload([Request(p, max_new_tokens, temperature) for p in prompts])


def poisson_workload(prompts: list[list[int]], requests_per_second: float,
                     max_new_tokens: int = 32, temperature: float = 0.0,
                     seed: int = 0) -> Workload:
    generator = np.random.default_rng(seed)
    gaps = generator.exponential(1.0 / requests_per_second, size=len(prompts))
    arrivals = np.cumsum(gaps)
    return Workload([Request(p, max_new_tokens, temperature, float(a))
                     for p, a in zip(prompts, arrivals)])


def shared_prefix_prompts(count: int, prefix_len: int, suffix_len: int, vocab_size: int,
                          seed: int = 0) -> list[list[int]]:
    if prefix_len + suffix_len > vocab_size:
        raise ValueError("prompt cannot be longer than the vocabulary it is drawn from")
    generator = np.random.default_rng(seed)
    prefix = list(range(1, 1 + prefix_len))
    return [prefix + generator.integers(prefix_len + 1, vocab_size, size=suffix_len).tolist()
            for _ in range(count)]


def distinct_prompts(count: int, prompt_len: int, vocab_size: int,
                     seed: int = 0) -> list[list[int]]:
    generator = np.random.default_rng(seed)
    return [generator.integers(1, vocab_size, size=prompt_len).tolist()
            for _ in range(count)]


def run_benchmark(engine: Engine, workload: Workload,
                  max_steps: int = 1_000_000) -> BenchmarkResult:
    started = time.perf_counter()
    traces: dict[int, RequestTrace] = {}
    pending = sorted(workload.requests, key=lambda r: r.arrival)
    next_index = 0
    sequences: dict[int, RequestTrace] = {}
    steps = 0

    def admit_due(now: float) -> None:
        nonlocal next_index
        while next_index < len(pending) and pending[next_index].arrival <= now:
            request = pending[next_index]
            seq = engine.submit(request.prompt_ids, max_new_tokens=request.max_new_tokens,
                                temperature=request.temperature)
            trace = RequestTrace(seq_id=seq.id, prompt_len=len(request.prompt_ids),
                                 submitted_at=started, arrived_at=started + request.arrival)
            traces[seq.id] = trace
            sequences[seq.id] = trace
            next_index += 1

    admit_due(0.0)

    while steps < max_steps:
        elapsed = time.perf_counter() - started
        admit_due(elapsed)

        if not engine.scheduler.has_work:
            if next_index >= len(pending):
                break
            time.sleep(min(0.001, max(0.0, pending[next_index].arrival - elapsed)))
            continue

        batch = engine.scheduler.schedule()
        if not batch:
            break

        _, logits = engine.forward(batch)
        temperatures = np.array([seq.temperature for seq in batch.sequences], dtype=np.float32)
        token_ids = sample_from_logits(logits, temperatures, engine.generator).tolist()

        before = {seq.id: seq.num_generated for seq in batch.sequences}
        engine.scheduler.finish_step(batch, token_ids)
        now = time.perf_counter()
        steps += 1

        for seq in batch.sequences:
            trace = sequences.get(seq.id)
            if trace is None:
                continue
            for _ in range(seq.num_generated - before[seq.id]):
                trace.token_times.append(now)
            if seq.is_finished and trace.finished_at is None:
                trace.finished_at = now

    wall = time.perf_counter() - started
    stats = engine.scheduler.stats
    return BenchmarkResult(
        traces=list(traces.values()),
        wall_time=wall,
        engine_steps=steps,
        prefill_steps=stats.prefill_steps,
        decode_steps=stats.decode_steps,
        preemptions=stats.preemptions,
        tokens_reused=stats.tokens_reused_from_cache,
    )
