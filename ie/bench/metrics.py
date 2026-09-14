from __future__ import annotations

from dataclasses import dataclass, field


def percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(slots=True)
class RequestTrace:
    seq_id: int
    prompt_len: int
    submitted_at: float
    arrived_at: float
    token_times: list[float] = field(default_factory=list)
    finished_at: float | None = None

    @property
    def num_generated(self) -> int:
        return len(self.token_times)

    @property
    def time_to_first_token(self) -> float | None:
        if not self.token_times:
            return None
        return self.token_times[0] - self.arrived_at

    @property
    def inter_token_latencies(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]

    @property
    def total_latency(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.arrived_at


@dataclass(slots=True)
class BenchmarkResult:
    traces: list[RequestTrace]
    wall_time: float
    engine_steps: int
    prefill_steps: int
    decode_steps: int
    preemptions: int
    tokens_reused: int

    @property
    def num_requests(self) -> int:
        return len(self.traces)

    @property
    def generated_tokens(self) -> int:
        return sum(trace.num_generated for trace in self.traces)

    @property
    def prompt_tokens(self) -> int:
        return sum(trace.prompt_len for trace in self.traces)

    @property
    def completed(self) -> list[RequestTrace]:
        return [t for t in self.traces if t.finished_at is not None]

    @property
    def ttft_samples(self) -> list[float]:
        return [t.time_to_first_token for t in self.traces if t.time_to_first_token is not None]

    @property
    def itl_samples(self) -> list[float]:
        return [x for t in self.traces for x in t.inter_token_latencies]

    @property
    def latency_samples(self) -> list[float]:
        return [t.total_latency for t in self.completed]

    @property
    def output_throughput(self) -> float:
        return self.generated_tokens / self.wall_time if self.wall_time else 0.0

    @property
    def request_throughput(self) -> float:
        return len(self.completed) / self.wall_time if self.wall_time else 0.0

    @property
    def tokens_per_step(self) -> float:
        return self.generated_tokens / self.engine_steps if self.engine_steps else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "requests": self.num_requests,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "wall_time": round(self.wall_time, 4),
            "ttft_p50": round(percentile(self.ttft_samples, 0.50), 4),
            "ttft_p90": round(percentile(self.ttft_samples, 0.90), 4),
            "ttft_p99": round(percentile(self.ttft_samples, 0.99), 4),
            "itl_p50": round(percentile(self.itl_samples, 0.50), 4),
            "itl_p90": round(percentile(self.itl_samples, 0.90), 4),
            "latency_p50": round(percentile(self.latency_samples, 0.50), 4),
            "latency_p99": round(percentile(self.latency_samples, 0.99), 4),
            "output_tok_per_s": round(self.output_throughput, 2),
            "requests_per_s": round(self.request_throughput, 3),
            "tokens_per_step": round(self.tokens_per_step, 3),
            "engine_steps": self.engine_steps,
            "preemptions": self.preemptions,
            "tokens_reused": self.tokens_reused,
        }

    def table(self) -> str:
        rows = self.summary()
        width = max(len(key) for key in rows)
        return "\n".join(f"  {key:<{width}}  {value}" for key, value in rows.items())
