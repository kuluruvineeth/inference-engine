from __future__ import annotations

from dataclasses import dataclass, field

from ..bench.metrics import percentile


@dataclass(slots=True)
class Histogram:
    name: str
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.samples.append(value)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def total(self) -> float:
        return sum(self.samples)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def quantile(self, fraction: float) -> float:
        return percentile(self.samples, fraction)


@dataclass(slots=True)
class ServerMetrics:
    requests_received: int = 0
    requests_finished: int = 0
    requests_cancelled: int = 0
    requests_failed: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    preemptions: int = 0
    iterations: int = 0

    running: int = 0
    waiting: int = 0
    kv_utilization: float = 0.0
    cache_reuse_rate: float = 0.0

    time_to_first_token: Histogram = field(default_factory=lambda: Histogram("ttft_seconds"))
    inter_token_latency: Histogram = field(default_factory=lambda: Histogram("itl_seconds"))
    end_to_end: Histogram = field(default_factory=lambda: Histogram("e2e_seconds"))

    @property
    def in_flight(self) -> int:
        return self.running + self.waiting

    @property
    def completion_rate(self) -> float:
        return self.requests_finished / self.requests_received if self.requests_received else 0.0

    def observe_queue(self, running: int, waiting: int, kv_utilization: float,
                      cache_reuse_rate: float) -> None:
        self.running = running
        self.waiting = waiting
        self.kv_utilization = kv_utilization
        self.cache_reuse_rate = cache_reuse_rate

    def snapshot(self) -> dict[str, float]:
        return {
            "requests_received": self.requests_received,
            "requests_finished": self.requests_finished,
            "requests_cancelled": self.requests_cancelled,
            "requests_failed": self.requests_failed,
            "requests_in_flight": self.in_flight,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "iterations": self.iterations,
            "preemptions": self.preemptions,
            "running": self.running,
            "waiting": self.waiting,
            "kv_utilization": round(self.kv_utilization, 4),
            "cache_reuse_rate": round(self.cache_reuse_rate, 4),
            "ttft_p50": round(self.time_to_first_token.quantile(0.50), 4),
            "ttft_p99": round(self.time_to_first_token.quantile(0.99), 4),
            "itl_p50": round(self.inter_token_latency.quantile(0.50), 4),
            "itl_p99": round(self.inter_token_latency.quantile(0.99), 4),
            "e2e_p50": round(self.end_to_end.quantile(0.50), 4),
            "e2e_p99": round(self.end_to_end.quantile(0.99), 4),
        }

    def prometheus_text(self, prefix: str = "inference") -> str:
        lines = []
        for key, value in self.snapshot().items():
            lines.append(f"{prefix}_{key} {value}")
        for histogram in (self.time_to_first_token, self.inter_token_latency, self.end_to_end):
            lines.append(f"{prefix}_{histogram.name}_count {histogram.count}")
            lines.append(f"{prefix}_{histogram.name}_sum {histogram.total:.6f}")
        return "\n".join(lines)
