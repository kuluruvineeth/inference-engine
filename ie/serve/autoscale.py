from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ScalingAction(Enum):
    SCALE_UP = auto()
    SCALE_DOWN = auto()
    HOLD = auto()


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    num_blocks: int
    block_size: int
    bytes_per_token: int

    @property
    def total_tokens(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def total_bytes(self) -> int:
        return self.total_tokens * self.bytes_per_token

    def blocks_for(self, tokens: int) -> int:
        return (tokens + self.block_size - 1) // self.block_size

    def concurrent_sequences(self, prompt_tokens: int, generated_tokens: int,
                             shared_prefix_tokens: int = 0) -> int:
        private = max(0, prompt_tokens - shared_prefix_tokens) + generated_tokens
        per_sequence = self.blocks_for(private)
        shared = self.blocks_for(shared_prefix_tokens) if shared_prefix_tokens else 0
        usable = self.num_blocks - shared
        return max(0, usable // per_sequence) if per_sequence else 0

    def blocks_needed_for(self, concurrency: int, prompt_tokens: int,
                          generated_tokens: int) -> int:
        return concurrency * self.blocks_for(prompt_tokens + generated_tokens)


@dataclass(frozen=True, slots=True)
class ScalingPolicy:
    target_per_replica: int = 8
    min_replicas: int = 1
    max_replicas: int = 16
    scale_up_at: float = 1.2
    scale_down_at: float = 0.5
    cold_start_seconds: float = 20.0
    scale_to_zero_after: float | None = None

    def __post_init__(self):
        if self.target_per_replica < 1:
            raise ValueError("target_per_replica must be at least 1")
        if self.min_replicas < 0 or self.max_replicas < max(1, self.min_replicas):
            raise ValueError("replica bounds are inconsistent")
        if self.scale_down_at >= self.scale_up_at:
            raise ValueError("scale_down_at must be below scale_up_at")


@dataclass(slots=True)
class ScalingDecision:
    action: ScalingAction
    replicas: int
    load_factor: float
    reason: str


class Autoscaler:
    def __init__(self, policy: ScalingPolicy, replicas: int | None = None) -> None:
        self.policy = policy
        self.replicas = replicas if replicas is not None else policy.min_replicas
        self.idle_seconds = 0.0
        self.history: list[ScalingDecision] = []

    def load_factor(self, in_flight: int) -> float:
        capacity = max(1, self.replicas) * self.policy.target_per_replica
        return in_flight / capacity

    def observe(self, in_flight: int, seconds_since_last: float = 0.0) -> ScalingDecision:
        policy = self.policy
        self.idle_seconds = self.idle_seconds + seconds_since_last if in_flight == 0 else 0.0

        if (policy.scale_to_zero_after is not None and in_flight == 0
                and self.idle_seconds >= policy.scale_to_zero_after and self.replicas > 0):
            return self._record(ScalingAction.SCALE_DOWN, 0,
                                self.load_factor(in_flight), "idle past the scale-to-zero window")

        if self.replicas == 0 and in_flight > 0:
            return self._record(ScalingAction.SCALE_UP, max(1, policy.min_replicas),
                                float("inf"), "cold start from zero")

        factor = self.load_factor(in_flight)

        if factor > policy.scale_up_at and self.replicas < policy.max_replicas:
            wanted = -(-in_flight // policy.target_per_replica)
            target = min(policy.max_replicas, max(self.replicas + 1, wanted))
            return self._record(ScalingAction.SCALE_UP, target, factor, "load above target")

        if factor < policy.scale_down_at and self.replicas > policy.min_replicas:
            return self._record(ScalingAction.SCALE_DOWN, self.replicas - 1, factor,
                                "load below target")

        return self._record(ScalingAction.HOLD, self.replicas, factor, "within target band")

    def _record(self, action: ScalingAction, replicas: int, factor: float,
                reason: str) -> ScalingDecision:
        decision = ScalingDecision(action=action, replicas=replicas,
                                   load_factor=factor, reason=reason)
        self.replicas = replicas
        self.history.append(decision)
        return decision

    def time_to_serve(self, was_cold: bool) -> float:
        return self.policy.cold_start_seconds if was_cold else 0.0


@dataclass(slots=True)
class AdmissionController:
    max_queue_depth: int
    max_prompt_tokens: int = 1 << 20
    admitted: int = 0
    rejected: int = 0

    def admit(self, queue_depth: int, prompt_tokens: int) -> tuple[bool, str | None]:
        if prompt_tokens > self.max_prompt_tokens:
            self.rejected += 1
            return False, "prompt exceeds the maximum accepted length"
        if queue_depth >= self.max_queue_depth:
            self.rejected += 1
            return False, "queue is full"
        self.admitted += 1
        return True, None

    @property
    def rejection_rate(self) -> float:
        total = self.admitted + self.rejected
        return self.rejected / total if total else 0.0
