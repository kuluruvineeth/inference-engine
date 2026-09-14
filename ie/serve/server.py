from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Iterator

import numpy as np

from ..engine.engine import Engine
from ..layers.functional import sample_from_logits
from .metrics import ServerMetrics


class RequestState(Enum):
    QUEUED = auto()
    STREAMING = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass(slots=True)
class ServedRequest:
    id: str
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float
    sequence: object = None
    state: RequestState = RequestState.QUEUED
    tokens: list[int] = field(default_factory=list)
    submitted_at: float = 0.0
    first_token_at: float | None = None
    last_token_at: float | None = None
    finished_at: float | None = None
    stop_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.state in (RequestState.QUEUED, RequestState.STREAMING)

    @property
    def time_to_first_token(self) -> float | None:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.submitted_at


@dataclass(slots=True)
class StreamChunk:
    request_id: str
    token_id: int
    index: int
    finished: bool = False
    stop_reason: str | None = None


class InferenceServer:
    def __init__(self, engine: Engine, clock=time.perf_counter) -> None:
        self.engine = engine
        self.clock = clock
        self.metrics = ServerMetrics()
        self.requests: dict[str, ServedRequest] = {}
        self._ids = count()

    def submit(self, prompt_ids: list[int], max_new_tokens: int = 32,
               temperature: float = 0.0, request_id: str | None = None) -> ServedRequest:
        if not prompt_ids:
            self.metrics.requests_failed += 1
            raise ValueError("a request needs at least one prompt token")

        identifier = request_id or f"req-{next(self._ids)}"
        if identifier in self.requests:
            raise ValueError(f"request {identifier} already exists")

        sequence = self.engine.submit(prompt_ids, max_new_tokens=max_new_tokens,
                                      temperature=temperature)
        served = ServedRequest(id=identifier, prompt_ids=list(prompt_ids),
                               max_new_tokens=max_new_tokens, temperature=temperature,
                               sequence=sequence, submitted_at=self.clock())
        self.requests[identifier] = served

        self.metrics.requests_received += 1
        self.metrics.prompt_tokens += len(prompt_ids)
        return served

    def cancel(self, request_id: str) -> bool:
        served = self.requests.get(request_id)
        if served is None or not served.is_open:
            return False

        scheduler = self.engine.scheduler
        sequence = served.sequence
        if sequence in scheduler.running:
            scheduler.running.remove(sequence)
        if sequence in scheduler.waiting:
            scheduler.waiting.remove(sequence)
        self.engine.blocks.free(sequence)

        served.state = RequestState.CANCELLED
        served.stop_reason = "cancelled"
        served.finished_at = self.clock()
        self.metrics.requests_cancelled += 1
        self._observe_queue()
        return True

    @property
    def has_work(self) -> bool:
        return self.engine.scheduler.has_work

    def _observe_queue(self) -> None:
        blocks = self.engine.blocks
        self.metrics.observe_queue(running=len(self.engine.scheduler.running),
                                   waiting=len(self.engine.scheduler.waiting),
                                   kv_utilization=blocks.utilization,
                                   cache_reuse_rate=blocks.reuse_rate)

    def step(self) -> list[StreamChunk]:
        batch = self.engine.scheduler.schedule()
        if not batch:
            self._observe_queue()
            return []

        _, logits = self.engine.forward(batch)
        temperatures = np.array([s.temperature for s in batch.sequences], dtype=np.float32)
        token_ids = sample_from_logits(logits, temperatures, self.engine.generator).tolist()

        before = {s.id: s.num_generated for s in batch.sequences}
        self.engine.scheduler.finish_step(batch, token_ids)
        now = self.clock()
        self.metrics.iterations += 1

        by_sequence = {served.sequence.id: served for served in self.requests.values()
                       if served.is_open and served.sequence is not None}
        chunks: list[StreamChunk] = []

        for sequence in batch.sequences:
            served = by_sequence.get(sequence.id)
            if served is None:
                continue
            for _ in range(sequence.num_generated - before[sequence.id]):
                index = len(served.tokens)
                token = sequence.token_ids[sequence.num_prompt + index]
                served.tokens.append(token)
                self.metrics.generated_tokens += 1

                if served.first_token_at is None:
                    served.first_token_at = now
                    served.state = RequestState.STREAMING
                    self.metrics.time_to_first_token.observe(now - served.submitted_at)
                elif served.last_token_at is not None:
                    self.metrics.inter_token_latency.observe(now - served.last_token_at)
                served.last_token_at = now

                chunks.append(StreamChunk(request_id=served.id, token_id=token, index=index))

            if sequence.is_finished:
                served.state = RequestState.COMPLETED
                served.stop_reason = sequence.stop_reason()
                served.finished_at = now
                self.metrics.requests_finished += 1
                self.metrics.end_to_end.observe(now - served.submitted_at)
                if chunks and chunks[-1].request_id == served.id:
                    chunks[-1].finished = True
                    chunks[-1].stop_reason = served.stop_reason

        self.metrics.preemptions = self.engine.scheduler.stats.preemptions
        self._observe_queue()
        return chunks

    def stream(self, max_iterations: int = 1_000_000) -> Iterator[StreamChunk]:
        for _ in range(max_iterations):
            if not self.has_work:
                return
            for chunk in self.step():
                yield chunk

    def drain(self, max_iterations: int = 1_000_000) -> None:
        for _ in self.stream(max_iterations):
            pass
