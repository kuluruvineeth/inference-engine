from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager
from .sequence import Sequence, Status


@dataclass(slots=True)
class Batch:
    sequences: list[Sequence] = field(default_factory=list)
    is_prefill: bool = False

    @property
    def num_tokens(self) -> int:
        return sum(seq.num_scheduled for seq in self.sequences)

    def __bool__(self) -> bool:
        return bool(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)


@dataclass(slots=True)
class SchedulerStats:
    steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    preemptions: int = 0
    finished: int = 0
    tokens_prefilled: int = 0
    tokens_decoded: int = 0
    tokens_reused_from_cache: int = 0


class Scheduler:
    def __init__(self, block_manager: BlockManager, max_batch_sequences: int = 256,
                 max_batch_tokens: int = 8192) -> None:
        self.blocks = block_manager
        self.max_batch_sequences = max_batch_sequences
        self.max_batch_tokens = max_batch_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: list[Sequence] = []
        self.stats = SchedulerStats()

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> Batch:
        batch = self._schedule_prefill()
        if batch:
            self.stats.steps += 1
            self.stats.prefill_steps += 1
            self.stats.tokens_prefilled += batch.num_tokens
            return batch

        batch = self._schedule_decode()
        if batch:
            self.stats.steps += 1
            self.stats.decode_steps += 1
            self.stats.tokens_decoded += batch.num_tokens
        return batch

    def _schedule_prefill(self) -> Batch:
        batch = Batch(is_prefill=True)
        budget = self.max_batch_tokens

        while self.waiting and len(batch) < self.max_batch_sequences and budget > 0:
            seq = self.waiting[0]

            if not seq.block_table:
                if not self.blocks.can_allocate(seq):
                    break
                self.stats.tokens_reused_from_cache += self.blocks.allocate(seq)

            remaining = seq.num_uncomputed
            if remaining > budget and batch:
                break

            seq.num_scheduled = min(remaining, budget)
            budget -= seq.num_scheduled

            if seq.num_computed + seq.num_scheduled >= seq.num_prompt:
                self.waiting.popleft()
                seq.status = Status.RUNNING
                self.running.append(seq)

            batch.sequences.append(seq)

        return batch

    def _schedule_decode(self) -> Batch:
        batch = Batch(is_prefill=False)

        while self.running and len(batch) < self.max_batch_sequences:
            seq = self.running.popleft()

            while not self.blocks.can_append(seq):
                if self.running:
                    self._preempt(self.running.pop())
                else:
                    self._preempt(seq)
                    seq = None
                    break

            if seq is None:
                break

            self.blocks.append_slot(seq)
            seq.num_scheduled = 1
            batch.sequences.append(seq)

        self.running.extendleft(reversed(batch.sequences))
        return batch

    def _preempt(self, seq: Sequence) -> None:
        self.blocks.free(seq)
        seq.discard_computation()
        self.waiting.appendleft(seq)
        self.stats.preemptions += 1

    def finish_step(self, batch: Batch, token_ids: list[int]) -> list[Sequence]:
        if len(token_ids) != len(batch):
            raise ValueError(f"expected {len(batch)} tokens, got {len(token_ids)}")

        just_finished: list[Sequence] = []

        for seq, token_id in zip(batch.sequences, token_ids):
            seq.commit_scheduled()
            self.blocks.share_computed_blocks(seq)

            if not seq.prompt_is_fully_computed:
                continue

            seq.append(token_id)

            if seq.stop_reason() is not None:
                self._retire(seq)
                just_finished.append(seq)

        return just_finished

    def _retire(self, seq: Sequence) -> None:
        seq.status = Status.FINISHED
        self.blocks.free(seq)
        if seq in self.running:
            self.running.remove(seq)
        self.finished.append(seq)
        self.stats.finished += 1
