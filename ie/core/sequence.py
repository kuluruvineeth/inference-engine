from __future__ import annotations

from enum import Enum, auto
from itertools import count


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    _ids = count()

    def __init__(self, prompt_ids: list[int], block_size: int, max_new_tokens: int = 64,
                 temperature: float = 1.0, eos_id: int | None = None) -> None:
        if not prompt_ids:
            raise ValueError("a sequence needs at least one prompt token")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.id = next(Sequence._ids)
        self.block_size = block_size
        self.token_ids = list(prompt_ids)
        self.num_prompt = len(prompt_ids)

        self.status = Status.WAITING
        self.num_computed = 0
        self.num_scheduled = 0
        self.block_table: list[int] = []

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eos_id = eos_id

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - self.num_prompt

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status is Status.FINISHED

    @property
    def prompt_is_fully_computed(self) -> bool:
        return self.num_computed >= self.num_prompt

    @property
    def num_uncomputed(self) -> int:
        return len(self) - self.num_computed

    @property
    def num_blocks(self) -> int:
        return (len(self) + self.block_size - 1) // self.block_size

    @property
    def num_computed_blocks(self) -> int:
        return self.num_computed // self.block_size

    def block_tokens(self, index: int) -> list[int]:
        if not 0 <= index < self.num_blocks:
            raise IndexError(f"block {index} out of range for {self.num_blocks} blocks")
        return self.token_ids[index * self.block_size : (index + 1) * self.block_size]

    def block_is_full(self, index: int) -> bool:
        return len(self.block_tokens(index)) == self.block_size

    def block_is_computed(self, index: int) -> bool:
        return (index + 1) * self.block_size <= self.num_computed

    @property
    def block_table_is_short(self) -> bool:
        return self.num_blocks > len(self.block_table)

    def append(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def commit_scheduled(self) -> None:
        self.num_computed += self.num_scheduled
        self.num_scheduled = 0

    def discard_computation(self) -> None:
        self.status = Status.WAITING
        self.num_computed = 0
        self.num_scheduled = 0
        self.block_table = []

    def stop_reason(self) -> str | None:
        if self.eos_id is not None and self.last_token == self.eos_id:
            return "eos"
        if self.num_generated >= self.max_new_tokens:
            return "length"
        return None

    def __repr__(self) -> str:
        return (f"Sequence(id={self.id}, {self.status.name}, len={len(self)}, "
                f"computed={self.num_computed}, blocks={len(self.block_table)})")
