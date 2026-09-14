from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol


class SlotCopier(Protocol):
    def read_block(self, block_id: int) -> Any: ...

    def write_block(self, block_id: int, payload: Any) -> None: ...


@dataclass(slots=True)
class TierStats:
    admitted: int = 0
    restored: int = 0
    evicted: int = 0
    rejected: int = 0

    @property
    def restore_rate(self) -> float:
        return self.restored / self.admitted if self.admitted else 0.0


class KVTier:
    def __init__(self, capacity_blocks: int) -> None:
        if capacity_blocks < 0:
            raise ValueError("capacity cannot be negative")
        self.capacity = capacity_blocks
        self.entries: OrderedDict[str, tuple[list[int], Any]] = OrderedDict()
        self.stats = TierStats()

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    @property
    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity

    def admit(self, key: str, token_ids: list[int], payload: Any) -> bool:
        if self.capacity == 0:
            self.stats.rejected += 1
            return False
        if key in self.entries:
            self.entries.move_to_end(key)
            return True
        while self.is_full:
            self.entries.popitem(last=False)
            self.stats.evicted += 1
        self.entries[key] = (list(token_ids), payload)
        self.stats.admitted += 1
        return True

    def holds(self, key: str, token_ids: list[int]) -> bool:
        entry = self.entries.get(key)
        return entry is not None and entry[0] == token_ids

    def take(self, key: str) -> tuple[list[int], Any] | None:
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.stats.restored += 1
        return entry

    def peek(self, key: str) -> tuple[list[int], Any] | None:
        return self.entries.get(key)
