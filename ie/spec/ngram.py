from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NgramProposer:
    max_context: int = 4
    min_context: int = 1
    lookahead: int = 4

    def __post_init__(self):
        if self.min_context < 1 or self.max_context < self.min_context:
            raise ValueError("need 1 <= min_context <= max_context")
        if self.lookahead < 1:
            raise ValueError("lookahead must be at least 1")

    def propose(self, token_ids: list[int]) -> list[int]:
        for width in range(min(self.max_context, len(token_ids) - 1), self.min_context - 1, -1):
            pattern = token_ids[-width:]
            found = self._last_earlier_match(token_ids, pattern)
            if found is None:
                continue
            start = found + width
            continuation = token_ids[start : start + self.lookahead]
            if continuation:
                return continuation
        return []

    @staticmethod
    def _last_earlier_match(token_ids: list[int], pattern: list[int]) -> int | None:
        width = len(pattern)
        for start in range(len(token_ids) - width - 1, -1, -1):
            if token_ids[start : start + width] == pattern:
                return start
        return None
