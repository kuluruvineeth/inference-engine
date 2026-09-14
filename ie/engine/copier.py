from __future__ import annotations

from typing import Any


class LayeredSlotCopier:
    def __init__(self, k_caches: list[Any], v_caches: list[Any], block_size: int) -> None:
        self.k_caches = k_caches
        self.v_caches = v_caches
        self.block_size = block_size
        self.blocks_read = 0
        self.blocks_written = 0

    def _span(self, block_id: int) -> slice:
        start = block_id * self.block_size
        return slice(start, start + self.block_size)

    def read_block(self, block_id: int) -> Any:
        span = self._span(block_id)
        self.blocks_read += 1
        return ([cache[span].copy() for cache in self.k_caches],
                [cache[span].copy() for cache in self.v_caches])

    def write_block(self, block_id: int, payload: Any) -> None:
        span = self._span(block_id)
        k_payload, v_payload = payload
        for cache, data in zip(self.k_caches, k_payload):
            cache[span] = data
        for cache, data in zip(self.v_caches, v_payload):
            cache[span] = data
        self.blocks_written += 1

    @property
    def bytes_moved(self) -> int:
        if not self.k_caches:
            return 0
        per_block = self.k_caches[0][self._span(0)].nbytes * 2 * len(self.k_caches)
        return (self.blocks_read + self.blocks_written) * per_block
