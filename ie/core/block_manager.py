from __future__ import annotations

import hashlib
from collections import deque

from .sequence import Sequence


def chain_hash(token_ids: list[int], parent: str | None = None) -> str:
    digest = hashlib.blake2b(digest_size=16)
    if parent is not None:
        digest.update(parent.encode())
        digest.update(b"|")
    digest.update(",".join(map(str, token_ids)).encode())
    return digest.hexdigest()


class Block:
    __slots__ = ("id", "ref_count", "hash", "token_ids")

    def __init__(self, block_id: int) -> None:
        self.id = block_id
        self.ref_count = 0
        self.hash: str | None = None
        self.token_ids: list[int] = []

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0

    @property
    def is_shareable(self) -> bool:
        return self.hash is not None

    def holds(self, token_ids: list[int]) -> bool:
        return self.token_ids == token_ids

    def __repr__(self) -> str:
        return f"Block({self.id}, refs={self.ref_count}, shareable={self.is_shareable})"


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_ids: deque[int] = deque(range(num_blocks))
        self.block_id_by_hash: dict[str, int] = {}
        self.blocks_reused = 0
        self.blocks_computed = 0

    @property
    def num_free(self) -> int:
        return len(self.free_ids)

    @property
    def num_total(self) -> int:
        return len(self.blocks)

    @property
    def utilization(self) -> float:
        return 1.0 - self.num_free / self.num_total

    @property
    def reuse_rate(self) -> float:
        total = self.blocks_reused + self.blocks_computed
        return self.blocks_reused / total if total else 0.0

    def _forget(self, block: Block) -> None:
        if block.is_shareable and self.block_id_by_hash.get(block.hash) == block.id:
            del self.block_id_by_hash[block.hash]
        block.hash = None
        block.token_ids = []

    def _claim_free_block(self) -> Block:
        block = self.blocks[self.free_ids.popleft()]
        assert block.is_free, f"{block} was on the free list but is still referenced"
        self._forget(block)
        block.ref_count = 1
        return block

    def _claim_shared_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.is_free:
            self.free_ids.remove(block_id)
        block.ref_count += 1

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.is_free:
            self.free_ids.append(block_id)

    def _reusable_prefix(self, seq: Sequence) -> list[int]:
        reusable: list[int] = []
        parent: str | None = None

        for index in range(seq.num_blocks - 1):
            if not seq.block_is_full(index):
                break
            tokens = seq.block_tokens(index)
            candidate = chain_hash(tokens, parent)
            block_id = self.block_id_by_hash.get(candidate)
            if block_id is None or not self.blocks[block_id].holds(tokens):
                break
            reusable.append(block_id)
            parent = candidate

        return reusable

    def _blocks_taken_from_pool(self, seq: Sequence, reusable: list[int]) -> int:
        revived = sum(1 for block_id in reusable if self.blocks[block_id].is_free)
        return revived + seq.num_blocks - len(reusable)

    def can_allocate(self, seq: Sequence) -> bool:
        reusable = self._reusable_prefix(seq)
        return self.num_free >= self._blocks_taken_from_pool(seq, reusable)

    def allocate(self, seq: Sequence) -> int:
        if seq.block_table:
            raise RuntimeError(f"sequence {seq.id} already holds blocks")

        reusable = self._reusable_prefix(seq)
        if self.num_free < self._blocks_taken_from_pool(seq, reusable):
            raise RuntimeError("no capacity; call can_allocate() first")

        for block_id in reusable:
            self._claim_shared_block(block_id)
            seq.block_table.append(block_id)
        for _ in range(len(reusable), seq.num_blocks):
            seq.block_table.append(self._claim_free_block().id)

        self.blocks_reused += len(reusable)
        self.blocks_computed += seq.num_blocks - len(reusable)
        seq.num_computed = len(reusable) * self.block_size
        return seq.num_computed

    def free(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            self._release(block_id)
        seq.block_table = []
        seq.num_computed = 0

    def can_append(self, seq: Sequence) -> bool:
        return not seq.block_table_is_short or self.num_free > 0

    def append_slot(self, seq: Sequence) -> None:
        if seq.block_table_is_short:
            if self.num_free == 0:
                raise RuntimeError("no free block; call can_append() first")
            seq.block_table.append(self._claim_free_block().id)

    def reserve(self, seq: Sequence, num_blocks: int) -> bool:
        while len(seq.block_table) < num_blocks:
            if self.num_free == 0:
                return False
            seq.block_table.append(self._claim_free_block().id)
        return True

    def share_computed_blocks(self, seq: Sequence) -> None:
        parent: str | None = None
        for index in range(seq.num_blocks):
            if not (seq.block_is_full(index) and seq.block_is_computed(index)):
                return
            block = self.blocks[seq.block_table[index]]
            tokens = seq.block_tokens(index)
            current = chain_hash(tokens, parent)
            if block.hash != current:
                self._forget(block)
                block.hash = current
                block.token_ids = tokens
                self.block_id_by_hash[current] = block.id
            parent = current

    def stats(self) -> dict[str, float | int]:
        return {
            "total": self.num_total,
            "free": self.num_free,
            "utilization": round(self.utilization, 4),
            "reused": self.blocks_reused,
            "computed": self.blocks_computed,
            "reuse_rate": round(self.reuse_rate, 4),
            "shareable": len(self.block_id_by_hash),
        }
