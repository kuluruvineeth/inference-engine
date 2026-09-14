"""Paged KV cache management.

The idea, from Kwon et al., "Efficient Memory Management for Large Language
Model Serving with PagedAttention" (SOSP 2023, arXiv:2309.06180): treat the
KV cache like virtual memory. Physical memory is a pool of fixed-size blocks;
each sequence holds a *block table* mapping its logical block i to some
physical block id. Nothing needs to be contiguous, so the only waste is the
unfilled tail of one block per sequence.

On top of that this module implements prefix sharing. Blocks are content-
addressed by a hash that is *chained* through the preceding blocks:

    h_0 = H(tokens_0)
    h_i = H(h_{i-1} || tokens_i)

The chain is what makes the hash safe to share on. A block's contents alone
are not enough — "the cat sat" means something different after a different
system prompt, and attention over it would be wrong. Chaining makes a hit
mean "every token from position 0 to the end of this block matches", which
is exactly the condition under which the cached K,V are reusable.

Hashes can collide, so a hit is confirmed by comparing the token ids before
the block is reused. This module never trusts a hash alone.
"""

from __future__ import annotations

import hashlib
from collections import deque

from .sequence import Sequence


class Block:
    """One physical slot in the KV cache."""

    __slots__ = ("id", "ref_count", "hash", "token_ids")

    def __init__(self, block_id: int) -> None:
        self.id = block_id
        self.ref_count = 0
        self.hash: str | None = None      # None => not shareable (partial/dirty)
        self.token_ids: list[int] = []

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0

    def __repr__(self) -> str:
        return f"Block({self.id}, refs={self.ref_count}, hashed={self.hash is not None})"


def chain_hash(token_ids: list[int], parent: str | None = None) -> str:
    """Content hash of a block, chained to its predecessor.

    blake2b at 16 bytes: fast, and far enough from birthday-collision range
    for any realistic number of resident blocks. The token-id check on lookup
    is the real safety net regardless.
    """
    h = hashlib.blake2b(digest_size=16)
    if parent is not None:
        h.update(parent.encode())
        h.update(b"|")
    h.update(",".join(map(str, token_ids)).encode())
    return h.hexdigest()


class BlockManager:
    """Owns the block pool, the free list, and the prefix-hash index."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_ids: deque[int] = deque(range(num_blocks))
        self.by_hash: dict[str, int] = {}

        # observability — the numbers that justify the whole design
        self.cache_hits = 0
        self.cache_misses = 0

    # ---- capacity ---------------------------------------------------------

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
    def hit_rate(self) -> float:
        seen = self.cache_hits + self.cache_misses
        return self.cache_hits / seen if seen else 0.0

    # ---- the free list ----------------------------------------------------

    def _take_free_block(self) -> Block:
        """Pop a block off the free list and make it exclusively ours.

        A free block may still be sitting in the hash index as a cached
        prefix. Evicting it means dropping that index entry, otherwise a
        later lookup would hand out a block whose contents we overwrote.
        """
        block = self.blocks[self.free_ids.popleft()]
        assert block.is_free, f"{block} was on the free list but is referenced"
        if block.hash is not None and self.by_hash.get(block.hash) == block.id:
            del self.by_hash[block.hash]
        block.hash = None
        block.token_ids = []
        block.ref_count = 1
        return block

    def _acquire_cached_block(self, block_id: int) -> None:
        """Take a reference to a block that is already populated."""
        block = self.blocks[block_id]
        if block.is_free:
            # resurrect it from the free list — it was evictable, not evicted
            self.free_ids.remove(block_id)
        block.ref_count += 1

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            # keep hash + contents: the block stays in the index and can be
            # re-acquired by a later matching prefix until it is reallocated
            self.free_ids.append(block_id)

    # ---- prefix lookup ----------------------------------------------------

    def _match_prefix(self, seq: Sequence) -> tuple[list[int], str | None]:
        """Walk the sequence's full blocks against the index.

        Returns the physical block ids that can be reused, and the running
        hash at the point matching stopped. Stops at the first miss — the
        chain makes anything after an unmatched block unreachable anyway.
        """
        matched: list[int] = []
        parent: str | None = None

        for i in range(seq.num_blocks):
            if not seq.is_block_full(i):
                break  # partial blocks are never shareable
            tokens = seq.block_tokens(i)
            h = chain_hash(tokens, parent)
            block_id = self.by_hash.get(h)
            if block_id is None:
                break
            # never trust the hash alone
            if self.blocks[block_id].token_ids != tokens:
                break
            matched.append(block_id)
            parent = h

        return matched, parent

    # ---- allocation -------------------------------------------------------

    def can_allocate(self, seq: Sequence) -> bool:
        matched, _ = self._match_prefix(seq)
        return self.num_free >= seq.num_blocks - len(matched)

    def allocate(self, seq: Sequence) -> int:
        """Give a sequence its block table. Returns how many tokens were
        served from cache (and therefore need no recomputation)."""
        if seq.block_table:
            raise RuntimeError(f"sequence {seq.id} already holds blocks")

        matched, _ = self._match_prefix(seq)
        if self.num_free < seq.num_blocks - len(matched):
            raise RuntimeError("allocate() called without capacity; check can_allocate()")

        for block_id in matched:
            self._acquire_cached_block(block_id)
            seq.block_table.append(block_id)
        for _ in range(len(matched), seq.num_blocks):
            seq.block_table.append(self._take_free_block().id)

        cached_tokens = len(matched) * self.block_size
        seq.num_computed = cached_tokens
        if matched:
            self.cache_hits += len(matched)
        self.cache_misses += seq.num_blocks - len(matched)
        return cached_tokens

    def free(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            self._release(block_id)
        seq.block_table = []
        seq.num_computed = 0

    # ---- growth -----------------------------------------------------------

    def can_append(self, seq: Sequence) -> bool:
        """Decode appends one token. That needs a new block only when the
        current last block is exactly full."""
        return not seq.needs_new_block or self.num_free > 0

    def append_slot(self, seq: Sequence) -> None:
        if seq.needs_new_block:
            if self.num_free == 0:
                raise RuntimeError("no free block; check can_append()")
            seq.block_table.append(self._take_free_block().id)

    def publish(self, seq: Sequence) -> None:
        """Index every full block the sequence has computed, so later
        sequences can share them.

        Only blocks whose tokens are fully computed are publishable — a block
        that is allocated but not yet filled by a forward pass holds garbage.
        """
        parent: str | None = None
        for i in range(seq.num_blocks):
            if not seq.is_block_full(i):
                break
            if (i + 1) * self.block_size > seq.num_computed:
                break  # not computed yet; its K,V are not real
            block = self.blocks[seq.block_table[i]]
            tokens = seq.block_tokens(i)
            h = chain_hash(tokens, parent)
            if block.hash != h:
                if block.hash is not None and self.by_hash.get(block.hash) == block.id:
                    del self.by_hash[block.hash]
                block.hash = h
                block.token_ids = tokens
                self.by_hash[h] = block.id
            parent = h

    # ---- introspection ----------------------------------------------------

    def stats(self) -> dict[str, float | int]:
        return {
            "total": self.num_total,
            "free": self.num_free,
            "utilization": round(self.utilization, 4),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "indexed": len(self.by_hash),
        }
