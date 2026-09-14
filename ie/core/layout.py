from __future__ import annotations

from dataclasses import dataclass

from .scheduler import Batch
from .sequence import Sequence


@dataclass(slots=True)
class BatchLayout:
    token_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    context_lens: list[int]
    query_lens: list[int]
    block_tables: list[list[int]]
    is_prefill: bool

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def cu_seqlens_q(self) -> list[int]:
        running = [0]
        for length in self.query_lens:
            running.append(running[-1] + length)
        return running

    @property
    def cu_seqlens_k(self) -> list[int]:
        running = [0]
        for length in self.context_lens:
            running.append(running[-1] + length)
        return running

    @property
    def max_query_len(self) -> int:
        return max(self.query_lens, default=0)

    @property
    def max_context_len(self) -> int:
        return max(self.context_lens, default=0)


def slot_for_position(seq: Sequence, position: int, block_size: int) -> int:
    block_index, offset = divmod(position, block_size)
    if block_index >= len(seq.block_table):
        raise IndexError(
            f"position {position} needs block {block_index} but sequence {seq.id} "
            f"holds {len(seq.block_table)}"
        )
    return seq.block_table[block_index] * block_size + offset


def computed_range(seq: Sequence) -> range:
    return range(seq.num_computed, seq.num_computed + seq.num_scheduled)


def build_layout(batch: Batch, block_size: int) -> BatchLayout:
    token_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []
    query_lens: list[int] = []
    block_tables: list[list[int]] = []

    for seq in batch.sequences:
        window = computed_range(seq)
        if not window:
            raise ValueError(f"sequence {seq.id} was scheduled with no tokens")

        for position in window:
            token_ids.append(seq.token_ids[position])
            positions.append(position)
            slot_mapping.append(slot_for_position(seq, position, block_size))

        query_lens.append(len(window))
        context_lens.append(window.stop)
        block_tables.append(list(seq.block_table))

    width = max((len(table) for table in block_tables), default=0)
    padded = [table + [-1] * (width - len(table)) for table in block_tables]

    return BatchLayout(
        token_ids=token_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        query_lens=query_lens,
        block_tables=padded,
        is_prefill=batch.is_prefill,
    )
