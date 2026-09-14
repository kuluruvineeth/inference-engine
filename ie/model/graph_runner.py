from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..core.layout import BatchLayout


def bucket_sizes(max_batch: int) -> list[int]:
    buckets = [1, 2, 4, 8]
    buckets += list(range(16, max_batch + 1, 16))
    if max_batch not in buckets:
        buckets.append(max_batch)
    return sorted(b for b in set(buckets) if b <= max_batch)


@dataclass(slots=True)
class GraphStats:
    captures: int = 0
    replays: int = 0
    eager_fallbacks: int = 0

    @property
    def replay_rate(self) -> float:
        total = self.replays + self.eager_fallbacks
        return self.replays / total if total else 0.0


@dataclass(slots=True)
class StaticBuffers:
    token_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    hidden: torch.Tensor


class CudaGraphRunner:
    def __init__(self, model, block_size: int, max_batch: int, max_blocks: int) -> None:
        self.model = model
        self.config = model.config
        self.block_size = block_size
        self.max_batch = max_batch
        self.max_blocks = max_blocks
        self.device = model.device
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.buffers: StaticBuffers | None = None
        self.pool = None
        self.stats = GraphStats()

    @property
    def is_captured(self) -> bool:
        return bool(self.graphs)

    def _allocate_buffers(self) -> StaticBuffers:
        zeros = lambda *shape, dtype: torch.zeros(shape, device=self.device, dtype=dtype)
        return StaticBuffers(
            token_ids=zeros(self.max_batch, dtype=torch.long),
            positions=zeros(self.max_batch, dtype=torch.long),
            slot_mapping=zeros(self.max_batch, dtype=torch.long),
            block_tables=zeros(self.max_batch, self.max_blocks, dtype=torch.long),
            hidden=zeros(self.max_batch, self.config.hidden_size, dtype=self.model.dtype),
        )

    def _layout_for(self, batch: int) -> BatchLayout:
        return BatchLayout(
            token_ids=[0] * batch,
            positions=[0] * batch,
            slot_mapping=[0] * batch,
            context_lens=[1] * batch,
            query_lens=[1] * batch,
            block_tables=[[0] * self.max_blocks for _ in range(batch)],
            is_prefill=False,
        )

    def capture(self, k_caches, v_caches) -> None:
        self.buffers = self._allocate_buffers()
        for batch in reversed(bucket_sizes(self.max_batch)):
            layout = self._layout_for(batch)
            self.model.forward(layout, k_caches, v_caches, self.block_size)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.pool):
                out = self.model.forward(layout, k_caches, v_caches, self.block_size)
                self.buffers.hidden[:batch] = out
            self.pool = graph.pool() if self.pool is None else self.pool
            self.graphs[batch] = graph
            self.stats.captures += 1
            torch.cuda.synchronize()

    def bucket_for(self, batch: int) -> int | None:
        return next((size for size in sorted(self.graphs) if size >= batch), None)

    def can_replay(self, layout: BatchLayout) -> bool:
        if layout.is_prefill or not self.is_captured:
            return False
        if any(length != 1 for length in layout.query_lens):
            return False
        return self.bucket_for(len(layout.query_lens)) is not None

    def replay(self, layout: BatchLayout) -> torch.Tensor:
        batch = len(layout.query_lens)
        bucket = self.bucket_for(batch)
        buffers = self.buffers

        buffers.token_ids[:batch] = torch.as_tensor(layout.token_ids, device=self.device)
        buffers.positions[:batch] = torch.as_tensor(layout.positions, device=self.device)
        buffers.slot_mapping[:batch] = torch.as_tensor(layout.slot_mapping, device=self.device)
        width = len(layout.block_tables[0])
        buffers.block_tables[:batch, :width] = torch.as_tensor(layout.block_tables,
                                                               device=self.device)

        self.graphs[bucket].replay()
        self.stats.replays += 1
        return buffers.hidden[:batch]
