from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.layout import BatchLayout
from .graph_runner import GraphStats, bucket_sizes


@dataclass(slots=True)
class DecodeBuffers:
    tokens: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    paged_tables: torch.Tensor


class DecodeGraphs:
    def __init__(self, model, block_size: int, max_batch: int, max_blocks: int) -> None:
        self.model = model
        self.block_size = block_size
        self.max_batch = max_batch
        self.max_blocks = max_blocks
        self.device = model.device
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.buffers: DecodeBuffers | None = None
        self.pool = None
        self.stats = GraphStats()

    @property
    def is_captured(self) -> bool:
        return bool(self.graphs)

    def _allocate(self) -> DecodeBuffers:
        zeros = lambda *shape, dtype: torch.zeros(shape, device=self.device, dtype=dtype)
        return DecodeBuffers(
            tokens=zeros(self.max_batch, dtype=torch.long),
            positions=zeros(self.max_batch, dtype=torch.long),
            slots=zeros(self.max_batch, dtype=torch.long),
            block_tables=zeros(self.max_batch, self.max_blocks, dtype=torch.long),
            context_lens=torch.ones(self.max_batch, device=self.device, dtype=torch.int32),
            paged_tables=zeros(self.max_batch, self.max_blocks, dtype=torch.int32),
        )

    def _plan(self, batch: int) -> dict:
        buffers = self.buffers
        return {
            "flash": self.model.uses_flash,
            "scale": self.model.config.head_dim ** -0.5,
            "slots": buffers.slots[:batch],
            "cu_q": None,
            "cu_k": None,
            "context_lens": buffers.context_lens[:batch],
            "paged_tables": None,
        }

    @staticmethod
    def _layout(batch: int) -> BatchLayout:
        return BatchLayout(token_ids=[0] * batch, positions=[0] * batch,
                           slot_mapping=[0] * batch, context_lens=[1] * batch,
                           query_lens=[1] * batch, block_tables=[[0]] * batch,
                           is_prefill=False)

    def capture(self, k_caches, v_caches) -> None:
        self.buffers = self._allocate()
        for batch in reversed(bucket_sizes(self.max_batch)):
            layout = self._layout(batch)
            plan = self._plan(batch)
            arguments = (self.buffers.tokens[:batch], self.buffers.positions[:batch],
                         self.buffers.block_tables[:batch], plan, layout,
                         k_caches, v_caches, self.block_size)

            self.model.run(*arguments)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.pool):
                self.outputs[batch] = self.model.run(*arguments)
            if self.pool is None:
                self.pool = graph.pool()
            self.graphs[batch] = graph
            self.stats.captures += 1
            torch.cuda.synchronize()

    def bucket_for(self, batch: int) -> int | None:
        return next((size for size in sorted(self.graphs) if size >= batch), None)

    def can_replay(self, layout: BatchLayout) -> bool:
        if layout.is_prefill or not self.is_captured:
            return False
        batch = len(layout.query_lens)
        if any(length != 1 for length in layout.query_lens):
            return False
        if len(layout.block_tables[0]) > self.max_blocks:
            return False
        return self.bucket_for(batch) is not None

    def replay(self, layout: BatchLayout, k_caches, v_caches) -> torch.Tensor:
        batch = len(layout.query_lens)
        bucket = self.bucket_for(batch)
        buffers = self.buffers
        width = len(layout.block_tables[0])

        as_tensor = lambda values, dtype: torch.as_tensor(values, device=self.device, dtype=dtype)
        buffers.tokens[:batch] = as_tensor(layout.token_ids, torch.long)
        buffers.positions[:batch] = as_tensor(layout.positions, torch.long)
        buffers.slots[:batch] = as_tensor(layout.slot_mapping, torch.long)
        buffers.context_lens[:batch] = as_tensor(layout.context_lens, torch.int32)
        buffers.block_tables[:batch, :width] = as_tensor(layout.block_tables, torch.long)
        if batch < bucket:
            buffers.context_lens[batch:bucket] = 1
            buffers.slots[batch:bucket] = 0

        self.graphs[bucket].replay()
        self.stats.replays += 1
        return self.outputs[bucket][:batch]
