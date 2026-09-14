from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.layout import build_layout
from ..core.scheduler import Batch
from ..engine.copier import LayeredSlotCopier
from ..engine.engine import Engine, EngineConfig
from ..layers.functional import sample_from_logits
from ..model.transformer import Transformer


@dataclass(slots=True)
class KVTransfer:
    token_ids: list[int]
    num_computed: int
    block_payloads: list[Any]
    bytes_moved: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.block_payloads)


@dataclass(slots=True)
class DisaggregationStats:
    prefills: int = 0
    decodes: int = 0
    transfers: int = 0
    blocks_transferred: int = 0
    bytes_transferred: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0


class PrefillWorker:
    def __init__(self, model: Transformer, config: EngineConfig) -> None:
        self.engine = Engine(model, config)
        self.copier = LayeredSlotCopier(self.engine.k_caches, self.engine.v_caches,
                                        config.block_size)

    def prefill(self, prompt_ids: list[int], temperature: float = 0.0) -> KVTransfer:
        seq = self.engine.submit(prompt_ids, max_new_tokens=1 << 30, temperature=temperature)
        while not seq.prompt_is_fully_computed:
            self.engine.step()

        payloads = [self.copier.read_block(block_id) for block_id in seq.block_table]
        transfer = KVTransfer(token_ids=list(seq.token_ids),
                              num_computed=seq.num_computed,
                              block_payloads=payloads,
                              bytes_moved=self._payload_bytes(len(payloads)))
        self.engine.blocks.free(seq)
        if seq in self.engine.scheduler.running:
            self.engine.scheduler.running.remove(seq)
        return transfer

    def _payload_bytes(self, num_blocks: int) -> int:
        if not self.engine.k_caches:
            return 0
        block_size = self.engine.config.block_size
        per_layer = self.engine.k_caches[0][0:block_size]
        per_block = int(np.asarray(per_layer).nbytes) * 2 * len(self.engine.k_caches)
        return num_blocks * per_block


class DecodeWorker:
    def __init__(self, model: Transformer, config: EngineConfig) -> None:
        self.engine = Engine(model, config)
        self.copier = LayeredSlotCopier(self.engine.k_caches, self.engine.v_caches,
                                        config.block_size)

    def admit(self, transfer: KVTransfer, max_new_tokens: int, temperature: float = 0.0):
        seq = self.engine.submit(transfer.token_ids[:-1], max_new_tokens=max_new_tokens,
                                 temperature=temperature)
        if not self.engine.blocks.reserve(seq, transfer.num_blocks):
            raise RuntimeError("decode worker has no room for the transferred cache")

        for block_id, payload in zip(seq.block_table, transfer.block_payloads):
            self.copier.write_block(block_id, payload)

        seq.token_ids = list(transfer.token_ids)
        seq.num_computed = transfer.num_computed
        self.engine.scheduler.waiting.remove(seq)

        if seq.stop_reason() is not None:
            seq.status = seq.status.__class__.FINISHED
            self.engine.blocks.free(seq)
            self.engine.scheduler.finished.append(seq)
            return seq

        seq.status = seq.status.__class__.RUNNING
        self.engine.scheduler.running.append(seq)
        return seq


class DisaggregatedEngine:
    def __init__(self, model: Transformer, config: EngineConfig | None = None) -> None:
        config = config or EngineConfig()
        self.prefill = PrefillWorker(model, config)
        self.decode = DecodeWorker(model, config)
        self.stats = DisaggregationStats()

    def generate(self, prompts: list[list[int]], max_new_tokens: int = 16,
                 temperature: float = 0.0) -> list[list[int]]:
        sequences = []
        for prompt in prompts:
            transfer = self.prefill.prefill(prompt, temperature)
            self.stats.prefills += 1
            self.stats.transfers += 1
            self.stats.blocks_transferred += transfer.num_blocks
            self.stats.bytes_transferred += transfer.bytes_moved
            self.stats.prefill_tokens += len(prompt)
            sequences.append(self.decode.admit(transfer, max_new_tokens, temperature))

        while self.decode.engine.scheduler.has_work:
            batch = self.decode.engine.scheduler.schedule()
            if not batch:
                break
            _, logits = self.decode.engine.forward(batch)
            temperatures = np.array([s.temperature for s in batch.sequences], dtype=np.float32)
            tokens = sample_from_logits(logits, temperatures,
                                        self.decode.engine.generator).tolist()
            self.decode.engine.scheduler.finish_step(batch, tokens)
            self.stats.decodes += 1
            self.stats.decode_tokens += len(batch)

        return [seq.token_ids[seq.num_prompt:] for seq in sequences]
