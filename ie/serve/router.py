from __future__ import annotations

from dataclasses import dataclass
from itertools import count

from ..core.block_manager import chain_hash


@dataclass(slots=True)
class RouterStats:
    routed: int = 0
    prefix_hits: int = 0
    blocks_matched: int = 0

    @property
    def hit_rate(self) -> float:
        return self.prefix_hits / self.routed if self.routed else 0.0


class ReplicaIndex:
    def __init__(self, replica_id: int, block_size: int) -> None:
        self.replica_id = replica_id
        self.block_size = block_size
        self.known_prefixes: set[str] = set()
        self.in_flight = 0

    def remember(self, token_ids: list[int]) -> None:
        parent = None
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            parent = chain_hash(token_ids[start : start + self.block_size], parent)
            self.known_prefixes.add(parent)

    def matching_blocks(self, token_ids: list[int]) -> int:
        parent = None
        matched = 0
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            parent = chain_hash(token_ids[start : start + self.block_size], parent)
            if parent not in self.known_prefixes:
                break
            matched += 1
        return matched


class RoundRobinRouter:
    def __init__(self, num_replicas: int, block_size: int) -> None:
        if num_replicas < 1:
            raise ValueError("need at least one replica")
        self.replicas = [ReplicaIndex(i, block_size) for i in range(num_replicas)]
        self.turns = count()
        self.stats = RouterStats()

    def route(self, token_ids: list[int]) -> int:
        chosen = next(self.turns) % len(self.replicas)
        self._record(chosen, token_ids)
        return chosen

    def _record(self, chosen: int, token_ids: list[int]) -> None:
        matched = self.replicas[chosen].matching_blocks(token_ids)
        self.stats.routed += 1
        self.stats.blocks_matched += matched
        if matched:
            self.stats.prefix_hits += 1
        self.replicas[chosen].remember(token_ids)
        self.replicas[chosen].in_flight += 1

    def release(self, replica_id: int) -> None:
        self.replicas[replica_id].in_flight = max(0, self.replicas[replica_id].in_flight - 1)


class CacheAwareRouter(RoundRobinRouter):
    def __init__(self, num_replicas: int, block_size: int, load_penalty: float = 0.5) -> None:
        super().__init__(num_replicas, block_size)
        self.load_penalty = load_penalty

    def route(self, token_ids: list[int]) -> int:
        def score(replica: ReplicaIndex) -> tuple[float, int, int]:
            return (replica.matching_blocks(token_ids) - self.load_penalty * replica.in_flight,
                    -replica.in_flight, -replica.replica_id)

        chosen = max(self.replicas, key=score).replica_id
        self._record(chosen, token_ids)
        return chosen
