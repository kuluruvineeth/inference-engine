from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto


class Readiness(Enum):
    STARTING = auto()
    READY = auto()
    DRAINING = auto()
    UNHEALTHY = auto()


@dataclass(slots=True)
class HealthProbe:
    weights_loaded: bool = False
    cache_allocated: bool = False
    warmed_up: bool = False
    draining: bool = False
    last_error: str | None = None

    @property
    def state(self) -> Readiness:
        if self.last_error is not None:
            return Readiness.UNHEALTHY
        if self.draining:
            return Readiness.DRAINING
        if self.weights_loaded and self.cache_allocated and self.warmed_up:
            return Readiness.READY
        return Readiness.STARTING

    @property
    def is_live(self) -> bool:
        return self.state is not Readiness.UNHEALTHY

    @property
    def accepts_traffic(self) -> bool:
        return self.state is Readiness.READY

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.name.lower(),
            "live": self.is_live,
            "ready": self.accepts_traffic,
            "weights_loaded": self.weights_loaded,
            "cache_allocated": self.cache_allocated,
            "warmed_up": self.warmed_up,
            "error": self.last_error,
        }


def probe_from_engine(engine, draining: bool = False) -> HealthProbe:
    return HealthProbe(
        weights_loaded=engine.model is not None,
        cache_allocated=bool(engine.k_caches) and engine.blocks.num_total > 0,
        warmed_up=engine.blocks.num_total > 0,
        draining=draining,
    )


def main() -> int:
    probe = HealthProbe(weights_loaded=True, cache_allocated=True, warmed_up=True)
    print(probe.snapshot())
    return 0 if probe.is_live else 1


if __name__ == "__main__":
    sys.exit(main())
