"""The unit of work: one generation request, and everything the engine
needs to know about it between forward passes.

A Sequence is deliberately dumb. It knows its tokens and which KV blocks
hold them; it does not know how to schedule itself or how to run a model.
"""

from __future__ import annotations

from enum import Enum, auto
from itertools import count


class Status(Enum):
    WAITING = auto()   # admitted, no KV blocks held yet (or preempted back)
    RUNNING = auto()   # prompt fully processed, now producing tokens
    FINISHED = auto()  # hit a stop condition; blocks released


class Sequence:
    """One request in flight.

    The token list grows by exactly one per decode step. Two counters track
    how much of it the engine has actually *processed*, which is what makes
    chunked prefill and prefix-cache hits expressible:

      num_computed  tokens whose K,V are already in the cache
      num_scheduled tokens the current forward pass will compute

    Invariant: num_computed + num_scheduled <= len(self)
    """

    _ids = count()

    def __init__(self, prompt_ids: list[int], block_size: int, max_new_tokens: int = 64,
                 temperature: float = 1.0, eos_id: int | None = None) -> None:
        if not prompt_ids:
            raise ValueError("a sequence needs at least one prompt token")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.id = next(Sequence._ids)
        self.block_size = block_size
        self.token_ids = list(prompt_ids)
        self.num_prompt = len(prompt_ids)

        self.status = Status.WAITING
        self.num_computed = 0
        self.num_scheduled = 0
        self.block_table: list[int] = []

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eos_id = eos_id

    # ---- size -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - self.num_prompt

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status is Status.FINISHED

    @property
    def prompt_done(self) -> bool:
        """True once every prompt token has its K,V in the cache."""
        return self.num_computed >= self.num_prompt

    # ---- block geometry ---------------------------------------------------

    @property
    def num_blocks(self) -> int:
        """Blocks needed to hold every token currently in the sequence."""
        return (len(self) + self.block_size - 1) // self.block_size

    def block_tokens(self, i: int) -> list[int]:
        """The token ids that belong in block i. May be a partial block."""
        if not 0 <= i < self.num_blocks:
            raise IndexError(f"block {i} out of range for {self.num_blocks} blocks")
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def is_block_full(self, i: int) -> bool:
        """Only full blocks may be hashed and shared — a partial block's
        contents can still change on the next decode step."""
        return len(self.block_tokens(i)) == self.block_size

    @property
    def needs_new_block(self) -> bool:
        """True when the tokens we hold no longer fit the blocks we own.

        Stated as a comparison rather than `len % block_size == k` because the
        modular form is only correct at one specific call site — it silently
        means different things before and after an append.
        """
        return self.num_blocks > len(self.block_table)

    # ---- mutation ---------------------------------------------------------

    def append(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def advance(self) -> None:
        """Commit the tokens the last forward pass computed."""
        self.num_computed += self.num_scheduled
        self.num_scheduled = 0

    def reset_for_recompute(self) -> None:
        """Preemption: the blocks are gone, so everything must be computed
        again from scratch. The tokens themselves survive — that is the whole
        point of recompute-style preemption versus swapping to host memory."""
        self.status = Status.WAITING
        self.num_computed = 0
        self.num_scheduled = 0
        self.block_table = []

    def stop_reason(self) -> str | None:
        if self.eos_id is not None and self.token_ids and self.last_token == self.eos_id:
            return "eos"
        if self.num_generated >= self.max_new_tokens:
            return "length"
        return None

    def __repr__(self) -> str:
        return (f"Sequence(id={self.id}, {self.status.name}, "
                f"len={len(self)}, computed={self.num_computed}, "
                f"blocks={len(self.block_table)})")
