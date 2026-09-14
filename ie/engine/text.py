from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..serve.server import InferenceServer
from .engine import Engine


@dataclass(slots=True)
class Completion:
    prompt: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    stop_reason: str | None


class TextEngine:
    def __init__(self, engine: Engine, tokenizer) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.server = InferenceServer(engine)

    def complete(self, prompts: list[str], max_new_tokens: int = 64,
                 temperature: float = 0.0) -> list[Completion]:
        eos = getattr(self.tokenizer, "eos_id", None)
        served = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt)
            request = self.server.submit(ids, max_new_tokens=max_new_tokens,
                                         temperature=temperature)
            request.sequence.eos_id = eos
            served.append((prompt, request))

        self.server.drain()
        return [Completion(prompt=prompt,
                           text=self.tokenizer.decode(request.tokens),
                           prompt_tokens=len(request.prompt_ids),
                           completion_tokens=len(request.tokens),
                           stop_reason=request.stop_reason)
                for prompt, request in served]

    def stream(self, prompt: str, max_new_tokens: int = 64,
               temperature: float = 0.0) -> Iterator[str]:
        ids = self.tokenizer.encode(prompt)
        request = self.server.submit(ids, max_new_tokens=max_new_tokens,
                                     temperature=temperature)
        request.sequence.eos_id = getattr(self.tokenizer, "eos_id", None)

        emitted = 0
        for chunk in self.server.stream():
            if chunk.request_id != request.id:
                continue
            text = self.tokenizer.decode(request.tokens)
            yield text[emitted:]
            emitted = len(text)
