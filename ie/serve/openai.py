from __future__ import annotations

import time
from dataclasses import dataclass

from .server import ServedRequest

FINISH_REASONS = {"eos": "stop", "length": "length", "cancelled": "cancelled"}


def finish_reason_for(stop_reason: str | None) -> str | None:
    if stop_reason is None:
        return None
    return FINISH_REASONS.get(stop_reason, stop_reason)


@dataclass(slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}


def completion_response(served: ServedRequest, model: str, text: str,
                        created: int | None = None) -> dict:
    usage = Usage(prompt_tokens=len(served.prompt_ids),
                  completion_tokens=len(served.tokens))
    return {
        "id": served.id,
        "object": "text_completion",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason_for(served.stop_reason),
            "logprobs": None,
        }],
        "usage": usage.as_dict(),
    }


def completion_chunk(request_id: str, model: str, text: str, index: int,
                     finish_reason: str | None = None, created: int | None = None) -> dict:
    return {
        "id": request_id,
        "object": "text_completion.chunk",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [{
            "index": index,
            "text": text,
            "finish_reason": finish_reason,
        }],
    }


def server_sent_event(payload: dict | str) -> str:
    import json

    body = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {body}\n\n"


def done_event() -> str:
    return server_sent_event("[DONE]")
