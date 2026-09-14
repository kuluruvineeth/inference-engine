from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class Tokenizer:
    def __init__(self, backend, eos_id: int | None = None,
                 bos_id: int | None = None) -> None:
        self.backend = backend
        self.eos_id = eos_id
        self.bos_id = bos_id

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Tokenizer":
        directory = Path(directory)
        path = directory / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"no tokenizer.json in {directory}")

        try:
            from tokenizers import Tokenizer as Backend
        except ImportError as error:
            raise ImportError("pip install tokenizers to load a real tokenizer") from error

        backend = Backend.from_file(str(path))
        eos_id, bos_id = cls._special_ids(directory, backend)
        return cls(backend, eos_id=eos_id, bos_id=bos_id)

    @staticmethod
    def _special_ids(directory: Path, backend) -> tuple[int | None, int | None]:
        config_path = directory / "generation_config.json"
        if config_path.exists():
            raw = json.loads(config_path.read_text())
            eos = raw.get("eos_token_id")
            if isinstance(eos, list):
                eos = eos[0] if eos else None
            return eos, raw.get("bos_token_id")

        config_path = directory / "tokenizer_config.json"
        if config_path.exists():
            raw = json.loads(config_path.read_text())
            eos_token = raw.get("eos_token")
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content")
            if eos_token:
                return backend.token_to_id(eos_token), None
        return None, None

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if add_bos and self.bos_id is not None:
            return [self.bos_id] + ids
        return ids

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        return self.backend.decode(token_ids, skip_special_tokens=skip_special)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]


class OfflineTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None, eos_id: int = 0) -> None:
        self.vocab = vocab or {}
        self.inverse = {value: key for key, value in self.vocab.items()}
        self.eos_id = eos_id
        self.bos_id = None

    @property
    def vocab_size(self) -> int:
        return max(self.vocab.values()) + 1 if self.vocab else 256

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        return [self.vocab.get(piece, ord(piece[0]) % self.vocab_size)
                for piece in text.split()] or [0]

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        return " ".join(self.inverse.get(token, f"<{token}>") for token in token_ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]


def apply_chat_template(messages: list[ChatMessage], add_generation_prompt: bool = True) -> str:
    parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages]
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)
