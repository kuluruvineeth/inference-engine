from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_SIZE_BYTES = 8

DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U8": np.dtype("<u1"),
    "BOOL": np.dtype("?"),
}


@dataclass(frozen=True, slots=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def num_elements(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


def bfloat16_to_float32(raw: np.ndarray) -> np.ndarray:
    widened = raw.astype(np.uint32) << 16
    return widened.view(np.float32)


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        size_bytes = handle.read(HEADER_SIZE_BYTES)
        if len(size_bytes) != HEADER_SIZE_BYTES:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_length,) = struct.unpack("<Q", size_bytes)
        header = json.loads(handle.read(header_length))
    return header, HEADER_SIZE_BYTES + header_length


def index_of(path: Path) -> dict[str, TensorEntry]:
    header, _ = read_header(path)
    entries = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        entries[name] = TensorEntry(name=name, dtype=spec["dtype"],
                                    shape=tuple(spec["shape"]), start=start, end=end)
    return entries


class SafetensorsFile:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no safetensors file at {self.path}")
        self.entries = index_of(self.path)
        _, self.data_start = read_header(self.path)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return sorted(self.entries)

    @property
    def total_parameters(self) -> int:
        return sum(entry.num_elements for entry in self.entries.values())

    def get(self, name: str) -> np.ndarray:
        entry = self.entries.get(name)
        if entry is None:
            raise KeyError(f"{name} is not in {self.path.name}")
        if entry.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {entry.dtype} for {name}")

        with open(self.path, "rb") as handle:
            handle.seek(self.data_start + entry.start)
            raw = handle.read(entry.nbytes)

        flat = np.frombuffer(raw, dtype=DTYPES[entry.dtype])
        if entry.dtype == "BF16":
            flat = bfloat16_to_float32(flat)
        return flat.reshape(entry.shape)


class ShardedCheckpoint:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        shards = sorted(self.directory.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors shards in {self.directory}")
        self.shards = [SafetensorsFile(shard) for shard in shards]
        self.owner: dict[str, SafetensorsFile] = {}
        for shard in self.shards:
            for name in shard.entries:
                self.owner[name] = shard

    def __contains__(self, name: str) -> bool:
        return name in self.owner

    def __len__(self) -> int:
        return len(self.owner)

    @property
    def names(self) -> list[str]:
        return sorted(self.owner)

    @property
    def total_parameters(self) -> int:
        return sum(shard.total_parameters for shard in self.shards)

    def get(self, name: str) -> np.ndarray:
        shard = self.owner.get(name)
        if shard is None:
            raise KeyError(f"{name} is not in any shard under {self.directory}")
        return shard.get(name)


def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    reverse = {value: key for key, value in DTYPES.items()}

    for name, array in tensors.items():
        contiguous = np.ascontiguousarray(array)
        dtype = reverse.get(contiguous.dtype.newbyteorder("<"))
        if dtype is None:
            raise ValueError(f"cannot store dtype {contiguous.dtype} for {name}")
        payload = contiguous.tobytes()
        header[name] = {"dtype": dtype, "shape": list(contiguous.shape),
                        "data_offsets": [offset, offset + len(payload)]}
        blobs.append(payload)
        offset += len(payload)

    encoded = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)
