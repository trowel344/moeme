from __future__ import annotations

import struct
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch

MAGIC = 0x4D4F454D
VERSION = 1
HEADER = struct.Struct("<III")
CHUNK_HEADER = struct.Struct("<I")


def inspect_activation_capture(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("truncated activation header")
        magic, version, width = HEADER.unpack(header)
        if magic != MAGIC:
            raise ValueError(f"invalid activation magic: {magic:#x}")
        if version != VERSION:
            raise ValueError(f"unsupported activation version: {version}")
        chunks = []
        tokens = 0
        while handle.tell() < size:
            value = handle.read(CHUNK_HEADER.size)
            if len(value) != CHUNK_HEADER.size:
                raise ValueError("truncated activation chunk header")
            chunk_tokens = CHUNK_HEADER.unpack(value)[0]
            payload_bytes = chunk_tokens * width * 4
            payload_offset = handle.tell()
            handle.seek(payload_bytes, 1)
            if handle.tell() > size:
                raise ValueError("truncated activation chunk payload")
            chunks.append(
                {
                    "tokens": chunk_tokens,
                    "payload_offset": payload_offset,
                    "payload_bytes": payload_bytes,
                }
            )
            tokens += chunk_tokens
        if handle.tell() != size:
            raise ValueError("activation capture has trailing or missing bytes")
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "version": version,
        "width": width,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "tokens": tokens,
    }


def load_activation_capture(path: str | Path) -> np.ndarray:
    path = Path(path)
    info = inspect_activation_capture(path)
    output = np.empty((info["tokens"], info["width"]), dtype=np.float32)
    cursor = 0
    for chunk in info["chunks"]:
        tokens = chunk["tokens"]
        values = np.memmap(
            path,
            mode="r",
            dtype=np.float32,
            offset=chunk["payload_offset"],
            shape=(tokens, info["width"]),
        )
        output[cursor : cursor + tokens] = values
        cursor += tokens
    return output


def activation_capture_statistics(path: str | Path) -> dict[str, Any]:
    """Compute finite/mean/std checks while holding only one capture chunk."""

    path = Path(path)
    info = inspect_activation_capture(path)
    value_count = info["tokens"] * info["width"]
    total = 0.0
    total_square = 0.0
    finite = True
    for chunk in info["chunks"]:
        values = np.memmap(
            path,
            mode="r",
            dtype=np.float32,
            offset=chunk["payload_offset"],
            shape=(chunk["tokens"], info["width"]),
        )
        finite = finite and bool(np.isfinite(values).all())
        total += float(values.sum(dtype=np.float64))
        total_square += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
    mean = total / value_count
    variance = max(0.0, total_square / value_count - mean * mean)
    return {**info, "finite": finite, "mean": mean, "std": variance**0.5}


def activation_capture_paths(directory: str | Path, layer: int) -> list[Path]:
    """Resolve one atomic capture or a completed deterministic shard campaign."""

    directory = Path(directory)
    single = directory / f"layer-{layer}.f32"
    if single.is_file():
        return [single]
    campaign_path = directory / "campaign.json"
    if not campaign_path.is_file():
        raise FileNotFoundError(f"no layer-{layer}.f32 or activation shard campaign in {directory}")
    import json

    campaign = json.loads(campaign_path.read_text())
    if campaign.get("status") != "passed":
        raise ValueError("activation shard campaign is not complete")
    shards = campaign.get("shards", {})
    expected = list(range(len(shards)))
    observed = sorted(int(key) for key in shards)
    if observed != expected:
        raise ValueError("activation shard campaign indices are not contiguous")
    paths = []
    for index in expected:
        receipt = shards[str(index)]
        if receipt.get("status") != "passed":
            raise ValueError(f"activation shard {index} did not pass")
        path = directory / f"shard-{index:04d}" / f"layer-{layer}.f32"
        if not path.is_file():
            raise FileNotFoundError(f"activation shard payload is missing: {path}")
        paths.append(path)
    if not paths:
        raise ValueError("activation shard campaign contains no shards")
    return paths


class ActivationTensorDataset:
    """Random-access activation rows without materializing whole captures in RAM."""

    def __init__(self, paths: list[Path]):
        if not paths:
            raise ValueError("at least one activation capture is required")
        self.paths = [Path(path) for path in paths]
        self.segments: list[dict[str, Any]] = []
        self.ends: list[int] = []
        width = None
        cursor = 0
        for path in self.paths:
            info = inspect_activation_capture(path)
            if width is None:
                width = int(info["width"])
            elif width != int(info["width"]):
                raise ValueError("activation captures have different widths")
            for chunk in info["chunks"]:
                tokens = int(chunk["tokens"])
                self.segments.append(
                    {
                        "path": path,
                        "start": cursor,
                        "tokens": tokens,
                        "payload_offset": int(chunk["payload_offset"]),
                    }
                )
                cursor += tokens
                self.ends.append(cursor)
        assert width is not None
        self.width = width
        self.shape = (cursor, width)

    @classmethod
    def from_directory(cls, directory: str | Path, layer: int) -> ActivationTensorDataset:
        return cls(activation_capture_paths(directory, layer))

    def __len__(self) -> int:
        return self.shape[0]

    def index_select(self, dimension: int, indices: torch.Tensor) -> torch.Tensor:
        if dimension != 0:
            raise ValueError("activation dataset supports index_select only on rows")
        return self[indices]

    def __getitem__(self, key: int | slice | list[int] | np.ndarray | torch.Tensor) -> torch.Tensor:
        scalar = isinstance(key, (int, np.integer))
        if isinstance(key, slice):
            indices = np.arange(*key.indices(len(self)), dtype=np.int64)
        elif isinstance(key, torch.Tensor):
            indices = key.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        else:
            indices = np.asarray([key] if scalar else key, dtype=np.int64).reshape(-1)
        indices = np.where(indices < 0, indices + len(self), indices)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("activation row index out of range")
        output = np.empty((len(indices), self.width), dtype=np.float32)
        by_segment: dict[int, list[tuple[int, int]]] = {}
        for output_row, index in enumerate(indices.tolist()):
            segment_index = bisect_right(self.ends, index)
            local_row = index - int(self.segments[segment_index]["start"])
            by_segment.setdefault(segment_index, []).append((output_row, local_row))
        for segment_index, rows in by_segment.items():
            segment = self.segments[segment_index]
            values = np.memmap(
                segment["path"],
                mode="r",
                dtype=np.float32,
                offset=segment["payload_offset"],
                shape=(segment["tokens"], self.width),
            )
            output_rows = [row[0] for row in rows]
            local_rows = [row[1] for row in rows]
            output[output_rows] = values[local_rows]
        tensor = torch.from_numpy(output)
        return tensor[0] if scalar else tensor


class IndexedActivationDataset:
    """A deterministic row view used for train/validation splits."""

    def __init__(self, source: ActivationTensorDataset, indices: torch.Tensor):
        self.source = source
        self.indices = indices.detach().cpu().to(dtype=torch.int64)
        self.shape = (len(self.indices), source.width)

    def __len__(self) -> int:
        return len(self.indices)

    def index_select(self, dimension: int, indices: torch.Tensor) -> torch.Tensor:
        if dimension != 0:
            raise ValueError("activation dataset supports index_select only on rows")
        return self[indices]

    def __getitem__(self, key: int | slice | list[int] | np.ndarray | torch.Tensor) -> torch.Tensor:
        return self.source[self.indices[key]]


def activation_block_split_indices(
    source: ActivationTensorDataset, validation_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hold out complete capture chunks, falling back only for a one-chunk fixture."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    generator = torch.Generator().manual_seed(seed)
    if len(source.segments) < 2:
        order = torch.randperm(len(source), generator=generator)
        validation_count = max(1, min(len(source) - 1, round(len(source) * validation_fraction)))
        return order[:validation_count], order[validation_count:]
    segment_order = torch.randperm(len(source.segments), generator=generator).tolist()
    validation_segments = max(
        1, min(len(source.segments) - 1, round(len(source.segments) * validation_fraction))
    )

    def rows(segment_indices: list[int]) -> torch.Tensor:
        return torch.cat(
            [
                torch.arange(
                    int(source.segments[index]["start"]),
                    int(source.segments[index]["start"] + source.segments[index]["tokens"]),
                )
                for index in segment_indices
            ]
        )

    return rows(segment_order[:validation_segments]), rows(segment_order[validation_segments:])
