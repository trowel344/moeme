from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .layout import ExpertLayout

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
FORMAT_VERSION = 2
MLP_PATTERN = re.compile(
    r"^(model\.language_model\.layers\.(\d+)\.mlp)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True, slots=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def bytes_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class SafeTensorFile:
    path: Path
    data_start: int
    tensors: tuple[TensorEntry, ...]
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class CopyPlan:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source: Path | None
    source_data_start: int
    source_offset: int
    byte_count: int
    row_count: int | None = None
    source_row_bytes: int | None = None
    row_offset_bytes: int | None = None
    output_row_bytes: int | None = None
    fill_byte: int | None = None

    def chunks(self, chunk_bytes: int = 8 * 1024 * 1024) -> Iterator[bytes]:
        if self.fill_byte is not None:
            remaining = self.byte_count
            block = bytes([self.fill_byte]) * min(chunk_bytes, max(remaining, 1))
            while remaining:
                piece = block[: min(len(block), remaining)]
                yield piece
                remaining -= len(piece)
            return
        if self.source is None:
            raise ValueError(f"copy plan {self.name} has no source")
        with self.source.open("rb") as handle:
            if self.row_count is None:
                handle.seek(self.source_data_start + self.source_offset)
                yield from _read_exact_chunks(handle, self.byte_count, chunk_bytes)
                return
            assert self.source_row_bytes is not None
            assert self.row_offset_bytes is not None
            assert self.output_row_bytes is not None
            for row in range(self.row_count):
                handle.seek(
                    self.source_data_start
                    + self.source_offset
                    + row * self.source_row_bytes
                    + self.row_offset_bytes
                )
                yield from _read_exact_chunks(handle, self.output_row_bytes, chunk_bytes)


def read_safetensors_header(path: str | Path) -> SafeTensorFile:
    path = Path(path)
    with path.open("rb") as handle:
        length_data = handle.read(8)
        if len(length_data) != 8:
            raise ValueError(f"{path}: truncated safetensors length")
        header_length = struct.unpack("<Q", length_data)[0]
        if header_length > path.stat().st_size - 8:
            raise ValueError(f"{path}: header length exceeds file size")
        try:
            header = json.loads(handle.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: invalid safetensors JSON header") from error
    data_start = 8 + header_length
    metadata = header.pop("__metadata__", {})
    tensors = []
    for name, value in header.items():
        start, end = value["data_offsets"]
        tensors.append(TensorEntry(name, value["dtype"], tuple(value["shape"]), start, end))
    tensors.sort(key=lambda entry: entry.start)
    previous_end = 0
    for entry in tensors:
        _validate_entry(path, entry, previous_end, path.stat().st_size - data_start)
        previous_end = entry.end
    return SafeTensorFile(path, data_start, tuple(tensors), metadata)


def plan_moeme_shard(source: SafeTensorFile, layout: ExpertLayout) -> list[CopyPlan]:
    plans: list[CopyPlan] = []
    router_layers: set[int] = set()
    for entry in source.tensors:
        match = MLP_PATTERN.match(entry.name)
        if not match:
            plans.append(_whole_tensor_plan(source, entry))
            continue
        prefix, layer_text, projection = match.groups()
        layer = int(layer_text)
        if projection == "gate_proj":
            router_layers.add(layer)
        plans.extend(_split_mlp_tensor(source, entry, prefix, projection, layout))
    for layer in sorted(router_layers):
        hidden_size = _hidden_size_for_layer(source, layer)
        gate_entry = _entry_by_name(
            source, f"model.language_model.layers.{layer}.mlp.gate_proj.weight"
        )
        plans.append(
            CopyPlan(
                name=f"model.language_model.layers.{layer}.mlp.router.weight",
                dtype="F32",
                shape=(layout.routed_experts, hidden_size),
                source=None,
                source_data_start=0,
                source_offset=0,
                byte_count=layout.routed_experts * hidden_size * 4,
                fill_byte=0,
            )
        )
        if layout.shared_width:
            plans.append(
                CopyPlan(
                    name=(f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"),
                    dtype=gate_entry.dtype,
                    shape=(1, hidden_size),
                    source=None,
                    source_data_start=0,
                    source_offset=0,
                    byte_count=hidden_size * DTYPE_BYTES[gate_entry.dtype],
                    fill_byte=0,
                )
            )
    return plans


def write_safetensors_streaming(
    path: str | Path,
    plans: list[CopyPlan],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    path = Path(path)
    if len({plan.name for plan in plans}) != len(plans):
        raise ValueError("output tensor names must be unique")
    header: dict[str, Any] = {}
    offset = 0
    for plan in plans:
        header[plan.name] = {
            "dtype": plan.dtype,
            "shape": list(plan.shape),
            "data_offsets": [offset, offset + plan.byte_count],
        }
        offset += plan.byte_count
    if metadata:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(struct.pack("<Q", len(header_bytes)))
            output.write(header_bytes)
            for plan in plans:
                written = 0
                for chunk in plan.chunks():
                    output.write(chunk)
                    written += len(chunk)
                if written != plan.byte_count:
                    raise OSError(
                        f"short streaming write for {plan.name}: {written} != {plan.byte_count}"
                    )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def convert_checkpoint_streaming(
    source_dir: str | Path,
    output_dir: str | Path,
    layout: ExpertLayout,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Convert dense MLP tensors while copying all other tensors byte-for-byte."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    index_path = source_dir / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        source_index = json.load(handle)
    shard_names = sorted(set(source_index["weight_map"].values()))
    missing = [name for name in shard_names if not (source_dir / name).is_file()]
    if missing and not allow_incomplete:
        raise FileNotFoundError(
            f"checkpoint is incomplete; missing {len(missing)} shards: {missing[:3]}"
        )
    available_shards = [name for name in shard_names if name not in missing]
    if not available_shards:
        raise FileNotFoundError("checkpoint has no completed safetensors shards yet")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "conversion-progress.json"
    fingerprint = _conversion_fingerprint(source_dir, source_index, layout)
    progress = _load_progress(progress_path, fingerprint)
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name in available_shards:
        source = read_safetensors_header(source_dir / shard_name)
        plans = plan_moeme_shard(source, layout)
        output_path = output_dir / shard_name
        expected_data_bytes = sum(plan.byte_count for plan in plans)
        completed = progress["completed"].get(shard_name)
        if not _completed_shard_is_valid(output_path, completed, expected_data_bytes, plans):
            write_safetensors_streaming(
                output_path,
                plans,
                metadata={**source.metadata, "moeme_layout": "disjoint-v1"},
            )
            progress["completed"][shard_name] = {
                "file_bytes": output_path.stat().st_size,
                "data_bytes": expected_data_bytes,
                "tensor_count": len(plans),
            }
            _atomic_json(progress_path, progress)
        for plan in plans:
            weight_map[plan.name] = shard_name
            total_size += plan.byte_count

    output_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    _atomic_json(output_dir / "model.safetensors.index.json", output_index)
    for name in (
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    manifest = {
        "format": f"moeme-safetensors-v{FORMAT_VERSION}",
        "source": str(source_dir.resolve()),
        "layout": {
            "intermediate_size": layout.intermediate_size,
            "groups": layout.groups,
            "shared_groups": layout.shared_groups,
            "routed_experts": layout.routed_experts,
            "initial_top_k": layout.routed_experts,
            "target_top_k": layout.top_k,
        },
        "complete": not missing,
        "converted_shards": len(available_shards),
        "expected_shards": len(shard_names),
        "missing_shards": missing,
        "tensors": len(weight_map),
        "total_size": total_size,
    }
    _atomic_json(output_dir / "moeme-manifest.json", manifest)
    progress["status"] = "complete" if not missing else "waiting-for-source-shards"
    _atomic_json(progress_path, progress)
    return manifest


def validate_converted_shards(
    source_dir: str | Path,
    output_dir: str | Path,
    layout: ExpertLayout,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Byte-compare converted tensor payloads with their exact source plans."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    with (source_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        source_index = json.load(handle)
    shard_names = sorted(set(source_index["weight_map"].values()))
    available = [
        name
        for name in shard_names
        if (source_dir / name).is_file() and (output_dir / name).is_file()
    ]
    missing = [name for name in shard_names if name not in available]
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"conversion is incomplete; {len(missing)} shards unavailable")
    if not available:
        raise FileNotFoundError("no source/output shard pairs are available")

    tensor_count = 0
    bytes_checked = 0
    for shard_name in available:
        source = read_safetensors_header(source_dir / shard_name)
        output = read_safetensors_header(output_dir / shard_name)
        plans = plan_moeme_shard(source, layout)
        output_entries = {entry.name: entry for entry in output.tensors}
        if set(output_entries) != {plan.name for plan in plans}:
            missing_names = sorted({plan.name for plan in plans} - set(output_entries))
            unexpected = sorted(set(output_entries) - {plan.name for plan in plans})
            raise ValueError(
                f"{shard_name}: tensor-name mismatch; missing={missing_names[:3]}, "
                f"unexpected={unexpected[:3]}"
            )
        for plan in plans:
            entry = output_entries[plan.name]
            if entry.dtype != plan.dtype or entry.shape != plan.shape:
                raise ValueError(f"{shard_name}:{plan.name}: dtype/shape mismatch")
            _compare_plan_to_output(plan, output, entry)
            tensor_count += 1
            bytes_checked += plan.byte_count
    return {
        "valid": True,
        "complete": not missing,
        "validated_shards": len(available),
        "expected_shards": len(shard_names),
        "missing_shards": missing,
        "tensors_checked": tensor_count,
        "bytes_checked": bytes_checked,
    }


def _split_mlp_tensor(
    source: SafeTensorFile,
    entry: TensorEntry,
    prefix: str,
    projection: str,
    layout: ExpertLayout,
) -> list[CopyPlan]:
    if entry.dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype for streaming split: {entry.dtype}")
    element_bytes = DTYPE_BYTES[entry.dtype]
    if projection in {"gate_proj", "up_proj"}:
        if len(entry.shape) != 2 or entry.shape[0] != layout.intermediate_size:
            raise ValueError(f"{entry.name}: unexpected projection shape {entry.shape}")
        hidden = entry.shape[1]
        row_bytes = hidden * element_bytes
        widths = [layout.shared_width] + [layout.expert_width] * layout.routed_experts
        names = ["shared_expert"] + [f"experts.{i}" for i in range(layout.routed_experts)]
        if layout.shared_width == 0:
            widths, names = widths[1:], names[1:]
        plans = []
        row_start = 0
        for name, width in zip(names, widths, strict=True):
            plans.append(
                CopyPlan(
                    name=f"{prefix}.{name}.{projection}.weight",
                    dtype=entry.dtype,
                    shape=(width, hidden),
                    source=source.path,
                    source_data_start=source.data_start,
                    source_offset=entry.start + row_start * row_bytes,
                    byte_count=width * row_bytes,
                )
            )
            row_start += width
        return plans

    if len(entry.shape) != 2 or entry.shape[1] != layout.intermediate_size:
        raise ValueError(f"{entry.name}: unexpected down projection shape {entry.shape}")
    hidden = entry.shape[0]
    source_row_bytes = layout.intermediate_size * element_bytes
    widths = [layout.shared_width] + [layout.expert_width] * layout.routed_experts
    names = ["shared_expert"] + [f"experts.{i}" for i in range(layout.routed_experts)]
    if layout.shared_width == 0:
        widths, names = widths[1:], names[1:]
    plans = []
    column_start = 0
    for name, width in zip(names, widths, strict=True):
        output_row_bytes = width * element_bytes
        plans.append(
            CopyPlan(
                name=f"{prefix}.{name}.{projection}.weight",
                dtype=entry.dtype,
                shape=(hidden, width),
                source=source.path,
                source_data_start=source.data_start,
                source_offset=entry.start,
                byte_count=hidden * output_row_bytes,
                row_count=hidden,
                source_row_bytes=source_row_bytes,
                row_offset_bytes=column_start * element_bytes,
                output_row_bytes=output_row_bytes,
            )
        )
        column_start += width
    return plans


def _whole_tensor_plan(source: SafeTensorFile, entry: TensorEntry) -> CopyPlan:
    return CopyPlan(
        name=entry.name,
        dtype=entry.dtype,
        shape=entry.shape,
        source=source.path,
        source_data_start=source.data_start,
        source_offset=entry.start,
        byte_count=entry.bytes_count,
    )


def _hidden_size_for_layer(source: SafeTensorFile, layer: int) -> int:
    name = f"model.language_model.layers.{layer}.mlp.gate_proj.weight"
    for entry in source.tensors:
        if entry.name == name:
            return entry.shape[1]
    raise ValueError(f"cannot derive hidden size for layer {layer}")


def _entry_by_name(source: SafeTensorFile, name: str) -> TensorEntry:
    for entry in source.tensors:
        if entry.name == name:
            return entry
    raise ValueError(f"missing source tensor {name}")


def _validate_entry(path: Path, entry: TensorEntry, previous_end: int, data_bytes: int) -> None:
    if entry.dtype not in DTYPE_BYTES:
        raise ValueError(f"{path}: unsupported dtype {entry.dtype}")
    if entry.start != previous_end or entry.end < entry.start or entry.end > data_bytes:
        raise ValueError(f"{path}: invalid/non-contiguous offsets for {entry.name}")
    elements = 1
    for dimension in entry.shape:
        if dimension < 0:
            raise ValueError(f"{path}: negative dimension for {entry.name}")
        elements *= dimension
    if elements * DTYPE_BYTES[entry.dtype] != entry.bytes_count:
        raise ValueError(f"{path}: byte size mismatch for {entry.name}")


def _read_exact_chunks(handle: BinaryIO, count: int, chunk_bytes: int) -> Iterator[bytes]:
    remaining = count
    while remaining:
        chunk = handle.read(min(remaining, chunk_bytes))
        if not chunk:
            raise OSError(f"unexpected EOF with {remaining} bytes remaining")
        remaining -= len(chunk)
        yield chunk


def _compare_plan_to_output(
    plan: CopyPlan,
    output: SafeTensorFile,
    entry: TensorEntry,
) -> None:
    expected_chunks = plan.chunks()
    with output.path.open("rb") as handle:
        handle.seek(output.data_start + entry.start)
        for expected in expected_chunks:
            actual = handle.read(len(expected))
            if actual != expected:
                raise ValueError(f"payload mismatch for {plan.name}")
        if handle.tell() != output.data_start + entry.end:
            raise ValueError(f"payload length mismatch for {plan.name}")


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _conversion_fingerprint(
    source_dir: Path, source_index: dict[str, Any], layout: ExpertLayout
) -> str:
    value = {
        "source": str(source_dir.resolve()),
        "index": source_index,
        "layout": {
            "intermediate_size": layout.intermediate_size,
            "groups": layout.groups,
            "shared_groups": layout.shared_groups,
            "routed_experts": layout.routed_experts,
            "top_k": layout.top_k,
        },
        "format": f"moeme-safetensors-v{FORMAT_VERSION}",
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _load_progress(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "format_version": FORMAT_VERSION,
            "fingerprint": fingerprint,
            "status": "running",
            "completed": {},
        }
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("format_version") is None:
        return {
            "format_version": FORMAT_VERSION,
            "fingerprint": fingerprint,
            "status": "migrating-format",
            "completed": {},
        }
    if value.get("fingerprint") != fingerprint:
        raise ValueError(f"{path}: conversion inputs/layout changed; use a new output directory")
    if not isinstance(value.get("completed"), dict):
        raise TypeError(f"{path}: invalid completed-shard state")
    return value


def _completed_shard_is_valid(
    path: Path,
    completed: dict[str, Any] | None,
    expected_data_bytes: int,
    plans: list[CopyPlan],
) -> bool:
    if not completed or not path.is_file() or path.stat().st_size != completed.get("file_bytes"):
        return False
    if completed.get("data_bytes") != expected_data_bytes:
        return False
    if completed.get("tensor_count") != len(plans):
        return False
    try:
        parsed = read_safetensors_header(path)
    except (OSError, ValueError):
        return False
    actual = {entry.name: (entry.dtype, entry.shape) for entry in parsed.tensors}
    expected = {plan.name: (plan.dtype, plan.shape) for plan in plans}
    return actual == expected
