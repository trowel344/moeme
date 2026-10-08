#!/usr/bin/env python3
"""Structurally validate and hash-bind a completed llama-imatrix GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from gguf import GGUFReader

REQUIRED_FIELDS = {
    "imatrix.chunk_count",
    "imatrix.chunk_size",
    "imatrix.datasets",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def receipt_imatrix(path: Path, receipt_path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"importance matrix is missing: {path}")
    reader = GGUFReader(path, "r")
    fields = sorted(field.name for field in reader.fields.values())
    tensor_count = len(reader.tensors)
    missing_fields = sorted(REQUIRED_FIELDS - set(fields))
    if missing_fields:
        raise ValueError("importance matrix GGUF is missing metadata: " + ", ".join(missing_fields))
    tensor_names = {tensor.name for tensor in reader.tensors}
    sums = {name.removesuffix(".in_sum2") for name in tensor_names if name.endswith(".in_sum2")}
    counts = {name.removesuffix(".counts") for name in tensor_names if name.endswith(".counts")}
    if not sums or sums != counts or tensor_count != len(sums) * 2:
        raise ValueError("importance matrix GGUF has incomplete sum/count tensor pairs")
    chunk_count_field = reader.fields["imatrix.chunk_count"]
    chunk_size_field = reader.fields["imatrix.chunk_size"]
    chunk_count = int(chunk_count_field.parts[chunk_count_field.data[0]][0])
    chunk_size = int(chunk_size_field.parts[chunk_size_field.data[0]][0])
    if chunk_count <= 0 or chunk_size <= 0:
        raise ValueError("importance matrix GGUF has an empty capture")
    receipt = {
        "format": "moeme-imatrix-receipt-v1",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "field_count": len(fields),
        "tensor_count": tensor_count,
        "tensor_pair_count": len(sums),
        "chunk_count": chunk_count,
        "chunk_size": chunk_size,
        "fields": fields,
    }
    del reader
    atomic_json(receipt_path, receipt)
    return receipt


def verify_imatrix_receipt(path: Path, receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != "moeme-imatrix-receipt-v1":
        raise ValueError("unsupported importance-matrix receipt format")
    if path.stat().st_size != receipt.get("bytes"):
        raise ValueError("importance-matrix size does not match its receipt")
    digest = sha256(path)
    if digest != receipt.get("sha256"):
        raise ValueError("importance-matrix SHA-256 does not match its receipt")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = (
        verify_imatrix_receipt(args.imatrix, args.receipt)
        if args.verify
        else receipt_imatrix(args.imatrix, args.receipt)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
