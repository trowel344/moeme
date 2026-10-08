#!/usr/bin/env python3
"""Recover and atomically publish a complete capture left in a temporary directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

try:
    from scripts.capture_activations import validate_capture
except ModuleNotFoundError:
    from capture_activations import validate_capture


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
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


def finalize_capture(
    partial_dir: Path,
    output_dir: Path,
    corpus: Path,
    layers: list[int],
    expected_tokens: int,
    from_chunk: int,
    cuda_graphs_disabled: bool,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"capture output already exists: {output_dir}")
    if not partial_dir.is_dir():
        raise FileNotFoundError(f"capture temporary directory is missing: {partial_dir}")
    layer_receipts = validate_capture(partial_dir, layers, expected_tokens)
    manifest = {
        "format": "moeme-dense-activation-capture-v1",
        "corpus_sha256": sha256(corpus),
        "layer_count": len(layers),
        "tokens_per_layer": expected_tokens,
        "from_chunk": from_chunk,
        "cuda_graphs_disabled": cuda_graphs_disabled,
        "layers": layer_receipts,
        "recovered_from_completed_temporary": True,
    }
    atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--expected-tokens", type=int, required=True)
    parser.add_argument("--from-chunk", type=int, default=0)
    parser.add_argument("--cuda-graphs-disabled", action="store_true")
    args = parser.parse_args()
    manifest = finalize_capture(
        args.partial_dir,
        args.output_dir,
        args.corpus,
        args.layer,
        args.expected_tokens,
        args.from_chunk,
        args.cuda_graphs_disabled,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
