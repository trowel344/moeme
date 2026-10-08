#!/usr/bin/env python3
"""Publish a hash-bound weight/partition seed for a free-cloud layer run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def prepare_seed(source: Path, output_dir: Path, layer: int) -> dict:
    required = {"expert_down", "router", "shared_indices", "expert_indices"}
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    if not required.issubset(keys):
        raise ValueError(f"training seed omits required tensors: {sorted(required - keys)}")
    if output_dir.exists():
        raise FileExistsError(f"training seed destination already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        tensor_file = temporary / "model.safetensors"
        subprocess.run(
            ["cp", "--reflink=auto", "--sparse=auto", str(source), str(tensor_file)],
            check=True,
        )
        digest = sha256(tensor_file)
        receipt = {
            "format": "moeme-training-seed-receipt-v1",
            "layer": layer,
            "tensor_file": tensor_file.name,
            "bytes": tensor_file.stat().st_size,
            "sha256": digest,
            "source": str(source.resolve()),
            "source_sha256": digest,
            "tensors": sorted(keys),
            "partial_projection_seed": not {"expert_gate", "expert_up"}.issubset(keys),
        }
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output_dir)
        return receipt
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    receipt = prepare_seed(args.source, args.output_dir, args.layer)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
