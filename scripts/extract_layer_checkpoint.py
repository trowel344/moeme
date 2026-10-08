#!/usr/bin/env python3
"""Extract one MoEMe layer into a small, portable training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def layer_tensor_names(index: dict[str, str], layer: int) -> list[str]:
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = sorted(name for name in index if name.startswith(prefix))
    if not names:
        raise ValueError(f"checkpoint contains no MLP tensors for layer {layer}")
    return names


def extract_layer_checkpoint(checkpoint: Path, layer: int, output_dir: Path) -> dict:
    checkpoint = checkpoint.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"destination already exists: {output_dir}")

    source_index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = source_index["weight_map"]
    names = layer_tensor_names(weight_map, layer)
    source_manifest = json.loads((checkpoint / "moeme-manifest.json").read_text())

    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    tensors = {}
    for filename, file_names in by_file.items():
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for name in file_names:
                tensors[name] = handle.get_tensor(name)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        tensor_file = temporary / "model.safetensors"
        save_file(tensors, tensor_file)
        portable_index = {
            "metadata": {"total_size": tensor_file.stat().st_size},
            "weight_map": {name: tensor_file.name for name in names},
        }
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(portable_index, indent=2, sort_keys=True) + "\n"
        )
        portable_manifest = {
            "complete": True,
            "format": "moeme-portable-layer-checkpoint-v1",
            "layer": layer,
            "layout": source_manifest["layout"],
            "source_format": source_manifest.get("format"),
            "tensor_count": len(names),
        }
        (temporary / "moeme-manifest.json").write_text(
            json.dumps(portable_manifest, indent=2, sort_keys=True) + "\n"
        )
        receipt = {
            "format": "moeme-portable-layer-checkpoint-receipt-v1",
            "layer": layer,
            "tensor_count": len(names),
            "tensor_file": tensor_file.name,
            "bytes": tensor_file.stat().st_size,
            "sha256": sha256(tensor_file),
        }
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output_dir)
        return receipt
    except BaseException:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = extract_layer_checkpoint(args.checkpoint, args.layer, args.output_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
