#!/usr/bin/env python3
"""Quantize one trained sparse layer and atomically patch a quantized GGUF.

This avoids materializing a second full BF16 model.  A small GGUF carrier holds
only the trained layer, llama-quantize converts those tensors using the supplied
importance matrix, and the resulting bytes replace the same tensors in a copy
of an already-validated quantized model.  Every other tensor is copied exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from gguf import (
    GGMLQuantizationType,
    GGUFReader,
    GGUFWriter,
)
from safetensors.torch import load_file

from moeme.ledger import ExperimentLedger

try:
    from scripts.patch_gguf_layer import TRAINED_KEYS, validate_trained_state
except ModuleNotFoundError:
    from patch_gguf_layer import TRAINED_KEYS, validate_trained_state


TENSOR_NAMES = {
    "expert_down": "ffn_down_exps.weight",
    "expert_gate": "ffn_gate_exps.weight",
    "expert_up": "ffn_up_exps.weight",
    "router": "ffn_gate_inp.weight",
    "shared_down": "ffn_down_shexp.weight",
    "shared_gate": "ffn_gate_shexp.weight",
    "shared_up": "ffn_up_shexp.weight",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def checkpoint_array(value: torch.Tensor) -> tuple[np.ndarray, GGMLQuantizationType]:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy(), GGMLQuantizationType.BF16
    if value.dtype == torch.float32:
        return value.numpy(), GGMLQuantizationType.F32
    raise TypeError(f"trained tensor must be BF16 or F32, got {value.dtype}")


def copy_model_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    """Copy the architecture metadata needed by llama-quantize.

    Tokenizer arrays are intentionally excluded: the carrier is never loaded as
    a model and copying them would add hundreds of thousands of irrelevant
    strings.  llama-quantize only needs the architecture fields below.
    """
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key.startswith("tokenizer."):
            continue
        subtype = field.types[1] if len(field.types) > 1 else None
        writer.add_key_value(key, field.contents(), field.types[0], subtype)


def write_layer_carrier(
    source: GGUFReader,
    state: dict[str, torch.Tensor],
    layer: int,
    path: Path,
) -> dict[str, str]:
    source_tensors = {tensor.name: tensor for tensor in source.tensors}
    writer = GGUFWriter(path, str(source.fields["general.architecture"].contents()))
    copy_model_metadata(source, writer)
    quant_types: dict[str, str] = {}
    for key, suffix in TENSOR_NAMES.items():
        name = f"blk.{layer}.{suffix}"
        target = source_tensors.get(name)
        if target is None:
            raise KeyError(f"quantized source is missing {name}")
        array, raw_dtype = checkpoint_array(state[key])
        if list(reversed(array.shape)) != target.shape.tolist():
            raise ValueError(
                f"checkpoint shape for {name} is {tuple(array.shape)}; "
                f"quantized target expects reversed shape {target.shape.tolist()}"
            )
        writer.add_tensor(name, array, raw_dtype=raw_dtype)
        quant_types[name] = target.tensor_type.name
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return quant_types


def quantize_layer_carrier(
    binary: Path,
    carrier: Path,
    output: Path,
    imatrix: Path,
    layer: int,
    quant_types: dict[str, str],
    threads: int,
    log: Path,
) -> None:
    command = [str(binary), "--imatrix", str(imatrix)]
    # llama.cpp treats --tensor-type selectors as regular expressions, but its
    # imatrix include list is a set of exact tensor names.  Repeat the option so
    # every mixed-precision tensor receives its captured importance weights.
    for name in sorted(quant_types):
        command += ["--include-weights", name]
    for name, quant_type in sorted(quant_types.items()):
        command += ["--tensor-type", f"^{re.escape(name)}$={quant_type}"]
    command += [str(carrier), str(output), "Q5_K_M", str(threads)]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def patch_quantized_copy(
    source_path: Path,
    quantized_layer_path: Path,
    output_path: Path,
    layer: int,
) -> dict[str, str]:
    temporary = output_path.with_name(f".{output_path.name}.partial")
    if output_path.exists() or temporary.exists():
        raise FileExistsError("output or partial output already exists")
    required = source_path.stat().st_size + 1_000_000_000
    free = shutil.disk_usage(output_path.parent).free
    if free < required:
        raise OSError(
            f"direct quantized-layer patch requires {required:,} free bytes; "
            f"only {free:,} are available"
        )
    subprocess.run(
        ["cp", "--sparse=auto", str(source_path), str(temporary)],
        check=True,
    )
    try:
        layer_reader = GGUFReader(quantized_layer_path, "r")
        target_reader = GGUFReader(temporary, "r+")
        layer_tensors = {tensor.name: tensor for tensor in layer_reader.tensors}
        target_tensors = {tensor.name: tensor for tensor in target_reader.tensors}
        hashes: dict[str, str] = {}
        expected_names = {f"blk.{layer}.{suffix}" for suffix in TENSOR_NAMES.values()}
        if set(layer_tensors) != expected_names:
            raise ValueError(
                "quantized carrier tensor set mismatch: "
                f"{sorted(set(layer_tensors) ^ expected_names)}"
            )
        for name in sorted(expected_names):
            source = layer_tensors[name]
            target = target_tensors.get(name)
            if target is None:
                raise KeyError(f"candidate model is missing {name}")
            if (
                source.tensor_type != target.tensor_type
                or source.data.shape != target.data.shape
                or source.data.nbytes != target.data.nbytes
            ):
                raise ValueError(f"quantized tensor layout mismatch for {name}")
            target.data[...] = source.data
            digest = hashlib.sha256(source.data.tobytes()).hexdigest()
            if hashlib.sha256(target.data.tobytes()).hexdigest() != digest:
                raise ValueError(f"post-write verification failed for {name}")
            hashes[name] = digest
        del target_reader
        del layer_reader
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        return hashes
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-quantized", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--minimum-router-nonzero", type=float, default=0.99)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()

    for path in (args.source_quantized, args.checkpoint, args.imatrix, args.binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.layer < 0:
        parser.error("--layer must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = load_file(args.checkpoint, device="cpu")
    health = validate_trained_state(state, args.layer, args.minimum_router_nonzero)
    missing = sorted(TRAINED_KEYS - set(state))
    if missing:
        raise ValueError(f"checkpoint must contain all trained projections: {missing}")

    configuration = {
        "source_quantized": str(args.source_quantized.resolve()),
        "source_sha256": sha256(args.source_quantized),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "imatrix": str(args.imatrix.resolve()),
        "imatrix_sha256": sha256(args.imatrix),
        "binary": str(args.binary.resolve()),
        "layer": args.layer,
        "output": str(args.output.resolve()),
        "threads": args.threads,
    }
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    ledger = ExperimentLedger(args.ledger)
    run_id = ledger.start("quantized-layer-patch", digest, configuration)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{args.output.name}.layer-", dir=args.output.parent
        ) as directory:
            carrier = Path(directory) / "layer-bf16.gguf"
            quantized = Path(directory) / "layer-quantized.gguf"
            source_reader = GGUFReader(args.source_quantized, "r")
            quant_types = write_layer_carrier(source_reader, state, args.layer, carrier)
            del source_reader
            quantize_layer_carrier(
                args.binary,
                carrier,
                quantized,
                args.imatrix,
                args.layer,
                quant_types,
                args.threads,
                args.log,
            )
            patched = patch_quantized_copy(
                args.source_quantized, quantized, args.output, args.layer
            )
        output_sha = sha256(args.output)
        summary = {
            "format": "moeme-quantized-layer-patch-v1",
            "layer": args.layer,
            "bytes": args.output.stat().st_size,
            "sha256": output_sha,
            "source_sha256": configuration["source_sha256"],
            "checkpoint_sha256": configuration["checkpoint_sha256"],
            "imatrix_sha256": configuration["imatrix_sha256"],
            "quant_types": quant_types,
            "patched_tensor_sha256": patched,
            "checkpoint_health": health,
            "unchanged_tensor_contract": "byte-exact copy outside named layer tensors",
        }
        atomic_json(args.receipt, summary)
        ledger.artifact(
            run_id,
            "quantized-layer-candidate",
            args.output,
            sha256=output_sha,
            bytes_count=args.output.stat().st_size,
        )
        ledger.finish(run_id, "passed", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        ledger.finish(run_id, "failed", {}, error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
