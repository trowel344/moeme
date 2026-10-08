#!/usr/bin/env python3
"""Atomically quantize and structurally validate a GGUF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from gguf import GGUFReader

from moeme.ledger import ExperimentLedger


def quantize_command(
    binary: Path,
    input_path: Path,
    output_path: Path,
    quant_type: str,
    threads: int,
    *,
    imatrix: Path | None = None,
    tensor_types: Sequence[str] = (),
    output_tensor_type: str | None = None,
    token_embedding_type: str | None = None,
    allow_requantize: bool = False,
) -> list[str]:
    """Assemble a llama-quantize command line for an optional mixed-precision run.

    ``tensor_types`` are ``name=ggml_type`` selectors (llama.cpp matches ``name``
    as a regex), used to raise precision only where it matters, e.g.
    ``ffn_.*_exps=Q5_K`` for routed experts.
    """
    command = [str(binary)]
    if allow_requantize:
        command.append("--allow-requantize")
    if imatrix is not None:
        command += ["--imatrix", str(imatrix)]
    for tensor_type in tensor_types:
        command += ["--tensor-type", tensor_type]
    if output_tensor_type is not None:
        command += ["--output-tensor-type", output_tensor_type]
    if token_embedding_type is not None:
        command += ["--token-embedding-type", token_embedding_type]
    command += [str(input_path), str(output_path), quant_type, str(threads)]
    return command


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", default="Q4_K_M")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, help="importance matrix for quant optimizations")
    parser.add_argument(
        "--tensor-type",
        action="append",
        default=[],
        metavar="NAME=TYPE",
        help="quantize matching tensors to TYPE; may be repeated (name is a regex)",
    )
    parser.add_argument("--output-tensor-type", help="ggml_type for output.weight")
    parser.add_argument("--token-embedding-type", help="ggml_type for token_embd.weight")
    parser.add_argument(
        "--allow-requantize",
        action="store_true",
        help="allow requantizing already-quantized tensors (accepts a Q4 input)",
    )
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()

    configuration = {
        "binary": str(args.binary.resolve()),
        "input": str(args.input.resolve()),
        "input_bytes": args.input.stat().st_size,
        "output": str(args.output.resolve()),
        "type": args.type,
        "threads": args.threads,
        "imatrix": str(args.imatrix.resolve()) if args.imatrix else None,
        "tensor_types": args.tensor_type,
        "output_tensor_type": args.output_tensor_type,
        "token_embedding_type": args.token_embedding_type,
        "allow_requantize": args.allow_requantize,
    }
    ledger = ExperimentLedger(args.ledger)
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    run_id = ledger.start("gguf-quantization", digest, configuration)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    try:
        if args.output.exists() or temporary.exists():
            raise FileExistsError("output or partial output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as log:
            subprocess.run(
                quantize_command(
                    args.binary,
                    args.input,
                    temporary,
                    args.type,
                    args.threads,
                    imatrix=args.imatrix,
                    tensor_types=args.tensor_type,
                    output_tensor_type=args.output_tensor_type,
                    token_embedding_type=args.token_embedding_type,
                    allow_requantize=args.allow_requantize,
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        reader = GGUFReader(temporary, "r")
        tensor_count = len(reader.tensors)
        if tensor_count == 0:
            raise ValueError("quantized GGUF contains no tensors")
        last = reader.tensors[-1]
        last_end = last.data_offset + last.data.nbytes
        if last_end > temporary.stat().st_size:
            raise ValueError("final tensor extends past end of quantized GGUF")
        del reader
        output_digest = sha256(temporary)
        os.replace(temporary, args.output)
        summary = {
            "output": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": output_digest,
            "tensor_count": tensor_count,
            "quantization": args.type,
        }
        ledger.artifact(
            run_id,
            "quantized-gguf",
            args.output,
            sha256=output_digest,
            bytes_count=args.output.stat().st_size,
        )
        ledger.finish(run_id, "passed", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        ledger.finish(
            run_id,
            "failed",
            {"partial_output": str(temporary.resolve()) if temporary.exists() else None},
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
