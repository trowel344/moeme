#!/usr/bin/env python3
"""Serve the final MoEMe artifact only after every release gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_top4_layers(value: str) -> tuple[int, ...]:
    """Parse an explicit Top-4 layer list; empty means leave routing native."""
    try:
        layers = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-4 layers must be comma-separated integers") from error
    if not layers or len(set(layers)) != len(layers) or not set(layers).issubset(range(64)):
        raise argparse.ArgumentTypeError("top-4 layers must be unique integers in [0, 63]")
    return layers


def release_environment(
    base: dict[str, str], top4_layers: tuple[int, ...] | None
) -> dict[str, str]:
    """Return the server environment, forcing Top-4 routing only when asked.

    The recommended Top-12 artifact already uses every expert, so exporting
    ``MOEME_TOP4_LAYERS`` unconditionally silently served a different model than
    the one the receipts qualified.
    """
    environment = dict(base)
    if top4_layers:
        environment["MOEME_TOP4_LAYERS"] = ",".join(map(str, top4_layers))
    else:
        environment.pop("MOEME_TOP4_LAYERS", None)
    return environment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required release receipt is missing: {path}")
    return json.loads(path.read_text())


def validate_release(
    model: Path,
    pipeline_manifest: Path,
    server_report: Path,
    parity_report: Path,
    verify_hash: bool = True,
) -> dict[str, object]:
    if not model.exists():
        raise FileNotFoundError(f"model artifact is missing: {model}")
    pipeline = read_json(pipeline_manifest)
    quantization = pipeline.get("phases", {}).get("quantization", {})
    evaluation = pipeline.get("phases", {}).get("evaluation", {})
    if quantization.get("status") != "passed":
        raise ValueError("pipeline quantization phase has not passed")
    if evaluation.get("status") != "passed":
        raise ValueError("pipeline runtime evaluation phase has not passed")
    server = read_json(server_report)
    if not (
        server.get("passed")
        and server.get("quality_passed")
        and server.get("stability_passed")
        and server.get("performance_passed")
    ):
        raise ValueError("capability, stability, or performance report has not passed")
    parity = read_json(parity_report)
    if not parity.get("passed"):
        raise ValueError("multi-domain source-logit parity report has not passed")
    expected_hash = quantization.get("sha256")
    if not isinstance(expected_hash, str):
        raise TypeError("quantization receipt does not contain an artifact SHA-256")
    if parity.get("candidate_sha256") != expected_hash:
        raise ValueError("source-logit parity receipt is not bound to the quantized artifact")
    actual_hash = sha256(model) if verify_hash else None
    if verify_hash and actual_hash != expected_hash:
        raise ValueError(f"model SHA-256 mismatch: {actual_hash} != {expected_hash}")
    return {
        "model": str(model.resolve()),
        "bytes": model.stat().st_size,
        "sha256": expected_hash,
        "server_suite_digest": server.get("suite_digest"),
        "parity_corpora": len(parity.get("results", [])),
    }


def server_command(
    binary: Path,
    model: Path,
    host: str,
    port: int,
    context: int,
    threads: int,
    cache_ram_mib: int,
    *,
    gpu_layers: int = 20,
    flash_attention: bool = True,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    context_checkpoints: int = 2,
) -> list[str]:
    """Build the resident server command for the 16 GiB target host.

    ``--fit off`` is required: the automatic fitter aborts or over-commits.
    ``-ngl 20`` is the largest offload the 19.9 GB artifact leaves room for.

    Batch and ubatch are left at llama.cpp's ``-b 2048 -ub 512`` defaults. They
    were lowered (to ``-b 256 -ub 64``) in an earlier revision on the strength
    of a short-context decode measurement; that was wrong and has been reverted.
    A controlled single-variable comparison on the long-context capability case
    (3905-token prompt) shows:

    * ``-ub 64`` aborts the server with ``CUBLAS_STATUS_INVALID_VALUE`` in
      ``ggml_cuda_op_mul_mat_cublas``; reproduced twice, and it does not depend
      on ``-b`` (``-b 256 -ub 512`` passes the same prompt).
    * ``-ub 64`` also cost ~60% of prefill throughput (54 t/s) against
      ``-ub 512`` (130 t/s) and ``-b 2048 -ub 512`` (216 t/s).

    See ``.moeme/serving-ubatch-abort.json`` for the receipt.

    Context checkpoints are bounded instead of left at the default 32. Each
    checkpoint holds ~150 MiB of RAM, so the default reserves up to ~4.8 GB of
    the 15.6 GiB host for a 19.9 GB artifact that is already partly disk-
    resident. With the default, a served 8K session decayed across consecutive
    requests (1.759 -> 0.721 t/s); bounded checkpoints keep it stable
    (2.289 -> 2.549 t/s on the same artifact) while still allowing the most
    recent turns to reuse their prompt state.
    """
    command = [
        str(binary),
        "-m",
        str(model),
        "-c",
        str(context),
        "-t",
        str(threads),
        "-tb",
        str(threads),
        "--fit",
        "off",
        "-ngl",
        str(gpu_layers),
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "--ctx-checkpoints",
        str(context_checkpoints),
        "--host",
        host,
        "--port",
        str(port),
        "--parallel",
        "1",
        "--cache-ram",
        str(cache_ram_mib),
    ]
    if flash_attention:
        command += ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/gguf/moeme-27b-top12-imatrix-q5_k_m.gguf"),
    )
    parser.add_argument(
        "--pipeline-manifest",
        type=Path,
        default=Path(".moeme/top12-release-manifest.json"),
    )
    parser.add_argument(
        "--server-report",
        type=Path,
        default=Path(".moeme/server-eval-top12-q5.json"),
    )
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=Path(".moeme/logit-parity-top12-imatrix-q5-chunks8.json"),
    )
    parser.add_argument(
        "--top4-layers",
        type=parse_top4_layers,
        default=None,
        help=(
            "comma-separated layers whose graph must route Top-4; omit to serve "
            "the artifact at its native expert count (required for Top-12)"
        ),
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=20,
        help="GPU offload layers; 20 is the largest that keeps an 8K context on this host",
    )
    parser.add_argument(
        "--cache-ram-mib",
        type=int,
        default=0,
        help="prompt-cache memory in MiB; disabled by default for this 16 GiB host",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="logical batch; llama.cpp default 2048 was fastest measured",
    )
    parser.add_argument(
        "--ubatch-size",
        type=int,
        default=512,
        help=(
            "physical batch; keep at the llama.cpp default 512. Values at or below 64 "
            "abort this model with a cuBLAS error on long prompts"
        ),
    )
    parser.add_argument(
        "--ctx-checkpoints",
        type=int,
        default=2,
        help=(
            "context checkpoints per slot; the default of 32 reserves up to ~4.8 GB "
            "of this host's RAM and collapses served throughput"
        ),
    )
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    release = validate_release(
        args.model,
        args.pipeline_manifest,
        args.server_report,
        args.parity_report,
        verify_hash=not args.skip_hash,
    )
    print(json.dumps(release, indent=2, sort_keys=True), flush=True)
    environment = release_environment(os.environ.copy(), args.top4_layers)
    command = server_command(
        args.binary,
        args.model,
        args.host,
        args.port,
        args.context,
        args.threads,
        args.cache_ram_mib,
        gpu_layers=args.gpu_layers,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        context_checkpoints=args.ctx_checkpoints,
    )
    os.execvpe(command[0], command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
