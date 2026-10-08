#!/usr/bin/env python3
"""Run and validate an atomic dense-FFN activation capture campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from moeme.activations import activation_capture_statistics
from moeme.ledger import ExperimentLedger

try:
    from scripts.receipt_imatrix import receipt_imatrix
except ModuleNotFoundError:
    from receipt_imatrix import receipt_imatrix


class CaptureInterrupted(RuntimeError):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum} during activation capture")
        self.signum = signum


class ChildSignalForwarder:
    """Forward user/provider interrupts to one child and restore parent handlers."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.signum: int | None = None
        self.previous: dict[int, object] = {}

    def _forward(self, signum, _frame) -> None:
        self.signum = signum
        if self.process.poll() is None:
            try:
                self.process.send_signal(signum)
            except ProcessLookupError:
                pass

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, self._forward)
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_capture(directory: Path, layer_ids: list[int], expected_tokens: int) -> dict:
    results = {}
    for layer in layer_ids:
        path = directory / f"layer-{layer}.f32"
        info = activation_capture_statistics(path)
        if info["tokens"] != expected_tokens:
            raise ValueError(
                f"layer {layer} captured {info['tokens']} tokens; expected {expected_tokens}"
            )
        if not info["finite"]:
            raise ValueError(f"layer {layer} contains non-finite activations")
        results[str(layer)] = {
            "bytes": info["bytes"],
            "chunks": info["chunk_count"],
            "tokens": info["tokens"],
            "width": info["width"],
            "mean": info["mean"],
            "std": info["std"],
            "sha256": sha256(path),
        }
    return results


def wait_for_capture_start(
    process: subprocess.Popen,
    capture_paths: list[Path],
    timeout_seconds: float,
    poll_seconds: float = 1.0,
) -> None:
    """Fail early when a live imatrix run never enters the capture hook."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        if any(path.is_file() and path.stat().st_size > 12 for path in capture_paths):
            return
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"llama-imatrix exited with status {returncode} before writing activations"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            expected = ", ".join(str(path) for path in capture_paths)
            raise TimeoutError(
                f"capture hook wrote no activation rows within {timeout_seconds:g}s; "
                f"expected one of: {expected}"
            )
        time.sleep(min(poll_seconds, remaining))


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imatrix-output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument(
        "--from-chunk",
        type=int,
        default=0,
        help="Skip this many context chunks before capture (for resumable sharded campaigns).",
    )
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--gpu-layers", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--layer", type=int, help="Capture only one language layer.")
    parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="Set GGML_CUDA_DISABLE_GRAPHS=1 for long captures that exhaust graph memory.",
    )
    parser.add_argument(
        "--capture-start-timeout",
        type=float,
        default=180.0,
        help="Abort if the selected activation file has no rows after this many seconds.",
    )
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()
    if args.from_chunk < 0:
        parser.error("--from-chunk cannot be negative")
    if args.capture_start_timeout <= 0:
        parser.error("--capture-start-timeout must be positive")

    corpus_digest = sha256(args.corpus)
    expected_tokens = args.chunks * args.ctx_size
    configuration = {
        "binary": str(args.binary.resolve()),
        "model": str(args.model.resolve()),
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": corpus_digest,
        "output_dir": str(args.output_dir.resolve()),
        "chunks": args.chunks,
        "from_chunk": args.from_chunk,
        "ctx_size": args.ctx_size,
        "gpu_layers": args.gpu_layers,
        "expected_tokens_per_layer": expected_tokens,
        "layer": args.layer,
        "cuda_graphs_disabled": args.disable_cuda_graphs,
        "format": "moeme-f32-v1",
    }
    ledger = ExperimentLedger(args.ledger)
    run_id = ledger.start(
        "dense-activation-capture",
        hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
        configuration,
    )
    temporary: Path | None = None
    try:
        if args.output_dir.exists():
            raise FileExistsError(
                f"capture destination already exists; refusing append: {args.output_dir}"
            )
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
        )
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.imatrix_output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.binary),
            "-m",
            str(args.model),
            "-f",
            str(args.corpus),
            "-c",
            str(args.ctx_size),
            "--chunks",
            str(args.chunks),
            "--chunk",
            str(args.from_chunk),
            "-ngl",
            str(args.gpu_layers),
            "-t",
            str(args.threads),
            "--no-ppl",
            "-o",
            str(args.imatrix_output),
        ]
        environment = os.environ.copy()
        environment["MOEME_CAPTURE_DIR"] = str(temporary)
        if args.layer is not None:
            environment["MOEME_CAPTURE_LAYER"] = str(args.layer)
        if args.disable_cuda_graphs:
            environment["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        layer_ids = [args.layer] if args.layer is not None else list(range(args.layers))
        capture_paths = [temporary / f"layer-{layer}.f32" for layer in layer_ids]
        with args.log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            with ChildSignalForwarder(process) as forwarder:
                try:
                    wait_for_capture_start(
                        process,
                        capture_paths,
                        args.capture_start_timeout,
                    )
                    returncode = process.wait()
                except BaseException:
                    terminate_process(process)
                    if forwarder.signum is not None:
                        raise CaptureInterrupted(forwarder.signum) from None
                    raise
            if forwarder.signum is not None:
                raise CaptureInterrupted(forwarder.signum)
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, command)
        imatrix_receipt_path = args.imatrix_output.with_suffix(".receipt.json")
        imatrix_receipt = receipt_imatrix(args.imatrix_output, imatrix_receipt_path)
        layers = validate_capture(temporary, layer_ids, expected_tokens)
        manifest = {
            "format": "moeme-dense-activation-capture-v1",
            "corpus_sha256": corpus_digest,
            "layer_count": len(layer_ids),
            "tokens_per_layer": expected_tokens,
            "from_chunk": args.from_chunk,
            "cuda_graphs_disabled": args.disable_cuda_graphs,
            "layers": layers,
            "imatrix": imatrix_receipt,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output_dir)
        temporary = None
        summary = {
            "manifest": str((args.output_dir / "manifest.json").resolve()),
            "layer_count": len(layer_ids),
            "tokens_per_layer": expected_tokens,
            "width": next(iter(layers.values()))["width"],
            "total_bytes": sum(item["bytes"] for item in layers.values()),
            "imatrix_receipt": str(imatrix_receipt_path.resolve()),
        }
        ledger.artifact(run_id, "activation-manifest", args.output_dir / "manifest.json")
        ledger.finish(run_id, "passed", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except CaptureInterrupted as error:
        summary = {"partial_directory": str(temporary.resolve()) if temporary else None}
        ledger.finish(run_id, "interrupted", summary, error=str(error))
        return 128 + error.signum
    except BaseException as error:
        summary = {"partial_directory": str(temporary.resolve()) if temporary else None}
        ledger.finish(run_id, "failed", summary, error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
