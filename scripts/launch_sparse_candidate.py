#!/usr/bin/env python3
"""Launch or resume a sparse-candidate acceptance pipeline from one config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.launch_free_cloud_job import run_forwarding_signals
except ModuleNotFoundError:
    from launch_free_cloud_job import run_forwarding_signals

REQUIRED = {
    "training_dir",
    "layer",
    "source_quantized",
    "candidate_quantized",
    "imatrix",
    "imatrix_receipt",
    "dense_reference",
    "llama_bin_dir",
    "manifest",
    "quantized_parity_report",
    "server_report",
    "corpora",
}
PATH_KEYS = {
    "training_dir",
    "source_quantized",
    "candidate_quantized",
    "imatrix",
    "imatrix_receipt",
    "dense_reference",
    "llama_bin_dir",
    "manifest",
    "quantized_parity_report",
    "server_report",
}


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


def load_configuration(path: Path, root: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("candidate configuration must be a JSON object")
    missing = sorted(REQUIRED - set(value))
    if missing:
        raise ValueError(f"candidate configuration is missing: {', '.join(missing)}")
    if not isinstance(value["corpora"], list) or not value["corpora"]:
        raise ValueError("candidate configuration needs at least one corpus")
    resolved = dict(value)
    for key in PATH_KEYS:
        item = Path(str(value[key]))
        resolved[key] = item.resolve() if item.is_absolute() else (root / item).resolve()
    resolved["corpora"] = [
        item.resolve() if item.is_absolute() else (root / item).resolve()
        for item in map(Path, value["corpora"])
    ]
    return resolved


def candidate_command(configuration: dict, root: Path) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts/run_sparse_candidate_pipeline.py"),
        "--training-dir",
        str(configuration["training_dir"]),
        "--layer",
        str(configuration["layer"]),
        "--source-quantized",
        str(configuration["source_quantized"]),
        "--candidate-quantized",
        str(configuration["candidate_quantized"]),
        "--imatrix",
        str(configuration["imatrix"]),
        "--imatrix-receipt",
        str(configuration["imatrix_receipt"]),
        "--dense-reference",
        str(configuration["dense_reference"]),
        "--llama-bin-dir",
        str(configuration["llama_bin_dir"]),
        "--manifest",
        str(configuration["manifest"]),
        "--quantized-parity-report",
        str(configuration["quantized_parity_report"]),
        "--server-report",
        str(configuration["server_report"]),
        "--port",
        str(configuration.get("port", 18082)),
        "--server-context",
        str(configuration.get("server_context", 8192)),
        "--gpu-layers",
        str(configuration.get("gpu_layers", 20)),
        "--max-median-ttft",
        str(configuration.get("max_median_ttft", 6.0)),
        "--max-ttft",
        str(configuration.get("max_ttft", 10.0)),
        "--max-long-context-ttft",
        str(configuration.get("max_long_context_ttft", 30.0)),
        "--min-median-decode-rate",
        str(configuration.get("min_median_decode_rate", 3.0)),
        "--min-long-decode-rate",
        str(configuration.get("min_long_decode_rate", 2.5)),
    ]
    for corpus in configuration["corpora"]:
        command += ["--corpus", str(corpus)]
    return command


def launch_status(returncode: int) -> str:
    if returncode == 0:
        return "passed"
    if returncode in {128 + signal.SIGINT, 128 + signal.SIGTERM}:
        return "interrupted"
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/layer63-candidate.json"))
    parser.add_argument(
        "--receipt", type=Path, default=Path(".moeme/layer63-candidate-launch.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (
        args.config.resolve() if args.config.is_absolute() else (root / args.config).resolve()
    )
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    configuration = load_configuration(config_path, root)
    command = candidate_command(configuration, root)
    configuration_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    state = {
        "format": "moeme-sparse-candidate-launch-v1",
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "configuration": str(config_path),
        "configuration_sha256": configuration_digest,
        "command": command,
    }
    atomic_json(receipt_path, state)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(root / "src"), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    log = receipt_path.with_suffix(".log")
    with log.open("a", encoding="utf-8") as handle:
        # Keep service stop/reboot signals connected to the resumable phase pipeline.
        returncode = run_forwarding_signals(
            command,
            cwd=root,
            environment=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    manifest = configuration["manifest"]
    state["returncode"] = returncode
    state["finished_at"] = datetime.now(UTC).isoformat()
    state["log"] = str(log)
    state["pipeline_manifest"] = json.loads(manifest.read_text()) if manifest.is_file() else None
    state["status"] = launch_status(returncode)
    atomic_json(receipt_path, state)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
