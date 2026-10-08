#!/usr/bin/env python3
"""Resume the complete train -> assemble -> quantize -> resident-eval campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
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


def begin_phase(path: Path, state: dict, phase: str, details: dict | None = None) -> None:
    """Persist phase ownership before starting a long or destructive operation."""
    now = datetime.now(UTC).isoformat()
    previous = state["phases"].get(phase, {})
    record = {
        "status": "running",
        "started_at": previous.get("started_at", now),
        "attempt": int(previous.get("attempt", 0)) + 1,
        **(details or {}),
    }
    if previous:
        record["resumed_at"] = now
    state["phases"][phase] = record
    state["current_phase"] = phase
    state["updated_at"] = now
    state.pop("last_error", None)
    atomic_json(path, state)


def pass_phase(path: Path, state: dict, phase: str, details: dict) -> None:
    now = datetime.now(UTC).isoformat()
    previous = state["phases"].get(phase, {})
    state["phases"][phase] = {
        **previous,
        "status": "passed",
        "finished_at": now,
        **details,
    }
    state["phases"][phase].pop("error", None)
    state["phases"][phase].pop("failed_at", None)
    if state.get("current_phase") == phase:
        state.pop("current_phase", None)
    state["updated_at"] = now
    atomic_json(path, state)


def fail_phase(path: Path, state: dict, error: BaseException) -> None:
    now = datetime.now(UTC).isoformat()
    message = f"{type(error).__name__}: {error}"
    phase = state.get("current_phase")
    if phase is not None:
        state["phases"][phase] = {
            **state["phases"].get(phase, {}),
            "status": "failed",
            "failed_at": now,
            "error": message,
        }
    state["last_error"] = message
    state["updated_at"] = now
    atomic_json(path, state)


def parse_layer_checkpoint(value: str) -> tuple[int, Path]:
    try:
        layer, checkpoint = value.split("=", 1)
        return int(layer), Path(checkpoint)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected LAYER=PATH") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(UTC).isoformat()}] {json.dumps(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command exited {completed.returncode}; see {log_path}")


def require_space(path: Path, minimum: int, phase: str) -> None:
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise OSError(f"{phase} requires at least {minimum:,} free bytes; only {free:,} available")


def wait_healthy(url: str, process: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("server did not become healthy within 180 seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--partition-checkpoint", type=Path, required=True)
    parser.add_argument("--source-bf16", type=Path, required=True)
    parser.add_argument("--diagnostic-q4", type=Path, required=True)
    parser.add_argument("--diagnostic-q4-sha256", required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--working-bf16", type=Path, required=True)
    parser.add_argument("--output-q4", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--llama-bin-dir", type=Path, required=True)
    parser.add_argument(
        "--seed-checkpoint", action="append", type=parse_layer_checkpoint, default=[]
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server-context", type=int, default=8192)
    args = parser.parse_args()

    seeds = dict(args.seed_checkpoint)
    if len(seeds) != len(args.seed_checkpoint):
        parser.error("seed layers must be unique")
    if not set(seeds).issubset(range(64)):
        parser.error("seed layers must be in [0, 63]")
    missing_layers = [layer for layer in range(64) if layer not in seeds]
    configuration = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "seed_checkpoint"
    }
    configuration["seed_checkpoints"] = {
        str(layer): str(path.resolve()) for layer, path in sorted(seeds.items())
    }
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if args.manifest.exists():
        state = json.loads(args.manifest.read_text())
        if state["configuration_digest"] != digest:
            raise ValueError("pipeline manifest configuration does not match this invocation")
    else:
        state = {
            "format": "moeme-all64-pipeline-v1",
            "configuration": configuration,
            "configuration_digest": digest,
            "phases": {},
            "started_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(args.manifest, state)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"

    def completed(phase: str) -> bool:
        return state["phases"].get(phase, {}).get("status") == "passed"

    try:
        if not completed("training"):
            begin_phase(args.manifest, state, "training", {"layers": len(missing_layers)})
            require_space(
                args.manifest.parent, len(missing_layers) * 540_000_000 + 5_000_000_000, "training"
            )
            command = [
                sys.executable,
                "scripts/run_training_campaign.py",
                "--activations",
                str(args.activations),
                "--checkpoint",
                str(args.partition_checkpoint),
                "--layers",
                *map(str, missing_layers),
                "--output-dir",
                str(args.training_dir),
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
            ]
            run_logged(command, args.training_dir / "pipeline-training.log", environment)
            campaign = json.loads((args.training_dir / "campaign.json").read_text())
            if set(map(int, campaign["layers"])) != set(missing_layers):
                raise ValueError("training campaign did not complete every requested layer")
            pass_phase(
                args.manifest,
                state,
                "training",
                {
                    "layers": len(missing_layers),
                    "campaign": str((args.training_dir / "campaign.json").resolve()),
                },
            )

        campaign = json.loads((args.training_dir / "campaign.json").read_text())
        checkpoints = {layer: path for layer, path in seeds.items()}
        checkpoints.update(
            {int(layer): Path(value["checkpoint"]) for layer, value in campaign["layers"].items()}
        )
        if set(checkpoints) != set(range(64)):
            raise ValueError("exactly one checkpoint is required for every layer")

        if not completed("injection"):
            begin_phase(args.manifest, state, "injection", {"layers": len(checkpoints)})
            if args.diagnostic_q4.exists():
                actual = sha256(args.diagnostic_q4)
                if actual != args.diagnostic_q4_sha256:
                    raise ValueError("refusing to delete diagnostic GGUF with unexpected checksum")
                args.diagnostic_q4.unlink()
            require_space(
                args.manifest.parent,
                args.source_bf16.stat().st_size + 5_000_000_000,
                "BF16 assembly",
            )
            partial = args.working_bf16.with_name(f".{args.working_bf16.name}.partial")
            args.working_bf16.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            command = [
                sys.executable,
                "scripts/patch_gguf_layer.py",
                "--source",
                str(args.source_bf16),
                "--output",
                str(args.working_bf16),
            ]
            for layer, checkpoint in sorted(checkpoints.items()):
                command.extend(("--layer-checkpoint", f"{layer}={checkpoint}"))
            run_logged(command, args.manifest.with_suffix(".injection.log"), environment)
            pass_phase(
                args.manifest, state, "injection", {"bytes": args.working_bf16.stat().st_size}
            )

        if not completed("checkpoint_cleanup"):
            begin_phase(args.manifest, state, "checkpoint_cleanup")
            removed = 0
            for checkpoint in checkpoints.values():
                if checkpoint.exists():
                    removed += checkpoint.stat().st_size
                    checkpoint.unlink()
            pass_phase(
                args.manifest,
                state,
                "checkpoint_cleanup",
                {"bytes_removed": removed, "reconstructible": True},
            )

        if not completed("quantization"):
            begin_phase(args.manifest, state, "quantization", {"type": "Q4_K_M"})
            require_space(args.manifest.parent, 20_000_000_000, "Q4_K_M quantization")
            args.output_q4.unlink(missing_ok=True)
            partial = args.output_q4.with_name(f".{args.output_q4.name}.partial")
            partial.unlink(missing_ok=True)
            run_logged(
                [
                    sys.executable,
                    "scripts/quantize_gguf.py",
                    "--binary",
                    str(args.llama_bin_dir / "llama-quantize"),
                    "--input",
                    str(args.working_bf16),
                    "--output",
                    str(args.output_q4),
                    "--type",
                    "Q4_K_M",
                    "--threads",
                    "8",
                    "--log",
                    str(args.manifest.with_suffix(".quantize.log")),
                ],
                args.manifest.with_suffix(".quantize-wrapper.log"),
                environment,
            )
            pass_phase(
                args.manifest,
                state,
                "quantization",
                {"bytes": args.output_q4.stat().st_size, "sha256": sha256(args.output_q4)},
            )

        if not completed("bf16_cleanup"):
            begin_phase(args.manifest, state, "bf16_cleanup")
            removed = args.working_bf16.stat().st_size if args.working_bf16.exists() else 0
            args.working_bf16.unlink(missing_ok=True)
            pass_phase(
                args.manifest,
                state,
                "bf16_cleanup",
                {"bytes_removed": removed, "reconstructible": True},
            )

        if not completed("evaluation"):
            begin_phase(args.manifest, state, "evaluation", {"context": args.server_context})
            url = f"http://127.0.0.1:{args.port}"
            server_environment = environment.copy()
            server_environment["MOEME_TOP4_LAYERS"] = ",".join(map(str, range(64)))
            server_log_path = args.manifest.with_suffix(".server.log")
            with server_log_path.open("a", encoding="utf-8") as server_log:
                server = subprocess.Popen(
                    [
                        str(args.llama_bin_dir / "llama-server"),
                        "-m",
                        str(args.output_q4),
                        "-c",
                        str(args.server_context),
                        "-t",
                        "8",
                        "-tb",
                        "8",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.port),
                        "--parallel",
                        "1",
                        "--cache-ram",
                        "0",
                    ],
                    env=server_environment,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    wait_healthy(url, server)
                    run_logged(
                        [
                            sys.executable,
                            "scripts/server_eval.py",
                            "--url",
                            url,
                            "--report",
                            str(args.report),
                        ],
                        args.manifest.with_suffix(".evaluation.log"),
                        environment,
                    )
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
            report = json.loads(args.report.read_text())
            pass_phase(args.manifest, state, "evaluation", report)

        state["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(args.manifest, state)
        print(json.dumps(state["phases"]["evaluation"], indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        fail_phase(args.manifest, state, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
