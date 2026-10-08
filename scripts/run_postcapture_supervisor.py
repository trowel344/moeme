#!/usr/bin/env python3
"""Deterministically advance capture -> preflight -> staging -> local smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from moeme.activations import inspect_activation_capture

try:
    from scripts.finalize_activation_capture import finalize_capture
except ModuleNotFoundError:
    from finalize_activation_capture import finalize_capture


class PostcaptureInterrupted(RuntimeError):
    def __init__(self, signum: int, stage: str):
        super().__init__(f"received signal {signum} during {stage}")
        self.signum = signum
        self.stage = stage


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def service_snapshot(name: str) -> dict:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            name,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "Result",
            "-p",
            "MainPID",
            "-p",
            "ExecMainStatus",
            "--no-pager",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    values = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    values["query_returncode"] = completed.returncode
    return values


def capture_is_running(snapshot: dict) -> bool:
    return (
        snapshot.get("ActiveState") in {"active", "activating"}
        and int(snapshot.get("MainPID", "0") or 0) > 0
    )


def recover_completed_capture(
    root: Path,
    *,
    output: Path | None = None,
    corpus: Path | None = None,
    layer: int = 63,
    expected_tokens: int = 204800,
    expected_width: int = 5120,
) -> dict | None:
    output = output or root / "cloud-inputs/layer63-200k-activations"
    corpus = corpus or root / "data/training/layer63-de-risk-200k.txt"
    if output.exists():
        return None
    candidates = []
    pattern = f".{output.name}.*"
    for directory in sorted(output.parent.glob(pattern)):
        payload = directory / f"layer-{layer}.f32"
        if not payload.is_file():
            continue
        try:
            info = inspect_activation_capture(payload)
        except (OSError, ValueError, EOFError):
            continue
        if info["tokens"] == expected_tokens and info["width"] == expected_width:
            candidates.append(directory)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(
            "multiple complete temporary captures exist; refusing ambiguous recovery: "
            + ", ".join(map(str, candidates))
        )
    manifest = finalize_capture(
        candidates[0],
        output,
        corpus,
        [layer],
        expected_tokens,
        0,
        True,
    )
    return {
        "partial_directory": str(candidates[0]),
        "output_directory": str(output),
        "manifest": manifest,
    }


def imatrix_receipt_command(root: Path, python: str) -> list[str]:
    command = [
        python,
        "scripts/receipt_imatrix.py",
        "--imatrix",
        ".moeme/layer63-200k-imatrix.gguf",
        "--receipt",
        ".moeme/layer63-200k-imatrix.receipt.json",
    ]
    if (root / ".moeme/layer63-200k-imatrix.receipt.json").is_file():
        command.append("--verify")
    return command


def validate_capture_ready(root: Path) -> dict:
    directory = root / "cloud-inputs/layer63-200k-activations"
    manifest_path = directory / "manifest.json"
    payload = directory / "layer-63.f32"
    manifest = json.loads(manifest_path.read_text())
    receipt = (manifest.get("layers") or {}).get("63")
    if not isinstance(receipt, dict):
        raise TypeError("capture manifest has no layer-63 receipt")
    expected = {
        "tokens": 204800,
        "width": 5120,
        "bytes": 4194305612,
    }
    observed = {
        "tokens": receipt.get("tokens"),
        "width": receipt.get("width"),
        "bytes": payload.stat().st_size,
    }
    if manifest.get("tokens_per_layer") != expected["tokens"] or observed != expected:
        raise ValueError(f"capture shape/size contract mismatch: {observed}")
    if receipt.get("bytes") != observed["bytes"]:
        raise ValueError("capture payload size does not match its manifest receipt")
    digest = sha256(payload)
    if digest != receipt.get("sha256"):
        raise ValueError("capture SHA-256 does not match its manifest")
    return {
        "manifest": str(manifest_path.resolve()),
        "payload": str(payload.resolve()),
        **observed,
        "sha256": digest,
    }


def gpu_compute_pids() -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi failed while waiting for the capture GPU to release")
    result = []
    for line in completed.stdout.splitlines():
        try:
            result.append(int(line.strip()))
        except ValueError:
            continue
    return result


def run_step(
    root: Path,
    state: dict,
    receipt_path: Path,
    name: str,
    command: list[str],
    *,
    required_outputs: tuple[Path, ...] = (),
    revalidate_passed: bool = False,
) -> None:
    previous = state.get("stages", {}).get(name) or {}
    resolved_outputs = tuple(
        path if path.is_absolute() else root / path for path in required_outputs
    )
    if (
        previous.get("status") == "passed"
        and not revalidate_passed
        and all(path.is_file() for path in resolved_outputs)
    ):
        return
    log = root / f".moeme/postcapture-logs/{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    state["status"] = "running"
    state["current_stage"] = name
    state.setdefault("stages", {})[name] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "command": command,
        "log": str(log),
    }
    atomic_json(receipt_path, state)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{root}:{root / 'src'}"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        received = {"signum": None}
        previous_handlers = {
            signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
        }

        def forward(signum, _frame) -> None:
            received["signum"] = signum
            if process.poll() is None:
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    pass

        for signum in previous_handlers:
            signal.signal(signum, forward)
        try:
            returncode = process.wait()
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    stage = state["stages"][name]
    stage["finished_at"] = datetime.now(UTC).isoformat()
    stage["returncode"] = returncode
    interrupted = received["signum"] is not None or returncode in {
        128 + signal.SIGINT,
        128 + signal.SIGTERM,
    }
    stage["status"] = "interrupted" if interrupted else "passed" if returncode == 0 else "failed"
    atomic_json(receipt_path, state)
    if interrupted:
        signum = int(received["signum"] or returncode - 128)
        raise PostcaptureInterrupted(signum, name)
    if returncode != 0:
        raise RuntimeError(f"post-capture stage failed: {name} (exit {returncode})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-service", default="moeme-layer63-capture-200k-v3.service")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-idle-timeout", type=float, default=900.0)
    parser.add_argument("--receipt", type=Path, default=Path(".moeme/layer63-postcapture-v1.json"))
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.gpu_idle_timeout <= 0:
        parser.error("poll and timeout values must be positive")
    root = args.root.resolve()
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    state = {
        "format": "moeme-postcapture-supervisor-v1",
        "status": "waiting",
        "current_stage": "capture",
        "capture_service": args.capture_service,
        "started_at": datetime.now(UTC).isoformat(),
        "stages": {},
    }
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        if previous.get("format") == state["format"]:
            state = previous
            state["status"] = "waiting"
            state["current_stage"] = "capture"
            state.pop("error", None)
    atomic_json(receipt_path, state)
    try:
        while True:
            snapshot = service_snapshot(args.capture_service)
            if capture_is_running(snapshot):
                time.sleep(args.poll_seconds)
                continue
            if (root / "cloud-inputs/layer63-200k-activations/manifest.json").is_file():
                break
            recovery = recover_completed_capture(root)
            if recovery is not None:
                state["capture_recovery"] = recovery
                atomic_json(receipt_path, state)
                break
            raise RuntimeError(f"capture terminated without a published manifest: {snapshot}")
        state["capture"] = validate_capture_ready(root)
        state["stages"]["capture"] = {
            "status": "passed",
            "finished_at": datetime.now(UTC).isoformat(),
            **state["capture"],
        }
        atomic_json(receipt_path, state)
        deadline = time.monotonic() + args.gpu_idle_timeout
        while gpu_compute_pids():
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU did not become idle after capture completion")
            time.sleep(args.poll_seconds)
        python = sys.executable
        run_step(
            root,
            state,
            receipt_path,
            "imatrix",
            imatrix_receipt_command(root, python),
            required_outputs=(
                Path(".moeme/layer63-200k-imatrix.gguf"),
                Path(".moeme/layer63-200k-imatrix.receipt.json"),
            ),
            revalidate_passed=True,
        )
        run_step(
            root,
            state,
            receipt_path,
            "production_preflight",
            [
                python,
                "scripts/run_free_cloud_training.py",
                "--config",
                "configs/free-cloud-layer63-200k.json",
                "--preflight-only",
                "--preflight-receipt",
                ".moeme/preflight/layer63-200k.json",
            ],
            required_outputs=(Path(".moeme/preflight/layer63-200k.json"),),
            revalidate_passed=True,
        )
        run_step(
            root,
            state,
            receipt_path,
            "cloud_stage",
            [python, "scripts/stage_free_cloud_job.py"],
            required_outputs=(Path("cloud-jobs/layer63-200k/upload-manifest.json"),),
            revalidate_passed=True,
        )
        run_step(
            root,
            state,
            receipt_path,
            "cloud_stage_verify",
            [
                python,
                "scripts/stage_free_cloud_job.py",
                "--verify-stage",
                "cloud-jobs/layer63-200k",
            ],
            required_outputs=(Path("cloud-jobs/layer63-200k/upload-manifest.json"),),
            revalidate_passed=True,
        )
        smoke_run = root / "cloud-results/local-seeded-smoke/cloud-run.json"
        if smoke_run.is_file():
            existing_smoke = json.loads(smoke_run.read_text())
            if existing_smoke.get("status") not in {"passed", "interrupted", "running"}:
                raise RuntimeError(
                    "local smoke has a terminal non-passing attempt; refusing auto-retry"
                )
            if existing_smoke.get("status") == "passed":
                state["stages"]["local_smoke"] = {
                    "status": "passed",
                    "existing_attempt": True,
                    "finished_at": existing_smoke.get("finished_at"),
                }
                atomic_json(receipt_path, state)
        if not smoke_run.is_file() or existing_smoke.get("status") != "passed":
            run_step(
                root,
                state,
                receipt_path,
                "local_smoke",
                [
                    python,
                    "scripts/run_free_cloud_training.py",
                    "--config",
                    "configs/local-layer63-seeded-smoke.json",
                ],
                required_outputs=(Path("cloud-results/local-seeded-smoke/cloud-run.json"),),
            )
        run_step(
            root,
            state,
            receipt_path,
            "local_smoke_verify",
            [
                python,
                "scripts/verify_cloud_result.py",
                "--directory",
                "cloud-results/local-seeded-smoke",
                "--receipt",
                ".moeme/layer63-seeded-smoke-verification.json",
            ],
            required_outputs=(Path(".moeme/layer63-seeded-smoke-verification.json"),),
            revalidate_passed=True,
        )
        run_step(
            root,
            state,
            receipt_path,
            "free_tier_budget",
            [python, "scripts/analyze_free_tier_budget.py"],
            required_outputs=(Path(".moeme/free-tier-budget.json"),),
            revalidate_passed=True,
        )
        state["status"] = "passed"
        state["current_stage"] = "complete"
        state["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(receipt_path, state)
        return 0
    except PostcaptureInterrupted as error:
        state["status"] = "interrupted"
        state["finished_at"] = datetime.now(UTC).isoformat()
        state["error"] = str(error)
        atomic_json(receipt_path, state)
        return 128 + error.signum
    except BaseException as error:
        state["status"] = "failed"
        state["finished_at"] = datetime.now(UTC).isoformat()
        state["error"] = f"{type(error).__name__}: {error}"
        atomic_json(receipt_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
