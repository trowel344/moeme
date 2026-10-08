#!/usr/bin/env python3
"""One-command, fail-closed bootstrap for a staged free-cloud MoEMe job."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

MINIMUM_VERSIONS = {"numpy": (1, 26), "safetensors": (0, 6)}


class BootstrapInterrupted(Exception):
    def __init__(self, signum: int, step: str):
        super().__init__(f"received signal {signum} during {step}")
        self.signum = signum
        self.step = step


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


def numeric_version(value: str) -> tuple[int, ...]:
    parts = []
    for component in value.split("."):
        match = re.match(r"\d+", component)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def missing_requirements() -> list[str]:
    missing = []
    for package, minimum in MINIMUM_VERSIONS.items():
        try:
            installed = numeric_version(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            installed = ()
        if installed < minimum:
            missing.append(f"{package}>={'.'.join(map(str, minimum))}")
    return missing


def detect_provider(requested: str) -> str:
    if requested != "auto":
        return requested
    if Path("/kaggle/working").is_dir() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if any(name.startswith("LIGHTNING_") for name in os.environ):
        return "lightning"
    return "local"


def resolved_workdir(stage: Path, provider: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    if provider == "kaggle":
        return Path("/kaggle/working/moeme").resolve()
    return stage.resolve()


def run_step(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: Path,
) -> dict:
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now(UTC).isoformat()}] {json.dumps(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        interrupted = {"signum": None}

        def forward(signum, _frame) -> None:
            interrupted["signum"] = signum
            if process.poll() is None:
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    pass

        previous = {
            signum: signal.signal(signum, forward) for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            returncode = process.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return {
        "name": name,
        "command": command,
        "returncode": returncode,
        "signal": interrupted["signum"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log": str(log),
    }


def launcher_command(
    stage: Path,
    config: Path,
    provider: str,
    workdir: Path,
    resume_from: Path | None,
    resume_terminal: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(stage / "scripts/launch_free_cloud_job.py"),
        "--stage",
        str(stage),
        "--config",
        str(config),
        "--provider",
        provider,
        "--workdir",
        str(workdir),
    ]
    if resume_from is not None:
        command += ["--resume-from", str(resume_from.resolve())]
    if resume_terminal:
        command.append("--resume-terminal")
    return command


def staged_configuration(stage: Path, requested: Path | None) -> Path:
    if requested is not None:
        path = requested.resolve() if requested.is_absolute() else (stage / requested).resolve()
    else:
        manifest = json.loads((stage / "upload-manifest.json").read_text())
        path = (stage / str(manifest["configuration"])).resolve()
    try:
        path.relative_to(stage)
    except ValueError as error:
        raise ValueError(f"cloud configuration escapes the staged job: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"staged cloud configuration is missing: {path}")
    return path


def configured_output(stage: Path, workdir: Path, config: Path) -> Path:
    value = json.loads(config.read_text())
    base = config.parent
    raw = Path(str(value["output_dir"]))
    source_output = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    try:
        relative = source_output.relative_to(stage)
    except ValueError as error:
        raise ValueError(f"cloud output escapes the staged job: {source_output}") from error
    return workdir / relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--provider", choices=("auto", "kaggle", "lightning", "local"), default="auto"
    )
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--resume-terminal",
        action="store_true",
        help="explicitly retry an existing failed or early-stopped scientific run",
    )
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    stage = args.stage.resolve()
    config = staged_configuration(stage, args.config)
    provider = detect_provider(args.provider)
    workdir = resolved_workdir(stage, provider, args.workdir)
    runtime = workdir / ".moeme-runtime"
    receipt_path = runtime / "bootstrap.json"
    log = runtime / "bootstrap.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(stage), str(stage / "src"), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    receipt = {
        "format": "moeme-free-cloud-bootstrap-v1",
        "started_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "stage": str(stage),
        "configuration": str(config),
        "workdir": str(workdir),
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "resume_terminal": args.resume_terminal,
        "steps": [],
        "status": "running",
    }
    atomic_json(receipt_path, receipt)

    try:
        requirements = [] if args.skip_install else missing_requirements()
        receipt["missing_requirements"] = requirements
        if requirements:
            step = run_step(
                "dependencies",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade-strategy",
                    "only-if-needed",
                    *requirements,
                ],
                cwd=stage,
                environment=environment,
                log=log,
            )
            receipt["steps"].append(step)
            atomic_json(receipt_path, receipt)
            if step["signal"] is not None:
                raise BootstrapInterrupted(step["signal"], step["name"])
            if step["returncode"] != 0:
                raise RuntimeError("dependency installation failed")

        verify = run_step(
            "stage-verification",
            [
                sys.executable,
                str(stage / "scripts/stage_free_cloud_job.py"),
                "--verify-stage",
                str(stage),
            ],
            cwd=stage,
            environment=environment,
            log=log,
        )
        receipt["steps"].append(verify)
        atomic_json(receipt_path, receipt)
        if verify["signal"] is not None:
            raise BootstrapInterrupted(verify["signal"], verify["name"])
        if verify["returncode"] != 0:
            raise RuntimeError("staged input verification failed")

        training = run_step(
            "training",
            launcher_command(
                stage,
                config,
                provider,
                workdir,
                args.resume_from,
                args.resume_terminal,
            ),
            cwd=stage,
            environment=environment,
            log=log,
        )
        receipt["steps"].append(training)
        output = configured_output(stage, workdir, config)
        cloud_run = output / "cloud-run.json"
        result_manifest = output / "cloud-result-manifest.json"
        receipt["cloud_run"] = json.loads(cloud_run.read_text()) if cloud_run.is_file() else None
        receipt["result_manifest"] = (
            json.loads(result_manifest.read_text()) if result_manifest.is_file() else None
        )
        if training["signal"] is not None:
            raise BootstrapInterrupted(training["signal"], training["name"])
        if training["returncode"] != 0:
            raise RuntimeError(f"training launcher exited {training['returncode']}")
        receipt["status"] = (
            "session_complete"
            if isinstance(receipt["cloud_run"], dict)
            and receipt["cloud_run"].get("stage") == "session-budget"
            else "passed"
        )
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(receipt_path, receipt)
        return 0
    except BootstrapInterrupted as error:
        receipt["status"] = "interrupted"
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipt["signal"] = error.signum
        receipt["interrupted_step"] = error.step
        atomic_json(receipt_path, receipt)
        return 128 + error.signum
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipt["error"] = f"{type(error).__name__}: {error}"
        atomic_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
