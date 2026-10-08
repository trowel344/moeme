#!/usr/bin/env python3
"""Adapt the portable free-cloud job to writable Lightning or Kaggle storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


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


def detect_provider(requested: str) -> str:
    if requested != "auto":
        return requested
    if Path("/kaggle/working").is_dir() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if any(name.startswith("LIGHTNING_") for name in os.environ):
        return "lightning"
    return "local"


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def runtime_configuration(stage: Path, workdir: Path, base_config: Path) -> tuple[dict, Path]:
    stage = stage.resolve()
    workdir = workdir.resolve()
    base = load_object(base_config)
    base_directory = base_config.resolve().parent
    for key in ("activations", "checkpoint", "seed_checkpoint"):
        value = Path(str(base[key]))
        source = value.resolve() if value.is_absolute() else (base_directory / value).resolve()
        if not source.exists():
            raise FileNotFoundError(f"staged cloud input is missing: {source}")
        try:
            source.relative_to(stage)
        except ValueError as error:
            raise ValueError(f"cloud input escapes the staged job: {source}") from error
        base[key] = str(source)
    configured_output = resolved_config_path(base_directory, str(base["output_dir"]))
    try:
        relative_output = configured_output.relative_to(stage)
    except ValueError as error:
        raise ValueError(f"cloud output escapes the staged job: {configured_output}") from error
    output = workdir / relative_output
    base["output_dir"] = str(output)
    runtime_config = workdir / ".moeme-runtime" / base_config.name
    return base, runtime_config


def resolved_config_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def restore_resume(source: Path, output: Path) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"resume directory is missing: {source}")
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"refusing to overlay non-empty output directory: {output}")
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.resume.", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        shutil.copytree(source, staged)
        os.replace(staged, output)


def run_forwarding_signals(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout=None,
    stderr=None,
) -> int:
    """Keep provider eviction and user interrupts connected to the trainer."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=stdout,
        stderr=stderr,
    )
    previous = {}

    def forward(signum, _frame) -> None:
        if process.poll() is None:
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, forward)
    try:
        return process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/free-cloud-layer63-200k.json"))
    parser.add_argument(
        "--provider", choices=("auto", "kaggle", "lightning", "local"), default="auto"
    )
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-terminal", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    stage = args.stage.resolve()
    provider = detect_provider(args.provider)
    if args.workdir is not None:
        workdir = args.workdir.resolve()
    elif provider == "kaggle":
        workdir = Path("/kaggle/working/moeme").resolve()
    else:
        workdir = stage
    base_config = (
        args.config.resolve() if args.config.is_absolute() else (stage / args.config).resolve()
    )
    try:
        base_config.relative_to(stage)
    except ValueError as error:
        raise ValueError(f"cloud configuration escapes the staged job: {base_config}") from error
    config, runtime_config = runtime_configuration(stage, workdir, base_config)
    output = Path(config["output_dir"])
    if args.resume_from is not None:
        restore_resume(args.resume_from, output)
    atomic_json(runtime_config, config)
    command = [
        sys.executable,
        str(stage / "scripts/run_free_cloud_training.py"),
        "--config",
        str(runtime_config),
    ]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.resume_terminal:
        command.append("--resume-terminal")
    environment = os.environ.copy()
    python_paths = (str(stage), str(stage / "src"))
    environment["PYTHONPATH"] = os.pathsep.join(
        (*python_paths, environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    launch_receipt = {
        "format": "moeme-free-cloud-launch-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "stage": str(stage),
        "workdir": str(workdir),
        "runtime_config": str(runtime_config),
        "base_config": str(base_config),
        "output_dir": str(output),
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "preflight_only": args.preflight_only,
        "resume_terminal": args.resume_terminal,
        "command": command,
    }
    atomic_json(workdir / ".moeme-runtime/launch.json", launch_receipt)
    return run_forwarding_signals(command, cwd=stage, environment=environment)


if __name__ == "__main__":
    raise SystemExit(main())
