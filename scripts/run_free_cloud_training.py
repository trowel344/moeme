#!/usr/bin/env python3
"""Preflight and run a resumable single-layer sparse-training experiment."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from moeme.activations import activation_capture_paths, inspect_activation_capture
from moeme.cloud_results import write_result_manifest


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def provider_name() -> str:
    if Path("/kaggle").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if any(name.startswith("LIGHTNING_") for name in os.environ):
        return "lightning"
    return "local-or-unknown"


def selected_compute_dtype(requested: str, bf16_supported: bool) -> str:
    if requested == "auto":
        return "bfloat16" if bf16_supported else "float16"
    if requested == "bfloat16" and not bf16_supported:
        raise RuntimeError("configuration requests bfloat16 but this GPU does not support it")
    if requested not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unknown compute dtype: {requested}")
    return requested


def gib(value: int) -> float:
    return value / (1024**3)


def system_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_receipt(path: Path, receipt: dict, label: str, verify_hash: bool) -> dict:
    actual_bytes = path.stat().st_size
    expected_bytes = receipt.get("bytes")
    if expected_bytes is not None and actual_bytes != int(expected_bytes):
        raise ValueError(f"{label} has {actual_bytes} bytes; receipt expects {expected_bytes}")
    expected_hash = receipt.get("sha256")
    actual_hash = sha256(path) if verify_hash and expected_hash else None
    if actual_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"{label} SHA-256 does not match its receipt")
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_hash or expected_hash,
        "hash_verified": actual_hash is not None,
    }


def existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def preflight_receipt_path(output: Path, override: Path | None, repository: Path) -> Path:
    if override is None:
        return output / "cloud-preflight.json"
    return override.resolve() if override.is_absolute() else (repository / override).resolve()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    required = {"activations", "checkpoint", "layer", "output_dir"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"cloud config is missing: {', '.join(missing)}")
    return config


def resolved_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def preflight(config: dict, config_path: Path) -> tuple[dict, dict[str, Path]]:
    base = config_path.resolve().parent
    paths = {
        key: resolved_path(config[key], base) for key in ("activations", "checkpoint", "output_dir")
    }
    if config.get("seed_checkpoint"):
        paths["seed_checkpoint"] = resolved_path(config["seed_checkpoint"], base)
    layer = int(config["layer"])
    checkpoint_receipt_path = paths["checkpoint"] / "receipt.json"
    required_files = [
        paths["checkpoint"] / "model.safetensors.index.json",
        paths["checkpoint"] / "moeme-manifest.json",
        checkpoint_receipt_path,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required cloud input(s): " + ", ".join(missing))

    verify_hashes = bool(config.get("verify_hashes", True))
    activation_files = activation_capture_paths(paths["activations"], layer)
    verified_activation_files = []
    activation_tokens = 0
    activation_bytes = 0
    activation_width = None
    for activation_file in activation_files:
        activation = inspect_activation_capture(activation_file)
        activation_manifest = json.loads((activation_file.parent / "manifest.json").read_text())
        activation_receipt = activation_manifest.get("layers", {}).get(str(layer))
        if activation_receipt is None:
            raise ValueError(f"activation manifest has no receipt for layer {layer}")
        if not isinstance(activation_receipt, dict):
            raise TypeError(f"activation receipt for layer {layer} is not an object")
        if activation["tokens"] != activation_manifest.get("tokens_per_layer"):
            raise ValueError("activation token count does not match its manifest")
        verified = verify_receipt(
            activation_file, activation_receipt, "activation capture", verify_hashes
        )
        verified_activation_files.append({**verified, "tokens": activation["tokens"]})
        activation_tokens += int(activation["tokens"])
        activation_bytes += int(activation["bytes"])
        if activation_width is None:
            activation_width = int(activation["width"])
        elif activation_width != int(activation["width"]):
            raise ValueError("activation shards have different widths")
    checkpoint_manifest = json.loads((paths["checkpoint"] / "moeme-manifest.json").read_text())
    checkpoint_receipt = json.loads(checkpoint_receipt_path.read_text())
    checkpoint_tensor = paths["checkpoint"] / checkpoint_receipt["tensor_file"]
    verified_checkpoint = verify_receipt(
        checkpoint_tensor, checkpoint_receipt, "portable checkpoint", verify_hashes
    )
    checkpoint_layer = checkpoint_manifest.get("layer")
    if checkpoint_layer is not None and int(checkpoint_layer) != layer:
        raise ValueError(
            f"portable checkpoint is for layer {checkpoint_layer}, requested layer {layer}"
        )
    verified_seed = None
    if "seed_checkpoint" in paths:
        seed_receipt_path = paths["seed_checkpoint"].parent / "receipt.json"
        if not seed_receipt_path.is_file():
            raise FileNotFoundError(f"training seed receipt is missing: {seed_receipt_path}")
        seed_receipt = json.loads(seed_receipt_path.read_text())
        if int(seed_receipt.get("layer", -1)) != layer:
            raise ValueError("training seed receipt is for a different layer")
        verified_seed = verify_receipt(
            paths["seed_checkpoint"], seed_receipt, "training seed", verify_hashes
        )
        verified_seed["partial_projection_seed"] = seed_receipt.get(
            "partial_projection_seed", False
        )

    usage = shutil.disk_usage(existing_ancestor(paths["output_dir"].parent))
    configured_minimum_free_gib = float(config.get("minimum_free_gib", 5.0))
    # Full-projection FP32 parameters plus Adam moments are roughly 6x the
    # portable BF16 payload. Atomic replacement temporarily keeps old and new
    # restart states, then the final BF16 checkpoint also needs room.
    estimated_atomic_output_gib = gib(verified_checkpoint["bytes"]) * 13 + 2.0
    minimum_free_gib = max(configured_minimum_free_gib, estimated_atomic_output_gib)
    if gib(usage.free) < minimum_free_gib:
        raise RuntimeError(
            f"only {gib(usage.free):.2f} GiB free; config requires {minimum_free_gib:.2f} GiB"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to spend a cloud session on CPU")
    system_gib = gib(system_memory_bytes())
    # Activations are indexed through bounded memory maps; only model, optimizer,
    # labels and active batches need resident memory.
    estimated_required_system_gib = min(gib(activation_bytes) * 0.15, 3.0) + 4.0
    if system_gib < estimated_required_system_gib:
        raise RuntimeError(
            f"host has {system_gib:.2f} GiB system RAM; this activation set needs an "
            f"estimated {estimated_required_system_gib:.2f} GiB"
        )

    bf16_supported = torch.cuda.is_bf16_supported()
    requested_dtype = str(config.get("compute_dtype", "auto"))
    selected_dtype = selected_compute_dtype(requested_dtype, bf16_supported)
    report = {
        "format": "moeme-free-cloud-preflight-v1",
        "checked_at": datetime.now(UTC).isoformat(),
        "provider": provider_name(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "safetensors")
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "bf16_supported": bf16_supported,
        "requested_compute_dtype": requested_dtype,
        "selected_compute_dtype": selected_dtype,
        "activation": {
            "files": verified_activation_files,
            "file_count": len(verified_activation_files),
            "bytes": activation_bytes,
            "tokens": activation_tokens,
            "width": activation_width,
            "gib": gib(activation_bytes),
            "memory_mapped": True,
        },
        "checkpoint": {
            **verified_checkpoint,
            "directory": str(paths["checkpoint"]),
            "format": checkpoint_manifest.get("format"),
            "layer": checkpoint_layer,
        },
        "seed_checkpoint": verified_seed,
        "storage": {
            "free_gib": gib(usage.free),
            "required_free_gib": minimum_free_gib,
            "configured_minimum_free_gib": configured_minimum_free_gib,
            "estimated_atomic_output_gib": estimated_atomic_output_gib,
        },
        "system_memory": {
            "total_gib": system_gib,
            "estimated_required_gib": estimated_required_system_gib,
        },
    }
    return report, paths


def trainer_command(config: dict, paths: dict[str, Path], repository: Path) -> list[str]:
    layer = int(config["layer"])
    output = paths["output_dir"]
    state_path = output / "training-state.pt"
    command = [
        sys.executable,
        str(repository / "scripts/train_sparse_layer.py"),
        "--activations",
        str(paths["activations"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--layer",
        str(layer),
        "--output-dir",
        str(output),
        "--ledger",
        str(output / "experiments.sqlite3"),
        "--state-path",
        str(state_path),
        "--device",
        "cuda",
        "--compute-dtype",
        str(config.get("compute_dtype", "auto")),
        "--partition-mode",
        str(config.get("partition_mode", "importance_contiguous")),
        "--groups",
        str(config.get("groups", 16)),
        "--shared-groups",
        str(config.get("shared_groups", 4)),
        "--steps-per-stage",
        str(config.get("steps_per_stage", 2000)),
        "--router-warmup-steps",
        str(config.get("router_warmup_steps", 300)),
        "--batch-size",
        str(config.get("batch_size", 4)),
        "--learning-rate",
        str(config.get("learning_rate", 3e-5)),
        "--feature-learning-rate",
        str(config.get("feature_learning_rate", 1e-5)),
        "--router-learning-rate",
        str(config.get("router_learning_rate", 1e-3)),
        "--routing-strategy",
        str(config.get("routing_strategy", "oracle")),
        "--train-projections",
        str(config.get("train_projections", "all")),
        "--checkpoint-every",
        str(config.get("checkpoint_every", 0)),
        "--state-every",
        str(config.get("state_every", 500)),
        "--progress-every",
        str(config.get("progress_every", 25)),
        "--evaluate-every",
        str(config.get("evaluate_every", 2500)),
        "--evaluation-tokens",
        str(config.get("evaluation_tokens", 2048)),
        "--label-chunk-tokens",
        str(config.get("label_chunk_tokens", 4096)),
        "--validation-fraction",
        str(config.get("validation_fraction", 0.25)),
        "--top-k-schedule",
        *[str(value) for value in config.get("top_k_schedule", [4])],
    ]
    if bool(config.get("train_shared", True)):
        command.append("--train-shared")
    if bool(config.get("systems_smoke", False)):
        command.append("--systems-smoke")
    if float(config.get("max_runtime_seconds", 0)) > 0:
        command.extend(("--max-runtime-seconds", str(config["max_runtime_seconds"])))
    if bool(config.get("early_stop_flat", False)):
        command.extend(
            (
                "--early-stop-flat",
                "--early-stop-min-step",
                str(config.get("early_stop_min_step", 15000)),
                "--early-stop-recent-points",
                str(config.get("early_stop_recent_points", 4)),
                "--early-stop-target-error",
                str(config.get("early_stop_target_error", 0.01)),
            )
        )
    seed_checkpoint = paths.get("seed_checkpoint")
    if seed_checkpoint is not None:
        command.extend(("--partition-indices-from", str(seed_checkpoint)))
    if state_path.exists():
        command.extend(("--resume-state", str(state_path)))
    elif seed_checkpoint is not None and bool(config.get("resume_seed", True)):
        command.extend(("--resume", str(seed_checkpoint), "--allow-partial-resume"))
    return command


def next_attempt(receipt_path: Path) -> int:
    try:
        previous = json.loads(receipt_path.read_text())
        return max(1, int(previous.get("attempt", 0)) + 1)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return 1


def existing_run_action(run: dict | None, allow_terminal_resume: bool) -> str:
    status = (run or {}).get("status")
    if status == "passed":
        return "complete"
    if status in {"failed", "early_stopped"} and not allow_terminal_resume:
        raise RuntimeError(
            f"existing cloud run is terminal ({status}); inspect it before an explicit retry"
        )
    return "run"


def acquire_run_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".training.lock"
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"another cloud trainer owns the run lock: {path}") from error
    return handle


def terminal_status(
    training_returncode: int,
    curve_analysis_returncode: int | None,
    report_passed: bool | None,
    *,
    require_report_passed: bool = True,
) -> tuple[str, str]:
    if training_returncode == 75:
        return "interrupted", "session-budget"
    if training_returncode in (128 + signal.SIGINT, 128 + signal.SIGTERM):
        return "interrupted", "training-interrupted"
    if training_returncode == 3:
        return "early_stopped", "curve-policy"
    if training_returncode != 0:
        return "failed", "training"
    if curve_analysis_returncode != 0:
        return "failed", "curve-analysis"
    if require_report_passed and report_passed is not True:
        return "failed", "training-report"
    return "passed", "complete" if report_passed is True else "systems-smoke-complete"


def run_and_tee(
    command: list[str], *, cwd: Path, environment: dict[str, str], log_path: Path
) -> tuple[int, int]:
    """Run a child visibly while retaining an append-only cloud-session log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        previous_handlers = {
            signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
        }

        def forward_termination(signum, _frame) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        for signum in previous_handlers:
            signal.signal(signum, forward_termination)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait(), process.pid
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--preflight-receipt",
        type=Path,
        help="alternate preflight receipt path; valid only with --preflight-only",
    )
    parser.add_argument(
        "--resume-terminal",
        action="store_true",
        help="explicitly resume a failed or scientifically early-stopped run",
    )
    args = parser.parse_args()
    if args.preflight_receipt is not None and not args.preflight_only:
        parser.error("--preflight-receipt requires --preflight-only")

    repository = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    config_base = args.config.resolve().parent
    configured_output = resolved_path(str(config["output_dir"]), config_base)
    existing_run_path = configured_output / "cloud-run.json"
    existing_run = (
        json.loads(existing_run_path.read_text()) if existing_run_path.is_file() else None
    )
    action = existing_run_action(existing_run, args.resume_terminal)
    if action == "complete" and not args.preflight_only:
        print(json.dumps({"status": "already_complete", "output": str(configured_output)}))
        return 0
    run_lock = None if args.preflight_only else acquire_run_lock(configured_output)
    report, paths = preflight(config, args.config)
    receipt_path = preflight_receipt_path(paths["output_dir"], args.preflight_receipt, repository)
    atomic_json(receipt_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return 0
    assert run_lock is not None

    command = trainer_command(config, paths, repository)
    environment = os.environ.copy()
    source_path = str(repository / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else f"{source_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    run_receipt_path = paths["output_dir"] / "cloud-run.json"
    attempt = next_attempt(run_receipt_path)
    started_at = datetime.now(UTC).isoformat()
    log_path = paths["output_dir"] / "cloud-training.log"
    running = {
        **report,
        "format": "moeme-free-cloud-run-v2",
        "status": "running",
        "stage": "training",
        "attempt": attempt,
        "started_at": started_at,
        "command": command,
        "log": str(log_path.resolve()),
        "resumed": (paths["output_dir"] / "training-state.pt").exists(),
        "resume_state": str((paths["output_dir"] / "training-state.pt").resolve()),
        "run_lock": str(Path(run_lock.name).resolve()),
    }
    atomic_json(run_receipt_path, running)
    try:
        training_returncode, child_pid = run_and_tee(
            command, cwd=repository, environment=environment, log_path=log_path
        )
    except BaseException as error:
        interrupted = {
            **running,
            "status": "interrupted",
            "finished_at": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_json(run_receipt_path, interrupted)
        raise
    curve_path = paths["output_dir"] / "curve.json"
    curve_analysis_path = paths["output_dir"] / "curve-analysis.json"
    curve_analysis_returncode = None
    if curve_path.exists():
        analyzed = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts/analyze_sparse_curve.py"),
                "--curve",
                str(curve_path),
                "--output",
                str(curve_analysis_path),
            ],
            cwd=repository,
            env=environment,
            check=False,
        )
        curve_analysis_returncode = analyzed.returncode
    training_report_path = paths["output_dir"] / "report.json"
    training_report = (
        json.loads(training_report_path.read_text()) if training_report_path.is_file() else None
    )
    report_passed = training_report.get("passed") if isinstance(training_report, dict) else None
    require_report_passed = bool(config.get("require_report_passed", True))
    status, terminal_stage = terminal_status(
        training_returncode,
        curve_analysis_returncode,
        report_passed,
        require_report_passed=require_report_passed,
    )
    result = {
        **running,
        "status": status,
        "stage": terminal_stage,
        "finished_at": datetime.now(UTC).isoformat(),
        "child_pid": child_pid,
        "returncode": training_returncode,
        "curve_analysis": (
            str(curve_analysis_path.resolve()) if curve_analysis_returncode == 0 else None
        ),
        "curve_analysis_returncode": curve_analysis_returncode,
        "training_report": (
            str(training_report_path.resolve()) if training_report_path.is_file() else None
        ),
        "training_report_passed": report_passed,
        "training_report_pass_required": require_report_passed,
    }
    result["result_manifest"] = str((paths["output_dir"] / "cloud-result-manifest.json").resolve())
    atomic_json(run_receipt_path, result)
    try:
        write_result_manifest(paths["output_dir"], int(config["layer"]))
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "stage": "result-manifest",
                "result_manifest_error": f"{type(error).__name__}: {error}",
            }
        )
        atomic_json(run_receipt_path, result)
        raise
    if training_returncode == 75:
        return 0
    if training_returncode != 0:
        return training_returncode
    if curve_analysis_returncode != 0:
        return 2
    return 0 if report_passed is True or not require_report_passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
