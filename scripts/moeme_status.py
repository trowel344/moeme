#!/usr/bin/env python3
"""Print one compact, authoritative snapshot of the all-layer MoEMe campaign."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import statistics
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL_PROGRESS = {"complete", "failed"}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def parse_service_show(output: str) -> dict[str, Any]:
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    pid = int(values.get("MainPID", "0") or 0)
    active = values.get("ActiveState", "unknown")
    try:
        started_monotonic_usec = int(values.get("ExecMainStartTimestampMonotonic", "0") or 0)
    except ValueError:
        started_monotonic_usec = 0
    return {
        "active": active == "active" and pid > 0,
        "running": active in {"active", "activating"} and pid > 0,
        "active_state": active,
        "sub_state": values.get("SubState", "unknown"),
        "main_pid": pid,
        "result": values.get("Result", "unknown"),
        "started_monotonic_usec": started_monotonic_usec,
    }


def service_status(name: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                name,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "MainPID",
                "-p",
                "Result",
                "-p",
                "ExecMainStartTimestampMonotonic",
                "--no-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"active": False, "error": f"{type(error).__name__}: {error}"}
    status = parse_service_show(result.stdout)
    status["query_returncode"] = result.returncode
    return status


def capture_progress_estimate(
    partial_files: list[dict[str, Any]], service: dict[str, Any], uptime_seconds: float
) -> dict[str, Any] | None:
    if not service.get("running") or not partial_files:
        return None
    progress = max(partial_files, key=lambda item: int(item.get("bytes", 0)))
    fraction = float(progress.get("completion_fraction", 0.0))
    started = int(service.get("started_monotonic_usec", 0)) / 1_000_000
    elapsed = uptime_seconds - started
    if fraction <= 0 or elapsed <= 0:
        return None
    estimated_total = elapsed / fraction
    return {
        "completion_fraction": fraction,
        "elapsed_seconds": round(elapsed),
        "estimated_total_seconds": round(estimated_total),
        "estimated_remaining_seconds": max(0, round(estimated_total - elapsed)),
        "method": "observed-bytes-linear",
    }


def process_table(proc_root: Path = Path("/proc")) -> dict[int, dict[str, Any]]:
    processes: dict[int, dict[str, Any]] = {}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = {
                key: value.strip()
                for key, value in (
                    line.split(":", 1)
                    for line in (directory / "status").read_text().splitlines()
                    if ":" in line
                )
            }
            command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
            pid = int(directory.name)
            processes[pid] = {
                "pid": pid,
                "ppid": int(fields.get("PPid", "0")),
                "state": fields.get("State", "unknown").split()[0],
                "rss_kib": int(fields.get("VmRSS", "0 kB").split()[0]),
                "command": command,
            }
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            ValueError,
            UnicodeDecodeError,
        ):
            continue
    return processes


def descendants(processes: dict[int, dict[str, Any]], root_pid: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        children = [value for value in processes.values() if value["ppid"] == parent]
        found.extend(children)
        pending.extend(value["pid"] for value in children)
    return sorted(found, key=lambda value: value["pid"])


def summarize_command(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return command[:240]
    if not parts:
        return ""
    summary = parts[:2]
    for option in ("--layer", "--output-dir", "--device", "--type"):
        if option in parts and parts.index(option) + 1 < len(parts):
            index = parts.index(option)
            summary.extend(parts[index : index + 2])
    return shlex.join(summary)[:240]


def parse_compute_apps(output: str) -> list[dict[str, int]]:
    applications = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            applications.append({"pid": int(fields[0]), "used_memory_mib": int(fields[1])})
        except ValueError:
            continue
    return applications


def gpu_status() -> dict[str, Any]:
    try:
        applications = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        device = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        fields = [field.strip() for field in device.stdout.splitlines()[0].split(",")]
        return {
            "available": applications.returncode == 0 and device.returncode == 0,
            "compute_apps": parse_compute_apps(applications.stdout),
            "memory_used_mib": int(fields[0]),
            "memory_total_mib": int(fields[1]),
            "utilization_percent": int(fields[2]),
            "temperature_c": int(fields[3]),
            "power_w": float(fields[4]),
        }
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value)


def duration_estimate(campaign: dict[str, Any], total_layers: int = 64) -> dict[str, Any]:
    finished = []
    started = campaign.get("started_at")
    if isinstance(started, str):
        previous = iso8601(started)
        for _, value in sorted(campaign.get("layers", {}).items(), key=lambda item: int(item[0])):
            timestamp = value.get("finished_at")
            if not isinstance(timestamp, str):
                continue
            current = iso8601(timestamp)
            elapsed = (current - previous).total_seconds()
            if elapsed > 0:
                finished.append(elapsed)
            previous = current
    if not finished:
        return {"median_layer_seconds": None, "training_seconds_remaining": None}
    completed = len(campaign.get("layers", {}))
    median = statistics.median(finished)
    return {
        "median_layer_seconds": round(median, 1),
        "training_seconds_remaining": round(max(total_layers - completed, 0) * median),
    }


def active_progress(training_dir: Path) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in training_dir.glob("layer-*/progress.json"):
        value = read_json(path)
        if value is None or value.get("phase") in TERMINAL_PROGRESS:
            continue
        try:
            layer = int(value.get("layer"))
        except (TypeError, ValueError):
            continue
        candidates.append((layer, path, value))
    if not candidates:
        return None
    layer, path, value = max(candidates, key=lambda item: item[1].stat().st_mtime_ns)
    return {
        "layer": layer,
        "phase": value.get("phase"),
        "step": value.get("step"),
        "steps": value.get("steps"),
        "loss": value.get("loss"),
        "reconstruction_loss": value.get("reconstruction_loss"),
        "routing_loss": value.get("routing_loss"),
        "updated_at": value.get("updated_at"),
        "receipt": str(path),
    }


def artifact(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {"exists": False, "path": str(path)}
    return {"exists": True, "bytes": size, "path": str(path)}


def build_status(root: Path) -> dict[str, Any]:
    pipeline_path = root / ".moeme/all64-pipeline-v3.json"
    diagnostic_path = root / ".moeme/corrected-prefix6-pipeline.json"
    absolute_sweep_path = root / ".moeme/prefix-parity-sweep.json"
    controlled_sweep_path = root / ".moeme/prefix-parity-sweep-vs-top12.json"
    sweep_path = controlled_sweep_path if controlled_sweep_path.exists() else absolute_sweep_path
    # The invalid all64-v2 campaign was retired after three recipe defects were
    # fixed; the corrected spec-layout campaign is the live one.
    campaign_path = root / "artifacts/router-training/campaign-spec12/campaign.json"
    # Release receipts for the shipped Top-12 artifact. The previous all64-top4
    # paths belonged to the retired Top-4 program (see docs/top4-feasibility.md),
    # and `final_path` pointed at that program's failed candidate.
    server_path = root / ".moeme/server-eval-top12-q5.json"
    parity_path = root / ".moeme/logit-parity-top12-imatrix-q5-chunks8.json"
    release_manifest_path = root / ".moeme/top12-release-manifest.json"
    iteration_dir = root / "artifacts/router-training/pilot-spec12/spec-oracle-down-4k"
    final_path = root / "artifacts/gguf/moeme-27b-top12-imatrix-q5_k_m.gguf"
    training_dir = campaign_path.parent

    pipeline = read_json(pipeline_path) or {}
    diagnostic = read_json(diagnostic_path) or {}
    sweep = read_json(sweep_path) or {}
    campaign = read_json(campaign_path) or {}
    server = read_json(server_path) or {}
    parity = read_json(parity_path) or {}
    release_manifest = read_json(release_manifest_path) or {}
    iteration_progress = read_json(iteration_dir / "progress.json")
    iteration_report = read_json(iteration_dir / "report.json")
    progress = active_progress(training_dir)
    layer_records = campaign.get("layers", {})
    checkpointed = len(layer_records) if isinstance(layer_records, dict) else 0
    phase_states = {
        name: value.get("status")
        for name, value in pipeline.get("phases", {}).items()
        if isinstance(value, dict)
    }
    inferred_phase = pipeline.get("current_phase")
    if inferred_phase is None and progress is not None:
        inferred_phase = "training"

    final_artifact = artifact(final_path)
    release_phases = release_manifest.get("phases") or {}
    quantization_status = (release_phases.get("quantization") or {}).get("status")
    evaluation_status = (release_phases.get("evaluation") or {}).get("status")
    receipt_gates = {
        "quantization": (
            quantization_status == "passed" or phase_states.get("quantization") == "passed"
        ),
        "server_evaluation": (
            (evaluation_status == "passed" or phase_states.get("evaluation") == "passed")
            and server.get("passed") is True
            and server.get("quality_passed") is True
            and server.get("stability_passed") is True
            and server.get("performance_passed") is True
        ),
        "logit_parity": parity.get("passed") is True,
        "final_artifact_present": final_artifact["exists"],
    }
    disk = shutil.disk_usage(root)
    services = {
        "pipeline": service_status("moeme-all64-pipeline.service"),
        "logit_parity": service_status("moeme-logit-parity.service"),
        "prefix6_diagnostic": service_status("moeme-prefix6-diagnostic.service"),
        "prefix_parity_sweep": service_status("moeme-prefix-parity-sweep.service"),
        "layer1_iteration": service_status("moeme-layer1-oracle-anneal.service"),
        "cloud_capture": service_status("moeme-layer63-capture-200k-v3.service"),
        "cloud_capture_retired_v2": service_status("moeme-layer63-capture-200k-v2.service"),
        "cloud_capture_retired_v1": service_status("moeme-layer63-capture-200k.service"),
        "cloud_postcapture": service_status("moeme-layer63-postcapture-v1.service"),
        "cloud_smoke": service_status("moeme-layer63-seeded-smoke-v1.service"),
        "cloud_candidate": service_status("moeme-layer63-candidate-v1.service"),
    }
    processes = process_table()
    pipeline_workers = descendants(processes, services["pipeline"].get("main_pid", 0))
    gpu = gpu_status()
    gpu_pids = {value["pid"] for value in gpu.get("compute_apps", [])}
    for worker in pipeline_workers:
        worker["command"] = summarize_command(worker["command"])
        worker["uses_gpu"] = worker["pid"] in gpu_pids

    cloud_checkpoint_receipt = read_json(root / "cloud-inputs/layer63-checkpoint/receipt.json")
    cloud_seed_receipt = read_json(root / "cloud-inputs/layer63-seed/receipt.json")
    cloud_corpus = read_json(root / "data/training/layer63-de-risk-200k.json")
    cloud_activation = read_json(root / "cloud-inputs/layer63-200k-activations/manifest.json")
    cloud_imatrix_receipt = read_json(root / ".moeme/layer63-200k-imatrix.receipt.json")
    cloud_progress = read_json(root / "cloud-results/layer63-200k/progress.json")
    cloud_curve = read_json(root / "cloud-results/layer63-200k/curve.json")
    cloud_curve_analysis = read_json(root / "cloud-results/layer63-200k/curve-analysis.json")
    cloud_run = read_json(root / "cloud-results/layer63-200k/cloud-run.json")
    cloud_result_manifest = read_json(
        root / "cloud-results/layer63-200k/cloud-result-manifest.json"
    )
    cloud_transfer = read_json(root / ".moeme/layer63-cloud-transfer.json")
    cloud_acceptance = read_json(root / ".moeme/layer63-candidate-pipeline.json")
    cloud_upload_manifest = read_json(root / "cloud-jobs/layer63-200k/upload-manifest.json")
    cloud_upload_receipt = read_json(root / ".moeme/cloud-upload.json")
    kaggle_kernel_receipt = read_json(root / ".moeme/kaggle-kernel.json")
    free_tier_budget = read_json(root / ".moeme/free-tier-budget.json")
    cloud_postcapture = read_json(root / ".moeme/layer63-postcapture-v1.json")
    smoke_run = read_json(root / "cloud-results/local-seeded-smoke/cloud-run.json")
    smoke_progress = read_json(root / "cloud-results/local-seeded-smoke/progress.json")
    smoke_result = read_json(root / "cloud-results/local-seeded-smoke/cloud-result-manifest.json")
    partial_capture_dirs = sorted(
        str(path.relative_to(root))
        for path in (root / "cloud-inputs").glob(".layer63-200k-activations.*")
    )
    # This runtime batches two 256-token contexts into each capture callback,
    # so the long run publishes 400 chunk headers around 204,800 token rows.
    expected_capture_bytes = 12 + 400 * 4 + 204800 * 5120 * 4
    partial_capture_files = []
    for path in sorted((root / "cloud-inputs").glob(".layer63-200k-activations.*/layer-63.f32")):
        size = path.stat().st_size
        partial_capture_files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": size,
                "expected_bytes": expected_capture_bytes,
                "completion_fraction": min(size / expected_capture_bytes, 1.0),
            }
        )
    try:
        uptime_seconds = float(Path("/proc/uptime").read_text().split()[0])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        uptime_seconds = 0.0
    cloud_capture_progress = capture_progress_estimate(
        partial_capture_files, services["cloud_capture"], uptime_seconds
    )

    return {
        "format": "moeme-compact-status-v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "pipeline": {
            "current_phase": inferred_phase,
            "phases": phase_states,
            "last_error": pipeline.get("last_error"),
            "manifest": str(pipeline_path),
        },
        "prefix6_diagnostic": {
            "current_phase": diagnostic.get("current_phase"),
            "phases": {
                name: value.get("status")
                for name, value in diagnostic.get("phases", {}).items()
                if isinstance(value, dict)
            },
            "last_error": diagnostic.get("last_error"),
            "manifest": str(diagnostic_path),
        },
        "prefix_parity_sweep": {
            "status": sweep.get("status"),
            "tested_prefixes": len(sweep.get("results", [])),
            "first_failed_prefix": sweep.get("first_failed_prefix"),
            "error": sweep.get("error"),
            "report": str(sweep_path),
            "absolute_dense_report": str(absolute_sweep_path),
        },
        "campaign": {
            "checkpointed_layers": checkpointed,
            "total_layers": 64,
            "gate_passed_layers": sum(
                value.get("status") == "passed" for value in layer_records.values()
            )
            if isinstance(layer_records, dict)
            else 0,
            "retained_gate_failed_layers": sum(
                value.get("status") == "gate_failed" for value in layer_records.values()
            )
            if isinstance(layer_records, dict)
            else 0,
            "active": progress if services["pipeline"]["active"] else None,
            "interrupted_progress": progress if not services["pipeline"]["active"] else None,
            "updated_at": campaign.get("updated_at"),
            **duration_estimate(campaign),
            "manifest": str(campaign_path),
        },
        "runtime": {
            "pipeline_workers": pipeline_workers,
            "active_gpu_worker": any(worker["uses_gpu"] for worker in pipeline_workers),
            "gpu": gpu,
        },
        "services": services,
        "free_cloud": {
            "portable_checkpoint": cloud_checkpoint_receipt,
            "training_seed": cloud_seed_receipt,
            "corpus": cloud_corpus,
            "activation_capture": {
                "service": services["cloud_capture"],
                "manifest": cloud_activation,
                "imatrix_receipt": cloud_imatrix_receipt,
                "expected_tokens": 204800,
                "ready": (
                    cloud_activation is not None
                    and cloud_activation.get("tokens_per_layer") == 204800
                    and "63" in cloud_activation.get("layers", {})
                ),
                "partial_directories": partial_capture_dirs,
                "partial_files": partial_capture_files,
                "progress": cloud_capture_progress,
            },
            "training": {
                "run": cloud_run,
                "result_manifest": cloud_result_manifest,
                "transfer_verification": cloud_transfer,
                "progress": cloud_progress,
                "curve_points": len((cloud_curve or {}).get("points", [])),
                "curve_analysis": cloud_curve_analysis,
            },
            "upload_stage": {
                "manifest": cloud_upload_manifest,
                "ready": cloud_upload_manifest is not None,
                "directory": str(root / "cloud-jobs/layer63-200k"),
            },
            "upload_transfer": cloud_upload_receipt,
            "kaggle_kernel": kaggle_kernel_receipt,
            "free_tier_budget": free_tier_budget,
            "postcapture_supervisor": cloud_postcapture,
            "acceptance": cloud_acceptance,
            "acceptance_service": services["cloud_candidate"],
            "local_smoke": {
                "service": services["cloud_smoke"],
                "run": smoke_run,
                "progress": smoke_progress,
                "result_manifest": smoke_result,
            },
        },
        "layer1_iteration": {
            "progress": iteration_progress,
            "report": iteration_report,
            "directory": str(iteration_dir),
        },
        "storage": {
            "free_bytes": disk.free,
            "total_bytes": disk.total,
        },
        "artifacts": {
            "diagnostic_q4": artifact(
                root / "artifacts/gguf/moeme-27b-hybrid-representative5-q4_k_m.gguf"
            ),
            "working_bf16": artifact(root / "artifacts/gguf/moeme-27b-all64-top4-bf16.gguf"),
            "prefix6_bf16": artifact(root / "artifacts/gguf/moeme-27b-corrected-prefix6-bf16.gguf"),
            "prefix6_q4": artifact(root / "artifacts/gguf/moeme-27b-corrected-prefix6-q4_k_m.gguf"),
            "final_q4": final_artifact,
        },
        "release": {
            "receipt_gates": receipt_gates,
            "receipts_ready_for_hash_validated_launch": all(receipt_gates.values()),
            "server_report": str(server_path),
            "parity_report": str(parity_path),
            "manifest": str(release_manifest_path),
        },
    }


def compact_summary(status: dict[str, Any]) -> dict[str, Any]:
    """Reduce the full diagnostic snapshot to heartbeat-sized authoritative state."""
    free_cloud = status.get("free_cloud") or {}
    capture = free_cloud.get("activation_capture") or {}
    training = free_cloud.get("training") or {}
    smoke = free_cloud.get("local_smoke") or {}
    acceptance = free_cloud.get("acceptance") or {}
    phases = acceptance.get("phases") or {}
    return {
        "format": "moeme-progress-summary-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "capture": {
            "service": capture.get("service"),
            "ready": capture.get("ready", False),
            "progress": capture.get("progress"),
            "partial_files": capture.get("partial_files", []),
            "manifest_present": capture.get("manifest") is not None,
            "imatrix_receipt_present": capture.get("imatrix_receipt") is not None,
        },
        "postcapture": free_cloud.get("postcapture_supervisor"),
        "upload_stage": free_cloud.get("upload_stage"),
        "upload_transfer": free_cloud.get("upload_transfer"),
        "kaggle_kernel": free_cloud.get("kaggle_kernel"),
        "free_tier_budget": free_cloud.get("free_tier_budget"),
        "smoke": {
            "service": smoke.get("service"),
            "run": smoke.get("run"),
            "result_manifest": smoke.get("result_manifest"),
        },
        "training": {
            "run": training.get("run"),
            "progress": training.get("progress"),
            "curve_points": training.get("curve_points", 0),
            "curve_analysis": training.get("curve_analysis"),
            "candidate_ready": bool((training.get("result_manifest") or {}).get("candidate_ready")),
        },
        "acceptance": {
            "passed": acceptance.get("passed", False),
            "current_phase": acceptance.get("current_phase"),
            "last_error": acceptance.get("last_error"),
            "phase_statuses": {
                name: value.get("status")
                for name, value in phases.items()
                if isinstance(value, dict)
            },
            "service": free_cloud.get("acceptance_service"),
        },
        "storage": status.get("storage"),
        "validated_local_artifact": (status.get("artifacts") or {}).get("final_q4"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    status = build_status(args.root.resolve())
    if args.summary:
        status = compact_summary(status)
    rendered = json.dumps(status, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        atomic_text(args.output, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
