from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CANDIDATE_CHECKPOINT = re.compile(r"^layer-(\d+)-top(\d+)\.safetensors$")


def configured_top_k(run: dict, report: dict) -> int:
    """The routing width a run actually trained, from its own configuration.

    A de-risk may route at a Top-K other than the shipped Top-4 and publishes
    ``layer-<layer>-top<K>.safetensors``. Assuming Top-4 would silently
    disqualify every such result, so the width is read from the run rather than
    hardcoded. Run configuration is authoritative; the launch command is the
    fallback for older receipts.
    """
    schedule = report.get("configuration", {}).get("top_k_schedule")
    if isinstance(schedule, (list, tuple)) and schedule:
        try:
            return int(schedule[-1])
        except (TypeError, ValueError):
            pass
    command = run.get("command")
    if isinstance(command, list) and "--top-k-schedule" in command:
        values: list[int] = []
        for token in command[command.index("--top-k-schedule") + 1 :]:
            if isinstance(token, str) and token.startswith("--"):
                break
            try:
                values.append(int(token))
            except (TypeError, ValueError):
                break
        if values:
            return values[-1]
    return 4


def candidate_checkpoint_name(layer: int, top_k: int) -> str:
    return f"layer-{layer}-top{top_k}.safetensors"


def scientific_candidate_eligibility(report: dict, curve: dict) -> dict:
    status = curve.get("status")
    try:
        baseline = float(curve["baseline_error"])
        target = float(curve["target_error"])
        curve_last = float(curve["last"]["relative_l2"])
        final = float(report["stages"][-1]["validation"]["relative_l2"])
    except (KeyError, IndexError, TypeError, ValueError):
        return {
            "passed": False,
            "curve_status": status,
            "reason": "missing calibrated curve or final held-out validation",
        }
    if not all(math.isfinite(value) for value in (baseline, target, curve_last, final)):
        return {
            "passed": False,
            "curve_status": status,
            "reason": "non-finite scientific eligibility metric",
        }
    if status == "target_reached":
        limit = target
    elif status == "promising" and baseline > target:
        limit = baseline - 0.25 * (baseline - target)
    else:
        return {
            "passed": False,
            "curve_status": status,
            "reason": "curve does not support candidate evaluation",
        }
    checks = {
        "training_report_passed": report.get("passed") is True,
        "last_curve_point_within_limit": curve_last <= limit,
        "final_validation_within_limit": final <= limit,
    }
    return {
        "passed": all(checks.values()),
        "curve_status": status,
        "baseline_error": baseline,
        "target_error": target,
        "candidate_error_limit": limit,
        "last_curve_relative_l2": curve_last,
        "final_validation_relative_l2": final,
        "checks": checks,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
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


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def result_payloads(directory: Path, layer: int) -> list[Path]:
    del layer
    manifest = directory / "cloud-result-manifest.json"
    symlinks = sorted(path for path in directory.rglob("*") if path.is_symlink())
    if symlinks:
        raise ValueError(
            "cloud result contains symlink(s): "
            + ", ".join(str(path.relative_to(directory)) for path in symlinks)
        )
    return sorted(path for path in directory.rglob("*") if path.is_file() and path != manifest)


def build_result_manifest(directory: Path, layer: int) -> dict:
    directory = directory.resolve()
    run = _json(directory / "cloud-run.json")
    report = _json(directory / "report.json") if (directory / "report.json").is_file() else {}
    curve = (
        _json(directory / "curve-analysis.json")
        if (directory / "curve-analysis.json").is_file()
        else {}
    )
    files = {}
    for path in result_payloads(directory, layer):
        name = str(path.relative_to(directory))
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    top_k = configured_top_k(run, report)
    checkpoint_name = candidate_checkpoint_name(layer, top_k)
    eligibility = scientific_candidate_eligibility(report, curve)
    candidate_ready = (
        run.get("status") == "passed" and checkpoint_name in files and eligibility["passed"]
    )
    return {
        "format": "moeme-cloud-result-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "layer": layer,
        "candidate_top_k": top_k,
        "run_status": run.get("status"),
        "curve_status": curve.get("status"),
        "candidate_ready": candidate_ready,
        "scientific_eligibility": eligibility,
        "files": files,
    }


def write_result_manifest(directory: Path, layer: int) -> dict:
    manifest = build_result_manifest(directory, layer)
    atomic_json(directory / "cloud-result-manifest.json", manifest)
    return manifest


def verify_result_manifest(directory: Path) -> dict:
    directory = directory.resolve()
    manifest = _json(directory / "cloud-result-manifest.json")
    if manifest.get("format") != "moeme-cloud-result-manifest-v1":
        raise ValueError("unsupported cloud result manifest format")
    expected_files = set(manifest.get("files", {}))
    symlinks = sorted(
        str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_symlink()
    )
    if symlinks:
        raise ValueError("cloud result contains symlink(s): " + ", ".join(symlinks))
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "cloud-result-manifest.json"
    }
    missing = sorted(expected_files - actual_files)
    unexpected = sorted(actual_files - expected_files)
    if missing or unexpected:
        raise ValueError(
            f"cloud result inventory mismatch; missing={missing}, unexpected={unexpected}"
        )
    verified = {}
    for name, receipt in manifest.get("files", {}).items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"cloud result payload is missing: {name}")
        actual_bytes = path.stat().st_size
        if actual_bytes != receipt.get("bytes"):
            raise ValueError(f"cloud result payload size mismatch: {name}")
        actual_hash = sha256(path)
        if actual_hash != receipt.get("sha256"):
            raise ValueError(f"cloud result payload hash mismatch: {name}")
        verified[name] = {"bytes": actual_bytes, "sha256": actual_hash}
    required = {"cloud-preflight.json", "cloud-run.json", "progress.json"}
    if manifest.get("run_status") == "passed":
        required.update(
            {
                "curve.json",
                "curve-analysis.json",
                "report.json",
                candidate_checkpoint_name(
                    int(manifest["layer"]), int(manifest.get("candidate_top_k", 4))
                ),
            }
        )
    missing_receipts = sorted(required - verified.keys())
    if missing_receipts:
        raise ValueError(
            "cloud result manifest omits required payloads: " + ", ".join(missing_receipts)
        )
    return {
        "format": "moeme-cloud-result-verification-v1",
        "verified_at": datetime.now(UTC).isoformat(),
        "layer": manifest.get("layer"),
        "run_status": manifest.get("run_status"),
        "curve_status": manifest.get("curve_status"),
        "candidate_ready": manifest.get("candidate_ready") is True,
        "scientific_eligibility": manifest.get("scientific_eligibility"),
        "files": verified,
    }
