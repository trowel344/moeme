#!/usr/bin/env python3
"""Prepare and optionally submit a private, hash-bound Kaggle training kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.stage_free_cloud_job import verify_stage
except ModuleNotFoundError:
    from stage_free_cloud_job import verify_stage

try:
    from scripts.upload_cloud_stage import verify_free_tier_budget
except ModuleNotFoundError:
    from upload_cloud_stage import verify_free_tier_budget

KAGGLE_ID = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
ACTIVE_LAUNCH_STATUSES = {"running", "submitted", "passed"}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_kaggle_id(value: str, label: str) -> str:
    if not KAGGLE_ID.fullmatch(value):
        raise ValueError(f"Kaggle {label} must be a single owner/slug identifier")
    return value


SLUG_WORD = re.compile(r"[a-z0-9]+")


def title_slug(title: str) -> str:
    """Reproduce the slug Kaggle derives from a kernel title."""
    return "-".join(SLUG_WORD.findall(title.lower()))


def kernel_metadata(kernel: str, dataset: str, title: str) -> dict:
    validate_kaggle_id(kernel, "kernel")
    validate_kaggle_id(dataset, "dataset")
    if not 6 <= len(title) <= 50 or any(character in title for character in "\r\n"):
        raise ValueError("Kaggle kernel title must be one line and 6-50 characters")
    owner, _, slug = kernel.partition("/")
    resolved = title_slug(title)
    if slug != resolved:
        # Kaggle creates the kernel at the slug derived from its title and ignores a
        # non-resolving id, so a mismatch would leave the receipted kernel id pointing
        # at a kernel that does not exist and break every later poll.
        raise ValueError(
            "Kaggle derives a kernel's slug from its title, so this title would push "
            f"{owner}/{resolved}, not {kernel}; use --kernel {owner}/{resolved} or a "
            "title that resolves to the requested id"
        )
    return {
        "id": kernel,
        "title": title,
        "code_file": "run_moeme.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def kernel_source(stage_manifest_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", stage_manifest_sha256):
        raise ValueError("stage manifest SHA-256 must be lowercase hexadecimal")
    return f'''#!/usr/bin/env python3
import hashlib
import subprocess
import sys
from pathlib import Path

EXPECTED_MANIFEST_SHA256 = "{stage_manifest_sha256}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# Kaggle currently mounts datasets at /kaggle/input/datasets/<owner>/<slug>/ rather than
# directly under /kaggle/input, so search the whole (small) input tree instead of a single
# level and bind to the staged manifest by content.
candidates = [
    path.parent
    for path in Path("/kaggle/input").rglob("upload-manifest.json")
    if sha256(path) == EXPECTED_MANIFEST_SHA256
]
if len(candidates) != 1:
    raise RuntimeError(
        "expected exactly one Kaggle input matching the staged manifest; "
        f"found {{len(candidates)}}"
    )
stage = candidates[0]
command = [
    sys.executable,
    str(stage / "scripts/cloud_bootstrap.py"),
    "--stage",
    str(stage),
    "--provider",
    "kaggle",
]
raise SystemExit(subprocess.run(command, check=False).returncode)
'''


def verify_kernel_view(view: Path, metadata: dict, source: str) -> dict:
    expected = {"kernel-metadata.json", "run_moeme.py"}
    actual = {
        str(path.relative_to(view))
        for path in view.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise ValueError(
            "Kaggle kernel view inventory mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    if any(path.is_symlink() for path in view.rglob("*")):
        raise ValueError("Kaggle kernel view contains a symlink")
    if json.loads((view / "kernel-metadata.json").read_text()) != metadata:
        raise ValueError("Kaggle kernel metadata does not match this invocation")
    if (view / "run_moeme.py").read_text() != source:
        raise ValueError("Kaggle kernel source does not match this invocation")
    return {
        "directory": str(view),
        "file_count": len(expected),
        "metadata_sha256": sha256(view / "kernel-metadata.json"),
        "source_sha256": sha256(view / "run_moeme.py"),
        "verified": True,
    }


def build_kernel_view(view: Path, metadata: dict, source: str) -> dict:
    if view.exists():
        return verify_kernel_view(view, metadata, source)
    view.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{view.name}.", dir=view.parent))
    try:
        (temporary / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "run_moeme.py").write_text(source)
        os.replace(temporary, view)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_kernel_view(view, metadata, source)


def verify_upload_receipt(path: Path, dataset: str, stage_manifest_sha256: str) -> dict:
    value = json.loads(path.read_text())
    expected = {
        "format": "moeme-cloud-upload-v1",
        "provider": "kaggle",
        "privacy": "private",
        "destination": dataset,
        "stage_manifest_sha256": stage_manifest_sha256,
        "status": "passed",
        "account_usage_checked": True,
    }
    mismatches = {
        key: {"expected": wanted, "observed": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"Kaggle upload receipt mismatch: {mismatches}")
    return {"path": str(path), "verified": True, **expected}


def kaggle_kernel_command(view: Path, executable: str = "kaggle") -> list[str]:
    return [
        executable,
        "kernels",
        "push",
        "--path",
        str(view.resolve()),
        "--accelerator",
        "NvidiaTeslaT4",
    ]


def refuse_duplicate_launch(receipt_path: Path, kernel: str) -> None:
    try:
        previous = json.loads(receipt_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if previous.get("kernel") == kernel and previous.get("status") in ACTIVE_LAUNCH_STATUSES:
        raise RuntimeError(
            f"Kaggle kernel already has authoritative status {previous['status']}; "
            "refusing to submit another version"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="private Kaggle dataset owner/slug")
    parser.add_argument("--kernel", required=True, help="private Kaggle kernel owner/slug")
    parser.add_argument("--stage", type=Path, default=Path("cloud-jobs/layer63-200k"))
    parser.add_argument("--budget", type=Path, default=Path(".moeme/free-tier-budget.json"))
    parser.add_argument("--upload-receipt", type=Path, default=Path(".moeme/cloud-upload.json"))
    parser.add_argument("--title", default="MoEMe Layer63 Training")
    parser.add_argument("--view", type=Path, default=Path(".moeme/kaggle-layer63-kernel"))
    parser.add_argument("--receipt", type=Path, default=Path(".moeme/kaggle-kernel.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--account-usage-checked",
        action="store_true",
        help="confirm free GPU and storage allowance before submitting the run",
    )
    args = parser.parse_args()
    if args.execute and not args.account_usage_checked:
        parser.error("--execute requires --account-usage-checked")
    stage = args.stage.resolve()
    view = args.view.resolve()
    receipt_path = args.receipt.resolve()
    metadata = kernel_metadata(args.kernel, args.dataset, args.title)
    stage_verification = verify_stage(stage)
    budget_verification = verify_free_tier_budget(stage, args.budget.resolve(), "kaggle")
    manifest_sha256 = sha256(stage / "upload-manifest.json")
    source = kernel_source(manifest_sha256)
    view_verification = build_kernel_view(view, metadata, source)
    receipt = {
        "format": "moeme-kaggle-kernel-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "provider": "kaggle",
        "privacy": "private",
        "dataset": args.dataset,
        "kernel": args.kernel,
        "accelerator": "NvidiaTeslaT4",
        "stage_manifest_sha256": manifest_sha256,
        "stage_verification": stage_verification,
        "budget_verification": budget_verification,
        "kernel_view_verification": view_verification,
        "command": kaggle_kernel_command(view),
        "executed": args.execute,
        "account_usage_checked": args.account_usage_checked,
        "status": "verified_dry_run",
    }
    if not args.execute:
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    refuse_duplicate_launch(receipt_path, args.kernel)
    receipt["upload_receipt"] = verify_upload_receipt(
        args.upload_receipt.resolve(), args.dataset, manifest_sha256
    )
    executable = shutil.which("kaggle")
    if executable is None:
        receipt["status"] = "failed"
        receipt["error"] = "Kaggle CLI is not installed or authenticated"
        atomic_json(receipt_path, receipt)
        raise FileNotFoundError(receipt["error"])
    command = kaggle_kernel_command(view, executable)
    receipt["command"] = command
    receipt["status"] = "running"
    receipt["started_at"] = datetime.now(UTC).isoformat()
    atomic_json(receipt_path, receipt)
    completed = subprocess.run(command, check=False)
    receipt["finished_at"] = datetime.now(UTC).isoformat()
    receipt["returncode"] = completed.returncode
    receipt["status"] = "submitted" if completed.returncode == 0 else "failed"
    atomic_json(receipt_path, receipt)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
