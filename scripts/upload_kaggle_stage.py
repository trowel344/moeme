#!/usr/bin/env python3
"""Prepare and optionally upload a verified private Kaggle dataset."""

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


DATASET_ID = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")


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


def validate_dataset_id(value: str) -> str:
    if not DATASET_ID.fullmatch(value):
        raise ValueError("Kaggle dataset must be a single owner/slug identifier")
    return value


def dataset_metadata(dataset: str, title: str) -> dict:
    validate_dataset_id(dataset)
    if not 6 <= len(title) <= 50 or any(character in title for character in "\r\n"):
        raise ValueError("Kaggle title must be one line and 6-50 characters")
    return {
        "title": title,
        "id": dataset,
        "licenses": [{"name": "other"}],
        "description": "Private MoEMe free-tier training stage; not for redistribution.",
    }


def expected_stage_files(stage: Path) -> set[str]:
    manifest = json.loads((stage / "upload-manifest.json").read_text())
    return set(manifest["files"]) | {"upload-manifest.json"}


def verify_upload_view(stage: Path, view: Path, metadata: dict) -> dict:
    expected = expected_stage_files(stage)
    actual = {
        str(path.relative_to(view))
        for path in view.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    required = expected | {"dataset-metadata.json"}
    if actual != required:
        raise ValueError(
            "Kaggle upload view inventory mismatch; "
            f"missing={sorted(required - actual)}, unexpected={sorted(actual - required)}"
        )
    symlinks = sorted(str(path.relative_to(view)) for path in view.rglob("*") if path.is_symlink())
    if symlinks:
        raise ValueError("Kaggle upload view contains symlink(s): " + ", ".join(symlinks))
    if json.loads((view / "dataset-metadata.json").read_text()) != metadata:
        raise ValueError("Kaggle upload metadata does not match this invocation")
    for relative in expected:
        source = stage / relative
        target = view / relative
        if not target.is_file() or not os.path.samefile(source, target):
            raise ValueError(f"Kaggle upload view is not bound to staged file: {relative}")
    return {
        "directory": str(view),
        "file_count": len(required),
        "hardlinked_stage_files": len(expected),
        "verified": True,
    }


def build_upload_view(stage: Path, view: Path, metadata: dict) -> dict:
    if view.exists():
        return verify_upload_view(stage, view, metadata)
    view.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{view.name}.", dir=view.parent))
    try:
        for relative in sorted(expected_stage_files(stage)):
            source = stage / relative
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
        (temporary / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, view)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_upload_view(stage, view, metadata)


def kaggle_command(
    view: Path,
    executable: str = "kaggle",
    *,
    version_notes: str | None = None,
) -> list[str]:
    """Build the Kaggle CLI upload command for this view.

    ``datasets create`` cannot update an already-published dataset, so refreshing
    a stage after a configuration change needs ``datasets version`` and its
    mandatory version notes instead.
    """
    if version_notes is None:
        return [
            executable,
            "datasets",
            "create",
            "--path",
            str(view.resolve()),
            "--dir-mode",
            "tar",
            "--keep-tabular",
        ]
    return [
        executable,
        "datasets",
        "version",
        "--path",
        str(view.resolve()),
        "--dir-mode",
        "tar",
        "--keep-tabular",
        "-m",
        version_notes,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=Path("cloud-jobs/layer63-200k"))
    parser.add_argument("--dataset", required=True, help="private Kaggle owner/slug")
    parser.add_argument("--budget", type=Path, default=Path(".moeme/free-tier-budget.json"))
    parser.add_argument("--title", default="MoEMe Layer 63 Private Training Stage")
    parser.add_argument("--view", type=Path, default=Path(".moeme/kaggle-layer63-upload"))
    parser.add_argument("--receipt", type=Path, default=Path(".moeme/cloud-upload.json"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the authenticated private upload; omission is a verified dry run",
    )
    parser.add_argument(
        "--account-usage-checked",
        action="store_true",
        help="confirm the signed-in account still has the required free allowance",
    )
    parser.add_argument(
        "--version-notes",
        default=None,
        help=(
            "publish a new version of an existing dataset with these notes; "
            "omit to create a brand-new dataset"
        ),
    )
    args = parser.parse_args()
    if args.execute and not args.account_usage_checked:
        parser.error("--execute requires --account-usage-checked")
    if args.version_notes is not None and not args.version_notes.strip():
        parser.error("--version-notes must not be blank")
    stage = args.stage.resolve()
    view = args.view.resolve()
    metadata = dataset_metadata(args.dataset, args.title)
    stage_verification = verify_stage(stage)
    budget_verification = verify_free_tier_budget(stage, args.budget.resolve(), "kaggle")
    view_verification = build_upload_view(stage, view, metadata)
    receipt = {
        "format": "moeme-cloud-upload-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "provider": "kaggle",
        "privacy": "private",
        "stage": str(stage),
        "stage_manifest_sha256": sha256(stage / "upload-manifest.json"),
        "stage_verification": stage_verification,
        "budget_verification": budget_verification,
        "upload_view_verification": view_verification,
        "destination": args.dataset,
        "command": kaggle_command(view, version_notes=args.version_notes),
        "mode": "version" if args.version_notes is not None else "create",
        "version_notes": args.version_notes,
        "executed": args.execute,
        "account_usage_checked": args.account_usage_checked,
        "status": "verified_dry_run",
    }
    receipt_path = args.receipt.resolve()
    atomic_json(receipt_path, receipt)
    if not args.execute:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    executable = shutil.which("kaggle")
    if executable is None:
        receipt["status"] = "failed"
        receipt["error"] = "Kaggle CLI is not installed or authenticated"
        atomic_json(receipt_path, receipt)
        raise FileNotFoundError(receipt["error"])
    command = kaggle_command(view, executable, version_notes=args.version_notes)
    receipt["command"] = command
    receipt["status"] = "running"
    receipt["started_at"] = datetime.now(UTC).isoformat()
    atomic_json(receipt_path, receipt)
    completed = subprocess.run(command, check=False)
    receipt["finished_at"] = datetime.now(UTC).isoformat()
    receipt["returncode"] = completed.returncode
    receipt["status"] = "passed" if completed.returncode == 0 else "failed"
    atomic_json(receipt_path, receipt)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
