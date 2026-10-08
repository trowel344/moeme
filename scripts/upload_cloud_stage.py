#!/usr/bin/env python3
"""Verify and optionally upload a staged MoEMe job with the Lightning CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.stage_free_cloud_job import verify_stage
except ModuleNotFoundError:
    from stage_free_cloud_job import verify_stage


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


def lightning_command(stage: Path, destination: str, executable: str = "lightning") -> list[str]:
    if not destination.startswith("lit://") or any(
        character in destination for character in "\r\n"
    ):
        raise ValueError("Lightning destination must be one single-line lit:// URI")
    return [executable, "cp", "-r", str(stage.resolve()), destination]


def verify_free_tier_budget(stage: Path, budget_path: Path, provider: str) -> dict:
    budget = json.loads(budget_path.read_text())
    if budget.get("format") != "moeme-free-tier-budget-v1" or budget.get("passed") is not True:
        raise ValueError("free-tier budget has not passed")
    manifest_digest = sha256(stage / "upload-manifest.json")
    if (budget.get("measured_from") or {}).get("stage_manifest_sha256") != manifest_digest:
        raise ValueError("free-tier budget is stale for this upload stage")
    provider_budget = (budget.get("providers") or {}).get(provider) or {}
    fit_key = "fits_unbilled_storage" if provider == "lightning" else "fits_working_storage"
    if provider_budget.get(fit_key) is not True:
        raise ValueError(f"measured free-tier budget does not fit {provider}")
    return {
        "budget": str(budget_path),
        "provider": provider,
        "stage_manifest_sha256": manifest_digest,
        "fit_key": fit_key,
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=Path("cloud-jobs/layer63-200k"))
    parser.add_argument("--destination", required=True)
    parser.add_argument("--budget", type=Path, default=Path(".moeme/free-tier-budget.json"))
    parser.add_argument("--receipt", type=Path, default=Path(".moeme/cloud-upload.json"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the authenticated external upload; omission is a verified dry run",
    )
    parser.add_argument(
        "--account-usage-checked",
        action="store_true",
        help="confirm the signed-in account still has the required free allowance",
    )
    args = parser.parse_args()
    if args.execute and not args.account_usage_checked:
        parser.error("--execute requires --account-usage-checked")
    stage = args.stage.resolve()
    verification = verify_stage(stage)
    budget_verification = verify_free_tier_budget(stage, args.budget.resolve(), "lightning")
    command = lightning_command(stage, args.destination)
    receipt = {
        "format": "moeme-cloud-upload-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "provider": "lightning",
        "stage": str(stage),
        "stage_manifest_sha256": sha256(stage / "upload-manifest.json"),
        "stage_verification": verification,
        "budget_verification": budget_verification,
        "destination": args.destination,
        "command": command,
        "executed": args.execute,
        "account_usage_checked": args.account_usage_checked,
        "status": "verified_dry_run",
    }
    receipt_path = args.receipt.resolve()
    atomic_json(receipt_path, receipt)
    if not args.execute:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    executable = shutil.which("lightning")
    if executable is None:
        receipt["status"] = "failed"
        receipt["error"] = "Lightning CLI is not installed; install lightning-sdk and sign in"
        atomic_json(receipt_path, receipt)
        raise FileNotFoundError(receipt["error"])
    command = lightning_command(stage, args.destination, executable)
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
