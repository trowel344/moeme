#!/usr/bin/env python3
"""Poll a submitted Kaggle kernel and atomically import its verified result."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from moeme.cloud_results import atomic_json, verify_result_manifest

try:
    from scripts.import_cloud_result import import_result, locate_result_directory
except ModuleNotFoundError:
    from import_cloud_result import import_result, locate_result_directory

STATUS_PATTERN = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
IN_PROGRESS = {"NEW_SCRIPT", "QUEUED", "RUNNING", "CANCEL_REQUESTED"}
FAILED = {"ERROR", "CANCEL_ACKNOWLEDGED"}


def parse_kernel_status(output: str) -> str:
    matches = STATUS_PATTERN.findall(output)
    if len(matches) != 1:
        raise ValueError(f"expected one Kaggle kernel status, found {len(matches)}")
    return matches[0]


def should_collect(
    remote_status: str,
    *,
    sync: bool,
    collect_failed: bool,
) -> bool:
    """Decide whether a terminal status should trigger a result download.

    A Kaggle ``ERROR`` is not proof there is no result: an early-stopped
    scientific run exits non-zero and still publishes a complete result tree.
    Collection is therefore opt-in per terminal class, and a missing tree is
    handled as a best-effort failure rather than a reason to skip the download.
    """
    if remote_status == "COMPLETE":
        return sync
    if remote_status in FAILED:
        return collect_failed
    return False


def status_command(kernel: str, executable: str = "kaggle") -> list[str]:
    return [executable, "kernels", "status", kernel]


def output_command(kernel: str, directory: Path, executable: str = "kaggle") -> list[str]:
    return [
        executable,
        "kernels",
        "output",
        kernel,
        "--path",
        str(directory.resolve()),
        "--quiet",
    ]


def load_launch_receipt(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("format") != "moeme-kaggle-kernel-v1":
        raise ValueError("unsupported Kaggle kernel receipt format")
    if value.get("provider") != "kaggle" or value.get("privacy") != "private":
        raise ValueError("Kaggle kernel receipt is not private Kaggle work")
    if not isinstance(value.get("kernel"), str):
        raise TypeError("Kaggle kernel receipt has no kernel identifier")
    return value


def promote_download(download: Path, inbox: Path) -> dict:
    candidate = locate_result_directory(download)
    verified = verify_result_manifest(candidate)
    if inbox.exists():
        existing = verify_result_manifest(inbox)
        if existing["files"] != verified["files"]:
            raise FileExistsError(f"refusing to replace a different downloaded result: {inbox}")
        return {**existing, "downloaded_to": str(inbox), "already_present": True}
    inbox.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate, inbox)
    return {**verified, "downloaded_to": str(inbox), "already_present": False}


def poll_status(receipt: dict, receipt_path: Path, executable: str) -> str:
    """Poll the receipted kernel once and record the raw evidence on the receipt."""
    checked = subprocess.run(
        status_command(receipt["kernel"], executable),
        capture_output=True,
        text=True,
        check=False,
    )
    receipt["last_polled_at"] = datetime.now(UTC).isoformat()
    receipt["status_returncode"] = checked.returncode
    receipt["status_output"] = (checked.stdout + checked.stderr).strip()
    if checked.returncode != 0:
        receipt["status"] = "poll_failed"
        atomic_json(receipt_path, receipt)
        return "poll_failed"
    remote_status = parse_kernel_status(receipt["status_output"])
    receipt["remote_status"] = remote_status
    return remote_status


def collect_result(
    receipt: dict,
    receipt_path: Path,
    executable: str,
    inbox: Path,
    target: Path,
    transfer_receipt: Path,
) -> int:
    """Download, verify, promote and import the run's result tree."""
    inbox = inbox.resolve()
    inbox.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{inbox.name}.kaggle.", dir=inbox.parent))
    try:
        command = output_command(receipt["kernel"], temporary, executable)
        receipt["download_command"] = command
        receipt["status"] = "downloading"
        atomic_json(receipt_path, receipt)
        downloaded = subprocess.run(command, check=False)
        receipt["download_returncode"] = downloaded.returncode
        if downloaded.returncode != 0:
            receipt["status"] = "download_failed"
            atomic_json(receipt_path, receipt)
            return downloaded.returncode
        receipt["download"] = promote_download(temporary, inbox)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    receipt["status"] = "downloaded"
    atomic_json(receipt_path, receipt)
    transfer = import_result(inbox, target.resolve(), transfer_receipt.resolve())
    receipt["transfer"] = transfer
    receipt["status"] = "imported"
    atomic_json(receipt_path, receipt)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=Path(".moeme/kaggle-kernel.json"))
    parser.add_argument("--inbox", type=Path, default=Path("cloud-inbox/layer63-200k"))
    parser.add_argument("--target", type=Path, default=Path("cloud-results/layer63-200k"))
    parser.add_argument(
        "--sync",
        action="store_true",
        help="download and import a completed result; omission only polls status",
    )
    parser.add_argument(
        "--collect-failed",
        action="store_true",
        help=(
            "best-effort download and import of a result left by a terminated (ERROR) "
            "run, e.g. an early-stopped sparse layer that exited non-zero"
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="poll until the kernel reaches a terminal state instead of returning after one poll",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="seconds between polls in --watch mode",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help="stop after this many polls in --watch mode; 0 means keep polling",
    )
    parser.add_argument(
        "--transfer-receipt",
        type=Path,
        default=Path(".moeme/layer63-cloud-transfer.json"),
    )
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    receipt = load_launch_receipt(receipt_path)
    if receipt.get("status") == "imported" and args.target.resolve().is_dir():
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    executable = shutil.which("kaggle")
    if executable is None:
        raise FileNotFoundError("Kaggle CLI is not installed or authenticated")

    polls = 0
    while True:
        remote_status = poll_status(receipt, receipt_path, executable)
        polls += 1
        if remote_status == "poll_failed":
            return receipt["status_returncode"]
        if remote_status in IN_PROGRESS:
            receipt["status"] = remote_status.lower()
            atomic_json(receipt_path, receipt)
            if not args.watch:
                print(json.dumps(receipt, indent=2, sort_keys=True))
                return 0
            if args.max_polls and polls >= args.max_polls:
                receipt["status"] = "watch_exhausted"
                atomic_json(receipt_path, receipt)
                print(json.dumps(receipt, indent=2, sort_keys=True))
                return 3
            time.sleep(max(args.interval, 0.0))
            continue
        break

    if remote_status not in FAILED and remote_status != "COMPLETE":
        receipt["status"] = "unknown_remote_status"
        atomic_json(receipt_path, receipt)
        raise RuntimeError(f"unsupported Kaggle kernel status: {remote_status}")

    finished = remote_status == "COMPLETE"
    receipt["status"] = "remote_complete" if finished else "failed"
    if not finished:
        receipt["finished_at"] = datetime.now(UTC).isoformat()
    collect = should_collect(
        remote_status,
        sync=args.sync,
        collect_failed=args.collect_failed,
    )
    if not collect:
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if finished else 2

    try:
        code = collect_result(
            receipt,
            receipt_path,
            executable,
            args.inbox,
            args.target,
            args.transfer_receipt,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        # Best-effort: a terminated run may publish no result tree at all, and the
        # download can fail for provider-side reasons. Both are recorded, not raised.
        receipt["status"] = "collection_failed" if finished else "failed"
        receipt["collection_error"] = f"{type(error).__name__}: {error}"
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    if code != 0:
        return code
    receipt["finished_at"] = datetime.now(UTC).isoformat()
    if not finished:
        receipt["status"] = "imported_after_failure"
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if finished else 2


if __name__ == "__main__":
    raise SystemExit(main())
