#!/usr/bin/env python3
"""Start candidate acceptance once after an eligible verified cloud transfer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def read_object(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


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


def service_snapshot(unit: str) -> dict:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "Result",
            "--no-pager",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    values = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    values["query_returncode"] = completed.returncode
    return values


def start_decision(
    transfer: dict | None,
    launch: dict | None,
    acceptance: dict | None,
    service: dict,
) -> dict:
    if acceptance and acceptance.get("passed") is True:
        return {"status": "complete", "start": False}
    if (
        service.get("ActiveState") in {"active", "activating"}
        and int(service.get("MainPID", "0") or 0) > 0
    ):
        return {"status": "running", "start": False}
    if launch and launch.get("status") == "failed":
        return {
            "status": "refused",
            "start": False,
            "reason": f"previous candidate launch is {launch['status']}; inspect before resume",
        }
    failed_phases = sorted(
        name
        for name, phase in ((acceptance or {}).get("phases") or {}).items()
        if isinstance(phase, dict) and phase.get("status") == "failed"
    )
    if failed_phases and (launch or {}).get("status") != "interrupted":
        return {
            "status": "refused",
            "start": False,
            "reason": (
                "candidate phase failure lacks an interruption receipt; inspect before resume: "
                + ", ".join(failed_phases)
            ),
        }
    if not transfer:
        return {"status": "not_ready", "start": False, "reason": "no transfer receipt"}
    if transfer.get("candidate_ready") is not True:
        return {
            "status": "refused",
            "start": False,
            "reason": "cloud transfer is intact but not scientifically eligible",
        }
    return {"status": "ready", "start": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--unit", default="moeme-layer63-candidate-v1.service")
    parser.add_argument(
        "--receipt", type=Path, default=Path(".moeme/layer63-candidate-service.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = args.receipt if args.receipt.is_absolute() else root / args.receipt
    decision = start_decision(
        read_object(root / ".moeme/layer63-cloud-transfer.json"),
        read_object(root / ".moeme/layer63-candidate-launch.json"),
        read_object(root / ".moeme/layer63-candidate-pipeline.json"),
        service_snapshot(args.unit),
    )
    result = {
        "format": "moeme-sparse-candidate-service-v1",
        "checked_at": datetime.now(UTC).isoformat(),
        "unit": args.unit,
        **decision,
    }
    if not decision["start"]:
        atomic_json(receipt, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if decision["status"] in {"complete", "running", "not_ready"} else 2

    command = [
        "systemd-run",
        "--user",
        f"--unit={args.unit.removesuffix('.service')}",
        "--collect",
        "--property=Type=exec",
        f"--property=WorkingDirectory={root}",
        f"--setenv=PYTHONPATH={root}:{root / 'src'}",
        "/usr/bin/python3",
        "scripts/launch_sparse_candidate.py",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result.update(
        {
            "status": "started" if completed.returncode == 0 else "failed_to_start",
            "start_command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    )
    atomic_json(receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
