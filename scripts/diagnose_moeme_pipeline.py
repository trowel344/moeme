#!/usr/bin/env python3
"""Report the first authoritative MoEMe pipeline gate needing attention."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.moeme_status import atomic_text, build_status
except ModuleNotFoundError:
    from moeme_status import atomic_text, build_status


def stage(name: str, status: str, evidence: str, action: str | None = None, **details) -> dict:
    value = {"name": name, "status": status, "evidence": evidence}
    if action is not None:
        value["action"] = action
    if details:
        value["details"] = details
    return value


def diagnose(status: dict[str, Any]) -> dict[str, Any]:
    free_cloud = status.get("free_cloud") or {}
    capture = free_cloud.get("activation_capture") or {}
    capture_service = capture.get("service") or {}
    stages = []
    if capture.get("ready"):
        capture_state = "passed"
        capture_action = None
    elif capture_service.get("running"):
        capture_state = "running"
        capture_action = "wait for the live capture service; do not start competing GPU work"
    elif capture_service.get("active_state") == "failed" or capture_service.get("result") not in {
        None,
        "success",
    }:
        capture_state = "failed"
        capture_action = "inspect the capture service and v3 log; do not restart automatically"
    else:
        capture_state = "pending"
        capture_action = "start capture only after confirming no authoritative service exists"
    stages.append(
        stage(
            "activation_capture",
            capture_state,
            ".moeme/layer63-200k-capture-v3.log",
            capture_action,
            progress=capture.get("progress"),
        )
    )

    postcapture = free_cloud.get("postcapture_supervisor") or {}
    post_status = postcapture.get("status")
    if post_status == "passed":
        normalized_post = "passed"
        post_action = None
    elif post_status == "failed":
        normalized_post = "failed"
        post_action = "inspect the named failed post-capture stage and its dedicated log"
    elif capture_state == "running":
        normalized_post = "pending"
        post_action = "leave the deterministic supervisor waiting on capture"
    elif post_status in {"running", "waiting"}:
        normalized_post = "running"
        post_action = "let the supervisor finish its current stage without duplication"
    else:
        normalized_post = "pending"
        post_action = "start only the authoritative post-capture supervisor"
    stages.append(
        stage(
            "postcapture",
            normalized_post,
            ".moeme/layer63-postcapture-v1.json",
            post_action,
            current_stage=postcapture.get("current_stage"),
            error=postcapture.get("error"),
        )
    )

    upload = free_cloud.get("upload_stage") or {}
    if upload.get("ready"):
        upload_state = "passed"
        upload_action = None
    elif normalized_post == "failed":
        upload_state = "blocked_upstream"
        upload_action = "repair the failed post-capture gate before rebuilding the stage"
    else:
        upload_state = "pending"
        upload_action = "allow the post-capture supervisor to build and verify the stage"
    stages.append(
        stage(
            "upload_stage",
            upload_state,
            "cloud-jobs/layer63-200k/upload-manifest.json",
            upload_action,
        )
    )

    smoke = free_cloud.get("local_smoke") or {}
    smoke_run = smoke.get("run") or {}
    if smoke_run.get("status") == "passed":
        smoke_state = "passed"
        smoke_action = None
    elif smoke_run.get("status") == "interrupted":
        smoke_state = "interrupted"
        smoke_action = "allow the supervisor to resume the exact atomic smoke state"
    elif smoke_run.get("status") == "failed":
        smoke_state = "failed"
        smoke_action = "inspect the smoke receipt; terminal failures are not retried"
    else:
        smoke_state = "pending"
        smoke_action = "allow the post-capture supervisor to run the single seeded smoke"
    stages.append(
        stage(
            "local_seeded_smoke",
            smoke_state,
            "cloud-results/local-seeded-smoke/cloud-run.json",
            smoke_action,
        )
    )

    training = free_cloud.get("training") or {}
    budget = free_cloud.get("free_tier_budget") or {}
    upload_transfer = free_cloud.get("upload_transfer") or {}
    kaggle_kernel = free_cloud.get("kaggle_kernel") or {}
    cloud_run = training.get("run") or {}
    run_status = cloud_run.get("status")
    if run_status == "passed":
        training_state = "passed"
        training_action = None
    elif run_status == "interrupted":
        training_state = "interrupted"
        training_action = "resume with cloud_bootstrap.py using the preserved exact state"
    elif run_status == "failed":
        training_state = "failed"
        training_action = "inspect cloud-run stage and training.log; do not retry unchanged"
    elif kaggle_kernel.get("status") in {"submitted", "queued", "running"}:
        training_state = "running"
        training_action = "poll only the receipted Kaggle kernel with sync_kaggle_kernel.py"
    elif kaggle_kernel.get("status") in {"remote_complete", "downloading", "downloaded"}:
        training_state = "running"
        training_action = "run sync_kaggle_kernel.py --sync to verify and import the result"
    elif kaggle_kernel.get("status") in {"failed", "poll_failed", "download_failed"}:
        training_state = "failed"
        training_action = "inspect the Kaggle kernel receipt; do not submit another version"
    elif upload.get("ready"):
        training_state = "user_action_required"
        provider = budget.get("recommended_provider")
        destination = f" to {provider}" if provider else ""
        if upload_transfer.get("status") == "passed":
            uploaded_provider = upload_transfer.get("provider") or provider or "cloud"
            if uploaded_provider == "kaggle":
                training_action = (
                    "submit the hash-bound private T4 job with launch_kaggle_kernel.py"
                )
            else:
                training_action = (
                    f"open the receipted private {uploaded_provider} destination, follow RUN.md "
                    "to run cloud_bootstrap.py, then download the result to cloud-inbox"
                )
        else:
            training_action = (
                f"privately upload the verified stage{destination}, then launch the "
                "receipted private GPU job"
            )
    else:
        training_state = "pending"
        training_action = "finish the local upload stage first"
    stages.append(
        stage(
            "cloud_training",
            training_state,
            "cloud-results/layer63-200k/cloud-run.json",
            training_action,
            stage=cloud_run.get("stage"),
            attempt=cloud_run.get("attempt"),
            recommended_provider=budget.get("recommended_provider"),
            measured_storage_fit=budget.get("passed"),
            upload_status=upload_transfer.get("status"),
            upload_destination=upload_transfer.get("destination"),
            kaggle_kernel_status=kaggle_kernel.get("status"),
            kaggle_remote_status=kaggle_kernel.get("remote_status"),
        )
    )

    curve = training.get("curve_analysis") or {}
    curve_status = curve.get("status")
    if curve_status in {"promising", "target_reached"}:
        curve_state = "passed"
        curve_action = None
    elif curve_status:
        curve_state = "rejected"
        curve_action = "retain the evidence and revise the scientific recipe before escalation"
    else:
        curve_state = "pending"
        curve_action = "wait for a completed cloud result and held-out curve analysis"
    stages.append(
        stage(
            "learning_curve",
            curve_state,
            "cloud-results/layer63-200k/curve-analysis.json",
            curve_action,
            scientific_status=curve_status,
            best=(curve.get("best") or {}).get("relative_l2"),
        )
    )

    transfer = training.get("transfer_verification") or {}
    if transfer.get("candidate_ready") is True:
        transfer_state = "passed"
        transfer_action = None
    elif transfer:
        transfer_state = "rejected"
        transfer_action = "retain the verified result; revise the recipe before candidate assembly"
    else:
        transfer_state = "pending"
        transfer_action = (
            "place the downloaded result folder or ZIP in cloud-inbox for atomic import"
        )
    stages.append(
        stage(
            "result_transfer",
            transfer_state,
            ".moeme/layer63-cloud-transfer.json",
            transfer_action,
            scientific_eligibility=transfer.get("scientific_eligibility"),
        )
    )

    acceptance = free_cloud.get("acceptance") or {}
    acceptance_service = free_cloud.get("acceptance_service") or {}
    if acceptance.get("passed") is True:
        acceptance_state = "passed"
        acceptance_action = None
    elif acceptance.get("last_error"):
        acceptance_state = "failed"
        acceptance_action = "inspect the first failed candidate phase; do not weaken its gate"
    elif acceptance_service.get("running") or acceptance.get("current_phase"):
        acceptance_state = "running"
        acceptance_action = "allow the resumable candidate pipeline to finish the current phase"
    else:
        acceptance_state = "pending"
        acceptance_action = "run candidate assembly only after verified eligible transfer"
    stages.append(
        stage(
            "candidate_acceptance",
            acceptance_state,
            ".moeme/layer63-candidate-pipeline.json",
            acceptance_action,
            current_phase=acceptance.get("current_phase"),
            error=acceptance.get("last_error"),
        )
    )

    first = next((item for item in stages if item["status"] != "passed"), None)
    return {
        "format": "moeme-pipeline-diagnosis-v1",
        "diagnosed_at": datetime.now(UTC).isoformat(),
        "overall": "passed" if first is None else first["status"],
        "current_gate": first["name"] if first else "complete",
        "next_action": first.get("action") if first else None,
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(build_status(args.root.resolve()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        atomic_text(args.output, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
