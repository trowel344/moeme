#!/usr/bin/env python3
"""Run a resumable sequence of independent sparse-layer distillations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from moeme.ledger import ExperimentLedger


def atomic_json(path: Path, value: dict) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()

    configuration = {
        "activations": str(args.activations.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "layers": args.layers,
        "output_dir": str(args.output_dir.resolve()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "recipe": "importance-shared-random-generalist-all-projections-and-shared-v2-router-aux",
    }
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output_dir / "campaign.json"
    if manifest_path.exists():
        campaign = json.loads(manifest_path.read_text())
        if campaign["configuration_digest"] != digest:
            raise ValueError("existing campaign configuration does not match this invocation")
    else:
        campaign = {
            "format": "moeme-training-campaign-v1",
            "configuration": configuration,
            "configuration_digest": digest,
            "layers": {},
            "started_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(manifest_path, campaign)

    ledger = ExperimentLedger(args.ledger)
    run_id = ledger.start("sparse-training-campaign", digest, configuration)
    try:
        for layer in args.layers:
            key = str(layer)
            existing = campaign["layers"].get(key)
            if existing and existing["status"] in ("passed", "gate_failed"):
                checkpoint_path = Path(existing["checkpoint"])
                report_path = Path(existing["report"])
                if checkpoint_path.exists() and report_path.exists():
                    continue
                raise FileNotFoundError(f"completed layer {layer} is missing retained artifacts")

            layer_dir = args.output_dir / f"layer-{layer}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            log_path = layer_dir / "train.log"
            command = [
                sys.executable,
                "scripts/train_sparse_layer.py",
                "--activations",
                str(args.activations),
                "--checkpoint",
                str(args.checkpoint),
                "--layer",
                str(layer),
                "--output-dir",
                str(layer_dir),
                "--partition-mode",
                "importance_shared",
                "--top-k-schedule",
                "4",
                "--steps-per-stage",
                str(args.steps),
                "--router-warmup-steps",
                "0",
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                "3e-5",
                "--feature-learning-rate",
                "1e-5",
                "--train-projections",
                "all",
                "--train-shared",
                "--routing-strategy",
                "random-generalist",
                "--checkpoint-every",
                "0",
                "--device",
                "cuda",
            ]
            environment = os.environ.copy()
            environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            environment["PYTHONPATH"] = "src"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            report_path = layer_dir / "report.json"
            checkpoint_path = layer_dir / f"layer-{layer}-top4.safetensors"
            if completed.returncode not in (0, 1) or not report_path.exists():
                raise RuntimeError(
                    f"layer {layer} process failed with return code {completed.returncode}"
                )
            report = json.loads(report_path.read_text())
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"layer {layer} did not retain its final checkpoint")
            status = "passed" if report["passed"] else "gate_failed"
            campaign["layers"][key] = {
                "status": status,
                "checkpoint": str(checkpoint_path.resolve()),
                "report": str(report_path.resolve()),
                "log": str(log_path.resolve()),
                "validation": report["stages"][-1]["validation"],
                "finished_at": datetime.now(UTC).isoformat(),
            }
            campaign["updated_at"] = datetime.now(UTC).isoformat()
            atomic_json(manifest_path, campaign)
            ledger.artifact(
                run_id,
                f"layer-{layer}-checkpoint",
                checkpoint_path,
                bytes_count=checkpoint_path.stat().st_size,
            )

        summary = {
            "manifest": str(manifest_path.resolve()),
            "layers_completed": len(campaign["layers"]),
            "layers_requested": len(args.layers),
            "gate_failures": sum(
                value["status"] == "gate_failed" for value in campaign["layers"].values()
            ),
        }
        campaign["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(manifest_path, campaign)
        ledger.artifact(run_id, "training-campaign-manifest", manifest_path)
        ledger.finish(run_id, "passed", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        ledger.finish(run_id, "failed", {}, error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
