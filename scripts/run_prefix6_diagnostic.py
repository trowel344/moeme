#!/usr/bin/env python3
"""Pause all-64 training at layer 5 and qualify a corrected prefix-6 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.run_all64_pipeline import (
        atomic_json,
        begin_phase,
        fail_phase,
        pass_phase,
        run_logged,
        sha256,
        wait_healthy,
    )
except ModuleNotFoundError:
    from run_all64_pipeline import (
        atomic_json,
        begin_phase,
        fail_phase,
        pass_phase,
        run_logged,
        sha256,
        wait_healthy,
    )

LAYERS = tuple(range(6))
MAIN_UNIT = "moeme-all64-pipeline.service"


def checkpoint_ready(training_dir: Path, layer: int) -> bool:
    progress_path = training_dir / f"layer-{layer}" / "progress.json"
    report_path = training_dir / f"layer-{layer}" / "report.json"
    checkpoint_path = training_dir / f"layer-{layer}" / f"layer-{layer}-top4.safetensors"
    try:
        progress = json.loads(progress_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return progress.get("phase") == "complete" and report_path.exists() and checkpoint_path.exists()


def wait_for_checkpoint(training_dir: Path, layer: int, poll_seconds: float) -> None:
    while not checkpoint_ready(training_dir, layer):
        progress = training_dir / f"layer-{layer}" / "progress.json"
        if progress.exists():
            value = json.loads(progress.read_text())
            if value.get("phase") == "failed":
                raise RuntimeError(f"layer {layer} failed before the diagnostic boundary")
        time.sleep(poll_seconds)


def recoverable_evaluation_report(state: dict, report_path: Path) -> dict | None:
    phase = state.get("phases", {}).get("evaluation", {})
    if phase.get("status") != "failed" or not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text())
        started_at = datetime.fromisoformat(phase["started_at"]).timestamp()
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
    if report_path.stat().st_mtime < started_at:
        return None
    required = ("passed", "quality_passed", "stability_passed", "performance_passed")
    return report if all(report.get(key) is True for key in required) else None


def stop_main_pipeline() -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", MAIN_UNIT],
        check=True,
        timeout=90,
    )


def main_pipeline_command(root: Path) -> list[str]:
    return [
        "/usr/bin/python3",
        "scripts/run_all64_pipeline.py",
        "--activations",
        "artifacts/activations/dense-q4-router-v3",
        "--partition-checkpoint",
        "artifacts/moeme-27b-initial",
        "--source-bf16",
        "artifacts/gguf/moeme-27b-top12-bf16.gguf",
        "--diagnostic-q4",
        "artifacts/gguf/moeme-27b-hybrid-representative5-q4_k_m.gguf",
        "--diagnostic-q4-sha256",
        "44d758929f86d2f249bb3abac812a6f74d90967633548732b5edff4fe67cb06a",
        "--training-dir",
        "artifacts/router-training/all64-v2",
        "--working-bf16",
        "artifacts/gguf/moeme-27b-all64-top4-bf16.gguf",
        "--output-q4",
        "artifacts/gguf/moeme-27b-all64-top4-q4_k_m.gguf",
        "--manifest",
        ".moeme/all64-pipeline-v3.json",
        "--report",
        ".moeme/server-eval-all64-top4.json",
        "--llama-bin-dir",
        "/home/cleanerbox/.cache/moeme/llama-nocache/build/bin",
        "--steps",
        "2000",
        "--batch-size",
        "4",
        "--server-context",
        "8192",
    ]


def resume_main_pipeline(root: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = subprocess.run(
            ["systemctl", "--user", "show", MAIN_UNIT, "-p", "LoadState", "--value"],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip() in ("", "not-found"):
            break
        time.sleep(1)
    command = [
        "systemd-run",
        "--user",
        "--unit=moeme-all64-pipeline",
        "--collect",
        f"--property=WorkingDirectory={root}",
        *main_pipeline_command(root),
    ]
    subprocess.run(command, check=True, timeout=30)
    time.sleep(2)
    active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", MAIN_UNIT], check=False)
    if active.returncode != 0:
        raise RuntimeError("all-64 pipeline did not become active after diagnostic resume")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir(root)

    training_dir = root / "artifacts/router-training/all64-v2"
    source = root / "artifacts/gguf/moeme-27b-top12-bf16.gguf"
    working = root / "artifacts/gguf/moeme-27b-corrected-prefix6-bf16.gguf"
    output = root / "artifacts/gguf/moeme-27b-corrected-prefix6-q4_k_m.gguf"
    manifest = root / ".moeme/corrected-prefix6-pipeline.json"
    server_report = root / ".moeme/server-eval-corrected-prefix6.json"
    parity_report = root / ".moeme/logit-parity-corrected-prefix6.json"
    llama_dir = Path("/home/cleanerbox/.cache/moeme/llama-nocache/build/bin")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    top4_layers = ",".join(map(str, LAYERS))

    configuration = {
        "layers": list(LAYERS),
        "source": str(source),
        "working": str(working),
        "output": str(output),
        "training_dir": str(training_dir),
        "server_context": 8192,
    }
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state["configuration_digest"] != digest:
            raise ValueError("prefix diagnostic manifest configuration changed")
    else:
        state = {
            "format": "moeme-prefix-diagnostic-v1",
            "configuration": configuration,
            "configuration_digest": digest,
            "phases": {},
            "started_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(manifest, state)

    def completed(phase: str) -> bool:
        return state["phases"].get(phase, {}).get("status") == "passed"

    try:
        if not completed("training_boundary"):
            begin_phase(manifest, state, "training_boundary", {"last_layer": LAYERS[-1]})
            wait_for_checkpoint(training_dir, LAYERS[-1], args.poll_seconds)
            stop_main_pipeline()
            pass_phase(
                manifest,
                state,
                "training_boundary",
                {"last_layer": LAYERS[-1], "main_pipeline_stopped": True},
            )

        checkpoints = {
            layer: training_dir / f"layer-{layer}" / f"layer-{layer}-top4.safetensors"
            for layer in LAYERS
        }
        if not all(path.exists() for path in checkpoints.values()):
            raise FileNotFoundError("one or more prefix checkpoints are missing")

        if not completed("injection"):
            begin_phase(manifest, state, "injection", {"layers": list(LAYERS)})
            needed = source.stat().st_size + 20_000_000_000
            if shutil.disk_usage(root).free < needed:
                raise OSError(f"prefix diagnostic needs {needed:,} free bytes")
            working.unlink(missing_ok=True)
            working.with_name(f".{working.name}.partial").unlink(missing_ok=True)
            command = [
                sys.executable,
                "scripts/patch_gguf_layer.py",
                "--source",
                str(source),
                "--output",
                str(working),
            ]
            for layer, checkpoint in checkpoints.items():
                command.extend(("--layer-checkpoint", f"{layer}={checkpoint}"))
            run_logged(command, manifest.with_suffix(".injection.log"), environment)
            pass_phase(manifest, state, "injection", {"bytes": working.stat().st_size})

        if not completed("quantization"):
            begin_phase(manifest, state, "quantization", {"type": "Q4_K_M"})
            output.unlink(missing_ok=True)
            output.with_name(f".{output.name}.partial").unlink(missing_ok=True)
            run_logged(
                [
                    sys.executable,
                    "scripts/quantize_gguf.py",
                    "--binary",
                    str(llama_dir / "llama-quantize"),
                    "--input",
                    str(working),
                    "--output",
                    str(output),
                    "--type",
                    "Q4_K_M",
                    "--threads",
                    "8",
                    "--log",
                    str(manifest.with_suffix(".quantize.log")),
                ],
                manifest.with_suffix(".quantize-wrapper.log"),
                environment,
            )
            pass_phase(
                manifest,
                state,
                "quantization",
                {"bytes": output.stat().st_size, "sha256": sha256(output)},
            )

        if not completed("bf16_cleanup"):
            begin_phase(manifest, state, "bf16_cleanup")
            removed = working.stat().st_size if working.exists() else 0
            working.unlink(missing_ok=True)
            pass_phase(manifest, state, "bf16_cleanup", {"bytes_removed": removed})

        if not completed("evaluation"):
            report = recoverable_evaluation_report(state, server_report)
            if report is None:
                begin_phase(manifest, state, "evaluation", {"context": 8192})
                server_environment = environment.copy()
                server_environment["MOEME_TOP4_LAYERS"] = top4_layers
                server_log_path = manifest.with_suffix(".server.log")
                with server_log_path.open("a", encoding="utf-8") as server_log:
                    server = subprocess.Popen(
                        [
                            str(llama_dir / "llama-server"),
                            "-m",
                            str(output),
                            "-c",
                            "8192",
                            "-t",
                            "8",
                            "-tb",
                            "8",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            "18081",
                            "--parallel",
                            "1",
                            "--cache-ram",
                            "0",
                        ],
                        env=server_environment,
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        wait_healthy("http://127.0.0.1:18081", server)
                        run_logged(
                            [
                                sys.executable,
                                "scripts/server_eval.py",
                                "--url",
                                "http://127.0.0.1:18081",
                                "--report",
                                str(server_report),
                                "--max-median-ttft",
                                "900",
                                "--max-ttft",
                                "900",
                                "--min-median-decode-rate",
                                "0",
                                "--min-long-decode-rate",
                                "0",
                            ],
                            manifest.with_suffix(".evaluation.log"),
                            environment,
                        )
                    finally:
                        server.terminate()
                        try:
                            server.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait()
                report = json.loads(server_report.read_text())
            pass_phase(
                manifest,
                state,
                "evaluation",
                {
                    **report,
                    "candidate_sha256": state["phases"]["quantization"]["sha256"],
                },
            )

        if not completed("logit_parity"):
            begin_phase(manifest, state, "logit_parity", {"layers": list(LAYERS)})
            run_logged(
                [
                    sys.executable,
                    "scripts/logit_parity_gate.py",
                    "--pipeline-manifest",
                    str(manifest),
                    "--reference",
                    "checkpoints/qwen3.8-27b-gguf-reference/Qwen3.8-27B-Q4_K_M.gguf",
                    "--candidate",
                    str(output),
                    "--corpus",
                    "data/calibration/router-v1.txt",
                    "--corpus",
                    "scripts/train_sparse_layer.py",
                    "--corpus",
                    "README.md",
                    "--binary",
                    str(llama_dir / "llama-perplexity"),
                    "--logits",
                    ".moeme/prefix6-reference-512.kld",
                    "--report",
                    str(parity_report),
                    "--reference-log",
                    ".moeme/logit-parity-prefix6-reference.log",
                    "--candidate-log",
                    ".moeme/logit-parity-prefix6-candidate.log",
                    "--context",
                    "512",
                    "--chunks",
                    "1",
                    "--top4-layers",
                    top4_layers,
                    "--max-ppl-ratio",
                    "1.01",
                    "--max-mean-kld",
                    "0.02",
                    "--min-same-top-percent",
                    "95",
                ],
                manifest.with_suffix(".parity-wrapper.log"),
                environment,
            )
            pass_phase(manifest, state, "logit_parity", json.loads(parity_report.read_text()))

        if not completed("all64_resume"):
            begin_phase(manifest, state, "all64_resume")
            resume_main_pipeline(root)
            pass_phase(manifest, state, "all64_resume", {"unit": MAIN_UNIT})

        state["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(manifest, state)
        return 0
    except BaseException as error:
        fail_phase(manifest, state, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
