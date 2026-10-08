#!/usr/bin/env python3
"""Validate and smoke-test the pinned dense reference GGUF.

This is intentionally a standalone campaign step: it can run under systemd
after the download unit, update the experiment ledger, and leave compact JSON
and text receipts without requiring an interactive Codex turn.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gguf import GGUFReader

from moeme.ledger import ExperimentLedger


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _timings(log: str) -> dict[str, float]:
    patterns = {
        "load_ms": r"load time\s*=\s*([0-9.]+) ms",
        "prompt_tokens_per_second": (
            r"(?:prompt eval time.*?([0-9.]+) tokens per second|Prompt:\s*([0-9.]+) t/s)"
        ),
        "decode_tokens_per_second": (
            r"(?:eval time.*?([0-9.]+) tokens per second|Generation:\s*([0-9.]+) t/s)"
        ),
    }
    values: dict[str, float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, log)
        if match:
            values[name] = float(next(group for group in match.groups() if group is not None))
    return values


def _finish_running_download(ledger: ExperimentLedger, summary: dict[str, Any]) -> None:
    run = next(
        (
            item
            for item in ledger.history()
            if item.stage == "reference-download" and item.status == "running"
        ),
        None,
    )
    if run is not None:
        ledger.artifact(
            run.id,
            "dense-reference-gguf",
            summary["path"],
            sha256=summary["sha256"],
            bytes_count=summary["bytes"],
        )
        ledger.metric(
            run.id,
            "format",
            "exact_size",
            summary["bytes"],
            "bytes",
            gate=f"=={summary['expected_bytes']}",
            passed=summary["bytes"] == summary["expected_bytes"],
        )
        ledger.finish(run.id, "passed", summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    parser.add_argument("--report", type=Path, default=Path(".moeme/reference-gate.json"))
    parser.add_argument("--log", type=Path, default=Path(".moeme/reference-chat.log"))
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--gpu-layers", type=int, default=0)
    args = parser.parse_args()

    ledger = ExperimentLedger(args.ledger)
    size = args.model.stat().st_size
    resolved_model = str(args.model.resolve())
    cached = next(
        (
            item.summary
            for item in ledger.history()
            if item.stage == "reference-download"
            and item.status == "passed"
            and item.summary.get("path") == resolved_model
            and item.summary.get("bytes") == size
            and item.summary.get("expected_bytes") == args.expected_bytes
            and item.summary.get("structurally_complete") is True
        ),
        None,
    )
    if cached is not None:
        validation = cached
        data_end = int(validation["data_end"])
    else:
        reader = GGUFReader(args.model)
        last = reader.tensors[-1]
        data_end = int(last.data_offset + last.data.nbytes)
        validation = {
            "path": resolved_model,
            "bytes": size,
            "expected_bytes": args.expected_bytes,
            "sha256": _sha256(args.model),
            "tensors": len(reader.tensors),
            "data_end": data_end,
            "structurally_complete": data_end == size,
        }
        # GGUFReader owns numpy memmaps. Release them before loading a 19 GB
        # model, or validator and runtime pages can overlap enough to OOM.
        del last, reader
        gc.collect()
    if size != args.expected_bytes or data_end != size:
        run = next(
            (
                item
                for item in ledger.history()
                if item.stage == "reference-download" and item.status == "running"
            ),
            None,
        )
        if run is not None:
            ledger.finish(run.id, "failed", validation, error="reference GGUF validation failed")
        _atomic_json(args.report, {"validation": validation, "quality": None})
        return 1
    _finish_running_download(ledger, validation)

    configuration = {
        "model": str(args.model.resolve()),
        "binary": str(args.binary.resolve()),
        "prompt": "What is the capital of France? Answer with only the city.",
        "reasoning": "off",
        "temperature": 0,
        "seed": 1,
        "gpu_layers": args.gpu_layers,
    }
    run_id = ledger.start(
        "reference-chat-factual",
        f"{validation['sha256']}-france-chat-v1",
        configuration,
    )
    command = [
        str(args.binary),
        "-m",
        str(args.model),
        "-c",
        "256",
        "-b",
        "64",
        "-t",
        "8",
        "-tb",
        "8",
        "-n",
        "16",
        "--temp",
        "0",
        "--seed",
        "1",
        "-p",
        configuration["prompt"],
        "-cnv",
        "-st",
        "--simple-io",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
        "--no-display-prompt",
        "--no-warmup",
    ]
    if args.gpu_layers:
        command.extend(["-ngl", str(args.gpu_layers)])
    started = time.monotonic()
    try:
        process = subprocess.run(command, text=True, capture_output=True, timeout=900, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        failure = {
            "failure_boundary": "runtime-infrastructure",
            "error": f"{type(error).__name__}: {error}",
            "wall_seconds": time.monotonic() - started,
        }
        ledger.finish(
            run_id,
            "failed",
            failure,
            error="dense reference runtime could not complete",
        )
        _atomic_json(args.report, {"validation": validation, "quality": failure})
        print(json.dumps({"validation": validation, "quality": failure}, indent=2))
        return 1
    elapsed = time.monotonic() - started
    merged = process.stdout + "\n" + process.stderr
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(merged)
    visible = process.stdout.strip()
    correct = "paris" in visible.lower()
    quality = {
        "returncode": process.returncode,
        "visible_output": visible,
        "correct": correct,
        "wall_seconds": elapsed,
        **_timings(merged),
    }
    ledger.artifact(run_id, "runtime-log", args.log, bytes_count=args.log.stat().st_size)
    ledger.metric(
        run_id,
        "factual-smoke",
        "correct_answer",
        float(correct),
        "boolean",
        gate="contains Paris",
        passed=correct,
    )
    status = "passed" if process.returncode == 0 and correct else "failed"
    error = None if status == "passed" else "dense reference factual chat smoke failed"
    ledger.finish(run_id, status, quality, error=error)
    _atomic_json(args.report, {"validation": validation, "quality": quality})
    print(json.dumps({"validation": validation, "quality": quality}, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Preserve a traceback in the supervising unit's log.
        print(f"reference gate crashed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
