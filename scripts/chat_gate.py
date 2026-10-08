#!/usr/bin/env python3
"""Run one deterministic chat-quality gate and persist its evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import time
from pathlib import Path

from moeme.ledger import ExperimentLedger


def _run_bounded(command: list[str], timeout: float, max_output: int = 4 * 1024 * 1024):
    """Run a noisy CLI while retaining only the tail of stdout."""

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if process.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    descriptor = process.stdout.fileno()
    output = bytearray()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(command, timeout)
        ready, _, _ = select.select([descriptor], [], [], min(1.0, remaining))
        if ready:
            chunk = os.read(descriptor, 256 * 1024)
            if chunk:
                output.extend(chunk)
                if len(output) > max_output:
                    del output[: len(output) - max_output]
            elif process.poll() is not None:
                break
        elif process.poll() is not None:
            chunk = os.read(descriptor, 256 * 1024)
            if chunk:
                output.extend(chunk)
                if len(output) > max_output:
                    del output[: len(output) - max_output]
            else:
                break
    return subprocess.CompletedProcess(
        command,
        process.wait(),
        output.decode("utf-8", errors="replace"),
        "",
    )


def _timing(log: str, *patterns: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, log)
        if match:
            return float(match.group(1))
    return None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifact-sha", required=True)
    parser.add_argument("--stage", default="runtime-chat-factual")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument(
        "--raw", action="store_true", help="use raw completion, not chat templating"
    )
    parser.add_argument("--predict", type=int, default=16)
    parser.add_argument(
        "--prompt", default="What is the capital of France? Answer with only the city."
    )
    parser.add_argument("--expected", default="Paris")
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ledger = ExperimentLedger(args.ledger)
    configuration = {
        "model": str(args.model.resolve()),
        "binary": str(args.binary.resolve()),
        "backend": args.backend,
        "gpu_layers": args.gpu_layers,
        "prompt": args.prompt,
        "expected": args.expected,
        "protocol": "raw" if args.raw else "chat",
        "reasoning": "off",
        "temperature": 0,
        "seed": 1,
    }
    run_id = ledger.start(
        args.stage,
        f"{args.artifact_sha}-{args.backend}-{'raw' if args.raw else 'chat'}-v1",
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
        str(args.predict),
        "--temp",
        "0",
        "--seed",
        "1",
        "-p",
        args.prompt,
        "--no-display-prompt",
        "--no-warmup",
        "--log-disable",
    ]
    if args.raw:
        command.append("-no-cnv")
    else:
        command.extend(
            [
                "-cnv",
                "-st",
                "--simple-io",
                "--reasoning",
                "off",
                "--reasoning-budget",
                "0",
                "--reasoning-format",
                "none",
            ]
        )
    if args.gpu_layers:
        command.extend(["-ngl", str(args.gpu_layers)])

    started = time.monotonic()
    try:
        process = _run_bounded(command, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as error:
        summary = {
            "failure_boundary": "runtime-infrastructure",
            "error": f"{type(error).__name__}: {error}",
            "wall_seconds": time.monotonic() - started,
        }
        ledger.finish(run_id, "failed", summary, error="chat runtime could not complete")
        _atomic_json(args.report, summary)
        return 1

    merged = process.stdout
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(merged)
    visible = process.stdout.strip()
    correct = args.expected.casefold() in visible.casefold()
    prompt_tps = _timing(
        merged,
        r"prompt eval time.*?([0-9.]+) tokens per second",
        r"Prompt:\s*([0-9.]+) t/s",
    )
    decode_tps = _timing(
        merged,
        r"(?<!prompt )eval time.*?([0-9.]+) tokens per second",
        r"Generation:\s*([0-9.]+) t/s",
    )
    summary = {
        "returncode": process.returncode,
        "visible_output": visible,
        "correct": correct,
        "wall_seconds": time.monotonic() - started,
        "prompt_tokens_per_second": prompt_tps,
        "decode_tokens_per_second": decode_tps,
    }
    ledger.artifact(run_id, "runtime-log", args.log, bytes_count=args.log.stat().st_size)
    ledger.metric(
        run_id,
        "factual-smoke",
        "correct_answer",
        float(correct),
        "boolean",
        gate=f"contains {args.expected}",
        passed=correct,
    )
    status = "passed" if process.returncode == 0 and correct else "failed"
    ledger.finish(
        run_id,
        status,
        summary,
        error=None if status == "passed" else "deterministic factual chat smoke failed",
    )
    _atomic_json(args.report, summary)
    print(json.dumps(summary, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
