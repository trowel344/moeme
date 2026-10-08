#!/usr/bin/env python3
"""Measure served decode throughput for a list of llama-server placements.

The shipped host cannot hold the artifact in RAM, so decode throughput is set by
which bytes are resident and where they are read from, not by FLOPs. That makes
placement (-ngl, --n-cpu-moe) and the compute-buffer batch sizes (-b, -ub) the
levers worth measuring, and it makes single-shot numbers useless: the same config
moves by more than 10% across sessions as the page cache warms.

One llama-server process is used per configuration, because this build aborts
(`CUDA error` inside `process_ubatch`) when a second model is loaded in an
already-used process. Each configuration is spawned, waited for, measured with
repeated requests, and torn down before the next one starts.

Every request disables the prompt cache so each sample replays the same prefill
and the decode figures are independent rather than cache-warmed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BINARY = Path("/home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-server")
DEFAULT_MODEL = Path("artifacts/gguf/moeme-27b-top12-imatrix-q5_k_m.gguf")
DEFAULT_PROMPT_SOURCE = Path("data/calibration/router-v1.txt")
DEFAULT_REPORT = Path(".moeme/serving-sweep.json")

DEFAULT_SWEEP: list[dict[str, Any]] = [
    {"name": "anchor-c512-defaults", "context": 512},
    {"name": "c1024-b256-ub64-ngl20", "context": 1024, "batch": 256, "ubatch": 64},
    {
        "name": "c1024-b256-ub64-ngl28",
        "context": 1024,
        "batch": 256,
        "ubatch": 64,
        "gpu_layers": 28,
    },
    {
        "name": "c1024-b256-ub64-ngl36",
        "context": 1024,
        "batch": 256,
        "ubatch": 64,
        "gpu_layers": 36,
    },
    {
        "name": "c1024-b256-ub64-ngl99-cmoe-all",
        "context": 1024,
        "batch": 256,
        "ubatch": 64,
        "gpu_layers": 99,
        "n_cpu_moe": 64,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def prompt_text(path: Path, characters: int) -> str:
    """Slice a fixed prefix of the calibration corpus.

    The same string is used for every configuration, so every configuration sees
    the same token count for a given tokenizer.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:characters]


def server_command(
    binary: Path,
    model: Path,
    config: dict[str, Any],
    port: int,
    threads: int,
) -> list[str]:
    command = [
        str(binary),
        "-m",
        str(model),
        "-c",
        str(config["context"]),
        "-t",
        str(threads),
        "-tb",
        str(threads),
        "--fit",
        "off",
        "-ngl",
        str(config.get("gpu_layers", 20)),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--parallel",
        "1",
        "--cache-ram",
        "0",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
    ]
    if config.get("batch"):
        command += ["-b", str(config["batch"])]
    if config.get("ubatch"):
        command += ["-ub", str(config["ubatch"])]
    if config.get("cpu_moe"):
        command.append("--cpu-moe")
    if config.get("n_cpu_moe") is not None:
        command += ["--n-cpu-moe", str(config["n_cpu_moe"])]
    command += [str(item) for item in config.get("extra", [])]
    return command


def post_completion(port: int, prompt: str, predict: int, timeout: float) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "n_predict": predict,
            "temperature": 0.0,
            "cache_prompt": False,
            "ignore_eos": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_for_health(port: int, process: subprocess.Popen[bytes], deadline: float) -> str:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return f"server exited early with rc={process.returncode}"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                body = json.loads(response.read().decode())
            if body.get("status") == "ok":
                return "ok"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            pass
        time.sleep(2)
    return "health check timed out"


def log_tail(path: Path, lines: int = 12) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def measure(
    binary: Path,
    model: Path,
    config: dict[str, Any],
    port: int,
    threads: int,
    prompt: str,
    reps: int,
    predict: int,
    load_timeout: float,
    log_dir: Path,
) -> dict[str, Any]:
    name = config["name"]
    log_path = log_dir / f"{name}.log"
    command = server_command(binary, Path(config.get("model", model)), config, port, threads)
    started = time.monotonic()
    with log_path.open("wb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    result: dict[str, Any] = {
        "name": name,
        "command": command,
        "context": config["context"],
        "gpu_layers": config.get("gpu_layers", 20),
        "batch": config.get("batch", 2048),
        "ubatch": config.get("ubatch", 512),
        "n_cpu_moe": config.get("n_cpu_moe"),
        "extra": config.get("extra", []),
        "model": str(config.get("model", model)),
        "log": str(log_path),
    }
    try:
        health = wait_for_health(port, process, time.monotonic() + load_timeout)
        result["load_seconds"] = round(time.monotonic() - started, 1)
        if health != "ok":
            result["status"] = "failed"
            result["error"] = health
            return result
        samples = []
        prefill = []
        prompt_tokens = []
        for index in range(reps):
            response = post_completion(port, prompt, predict, timeout=900)
            timings = response.get("timings") or {}
            samples.append(round(float(timings.get("predicted_per_second", 0.0)), 3))
            prefill.append(round(float(timings.get("prompt_per_second", 0.0)), 1))
            prompt_tokens.append(int(timings.get("prompt_n", 0)))
            if index == 0:
                result["first_sample_text"] = str(response.get("content", ""))[:160]
        result["status"] = "passed"
        result["decode_samples"] = samples
        result["prefill_samples"] = prefill
        result["prompt_tokens"] = prompt_tokens
        result["median_decode_tps"] = round(sorted(samples)[len(samples) // 2], 3)
        return result
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        return result
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if result.get("status") != "passed":
            result["log_tail"] = log_tail(log_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-dir", type=Path, default=Path(".moeme/serving-sweep-logs"))
    parser.add_argument("--prompt-source", type=Path, default=DEFAULT_PROMPT_SOURCE)
    parser.add_argument("--prompt-characters", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--port", type=int, default=18099)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--predict", type=int, default=96)
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument(
        "--expected-model-sha256",
        help="Refuse to benchmark unless the model matches this SHA-256.",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        help="JSON file holding the sweep; defaults to the built-in placement sweep",
    )
    args = parser.parse_args()

    model_sha256 = None
    if args.expected_model_sha256 is not None:
        model_sha256 = sha256(args.model)
        if model_sha256 != args.expected_model_sha256:
            raise ValueError("model SHA-256 does not match --expected-model-sha256")

    configs = json.loads(args.configs.read_text()) if args.configs is not None else DEFAULT_SWEEP
    prompt = prompt_text(args.prompt_source, args.prompt_characters)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "format": "moeme-serving-sweep-v1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(args.model),
        "model_sha256": model_sha256,
        "binary": str(args.binary),
        "threads": args.threads,
        "reps": args.reps,
        "predict": args.predict,
        "prompt_characters": args.prompt_characters,
        "configs": [],
    }
    for config in configs:
        print(f"== {config['name']}", flush=True)
        measured = measure(
            args.binary,
            args.model,
            config,
            args.port,
            args.threads,
            prompt,
            args.reps,
            args.predict,
            args.load_timeout,
            args.log_dir,
        )
        receipt["configs"].append(measured)
        print(json.dumps({k: measured[k] for k in measured if k not in {"command"}}), flush=True)
        atomic_text(args.report, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_text(args.report, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
