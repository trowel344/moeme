#!/usr/bin/env python3
"""
bench_inference.py -- measure real quantised inference on this machine.

Records, per (model, num_gpu) configuration:
  * decode rate (generated tokens/s)   <- the number that actually matters
  * prefill rate (prompt tokens/s)
  * load time and time-to-first-token
  * peak GPU VRAM used during the run
  * minimum available system RAM during the run (i.e. thrash pressure)

Uses the ollama HTTP API rather than the interactive CLI so results are
machine-parseable and reproducible. Sampling runs on a background thread while
generation is in flight.

Usage:
    python3 tools/bench_inference.py \
        --spec qwen3:4b:-1:64 \
        --spec qwen3.8:latest:-1:8 \
        --out results/inference.json

--spec format:  model:num_gpu:num_predict
    num_gpu -1  -> let ollama decide (full offload of what fits)
    num_gpu  0  -> CPU/CUDA-free (pure RAM + disk streaming)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

API = "http://127.0.0.1:11434"


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{API}/api/version", timeout=5) as r:
            json.load(r)
        return True
    except Exception:
        return False


def ensure_server() -> bool:
    if server_up():
        return True
    print("starting ollama serve ...", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(40):
        time.sleep(0.5)
        if server_up():
            return True
    return False


def mem_available_bytes() -> int:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024
    return 0


def mem_used_bytes() -> int:
    d = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            d[k.strip()] = int(v.split()[0]) * 1024
    return d["MemTotal"] - d["MemAvailable"]


def gpu_used_mib():
    txt = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return int(txt.splitlines()[0])
    except Exception:
        return None


class Sampler(threading.Thread):
    """Poll GPU VRAM + system RAM while a generation is in flight."""

    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.peak_vram = 0
        self.peak_ram = 0
        self.min_avail = None

    def run(self) -> None:
        while not self.stop_evt.is_set():
            v = gpu_used_mib()
            if v is not None:
                self.peak_vram = max(self.peak_vram, v)
            self.peak_ram = max(self.peak_ram, mem_used_bytes())
            a = mem_available_bytes()
            self.min_avail = a if self.min_avail is None else min(self.min_avail, a)
            self.stop_evt.wait(self.interval)

    def stop(self) -> None:
        self.stop_evt.set()
        self.join(timeout=5)


PROMPT = (
    "Write a Python function that merges two sorted lists into one sorted list "
    "in O(n) time. Then explain its time and space complexity in two sentences."
)


def run_one(model: str, num_gpu: int, num_predict: int,
            timeout_s: float = 1800.0) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "keep_alive": "30s",
        "options": {
            "num_predict": num_predict,
            "temperature": 0.0,
            "seed": 1234,
            "num_gpu": num_gpu,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API}/api/generate", data=data,
        headers={"Content-Type": "application/json"},
    )

    sampler = Sampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            resp = json.load(r)
        err = None
    except Exception as e:
        resp, err = None, f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    sampler.stop()

    out: dict = {
        "model": model,
        "num_gpu": num_gpu,
        "num_predict": num_predict,
        "wall_s": wall,
        "peak_vram_mib": sampler.peak_vram,
        "peak_ram_used_bytes": sampler.peak_ram,
        "min_ram_available_bytes": sampler.min_avail,
        "error": err,
    }
    if resp:
        ec = resp.get("eval_count") or 0
        ed = resp.get("eval_duration") or 0
        pc = resp.get("prompt_eval_count") or 0
        pd = resp.get("prompt_eval_duration") or 0
        out.update(
            eval_count=ec,
            prompt_eval_count=pc,
            load_s=(resp.get("load_duration") or 0) / 1e9,
            prompt_eval_s=pd / 1e9,
            decode_s=ed / 1e9,
            decode_tok_s=(ec / (ed / 1e9)) if ed else None,
            prefill_tok_s=(pc / (pd / 1e9)) if pd else None,
            ttft_s=((resp.get("load_duration") or 0)
                    + (resp.get("prompt_eval_duration") or 0)) / 1e9,
            response_chars=len(resp.get("response") or ""),
            response_preview=(resp.get("response") or "")[:200],
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", default=[],
                    help="model:num_gpu:num_predict (repeatable)")
    ap.add_argument("--out", default="results/inference.json")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    specs = []
    for s in args.spec:
        parts = s.rsplit(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"bad --spec {s!r}, want model:num_gpu:num_predict")
        specs.append((parts[0], int(parts[1]), int(parts[2])))
    if not specs:
        raise SystemExit("give at least one --spec")

    if not ensure_server():
        raise SystemExit("could not reach or start the ollama server")

    print(f"baseline before run: ram_used={mem_used_bytes()/1e9:.2f} GB "
          f"vram_used={gpu_used_mib()} MiB\n")

    results = []
    for model, num_gpu, npred in specs:
        print(f"=== {model}  num_gpu={num_gpu}  num_predict={npred} ===", flush=True)
        r = run_one(model, num_gpu, npred, args.timeout)
        results.append(r)
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
        else:
            d = r.get("decode_tok_s")
            print(f"  decode      : {d:.3f} tok/s" if d else "  decode: n/a")
            print(f"  prefill     : {r.get('prefill_tok_s')}")
            print(f"  load        : {r.get('load_s'):.1f} s")
            print(f"  peak VRAM   : {r.get('peak_vram_mib')} MiB")
            print(f"  min avail RAM: {(r.get('min_ram_available_bytes') or 0)/1e9:.2f} GB")
        print(flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
