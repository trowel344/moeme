#!/usr/bin/env python3
"""
eval_llama_engine.py -- score a model through llama.cpp at a chosen placement.

Why this exists
---------------
`tools/bench_inference.py` measures the ollama tag, and `results/moe_placement.json`
measured placements with `llama-bench`, which generates text but never checks it.
Neither answers the question this repo keeps asking: if the engine and the
placement change, does the *output* stay right? An inference engine decides where
bytes come from, so it can change speed; it cannot change which weights the model
needs, so it must not change the answers.

This runs the repo's own task set (`experiments/eval_gate.py`, `--backend llama`)
against a llama-server started with explicit flags. The tasks, prompts, decode
settings and scorer are identical to the ollama runs, so the resulting report is
directly comparable to `results/eval_<tag>_*.json` row by row.

Memory safety
-------------
The whole point is to run a 13 GB model on an 8 GB card without taking the
machine down, so this refuses to start unless free VRAM and available RAM clear
explicit thresholds, and it samples peak VRAM and peak RSS while running. Use
--force to override; the thresholds are conservative, not physical limits.

Usage
-----
    python3 tools/eval_llama_engine.py \
        --server ~/llama.cpp-upstream/build/bin/llama-server \
        --model  ~/models/Ornith-1.5-35B-A3B-Q2_K.gguf \
        --placement "-ngl 42 -ncmoe 24" \
        --tasks r_bat_ball,r_work_rate,r_lily,r_sequence \
        --num-predict 768 --out results/eval_llama_ngl42_ncmoe24.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# preflight: what must be free before a 13 GB model is mapped in
# ---------------------------------------------------------------------------

def _read_meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            out[k] = int(v.strip().split()[0]) * 1024
    return out


def vram_used_mib() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        used, total = (int(x) for x in p.stdout.strip().splitlines()[0].split(","))
        return used
    except Exception:
        return None


def vram_total_mib() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(p.stdout.strip().splitlines()[0])
    except Exception:
        return None


def preflight(need_vram_mib: int, need_ram_gib: float, force: bool) -> dict:
    """Return the numbers the run started from, or exit with a reason.

    A failed allocation on this host takes the desktop with it (the CPU-only
    control run in docs/expert_offloading.md was OOM-killed), so the check is
    deliberately conservative and it prints what it saw either way.
    """
    mi = _read_meminfo()
    ram_avail = mi["MemAvailable"]
    used, total = vram_used_mib(), vram_total_mib()
    facts = {
        "ram_available_bytes": ram_avail,
        "vram_used_mib_before": used,
        "vram_total_mib": total,
    }
    print("preflight:")
    print(f"  RAM available   : {ram_avail/2**30:.2f} GiB "
          f"(need {need_ram_gib:.1f} GiB)")
    if used is not None and total is not None:
        print(f"  VRAM            : {used} MiB used of {total} MiB "
              f"(need {need_vram_mib} MiB free)")
    problems = []
    if ram_avail < need_ram_gib * 2**30:
        problems.append(f"only {ram_avail/2**30:.2f} GiB RAM available")
    if used is not None and total is not None and total - used < need_vram_mib:
        problems.append(f"only {total - used} MiB VRAM free (need {need_vram_mib})")
    if problems:
        joined = "; ".join(problems)
        if force:
            print(f"  WARNING: {joined} -- continuing because --force was given")
        else:
            raise SystemExit(
                f"refusing to start: {joined}.\n"
                f"Free the memory first -- `ollama stop <model>` unloads a resident "
                f"model, and closing a browser releases VRAM -- or pass --force.")
    return facts


# ---------------------------------------------------------------------------
# sampling while a run is in flight
# ---------------------------------------------------------------------------

class Sampler:
    """Peak VRAM, peak RSS and peak system RAM, sampled until stopped.

    Peak is what matters: an average hides the moment the run got close to the
    edge, and on this host the edge is the whole desktop. System RAM is sampled
    as well as the process's own RSS because a shim spawns llama-server as a
    child -- measuring only the wrapper would report a rounding error while the
    real allocation sits in a different process.
    """

    def __init__(self, pid: int | None = None, interval: float = 0.5):
        self.pid = pid
        self.interval = interval
        self.stop = threading.Event()
        self.peak_vram = 0
        self.peak_rss = 0
        self.peak_ram_used = 0
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop.wait(self.interval):
            u = vram_used_mib()
            if u:
                self.peak_vram = max(self.peak_vram, u)
            try:
                mi = _read_meminfo()
                self.peak_ram_used = max(self.peak_ram_used,
                                        mi["MemTotal"] - mi["MemAvailable"])
            except Exception:
                pass
            if self.pid:
                self.peak_rss = max(self.peak_rss, self._rss(self.pid))

    @staticmethod
    def _rss(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0

    def start(self) -> "Sampler":
        self._t.start()
        return self

    def finish(self) -> dict:
        self.stop.set()
        self._t.join(timeout=2)
        return {"peak_vram_mib": self.peak_vram,
                "peak_rss_bytes": self.peak_rss,
                "peak_ram_used_bytes": self.peak_ram_used}


# ---------------------------------------------------------------------------
# llama-server lifecycle
# ---------------------------------------------------------------------------

def wait_ready(api: str, timeout: float, proc: subprocess.Popen,
               path: str = "/health", what: str = "llama-server") -> None:
    """Poll a readiness path. 503 means still loading, which is not the same as
    unreachable -- and a load that dies (not enough memory) must be reported as
    that, not as a slow start.

    `path` differs by engine: llama-server has /health, surplus's Ollama shim
    only has Ollama's own routes.
    """
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        code = proc.poll()
        if code is not None:
            raise SystemExit(
                f"{what} exited with code {code} during load. "
                f"Check its log; on this host that is almost always an "
                f"allocation failure from too many GPU layers.")
        try:
            with urllib.request.urlopen(f"{api}{path}", timeout=3) as r:
                if r.status == 200:
                    print(f"  {what} ready at {api} "
                          f"({time.time() - (deadline - timeout):.1f}s)")
                    return
                last = f"status {r.status}"
        except urllib.error.HTTPError as e:
            last = f"status {e.code} (loading)"
        except Exception as e:
            last = str(e)
        time.sleep(2)
    raise SystemExit(f"{what} not ready after {timeout:.0f}s (last: {last})")


def start_server(args) -> tuple[subprocess.Popen, object]:
    log_path = os.path.join(ROOT, "logs",
                            f"llama_server_{time.strftime('%Y%m%dT%H%M%S')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    placement = args.placement.split()
    cmd = [os.path.expanduser(args.server), "-m", os.path.expanduser(args.model),
           "--port", str(args.port), "--host", "127.0.0.1",
           "-c", str(args.ctx), "-t", str(args.threads)] + placement
    print("launching:")
    print("  " + " ".join(cmd))
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="path to llama-server")
    ap.add_argument("--model", required=True, help="path to the .gguf")
    ap.add_argument("--placement", default="-ngl 42 -ncmoe 24",
                    help="llama.cpp placement flags; the default is the placement "
                         "surplus's planner picked for the Q2_K MoE")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task ids (default: the eval_gate subset "
                         "used for the ollama runs)")
    ap.add_argument("--num-predict", type=int, default=768)
    ap.add_argument("--think", default="off", choices=("auto", "off", "on"))
    ap.add_argument("--template-kwargs", default='{"enable_thinking": false}',
                    help="JSON forwarded as chat_template_kwargs, so a thinking "
                         "model's <think> block does not end up in the answer. "
                         "The ollama baseline runs --think off; without this the "
                         "comparison measures template defaults, not the engine")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-timeout", type=float, default=900.0)
    ap.add_argument("--need-vram-mib", type=int, default=6000)
    ap.add_argument("--need-ram-gib", type=float, default=6.0)
    ap.add_argument("--force", action="store_true",
                    help="start even if the memory preflight fails")
    ap.add_argument("--keep-server", action="store_true",
                    help="leave llama-server running afterwards")
    args = ap.parse_args()

    facts = preflight(args.need_vram_mib, args.need_ram_gib, args.force)
    api = f"http://127.0.0.1:{args.port}"

    proc, log = start_server(args)
    sampler = Sampler(proc.pid).start()
    started = time.time()
    ok = False
    try:
        wait_ready(api, args.load_timeout, proc)
        load_s = time.time() - started
        cmd = [sys.executable, os.path.join(ROOT, "experiments", "eval_gate.py"),
               "--model", os.path.basename(args.model), "--backend", "llama",
               "--api", api, "--num-predict", str(args.num_predict),
               "--think", args.think, "--out", args.out]
        if args.template_kwargs:
            cmd += ["--chat-template-kwargs", args.template_kwargs]
        if args.tasks:
            cmd += ["--tasks", args.tasks]
        print(f"\nrunning the gate through llama.cpp ({args.num_predict} tokens/task):\n")
        rc = subprocess.call(cmd, cwd=ROOT)
        ok = rc == 0
        if not ok:
            print(f"eval_gate exited {rc}; partial results are checkpointed in "
                  f"{args.out}")
    finally:
        mem = sampler.finish()
        if not args.keep_server:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()

    # Annotate the report: which engine, which placement, and what it cost on
    # this host. Without this the file cannot be told apart from an ollama run.
    if ok and os.path.exists(args.out):
        with open(args.out) as fh:
            report = json.load(fh)
        report["engine"] = {
            "kind": "llama.cpp llama-server",
            "model_path": args.model,
            "placement_flags": args.placement,
            "ctx": args.ctx,
            "threads": args.threads,
            "kv_cache_type": "f16 (llama-server default; the llama-bench placement "
                            "sweep in results/moe_placement.json used q8_0)",
            "chat_template_kwargs": args.template_kwargs,
            "load_s": load_s,
        }
        report["memory"] = {**facts, **mem}
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)

    print(f"\npeak VRAM {mem['peak_vram_mib']} MiB, peak server RSS "
          f"{mem['peak_rss_bytes']/2**30:.2f} GiB, load {load_s:.1f}s")
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
