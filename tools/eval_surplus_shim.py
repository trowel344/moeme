#!/usr/bin/env python3
"""
eval_surplus_shim.py -- score a model through *surplus's* engine, same tasks.

Why the shim and not a llama.cpp command line
--------------------------------------------
`surplus serve` is OpenAI-shaped and `surplus shim` is Ollama-shaped. The shim
matters for this repo: the existing gate runs against Ollama's /api/generate, so
pointing it at `http://127.0.0.1:11435` compares surplus to the ollama tag through
the *same* request payloads, the same prompts, the same greedy decode and the
same scorer. Nothing is translated by hand, so a difference in the score is a
difference in the engine, not in how the comparison was set up.

The placement is not passed on the command line: surplus's planner decides it
(moe-split, `--n-cpu-moe 24` for the Q2_K artefact) and this records the plan
alongside the result, including its predicted decode rate, so the prediction can
be checked against what was measured.

Memory
------
surplus's server is launched with its "RAM cage" enabled by default -- surplus's
own docs say an uncaged server can OOM the desktop. That is the same risk this
repo measured the hard way (the CPU-only control run in docs/expert_offloading.md
was OOM-killed), so this refuses to start on a busy GPU the same way
tools/eval_llama_engine.py does, and samples peak VRAM / RAM while running.

Usage
-----
    python3 tools/eval_surplus_shim.py \
        --model ~/models/Ornith-1.5-35B-A3B-Q2_K.gguf \
        --tasks r_bat_ball,r_work_rate,r_lily,r_sequence \
        --num-predict 768 --out results/eval_surplus_shim.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_llama_engine import Sampler, preflight, wait_ready  # noqa: E402

SHORT19 = ("c_sum_evens,c_reverse_words,c_gcd,r_bat_ball,r_work_rate,r_lily,"
           "r_sequence,f_json_three_actions,f_json_tool_call,f_json_no_prose,"
           "f_json_nested,f_json_array,f_json_nulls,i_exact_three_bullets,"
           "i_only_word,i_uppercase,i_five_words,i_lowercase_start,i_no_letter_e")


def surplus_plan(shim: str, model: str) -> dict:
    """The planner's own answer for this artifact, kept as evidence.

    `surplus plan` prints two human lines before the JSON object, so the object
    is located rather than assumed to be the whole output.
    """
    try:
        p = subprocess.run([shim, "plan", model], capture_output=True, text=True,
                           timeout=300)
    except Exception as e:
        return {"error": str(e)}
    out = p.stdout
    i = out.find("{")
    if p.returncode != 0 or i < 0:
        return {"error": (p.stderr or out).strip()[:400]}
    try:
        # The plan object is followed by more output (advisory text), so decode
        # the first JSON value rather than requiring the rest to be empty.
        plan, _ = json.JSONDecoder().raw_decode(out[i:])
    except ValueError as e:
        return {"error": f"could not parse plan output: {e}"}
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", default="surplus", help="the surplus binary")
    ap.add_argument("--model", required=True, help="path to the .gguf")
    ap.add_argument("--port", type=int, default=11435,
                    help="the shim's default; it deliberately never uses 11434")
    ap.add_argument("--alias", default="surplus")
    ap.add_argument("--ctx-size", default=None)
    ap.add_argument("--engine-dir", default=None,
                    help="a llama.cpp CHECKOUT ROOT (surplus looks for "
                         "<root>/build/bin/llama-server). Needed for MoE: the "
                         "engine surplus otherwise picks (llama.cpp-moe-cache) "
                         "segfaults on any explicit -ncmoe in its llama-bench, "
                         "which is the placement the planner picks for this "
                         "artifact. Passing a build directory instead of the "
                         "checkout root is silently ignored: surplus reports "
                         "'has no llama-server' and falls back.")
    ap.add_argument("--no-cage", action="store_true",
                    help="disable surplus's RAM cage (its docs call an uncaged "
                         "server able to OOM the desktop)")
    ap.add_argument("--tasks", default=SHORT19)
    ap.add_argument("--num-predict", type=int, default=768)
    ap.add_argument("--think", default="off", choices=("auto", "off", "on"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-timeout", type=float, default=900.0)
    ap.add_argument("--need-vram-mib", type=int, default=6000)
    ap.add_argument("--need-ram-gib", type=float, default=6.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-server", action="store_true")
    args = ap.parse_args()

    model = os.path.expanduser(args.model)
    plan = surplus_plan(args.shim, model)
    print("surplus plan for this artifact:")
    if "error" in plan:
        print(f"  unavailable ({plan['error'][:200]})")
    else:
        print(f"  gpu_layers={plan.get('gpu_layers')} "
              f"n_cpu_moe={plan.get('n_cpu_moe_layers')} "
              f"ctx={plan.get('context_size')} "
              f"kv={plan.get('cache_type_k')}/{plan.get('cache_type_v')}")
        print(f"  predicted decode {plan.get('predicted_decode_tps')} tok/s")

    facts = preflight(args.need_vram_mib, args.need_ram_gib, args.force)
    api = f"http://127.0.0.1:{args.port}"

    cmd = [args.shim, "shim", model, "--port", str(args.port),
           "--alias", args.alias]
    if args.ctx_size:
        cmd += ["--ctx-size", str(args.ctx_size)]
    if args.engine_dir:
        cmd += ["--engine-dir", os.path.expanduser(args.engine_dir)]
    if args.no_cage:
        cmd += ["--no-cage"]
    log_path = os.path.join(ROOT, "logs",
                            f"surplus_shim_{time.strftime('%Y%m%dT%H%M%S')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print("launching:\n  " + " ".join(cmd))
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    sampler = Sampler(proc.pid).start()
    started = time.time()
    ok = False
    try:
        wait_ready(api, args.load_timeout, proc, path="/api/tags",
                   what="surplus shim")
        load_s = time.time() - started
        cmd = [sys.executable, os.path.join(ROOT, "experiments", "eval_gate.py"),
               "--model", args.alias, "--api", api, "--backend", "ollama",
               "--num-predict", str(args.num_predict), "--think", args.think,
               "--keep-alive", "10m", "--tasks", args.tasks, "--out", args.out]
        print(f"\nrunning the gate through the surplus shim:\n")
        rc = subprocess.call(cmd, cwd=ROOT)
        ok = rc == 0
    finally:
        mem = sampler.finish()
        if not args.keep_server:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()

    if ok and os.path.exists(args.out):
        with open(args.out) as fh:
            report = json.load(fh)
        report["engine"] = {
            "kind": "surplus shim (Ollama /api routes)",
            "surplus_plan": plan,
            "engine_dir": args.engine_dir or "surplus default (pinned)",
            "ram_cage": not args.no_cage,
            "load_s": load_s,
            "note": "the plan's KV type is q8_0; the ollama baseline runs f16 KV, "
                    "so this is surplus as configured rather than a pure "
                    "engine swap",
        }
        report["memory"] = {**facts, **mem}
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)

    print(f"\npeak VRAM {mem['peak_vram_mib']} MiB, peak system RAM used "
          f"{mem['peak_ram_used_bytes']/2**30:.2f} GiB, load {load_s:.1f}s")
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
