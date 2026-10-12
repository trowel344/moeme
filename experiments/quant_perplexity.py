#!/usr/bin/env python3
"""
quant_perplexity.py -- how much does a quantisation cost the model?

Why this exists
---------------
The repo's quality gate (experiments/eval_gate.py) is pass/fail on 19-30 tasks
with exact-match checks. On those tasks Q2_K and Q4_K_M of the same model score
the same, and the failures move inside the same handful of brittle constraints --
so the gate says "no measurable difference" and cannot say "Q2 is fine". It is
not the right instrument for the question a user actually asks ("will the small
quant make my model dumber?"), because the answer lives in the *distribution*,
not in whether an easy answer is right.

Perplexity is that instrument. Same weights, same text, same token stream, only
the quantisation differs; the difference in negative log-likelihood per token is
the damage. It is also the measurement that shows why the quant sizes in this
repo are what they are.

What it does
------------
1. builds one fixed corpus from this repo's own prose and code (the text the user
   of this repo is actually pointing the model at), and records its sha256;
2. runs `llama-perplexity` once per `--model LABEL=PATH`, identically
   (`-c/--chunks/--threads` held fixed), optionally under `tools/mem_watchdog.py`;
3. parses each run's final estimate and its per-chunk series, and writes a JSON
   report with the command, the corpus hash, the runtime, and whether a guard
   ended the run.

A detail that matters when reading the output: llama-perplexity prints a
*CUMULATIVE* perplexity after each chunk, not a per-chunk value -- the last
number equals the final estimate exactly. The per-chunk list is recorded here for
provenance only; do not difference it into per-chunk perplexities.

Usage
-----
    python3 experiments/quant_perplexity.py \\
        --model q2=$HOME/models/Ornith-1.5-35B-A3B-Q2_K.gguf \\
        --model q4=$HOME/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf \\
        --out results/quant_perplexity.json

`--self-test` checks the parser and the corpus builder offline, with no model.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The unsloth build is the one that runs on this host: the llama.cpp-upstream
# build ships a llama-perplexity that segfaults on any model here (SIGSEGV before
# it prints anything). Override with --perplexity.
DEFAULT_PERPLEXITY = os.path.expanduser("~/.unsloth/llama.cpp/llama-perplexity")
WATCHDOG = os.path.join(REPO, "tools", "mem_watchdog.py")

# Prose first, then code: the same ordering every run, so two runs see an
# identical token stream even if one model tokenises slightly differently.
CORPUS_SOURCES = ["README.md", "docs/*.md", "src/moeme/*.py", "experiments/*.py"]

FINAL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")
CHUNK_RE = re.compile(r"\[(\d+)\]([0-9.]+)")
# "perplexity: calculating perplexity over 8 chunks, n_ctx=512" -- the window a run
# actually scored, which is what makes two runs comparable.
WINDOW_RE = re.compile(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)")


def build_corpus(out_path: str, sources: list[str] | None = None) -> dict:
    """Concatenate fixed, sorted sources into one corpus file.

    Returns {path, sha256, bytes, files}. Sorted globs keep two runs -- and two
    architectures -- on the same tokens, which is the whole basis of the
    comparison.
    """
    sources = sources if sources is not None else CORPUS_SOURCES
    paths: list[str] = []
    for pattern in sources:
        full = os.path.join(REPO, pattern)
        if any(c in pattern for c in "*?["):
            paths.extend(sorted(glob.glob(full)))
        elif os.path.exists(full):
            paths.append(full)
    if not paths:
        raise SystemExit(f"no corpus sources found under {REPO} for {sources}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as out:
        for p in paths:
            out.write(f"\n\n===== {os.path.relpath(p, REPO)} =====\n\n".encode())
            with open(p, "rb") as fh:
                out.write(fh.read())
    data = open(out_path, "rb").read()
    return {"path": out_path, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "files": [os.path.relpath(p, REPO) for p in paths]}


def parse_ppl(text: str) -> dict:
    """Pull the final estimate and the printed series out of a perplexity log."""
    final = FINAL_RE.search(text)
    series = [float(v) for _, v in CHUNK_RE.findall(text)]
    window = WINDOW_RE.search(text)
    chunks, ctx = (int(window.group(1)), int(window.group(2))) if window else (None, None)
    return {
        "ppl": float(final.group(1)) if final else None,
        "stderr": float(final.group(2)) if final else None,
        # Cumulative, not per-chunk -- see the module docstring.
        "cumulative_series": series,
        "chunks_reported": len(series),
        "chunks_scored": chunks,
        "ctx": ctx,
        "tokens_scored": chunks * ctx if chunks and ctx else None,
        "final_estimate_line": final.group(0) if final else None,
    }


def window_note(run: dict, corpus_bytes: int, bytes_per_token: float = 4.0) -> str | None:
    """Say out loud how little of the corpus a run scored, or None if it scored it all.

    llama-perplexity scores the first `chunks * n_ctx` tokens and prints that window
    itself. Recording `chunks: 8` beside a 400 KB corpus invites the reader to think
    the whole corpus was measured; at 512 x 8 it is roughly the first 4,000 tokens.
    That this is not pedantry was demonstrated by accident: a corpus that grew from
    389,996 to 397,855 bytes produced a byte-identical estimate, because all the new
    text landed past the end of the window. Quote the window, not the corpus.
    """
    tokens = run.get("tokens_scored")
    if not tokens:
        return None
    approx_total = max(corpus_bytes / bytes_per_token, 1.0)
    if tokens >= 0.9 * approx_total:
        return None
    return (f"{tokens} tokens scored ~ {100 * tokens / approx_total:.0f}% of the "
            f"{corpus_bytes}-byte corpus -- the FIRST {tokens} tokens only")


def run_one(label: str, model: str, corpus: str, ctx: int, chunks: int,
            threads: int, perplexity: str, guard: bool, min_ram_gib: float,
            gpu_layers: int, cpu_moe: bool, log_dir: str,
            extra: list[str] | None = None) -> dict:
    """Run llama-perplexity once, optionally under the memory watchdog."""
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"ppl_{label}.log")
    argv = [perplexity, "-m", model, "-f", corpus, "-c", str(ctx),
            "--chunks", str(chunks), "-t", str(threads),
            "-ngl", str(gpu_layers)]
    if cpu_moe:
        argv.append("-cmoe")
    argv.extend(extra or [])
    cmd = " ".join(argv)
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT)
    guard_out = ""
    guard_log = os.path.join(log_dir, f"ppl_{label}.watch.log")
    if guard:
        w = subprocess.run([sys.executable, WATCHDOG, "--pid", str(proc.pid),
                            "--min-ram-gib", str(min_ram_gib),
                            "--grace-samples", "2", "--interval", "3",
                            "--log", guard_log], capture_output=True, text=True)
        guard_out = (w.stdout or "").strip()
    rc = proc.wait()
    wall = time.perf_counter() - t0
    text = open(log, errors="replace").read()
    parsed = parse_ppl(text)
    parsed.update({"label": label, "model": model, "gguf_bytes": os.path.getsize(model),
                   "exit_code": rc, "wall_s": wall, "command": cmd,
                   "guard": guard_out or None,
                   "guard_ended_it": "GUARD TERMINATED" in guard_out})
    print(f"{label:<10} ppl={parsed['ppl']} exit={rc} wall={wall:.0f}s "
          f"{'-- ' + guard_out if guard_out else ''}", flush=True)
    return parsed


# GGUF metadata value types (llama.cpp gguf.h). Only metadata is read here; the
# tensor data is never touched.
_GGUF_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
                 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_GGUF_STRING = 8
_GGUF_ARRAY = 9

# general.file_type, the GGUF file-type enum -- a property of the artifact, not a
# guess made from its filename.
FILE_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_SOME_F16", 5: "BF16",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "IQ2_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ2_M",
}


def _read_gguf_value(fh, vtype: int, *, keep: bool = True):
    """Read one metadata value, or consume it without keeping it.

    Every value must be *consumed* even when it is not wanted: a reader that skips
    a key it does not recognise loses alignment and misreads every key after it.
    """
    if vtype in _GGUF_SCALARS:
        fmt, size = _GGUF_SCALARS[vtype]
        raw = fh.read(size)
        if len(raw) != size:
            raise ValueError("truncated GGUF metadata")
        return struct.unpack(fmt, raw)[0] if keep else None
    if vtype == _GGUF_STRING:
        n, = struct.unpack("<Q", fh.read(8))
        raw = fh.read(n)
        if len(raw) != n:
            raise ValueError("truncated GGUF string")
        return raw.decode("utf-8", "replace") if keep else None
    if vtype == _GGUF_ARRAY:
        elem, = struct.unpack("<I", fh.read(4))
        count, = struct.unpack("<Q", fh.read(8))
        if not keep:
            for _ in range(count):
                _read_gguf_value(fh, elem, keep=False)
            return None
        return [_read_gguf_value(fh, elem) for _ in range(count)]
    raise ValueError(f"unknown GGUF metadata type {vtype}")


def gguf_metadata(path: str, keys: set[str] | None = None) -> dict:
    """Read a GGUF's metadata (never its tensors), optionally only `keys`.

    GGUF v1/v2 store the two counts as uint32; v3 widened them to uint64.
    """
    meta: dict = {}
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version, = struct.unpack("<I", fh.read(4))
        wide = "<Q" if version >= 3 else "<I"
        struct.unpack(wide, fh.read(struct.calcsize(wide)))  # n_tensors
        n_kv, = struct.unpack(wide, fh.read(struct.calcsize(wide)))
        for _ in range(n_kv):
            n, = struct.unpack("<Q", fh.read(8))
            key = fh.read(n).decode("utf-8", "replace")
            vtype, = struct.unpack("<I", fh.read(4))
            want = keys is None or key in keys
            value = _read_gguf_value(fh, vtype, keep=want)
            if want:
                meta[key] = value
    return meta


def fmt_bpw(path: str) -> str:
    """Quantisation from the GGUF's own metadata (Q2_K / Q4_K_M / ...), or "?".

    "?" is a real answer, not a failure to try. A quantisation tool may omit
    general.file_type, and a name inferred from the *filename* would put an
    unverified string into a results file that is quoted as a measurement.

    Fixing this to read real artifacts mattered: the tokenizer's token array sits
    in front of general.file_type in these files, and the first version of this
    reader -- scalar types only, arrays skipped by their printed width -- lost
    alignment on it and reported "?" for both GGUFs, i.e. it silently failed to
    report the exact property the report exists to record.
    """
    try:
        value = gguf_metadata(path, keys={"general.file_type"}).get(
            "general.file_type")
    except (OSError, ValueError, struct.error):
        return "?"
    return FILE_TYPE_NAMES.get(value, "?") if isinstance(value, int) else "?"


def self_test() -> int:
    """Offline: the parser and the corpus builder, no model and no GPU."""
    sample = (
        "0.05.265.837 I perplexity: calculating perplexity over 2 chunks, n_ctx=512\n"
        "[1]8.3929,[2]6.5966,\n"
        "1.29.771.068 I perplexity: 74.63 seconds per pass - ETA 2.48 minutes\n"
        "2.46.871.431 I Final estimate: PPL = 6.5966 +/- 0.38006\n")
    got = parse_ppl(sample)
    assert got["ppl"] == 6.5966, got
    assert got["stderr"] == 0.38006, got
    assert got["cumulative_series"] == [8.3929, 6.5966], got
    assert got["chunks_reported"] == 2, got
    # A run the OOM killer ended has no final estimate; that must read as "no
    # number", never as a pass or a zero.
    killed = parse_ppl("0.00.930.529 W llama_model_loader: tensor overrides\n")
    assert killed["ppl"] is None and killed["cumulative_series"] == [], killed
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        c = build_corpus(os.path.join(tmp, "c.txt"))
        assert c["bytes"] > 1000 and len(c["sha256"]) == 64, c
        again = build_corpus(os.path.join(tmp, "c2.txt"))
        assert again["sha256"] == c["sha256"], "corpus must be deterministic"
        # A real artifact states its quantisation *after* the tokenizer's token
        # array. A reader that cannot consume an array loses alignment there and
        # reports "?" for a file that does state it -- which is how the first
        # version of this knowingly failed. This is that regression.
        meta = os.path.join(tmp, "m.gguf")
        key1, key2 = b"tokenizer.ggml.tokens", b"general.file_type"
        with open(meta, "wb") as fh:
            fh.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0))
            fh.write(struct.pack("<Q", 2))
            fh.write(struct.pack("<Q", len(key1)) + key1
                     + struct.pack("<I", _GGUF_ARRAY) + struct.pack("<I", _GGUF_STRING)
                     + struct.pack("<Q", 2000)
                     + b"".join(struct.pack("<Q", 2) + b"xy" for _ in range(2000)))
            fh.write(struct.pack("<Q", len(key2)) + key2
                     + struct.pack("<I", 4) + struct.pack("<I", 10))
        assert fmt_bpw(meta) == "Q2_K", fmt_bpw(meta)
    print("quant_perplexity self-test: ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                    help="a GGUF to score; repeat for each quantisation")
    ap.add_argument("--out", default="results/quant_perplexity.json")
    ap.add_argument("--corpus", default="results/ppl_corpus.txt")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--gpu-layers", type=int, default=99,
                    help="layers to offload; 99 = all that fit")
    ap.add_argument("--cpu-moe", action="store_true",
                    help="keep every expert on the CPU. Fatal for a 21 GB MoE on "
                         "a 15 GiB host: loading it that way was OOM-killed here")
    ap.add_argument("--perplexity", default=DEFAULT_PERPLEXITY)
    ap.add_argument("--no-guard", action="store_true",
                    help="do not wrap each run in tools/mem_watchdog.py")
    ap.add_argument("--min-ram-gib", type=float, default=1.5)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.model:
        raise SystemExit("give at least one --model LABEL=PATH")
    if not shutil.which(args.perplexity) and not os.path.exists(args.perplexity):
        raise SystemExit(f"no llama-perplexity at {args.perplexity}; "
                         f"pass --perplexity")

    corpus = build_corpus(args.corpus)
    print(f"corpus {corpus['path']}: {corpus['bytes']} bytes from "
          f"{len(corpus['files'])} files, sha256 {corpus['sha256'][:16]}...\n")

    report = {"corpus": corpus, "ctx": args.ctx, "chunks": args.chunks,
              "threads": args.threads, "gpu_layers": args.gpu_layers,
              "cpu_moe": args.cpu_moe,
              "perplexity_binary": args.perplexity, "runs": {}}
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"bad --model {spec!r}, want LABEL=PATH")
        label, path = spec.split("=", 1)
        run = run_one(label, path, corpus["path"], args.ctx, args.chunks,
                      args.threads, args.perplexity, not args.no_guard,
                      args.min_ram_gib, args.gpu_layers, args.cpu_moe,
                      args.log_dir)
        run["quant"] = fmt_bpw(path)
        # Stored in the report, not only printed: a summary that re-reads this file
        # must not have to re-derive what the run covered.
        run["scored_window_note"] = window_note(run, corpus["bytes"])
        report["runs"][label] = run
        save(report, args.out)

    # The comparison is only meaningful when two runs saw the same tokens.
    scored = {k: v["ppl"] for k, v in report["runs"].items() if v["ppl"]}
    if len(scored) >= 2:
        print()
        for k, v in scored.items():
            print(f"  {k:<10} PPL {v:.4f}")
        lo, hi = min(scored, key=scored.get), max(scored, key=scored.get)
        delta = scored[hi] - scored[lo]
        print(f"  {hi} is {delta:+.4f} PPL ({100 * delta / scored[lo]:+.1f}%) "
              f"against {lo}")
        print(f"  resolution: +/-{delta / (report['runs'][hi]['stderr'] or 1):.2f} sigma "
              f"on the printed standard errors -- a paired difference on the same "
              f"tokens is tighter than that, but a gap near 1 sigma is not a result")
        windows = {v["tokens_scored"] for v in report["runs"].values() if v["ppl"]}
        if len(windows) > 1:
            print(f"  CAREFUL: the runs scored different windows ({sorted(windows)}) "
                  f"-- this is not a matched pair")
    for k, v in report["runs"].items():
        note = window_note(v, corpus["bytes"])
        if note:
            print(f"  {k}: {note}")
    missing = [k for k, v in report["runs"].items() if not v["ppl"]]
    if missing:
        print(f"\n  no estimate for {', '.join(missing)} -- read the recorded "
              f"exit code and guard verdict before comparing anything")
    print(f"wrote {args.out}")
    return 0


def save(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
