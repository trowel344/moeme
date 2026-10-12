#!/usr/bin/env python3
"""
mem_watchdog.py -- kill a run before RAM exhaustion reaches the OOM killer.

Why this exists
---------------
Loading a model larger than the fast tiers is the one operation in this repo that
can take the desktop down, and it already has: the CPU-only control run in
`docs/expert_offloading.md` was OOM-killed. The kernel's OOM killer picks its
victim without asking, so the safe design is to end the expensive process
*first*, at a threshold the operator chose, and record that it happened.

This host makes that sharper: swap is effectively full (see `make doctor`), so
there is no swap headroom to absorb a surge of anonymous memory.

What it does
------------
Samples MemAvailable, SwapFree and VRAM every `--interval` seconds while a target
PID lives, writes one line per sample (so the run has evidence, not just a
verdict), and sends SIGTERM, then SIGKILL, when available RAM stays below
`--min-ram-gib` for `--grace-samples` consecutive samples. Transient dips are
normal; a *sustained* one is the failure mode.

Exit codes: 0 target exited on its own, 2 the watchdog ended the target.

Usage
-----
    python3 tools/mem_watchdog.py --pid 1234 --min-ram-gib 1.5 \\
        --log logs/mem_watch_run.log
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval_llama_engine import _read_meminfo, vram_used_mib  # noqa: E402

GIB = 2 ** 30


def should_act(available_bytes: int, floor_bytes: int,
               consecutive_below: int, grace_samples: int) -> bool:
    """True when the floor has been breached for long enough to act on.

    A single low sample is not worth killing a multi-hour run over -- the page
    cache is reclaimable and dips recover -- so the floor must hold for
    `grace_samples` in a row.
    """
    return available_bytes < floor_bytes and consecutive_below >= grace_samples


def process_state(pid: int) -> str:
    """The single-letter state from /proc/<pid>/stat, or "?" if unreadable.

    Read from the end of the line because the comm field can contain spaces and
    parentheses; the state is the first field after the last ')'. "Z" is a
    zombie: the PID still exists, the process does not.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return "?"


def target_alive(pid: int) -> bool:
    """False once the process is gone *or* has become a zombie.

    `os.kill(pid, 0)` alone reports a zombie as alive, because the PID is still
    held by the process table until the parent reaps it. A watchdog that misses
    that never leaves its loop on a finished run, so it never writes its summary
    and never returns -- the failure this test exists to prevent.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return process_state(pid) != "Z"


def target_rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None, help="process to protect")
    ap.add_argument("--pidfile", default=None,
                    help="read the target PID from this file instead of --pid, "
                         "waiting for it to appear (a run that writes its own "
                         "PID cannot be guarded by mistake against a wrapper)")
    ap.add_argument("--wait-pidfile", type=float, default=180.0)
    ap.add_argument("--min-ram-gib", type=float, default=1.5,
                    help="kill the target when MemAvailable stays below this")
    ap.add_argument("--grace-samples", type=int, default=3,
                    help="consecutive samples below the floor before acting")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--log", default="logs/mem_watch.log")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop watching after this long (0 = until the target exits)")
    ap.add_argument("--warn-ram-gib", type=float, default=2.5)
    args = ap.parse_args()

    if args.pid is None and not args.pidfile:
        raise SystemExit("give --pid or --pidfile")
    if args.pidfile and args.pid is None:
        deadline = time.time() + args.wait_pidfile
        while time.time() < deadline:
            try:
                with open(args.pidfile) as fh:
                    args.pid = int(fh.read().strip())
                break
            except (OSError, ValueError):
                time.sleep(0.5)
        if args.pid is None:
            raise SystemExit(f"no PID in {args.pidfile} after "
                             f"{args.wait_pidfile:.0f}s")
        # The file can appear before the process is visible to the kernel.
        for _ in range(50):
            if target_alive(args.pid):
                break
            time.sleep(0.2)

    floor = int(args.min_ram_gib * GIB)
    warn = int(args.warn_ram_gib * GIB)
    deadline = (time.time() + args.max_minutes * 60) if args.max_minutes else None

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    below = 0
    min_avail = None
    peak_vram = 0
    peak_rss = 0
    killed = False
    samples = 0
    with open(args.log, "w") as log:
        log.write(f"# watching pid {args.pid}, floor {args.min_ram_gib} GiB, "
                  f"grace {args.grace_samples} samples, interval {args.interval}s\n")
        log.write("# time avail_gib swap_free_gib vram_used_mib target_rss_gib note\n")
        log.flush()
        while target_alive(args.pid):
            if deadline and time.time() > deadline:
                log.write("# max-minutes reached; target left running\n")
                break
            mi = _read_meminfo()
            avail = mi["MemAvailable"]
            swap_free = mi["SwapFree"]
            vram = vram_used_mib() or 0
            rss = target_rss(args.pid)
            samples += 1
            min_avail = avail if min_avail is None else min(min_avail, avail)
            peak_vram = max(peak_vram, vram)
            peak_rss = max(peak_rss, rss)
            below = below + 1 if avail < floor else 0
            note = ""
            if avail < floor:
                note = f"BELOW FLOOR ({below}/{args.grace_samples})"
            elif avail < warn:
                note = "low"
            log.write(f"{time.strftime('%H:%M:%S')} {avail / GIB:.2f} "
                      f"{swap_free / GIB:.2f} {vram} {rss / GIB:.2f} {note}\n")
            log.flush()
            if should_act(avail, floor, below, args.grace_samples):
                log.write(f"# GUARD: MemAvailable {avail / GIB:.2f} GiB below "
                          f"{args.min_ram_gib} GiB for {below} samples -- "
                          f"terminating {args.pid} before the OOM killer picks "
                          f"its own victim\n")
                log.flush()
                killed = True
                try:
                    os.kill(args.pid, signal.SIGTERM)
                    for _ in range(30):
                        if not target_alive(args.pid):
                            break
                        time.sleep(1)
                    if target_alive(args.pid):
                        os.kill(args.pid, signal.SIGKILL)
                except OSError as e:
                    log.write(f"# could not signal the target: {e}\n")
                break
            time.sleep(args.interval)
        else:
            log.write("# target exited on its own\n")

    print(f"watched {samples} sample(s) of pid {args.pid}: "
          f"min RAM available {(min_avail or 0) / GIB:.2f} GiB, "
          f"peak VRAM {peak_vram} MiB, peak target RSS {peak_rss / GIB:.2f} GiB"
          + ("  -- GUARD TERMINATED THE TARGET" if killed else ""))
    print(f"samples: {args.log}")
    return 2 if killed else 0


if __name__ == "__main__":
    sys.exit(main())
