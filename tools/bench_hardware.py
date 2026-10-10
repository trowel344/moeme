#!/usr/bin/env python3
"""
bench_hardware.py -- non-destructive hardware characterisation for edge LLM work.

Measures the quantities that actually decide whether a MoE-on-NVMe design can
work on this class of machine:

  * usable VRAM (total minus what the desktop already holds)
  * usable RAM (total minus current usage, plus swap headroom)
  * system RAM bandwidth (this upper-bounds CPU-side expert execution)
  * GPU memory bandwidth (upper-bounds VRAM-resident decode)
  * NVMe sequential read throughput
  * NVMe random-read throughput at 4K / 64K / 256K / 1M
  * mmap cold-page fault latency for weight-sized random reads

Design notes
------------
* The disk test writes a temp file LARGER THAN TOTAL RAM so the page cache
  cannot satisfy the reads. Without that, sequential numbers are fictional.
* All temp files are removed on exit, including on failure.
* GPU experiments stay small (<= 1 GiB) so we do not evict the user's desktop.
* No sudo, no writes outside the temp file, no changes to system state.

Usage:
    python3 tools/bench_hardware.py --out results/hardware.json
    python3 tools/bench_hardware.py --quick          # skip the disk test
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

GiB = 1024 ** 3


# --------------------------------------------------------------------------
# system inventory
# --------------------------------------------------------------------------

def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"<error: {e}>"


def meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            out[k.strip()] = int(v.strip().split()[0]) * 1024  # kB -> bytes
    return out


def gpu_info() -> dict:
    q = ("index,name,memory.total,memory.used,memory.free,"
         "utilization.gpu,temperature.gpu,clocks.max.memory,"
         "clocks.max.sm,power.limit")
    txt = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if not txt or txt.startswith("<error"):
        return {"available": False}

    def to_int(s: str):
        """nvidia-smi emits '[N/A]' / 'N/A' for unsupported fields."""
        s = s.strip().strip("[]")
        try:
            return int(float(s))
        except ValueError:
            return None

    d = {"available": True}
    for row in txt.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 10:
            continue
        d.update(
            name=parts[1],
            vram_total_mib=to_int(parts[2]),
            vram_used_mib=to_int(parts[3]),
            vram_free_mib=to_int(parts[4]),
            gpu_util_pct=to_int(parts[5]),
            temp_c=to_int(parts[6]),
            mem_clock_max_mhz=to_int(parts[7]),
            sm_clock_max_mhz=to_int(parts[8]),
            power_limit_w=to_int(parts[9]),
        )
    return d


def inventory() -> dict:
    mi = meminfo()
    inv = {
        "hostname": os.uname().nodename,
        "cpu_model": next(
            (l.split(":", 1)[1].strip()
             for l in open("/proc/cpuinfo") if l.startswith("model name")),
            "unknown",
        ),
        "cpu_logical": os.cpu_count(),
        "kernel": os.uname().release,
        "ram_total_bytes": mi["MemTotal"],
        "ram_available_bytes": mi["MemAvailable"],
        "ram_free_bytes": mi["MemFree"],
        "ram_buffcache_bytes": mi.get("Buffers", 0) + mi.get("Cached", 0),
        "swap_total_bytes": mi.get("SwapTotal", 0),
        "swap_free_bytes": mi.get("SwapFree", 0),
        "gpu": gpu_info(),
    }
    try:
        u = shutil.disk_usage(os.getcwd())
        inv["disk_free_bytes"] = u.free
        inv["disk_total_bytes"] = u.total
    except Exception:
        pass
    return inv


# --------------------------------------------------------------------------
# RAM bandwidth
# --------------------------------------------------------------------------

def bench_ram_bandwidth(size_mb: int = 1024, iters: int = 24) -> dict:
    """Copy + read + write bandwidth on a buffer larger than any cache."""
    n = size_mb * 1024 * 1024
    try:
        import numpy as np
    except ImportError:
        return {"skipped": "numpy unavailable"}
    src = np.ones(n, dtype=np.uint8)
    dst = np.empty(n, dtype=np.uint8)
    # warm up
    for _ in range(2):
        np.copyto(dst, src)

    def timeit(fn) -> float:
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    t_copy = timeit(lambda: np.copyto(dst, src))
    t_read = timeit(lambda: float(np.bitwise_xor.reduce(src)))
    return {
        "buffer_mb": size_mb,
        "copy_bw_gbps": (2 * n / t_copy) / 1e9,      # read+write
        "read_bw_gbps": (n / t_read) / 1e9,
    }


# --------------------------------------------------------------------------
# GPU bandwidth + a representative quantised-style decode kernel
# --------------------------------------------------------------------------

def bench_gpu(alloc_mb: int = 1024, iters: int = 40) -> dict:
    try:
        import torch
    except ImportError:
        return {"skipped": "torch unavailable"}
    if not torch.cuda.is_available():
        return {"skipped": "cuda unavailable"}

    dev = "cuda"
    n_bytes = alloc_mb * 1024 * 1024
    n = n_bytes // 2  # bf16
    src = torch.ones(n, dtype=torch.bfloat16, device=dev)
    dst = torch.empty_like(src)

    def timeit(fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    t_copy = timeit(lambda: dst.copy_(src))
    t_sum = timeit(lambda: torch.sum(src))

    # Representative decode GEMV: ONE expert FFN up-projection in the real
    # Qwen3.8-27B geometry. torch.nn.functional uses (out, in) layout, matching
    # the stored HF weight shape (intermediate 17408 x hidden 5120).
    # This is exactly the memory-bound regime MoE decode lives in.
    w = torch.empty(17408, 5120, dtype=torch.bfloat16, device=dev)
    x = torch.empty(5120, dtype=torch.bfloat16, device=dev)
    t_gemv = timeit(lambda: torch.mv(w, x))
    gemv_bytes = 17408 * 5120 * 2  # read the whole weight matrix once

    out = {
        "alloc_mb": alloc_mb,
        "copy_bw_gbps": (2 * n_bytes / t_copy) / 1e9,
        "read_bw_gbps": (n_bytes / t_sum) / 1e9,
        "gemv_ms": t_gemv * 1e3,
        "gemv_effective_bw_gbps": (gemv_bytes / t_gemv) / 1e9,
        "device_name": torch.cuda.get_device_name(0),
        "peak_alloc_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }
    del w, x, src, dst
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# NVMe
# --------------------------------------------------------------------------

def _drop_cache(path: str) -> None:
    """Evict this file's *clean* pages from the page cache.

    MADV_DONTNEED on a MAP_SHARED file mapping only drops the process PTEs;
    the next fault re-attaches the same cached page, so it does NOT produce a
    cold read. POSIX_FADV_DONTNEED on the file itself is the operation that
    actually discards clean file pages. Requires the file to be already
    fsync'd (dirty pages are not dropped).
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def bench_disk(tmpdir: str | None, file_gb: float = 20.0) -> dict:
    """Sequential + random read throughput on a file larger than total RAM.

    The file must exceed total RAM or the page cache serves the reads and the
    numbers describe DRAM rather than the SSD. We also fsync, then wait for
    writeback to drain before timing reads, because measuring reads while our
    own dirty pages are still flushing understates read throughput badly.
    """
    total_ram = meminfo()["MemTotal"]
    size = max(int(file_gb * GiB), total_ram + 4 * GiB)  # must exceed cache
    fd, path = tempfile.mkstemp(prefix="moeme_diskbench_", dir=tmpdir)
    try:
        block = b"\xa5" * (8 * 1024 * 1024)
        written = 0
        t0 = time.perf_counter()
        with os.fdopen(fd, "wb", buffering=0) as fh:
            while written < size:
                fh.write(block)
                written += len(block)
            fh.flush()
            os.fsync(fh.fileno())
        write_s = time.perf_counter() - t0
        del block

        # let asynchronous writeback fully drain, then drop clean pages
        time.sleep(5.0)
        _drop_cache(path)

        results: dict = {
            "file_bytes": written,
            "write_s": write_s,
            "sequential_write_gbps": (written / write_s) / 1e9,
        }

        # --- sequential read, two passes; the 2nd is the sustained figure ---
        def seq_read() -> float:
            t0 = time.perf_counter()
            rd = 0
            with open(path, "rb", buffering=0) as fh:
                while True:
                    b = fh.read(8 * 1024 * 1024)
                    if not b:
                        break
                    rd += len(b)
            return (rd / (time.perf_counter() - t0)) / 1e9

        results["sequential_read_gbps_pass1"] = seq_read()
        _drop_cache(path)
        time.sleep(2.0)
        results["sequential_read_gbps_pass2"] = seq_read()
        _drop_cache(path)

        # --- random reads at several block sizes (cache dropped per read) ---
        rng = random.Random(1234)
        rand: dict[str, dict] = {}
        with open(path, "rb", buffering=0) as fh:
            for bs_kb in (4, 64, 256, 1024):
                bs = bs_kb * 1024
                n_reads = max(32, min(256, (256 << 20) // bs))
                offsets = [rng.randrange(0, written - bs) & ~511 for _ in range(n_reads)]
                ts = []
                for off in offsets:
                    _drop_cache(path)
                    t0 = time.perf_counter()
                    fh.seek(off)
                    fh.read(bs)
                    ts.append(time.perf_counter() - t0)
                med = statistics.median(ts)
                rand[f"random_{bs_kb}K"] = {
                    "median_latency_us": med * 1e6,
                    "throughput_gbps": (bs / med) / 1e9,
                    "iops": 1.0 / med,
                    "n_reads": n_reads,
                }
        results["random"] = rand
        results["random_cold_note"] = "page cache dropped before every read"
        return results
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def bench_mmap_cold(n_reads: int = 128, chunk: int = 1 << 20,
                    file_gb: float = 20.0, tmpdir: str | None = None) -> dict:
    """Cold mmap page-fault cost for weight-sized random reads.

    This is what an expert-streaming design actually pays: a random 1 MiB
    weight block touched for the first time. The backing file is larger than
    RAM and clean pages are dropped from the page cache before each touch, so
    these are genuine storage faults rather than cache hits.
    """
    total_ram = meminfo()["MemTotal"]
    size = max(int(file_gb * GiB), total_ram + 4 * GiB)
    fd, path = tempfile.mkstemp(prefix="moeme_mmapbench_", dir=tmpdir)
    try:
        with os.fdopen(fd, "wb", buffering=0) as fh:
            blk = b"\x5a" * (8 << 20)
            for _ in range(size // len(blk)):
                fh.write(blk)
            fh.flush()
            os.fsync(fh.fileno())
        time.sleep(5.0)
        _drop_cache(path)

        rng = random.Random(99)
        ts = []
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            for _ in range(n_reads):
                off = rng.randrange(0, size - chunk) & ~4095
                _drop_cache(path)
                t0 = time.perf_counter()
                mm[off:off + chunk]
                ts.append(time.perf_counter() - t0)
            mm.close()
        med = statistics.median(ts)
        return {
            "chunk_bytes": chunk,
            "median_latency_ms": med * 1e3,
            "throughput_gbps": (chunk / med) / 1e9,
            "n_reads": n_reads,
            "cold": True,
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def storage_info() -> dict:
    """Drive model + whether the filesystem sits on an encrypted mapper."""
    info = {
        "lsblk": sh(["lsblk", "-d", "-o", "NAME,MODEL,SIZE,ROTA,TYPE"]),
        "rotational": {},
        "scheduler": {},
        "crypt": sh(["bash", "-lc",
                     "lsblk -o NAME,TYPE,FSTYPE,MOUNTPOINT | grep -i crypt || true"]),
    }
    for dev in os.listdir("/sys/block"):
        base = f"/sys/block/{dev}"
        try:
            with open(f"{base}/queue/rotational") as fh:
                info["rotational"][dev] = fh.read().strip()
            with open(f"{base}/queue/scheduler") as fh:
                info["scheduler"][dev] = fh.read().strip()
        except OSError:
            pass
    return info


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hardware.json")
    ap.add_argument("--quick", action="store_true", help="skip disk + mmap benchmarks")
    ap.add_argument("--disk-gb", type=float, default=20.0)
    ap.add_argument("--tmpdir", default=None, help="where to put the temp file")
    args = ap.parse_args()

    rep: dict = {"inventory": inventory(), "generated_unix": time.time()}
    print("== inventory ==")
    print(json.dumps(rep["inventory"], indent=2))
    if not args.quick:
        print("\n== storage device ==")
        rep["storage"] = storage_info()
        print(json.dumps(rep["storage"], indent=2))

    if not args.quick:
        print("\n== RAM bandwidth ==", flush=True)
        rep["ram_bandwidth"] = bench_ram_bandwidth()
        print(json.dumps(rep["ram_bandwidth"], indent=2))

    print("\n== GPU ==", flush=True)
    rep["gpu_bench"] = bench_gpu()
    print(json.dumps(rep["gpu_bench"], indent=2))

    if not args.quick:
        print(f"\n== NVMe (needs {args.disk_gb:.0f} GB free, temp file removed after) ==",
              flush=True)
        rep["disk"] = bench_disk(args.tmpdir, args.disk_gb)
        print(json.dumps(rep["disk"], indent=2))
        print("\n== mmap cold-read (4 GB file, cache dropped per touch) ==", flush=True)
        rep["mmap_cold"] = bench_mmap_cold(file_gb=4.0, tmpdir=args.tmpdir)
        print(json.dumps(rep["mmap_cold"], indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
