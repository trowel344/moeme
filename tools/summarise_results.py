#!/usr/bin/env python3
"""
summarise_results.py -- re-print every headline number from results/*.json.

Keeps the reports and the raw evidence in sync: if a number is quoted in
docs/, it came out of this script.

Usage:
    python3 tools/summarise_results.py
"""

from __future__ import annotations

import json
import os
import sys

R = "results"


def load(name: str):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def section_hardware() -> None:
    d = load("hardware.json")
    if not d:
        print("hardware.json missing -- run `make doctor`")
        return
    hr("HARDWARE (measured, results/hardware.json)")
    inv, gpu = d["inventory"], d.get("gpu_bench", {})
    print(f"  cpu            : {inv['cpu_model']} ({inv['cpu_logical']} threads)")
    print(f"  ram total      : {inv['ram_total_bytes']/1e9:.2f} GB")
    print(f"  ram available  : {inv['ram_available_bytes']/1e9:.2f} GB")
    print(f"  swap free      : {inv['swap_free_bytes']/1e9:.2f} GB of "
          f"{inv['swap_total_bytes']/1e9:.2f} GB")
    g = inv.get("gpu", {})
    if g.get("available"):
        print(f"  gpu            : {g['name']}")
        print(f"  vram           : {g['vram_total_mib']} MiB total, "
              f"{g['vram_used_mib']} MiB already used by the desktop, "
              f"{g['vram_free_mib']} MiB free")
    if gpu:
        print(f"  gpu bw         : {gpu.get('copy_bw_gbps', 0):.1f} GB/s copy, "
              f"{gpu.get('read_bw_gbps', 0):.1f} GB/s read")
        if "gemv_ms" in gpu:
            print(f"  decode GEMV    : {gpu['gemv_ms']:.3f} ms for one "
                  f"17408x5120 block ({gpu['gemv_effective_bw_gbps']:.1f} GB/s)")
    rb = d.get("ram_bandwidth")
    if rb:
        print(f"  ram bw         : {rb['copy_bw_gbps']:.1f} GB/s copy, "
              f"{rb['read_bw_gbps']:.1f} GB/s single-thread read")
    disk = d.get("disk")
    if disk:
        print(f"  nvme seq read  : {disk['sequential_read_gbps_pass2']:.2f} GB/s "
              f"(file larger than RAM, writeback drained)")
        for k, v in disk.get("random", {}).items():
            print(f"  nvme {k:14s}: {v['throughput_gbps']:.2f} GB/s  "
                  f"({v['median_latency_us']:.0f} us, {v['iops']:.0f} IOPS)")
    mm = d.get("mmap_cold")
    if mm:
        print(f"  mmap cold 1M   : {mm['throughput_gbps']:.2f} GB/s "
              f"({mm['median_latency_ms']:.2f} ms per 1 MiB touch)")


def section_inference() -> None:
    hr("INFERENCE BASELINES (measured, ollama on this machine)")
    rows = []
    for f in ("inference_safe.json", "inference_27b.json"):
        d = load(f)
        if d:
            rows += d["results"]
    if not rows:
        print("  no inference results -- run `make bench-inference`")
        return
    print(f"  {'model':>18} {'decode tok/s':>13} {'prefill tok/s':>14} "
          f"{'peak VRAM':>10} {'load s':>7}")
    for r in rows:
        if r.get("error"):
            print(f"  {r['model']:>18}   ERROR: {r['error'][:60]}")
            continue
        d = r.get("decode_tok_s") or 0.0
        p = r.get("prefill_tok_s") or 0.0
        print(f"  {r['model']:>18} {d:>13.3f} {p:>14.2f} "
              f"{r['peak_vram_mib']:>8} MiB {r.get('load_s', 0):>7.1f}")


def section_real_ffn() -> None:
    d = load("real_ffn.json")
    if not d:
        print("\nreal_ffn.json missing -- run `make verify-real-ffn`")
        return
    hr("REAL Qwen3.8-27B WEIGHTS: partitioning exactness (results/real_ffn.json)")
    g = d["geometry"]
    print(f"  layer {d['layer']}: hidden={g['hidden']} intermediate={g['intermediate']} "
          f"params={g['params']:,}")
    worst = max(v for row in d["exactness_worst_rel_error"].values()
                for v in row.values())
    print(f"  worst relative L2 error, all experts active vs dense: {worst:.3e}")
    print("  -> partitioning is exactly dense-equivalent on the real artefact")
    tp = d.get("throughput", {})
    if "dense_ms" in tp:
        print(f"\n  throughput at batch 1 (GPU):")
        print(f"    dense           {tp['dense_ms']:.3f} ms")
        for k, v in tp.items():
            if k.endswith("_all_active_ms"):
                e = k.split("_")[0]
                print(f"    {e:4s} all-active  {v:.3f} ms  "
                      f"({tp[e+'_all_active_overhead_vs_dense']:.2f}x dense)")


def section_pilot() -> None:
    d = load("pilot_sparsity.json")
    if not d:
        print("\npilot_sparsity.json missing -- run `make pilot`")
        return
    hr("PILOT SPARSITY STUDY (results/pilot_sparsity.json)")
    print(f"  model: {d['model']}  calibration: {d['calibration_chars']:,} chars")
    print("  metric: relative L2 residual of the k-expert reconstruction\n")
    print(f"  {'layer':>5} {'plan':>12} {'E':>4} {'k':>3} {'active':>7} "
          f"{'largest':>8} {'oracle':>7} {'learned':>8}")
    for L in sorted(d["layers"], key=int):
        for key, v in d["layers"][L].items():
            if not isinstance(v, dict) or "oracle" not in v:
                continue
            E = int(key.split("_E")[1])
            plan = key.split("_E")[0]
            for k, o in sorted(v["oracle"].items(), key=lambda kv: int(kv[0])):
                if k == "1" or int(k) == E // 8:
                    print(f"  {L:>5} {plan:>12} {E:>4} {k:>3} "
                          f"{100*int(k)/E:>6.1f}% "
                          f"{v['expert_sizes']['largest_share']:>8.3f} "
                          f"{o['mean_rel_residual']:>7.4f} "
                          f"{v['learned_router'][k]['mean_rel_residual']:>8.4f}")


def main() -> int:
    print("moeme result summary")
    section_hardware()
    section_inference()
    section_real_ffn()
    section_pilot()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
