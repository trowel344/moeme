#!/usr/bin/env python3
"""
residency_table.py -- what quantisation level fits this machine resident, and
what decode rate follows?

The central engineering fact of this project: decode reads every active weight
once per token, so throughput is bandwidth-bound. The only regime with usable
speed is FULL RESIDENCY (no per-token storage traffic). This script computes the
residency threshold from measured tier capacities and the measured tier
bandwidths, so the claim can be re-derived rather than trusted.

Constants come from results/hardware.json (see `make doctor`).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moeme.accounting import (  # noqa: E402
    MemoryTier,
    Qwen38_27B,
    predicted_decode_tok_s,
)

GIB = 1024 ** 3

# measured effective capacities and bandwidths (configs/tiers.json)
VRAM_USABLE_GIB = 3.6
RAM_USABLE_GIB = 10.2
BW_GPU = 246.8e9
BW_RAM = 40.7e9
BW_DISK = 1.5e9

QUANTS = [
    ("Q2_K", 2.8), ("Q3_K_M", 3.4), ("Q4_K_S", 4.4), ("Q4_K_M", 4.9),
    ("Q5_K_M", 5.6), ("Q6_K", 6.6), ("Q8_0", 8.5), ("bf16", 16.0),
]


def tiers() -> tuple[MemoryTier, MemoryTier, MemoryTier]:
    return (
        MemoryTier("gpu", VRAM_USABLE_GIB * GIB, BW_GPU),
        MemoryTier("ram", RAM_USABLE_GIB * GIB, BW_RAM),
        MemoryTier("disk", 1e15, BW_DISK),
    )


def main() -> int:
    params = Qwen38_27B().measured_total
    resident_gb = (VRAM_USABLE_GIB + RAM_USABLE_GIB) * GIB / 1e9

    print(f"Qwen3.8-27B: {params:,} params")
    print(f"usable fast memory: {VRAM_USABLE_GIB} GiB VRAM + {RAM_USABLE_GIB} GiB RAM "
          f"= {resident_gb:.1f} GB")
    print(f"measured bandwidths: gpu {BW_GPU/1e9:.0f} GB/s, ram {BW_RAM/1e9:.0f} GB/s, "
          f"disk {BW_DISK/1e9:.2f} GB/s\n")
    print(f"{'quant':>8} {'bpw':>5} {'model GB':>9} {'per-token':>10} "
          f"{'streamed GB':>12} {'pred tok/s':>11} {'regime':>12}")
    print("-" * 72)
    for name, bpw in QUANTS:
        r = predicted_decode_tok_s(
            active_params=params, bits_per_param=bpw, gpu=tiers()[0],
            ram=tiers()[1], disk=tiers()[2],
        )
        streamed = r["placement"].get("disk", {}).get("bytes", 0.0)
        regime = "disk-bound" if streamed > 0.1e9 else "RESIDENT"
        print(f"{name:>8} {bpw:>5.1f} {params*bpw/8/1e9:>9.1f} "
              f"{params*bpw/8/1e9:>10.1f} {streamed/1e9:>12.2f} "
              f"{r['tok_s']:>11.2f} {regime:>12}")

    print()
    print("Residency threshold (largest model that fits without streaming):")
    resident_bytes = (VRAM_USABLE_GIB + RAM_USABLE_GIB) * GIB
    for name, bpw in QUANTS:
        cap = resident_bytes / (bpw / 8) / 1e9
        mark = "  <- Qwen3.8-27B fits" if cap >= params / 1e9 else ""
        print(f"  at {name:>7} ({bpw} bpw, {bpw/8:.3f} bytes/param): "
              f"{cap:>5.1f}B params{mark}")
    print(f"\n  Qwen3.8-27B is {params/1e9:.2f}B -> above the Q4 threshold, "
          f"below the Q3 threshold.")

    print()
    print("The 9B-active question, separated from the total-size question:")
    r = predicted_decode_tok_s(active_params=9e9, bits_per_param=4.9,
                               gpu=tiers()[0], ram=tiers()[1], disk=tiers()[2])
    print(f"  9B active at 4.9 bpw = {9e9*4.9/8/1e9:.2f} GB read per token.")
    print(f"  If fully resident: {r['tok_s']:.1f} tok/s (bottleneck: {r['bottleneck']}).")
    print("  So 9B *active* is not itself a speed problem -- a fully resident 9B-active")
    print("  model would decode fine. The problem is that a model with 9B active and")
    print("  near-27B quality needs far more than 9B *total*, and the total has to fit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
