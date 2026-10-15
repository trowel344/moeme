#!/usr/bin/env python3
"""mocm_bytes.py -- byte economics of converting the REAL Qwen3.8-27B.

For a spread of real 27B FFN layers (range-fetched, cached), measures the
frozen oracle residual at k=1 over E=8/16/32/64 (the quality proxy for a
training-free cut), then converts each width to exact bytes/token and
predicted tok/s on THIS machine via the measured tier model.

Answers: is there any (width -> bytes -> speed) point where a converted 27B
both fits the fast tiers and keeps the layer output? No training involved;
the oracle is the ceiling -- a real router can only do worse.

  python3 experiments/mocm_bytes.py --out results/mocm_bytes.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import torch

from moeme.ffn import (DenseFFN, PartitionedFFN, contribution_concentration,
                       oracle_topk_residual, random_plan)
from moeme.accounting import (MemoryTier, Qwen38_27B, bytes_for_params,
                              predicted_decode_tok_s)
from hf_tensor import get_tensor, index_for

REPO = "Qwen/Qwen3.8-27B"
LAYERS = [0, 8, 23, 32, 47, 63]
EXPERTS = [8, 16, 32, 64]


def tensor_names(layer: int) -> list[str]:
    p = f"model.language_model.layers.{layer}.mlp"
    return [f"{p}.gate_proj.weight", f"{p}.up_proj.weight",
            f"{p}.down_proj.weight"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mocm_bytes.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    geo = Qwen38_27B()
    idx = index_for(REPO)
    per_layer, per_cfg = {}, {E: [] for E in EXPERTS}
    for li in LAYERS:
        names = tensor_names(li)
        print(f"layer {li}: fetching ...", flush=True)
        tensors = [get_tensor(REPO, n, idx)[0] for n in names]
        dense = DenseFFN(gate=tensors[0].float(), up=tensors[1].float(),
                         down=tensors[2].float())
        H, I = dense.hidden, dense.intermediate
        # synthetic heavy-tailed inputs (same caveat as real_ffn.json)
        g = torch.Generator().manual_seed(args.seed)
        x = torch.empty(256, H).t_().exponential_().t() \
            * torch.randint(0, 2, (256, H), generator=g).float().mul(2).add(-1)
        x = x * (H ** -0.5) * 8
        for E in EXPERTS:
            plan = random_plan(I, E, seed=args.seed)
            part = PartitionedFFN(dense, plan)
            outs = part.expert_outputs(x)
            r = oracle_topk_residual(outs, 1)
            c = contribution_concentration(outs)
            per_cfg[E].append(r["mean_rel_residual"])
            per_layer.setdefault(li, {})[E] = {
                "oracle_k1": r["mean_rel_residual"],
                "participation_ratio": c["participation_ratio"]}
        print(f"  layer {li}: " + " ".join(
            f"E{E}={per_layer[li][E]['oracle_k1']:.3f}" for E in EXPERTS),
            flush=True)
        del dense, tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # bytes/token + predicted speed per width on this machine's tiers
    tiers = json.load(open("configs/tiers.json"))
    bw = tiers["measured_bandwidth_bps"]
    cap = tiers["effective_capacity"]
    gpu = MemoryTier("gpu", cap["vram_usable_gib"] * 2 ** 30,
                     bw["gpu_memory"])
    ram = MemoryTier("ram", cap["ram_usable_gb"] * 1e9, bw["ram_copy"])
    disk = MemoryTier("disk", float("inf"), bw["nvme_sequential_read_1M_direct"])
    backbone = geo.irreducible_active_params()
    ffn_full = geo.ffn_params()
    rows = {}
    for E in EXPERTS:
        ffn_active = ffn_full // E
        active = backbone + ffn_active
        # Q4_K_M ~ 4.5 bits/param effective on this model class
        pred = predicted_decode_tok_s(active_params=active, bits_per_param=4.5,
                                      gpu=gpu, ram=ram, disk=disk)
        rows[E] = {
            "active_B": active / 1e9,
            "active_ffn_B": ffn_active / 1e9,
            "gb_per_token": pred["active_bytes"] / 1e9,
            "pred_tok_s": pred["tok_s"],
            "bottleneck": pred["bottleneck"],
            "mean_oracle_k1": sum(per_cfg[E]) / len(per_cfg[E]),
        }
        print(f"E={E:2d} active {active/1e9:5.2f}B "
              f"{pred['active_bytes']/1e9:5.2f} GB/tok -> "
              f"{pred['tok_s']:6.2f} tok/s ({pred['bottleneck']}) | "
              f"oracle {rows[E]['mean_oracle_k1']:.3f}", flush=True)
    json.dump({"layers": per_layer, "widths": rows,
               "backbone_B": backbone / 1e9, "ffn_full_B": ffn_full / 1e9},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
