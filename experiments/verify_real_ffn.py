#!/usr/bin/env python3
"""
verify_real_ffn.py -- verify dense-equivalent FFN partitioning on the REAL
Qwen3.8-27B weights (not synthetic tensors).

Why only one layer: the checkpoint is 55.6 GB, but every one of the 64 FFN
blocks has identical geometry (267,386,880 params, verified from the safetensors
headers). One block is therefore sufficient to verify the maths on real weights,
and it costs ~535 MB of ranged download instead of 55.6 GB.

What this establishes
---------------------
  1. EXACTNESS  -- the partitioned layer with all experts active reproduces the
     real dense layer to floating-point tolerance, for every plan and expert
     count, on real data.
  2. THROUGHPUT -- partitioning adds no meaningful cost to the all-active path.
  3. STRUCTURE  -- how concentrated the per-expert contributions are, i.e.
     whether a trained dense FFN has any natural structure a top-k router could
     exploit. This is the quantity that decides whether post-hoc FFN sparsifi-
     cation is viable at all.

What this does NOT establish
----------------------------
It does not measure end-to-end model quality. That needs real activations from
the full model (see docs/feasibility_report.md, "Limitations"). Inputs here are
synthetic distributions; the oracle numbers are therefore a structure probe, not
a quality prediction.

Usage:
    python3 experiments/verify_real_ffn.py --out results/real_ffn.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import torch

from moeme.ffn import (
    DenseFFN,
    PartitionedFFN,
    clustered_plan,
    contiguous_plan,
    contribution_concentration,
    interleaved_plan,
    oracle_topk_residual,
    random_plan,
)
from hf_tensor import get_tensor, index_for  # noqa: E402

REPO = "Qwen/Qwen3.8-27B"
LAYER = 1  # a Gated DeltaNet layer; FFN geometry is identical in all 64

def tensor_names(layer: int) -> list[str]:
    p = f"model.language_model.layers.{layer}.mlp"
    return [f"{p}.gate_proj.weight", f"{p}.up_proj.weight", f"{p}.down_proj.weight"]


def load_real_ffn(layer: int = LAYER) -> DenseFFN:
    idx = index_for(REPO)
    names = tensor_names(layer)
    print(f"fetching {len(names)} tensors for layer {layer} (range requests, cached)...")
    t0 = time.perf_counter()
    tensors = [get_tensor(REPO, n, idx)[0] for n in names]
    print(f"  fetched in {time.perf_counter() - t0:.1f}s  "
          f"({sum(t.numel() for t in tensors) * 2 / 1e9:.3f} GB bf16)")
    gate, up, down = tensors
    return DenseFFN(gate=gate.float(), up=up.float(), down=down.float())


def make_inputs(hidden: int, n: int, kind: str, seed: int = 0) -> torch.Tensor:
    """Synthetic inputs. 'student_t' is heavy-tailed, closer to real LLM
    activation statistics than a Gaussian, but it is NOT a real activation."""
    g = torch.Generator().manual_seed(seed)
    if kind == "gaussian":
        return torch.randn(n, hidden, generator=g)
    if kind == "student_t":
        # df=3 gives realistic outlier kurtosis; scale to unit variance
        return (torch.randn(n, hidden, generator=g)
                / torch.randn(n, hidden, generator=g).pow(2).sqrt().clamp_min(1e-6)
                / (3 ** 0.5))
    raise ValueError(kind)


def weight_structure(ffn: DenseFFN) -> dict:
    """Basic statistics that show whether the intermediate axis is homogeneous."""
    gnorm = ffn.gate.norm(dim=1)
    unorm = ffn.up.norm(dim=1)
    feats = torch.cat([ffn.gate, ffn.up], dim=1)
    # per-axis mean/median/std give a cheap view of heterogeneity
    def stats(t: torch.Tensor) -> dict:
        return {
            "mean": float(t.mean()), "std": float(t.std()),
            "min": float(t.min()), "max": float(t.max()),
            "cv": float(t.std() / t.mean().clamp_min(1e-12)),
        }
    gn = stats(gnorm)
    un = stats(unorm)
    return {
        "intermediate": ffn.intermediate,
        "hidden": ffn.hidden,
        "gate_row_norm": gn,
        "up_row_norm": un,
        "gate_row_norm_cv": gn["cv"],
        "up_row_norm_cv": un["cv"],
        "feature_dim": int(feats.shape[1]),
    }


# k-means over 17,408 points in 10,240 dimensions is O(iters * n * E * d);
# keep E modest for the clustered plan so the experiment stays tractable.
PLANS = {
    "contiguous": (lambda ffn, E: contiguous_plan(ffn.intermediate, E), (8, 32, 128)),
    "interleaved": (lambda ffn, E: interleaved_plan(ffn.intermediate, E), (8, 32, 128)),
    "random": (lambda ffn, E: random_plan(ffn.intermediate, E, seed=0), (8, 32, 128)),
    "clustered": (lambda ffn, E: clustered_plan(ffn.gate, ffn.up, E, seed=0), (8, 32)),
}


def run(args) -> dict:
    out: dict = {"repo": REPO, "layer": LAYER,
                 "note": "real Qwen3.8-27B FFN weights, synthetic inputs"}
    ffn = load_real_ffn(LAYER)
    out["geometry"] = {"hidden": ffn.hidden, "intermediate": ffn.intermediate,
                       "params": ffn.params()}
    print(f"real FFN: hidden={ffn.hidden} intermediate={ffn.intermediate} "
          f"params={ffn.params():,}")
    out["weight_structure"] = weight_structure(ffn)
    print("weight structure:", json.dumps(out["weight_structure"], indent=2))

    # ---------------- exactness ----------------
    print("\n== exactness (all experts active vs dense, real weights) ==")
    exactness = {}
    for kind in ("gaussian", "student_t"):
        x = make_inputs(ffn.hidden, 16, kind)
        row = {}
        for pname, (fn, es) in PLANS.items():
            for E in es:
                p = PartitionedFFN(ffn, fn(ffn, E))
                err = p.dense_equivalent_error(x)
                row[f"{pname}_E{E}"] = err
                assert p.plan.covers_all(), f"{pname} E={E} plan does not cover all"
        exactness[kind] = row
        worst = max(row.values())
        print(f"  {kind:10s} worst relative L2 error = {worst:.3e}")
        assert worst < 1e-5, f"exactness violated for {kind}: {worst}"
    out["exactness_worst_rel_error"] = exactness
    print("  ALL EXACT within 1e-5  (checked: 4 plans x E in {8,128} x 2 input dists)")

    # ---------------- structure / oracle ----------------
    print("\n== contribution structure (does a top-k router have anything to find?) ==")
    oracle = {}
    for kind in ("gaussian", "student_t"):
        x = make_inputs(ffn.hidden, 256, kind)
        per_kind = {}
        for pname, (fn, es) in PLANS.items():
            for E in [e for e in es if e >= 32]:
                p = PartitionedFFN(ffn, fn(ffn, E))
                outs = p.expert_outputs(x)
                conc = contribution_concentration(outs)
                res = {k: oracle_topk_residual(outs, k)["mean_rel_residual"]
                       for k in (1, 2, 4, 8, 16) if k <= E}
                per_kind[f"{pname}_E{E}"] = {"concentration": conc,
                                             "oracle_mean_rel_residual": res}
        oracle[kind] = per_kind
        print(f"  -- {kind} --")
        for key, v in per_kind.items():
            best = min(v["oracle_mean_rel_residual"].items(),
                       key=lambda kv: kv[1])
            print(f"    {key:20s} participation_ratio="
                  f"{v['concentration']['participation_ratio']:7.2f}  "
                  f"best oracle residual (k={best[0]}) = {best[1]:.4f}")
    out["oracle_structure"] = oracle

    # ---------------- throughput ----------------
    print("\n== throughput (real weights, GPU if available) ==")
    tp = {}
    if torch.cuda.is_available():
        dev = "cuda"
        try:
            g = ffn.gate.to(dev).to(torch.bfloat16)
            u = ffn.up.to(dev).to(torch.bfloat16)
            d = ffn.down.to(dev).to(torch.bfloat16)
            gffn = DenseFFN(gate=g, up=u, down=d)
            x = torch.randn(1, gffn.hidden, dtype=torch.bfloat16, device=dev)

            def timeit(fn, iters=30):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                ts.sort()
                return ts[len(ts) // 2]

            t_dense = timeit(lambda: gffn.forward(x))
            tp["dense_ms"] = t_dense * 1e3
            for E, k in ((32, 2), (128, 4)):
                p = PartitionedFFN(gffn, contiguous_plan(gffn.intermediate, E))
                t_all = timeit(lambda: p.all_active_forward(x))
                idx = torch.topk(torch.rand(1, E, device=dev), k, dim=1).indices
                t_topk = timeit(lambda: p.topk_forward(x, idx))
                tp[f"E{E}_all_active_ms"] = t_all * 1e3
                tp[f"E{E}_topk{k}_ms"] = t_topk * 1e3
                tp[f"E{E}_all_active_overhead_vs_dense"] = t_all / t_dense
            print(json.dumps(tp, indent=2))
            del g, u, d, gffn
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            tp["error"] = f"cuda OOM: {e}"
            print("  skipped: ", tp["error"])
    else:
        tp["error"] = "no cuda"
    out["throughput"] = tp

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real_ffn.json")
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
