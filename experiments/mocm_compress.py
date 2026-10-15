#!/usr/bin/env python3
"""mocm_compress.py -- what compresses a REAL Qwen3.8-27B FFN layer?

Three training-free-or-cheap probes on range-fetched 27B weights, one layer:
  A. SVD: relative output error vs kept rank (per matrix, gate/up/down).
  B. Magnitude pruning: error vs fraction zeroed (unstructured).
  C. Narrow student: train a 25%-width swiGLU to mimic the layer on synthetic
     inputs; compare against the slicing oracle (0.91) -- learning vs cutting.

  python3 experiments/mocm_compress.py --layer 23 --out results/mocm_compress.json
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
import torch.nn.functional as F

from hf_tensor import get_tensor, index_for

REPO = "Qwen/Qwen3.8-27B"


def tensor_names(layer: int) -> list[str]:
    p = f"model.language_model.layers.{layer}.mlp"
    return [f"{p}.gate_proj.weight", f"{p}.up_proj.weight",
            f"{p}.down_proj.weight"]


def dense_out(x, g, u, d):
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def synth_inputs(H, n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.empty(n, H).t_().exponential_().t() \
        * torch.randint(0, 2, (n, H), generator=g).float().mul(2).add(-1)
    return t * (H ** -0.5) * 8


@torch.no_grad()
def svd_curve(W, x_full, ranks=(512, 1024, 2048, 4096)):
    """Error of rank-r approx used inside the layer: needs gate+up factors
    applied jointly, so measure per-matrix spectral tail instead (upper bound
    signal) plus a true joint test at one rank."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    total = (S ** 2).sum()
    tail = {r: float((S[r:] ** 2).sum() / total) for r in ranks}
    return tail, (U, S, Vh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--out", default="results/mocm_compress.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student-steps", type=int, default=1000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = index_for(REPO)
    print(f"fetching layer {args.layer} ...", flush=True)
    t0 = time.perf_counter()
    tensors = [get_tensor(REPO, n, idx)[0] for n in tensor_names(args.layer)]
    print(f"  fetched in {time.perf_counter()-t0:.1f}s", flush=True)
    gate, up, down = [t.float().cpu() for t in tensors]
    H, I = gate.shape[1], gate.shape[0]
    print(f"hidden {H} intermediate {I}", flush=True)
    rep = {"layer": args.layer, "hidden": H, "intermediate": I}

    x = synth_inputs(H).float()
    with torch.no_grad():
        ref = dense_out(x, gate, up, down)

    # A. spectral tails (fraction of weight energy beyond rank r)
    print("SVD tails ...", flush=True)
    rep["svd"] = {}
    factors = {}
    for name, W in (("gate", gate), ("up", up), ("down", down)):
        tail, (U, S, Vh) = svd_curve(W, x)
        rep["svd"][name] = tail
        factors[name] = (U, S, Vh)
        print(f"  {name}: " + " ".join(f"r{r}:{tail[r]:.4f}" for r in tail),
              flush=True)
    # joint rank test: truncate gate+up to r=2048 (keep down full) -> error
    with torch.no_grad():
        r = 2048
        Ug, Sg, Vg = factors["gate"]
        Uu, Su, Vu = factors["up"]
        gg = (Ug[:, :r] * Sg[:r]) @ Vg[:r]
        uu = (Uu[:, :r] * Su[:r]) @ Vu[:r]
        approx = dense_out(x, gg, uu, down)
        rep["svd_joint_r2048_err"] = \
            float((ref - approx).norm() / ref.norm().clamp_min(1e-12))
    print(f"  joint r=2048 (gate+up) layer-output err: "
          f"{rep['svd_joint_r2048_err']:.4f}", flush=True)

    # B. magnitude pruning (whole-layer, unstructured)
    print("magnitude pruning ...", flush=True)
    rep["prune"] = {}
    with torch.no_grad():
        allw = torch.cat([gate.reshape(-1), up.reshape(-1), down.reshape(-1)])
        sample = allw[torch.randperm(allw.numel(),
                                     generator=torch.Generator().manual_seed(0))[:10_000_000]]
        for frac in (0.10, 0.25, 0.50, 0.75, 0.90):
            thr = torch.quantile(sample.abs(), frac)
            gp, up_, dp = (torch.where(w.abs() < thr, torch.zeros(()), w)
                           for w in (gate, up, down))
            got = dense_out(x, gp, up_, dp)
            e = float((ref - got).norm() / ref.norm().clamp_min(1e-12))
            rep["prune"][frac] = e
            print(f"  zeros={frac:.0%} err {e:.4f}", flush=True)

    # C. narrow student vs slicing oracle, same layer + inputs
    print("narrow student (25% width) ...", flush=True)
    from moeme.ffn import PartitionedFFN, DenseFFN, random_plan, \
        oracle_topk_residual
    dense = DenseFFN(gate=gate, up=up, down=down)
    part = PartitionedFFN(dense, random_plan(I, 8, seed=args.seed))
    oracle = oracle_topk_residual(part.expert_outputs(x), 1)["mean_rel_residual"]
    rep["slice_oracle_E8k1"] = oracle
    print(f"  slice oracle E8/k1: {oracle:.4f}", flush=True)

    SI = I // 4
    gg = torch.nn.Parameter(torch.randn(SI, H) * 0.02)
    uu = torch.nn.Parameter(torch.randn(SI, H) * 0.02)
    dd = torch.nn.Parameter(torch.randn(H, SI) * 0.02)
    opt = torch.optim.AdamW([gg, uu, dd], lr=3e-4, betas=(0.9, 0.95))
    warm = max(1, args.student_steps // 10)

    def lr_at(s):
        import math
        if s < warm:
            return 3e-4 * (s + 1) / warm
        t = (s - warm) / max(1, args.student_steps - warm)
        return 3e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.perf_counter()
    for step in range(args.student_steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        xb = synth_inputs(H, n=64, seed=args.seed + step + 999).float()
        with torch.no_grad():
            yt = dense_out(xb, gate, up, down)
        h = F.silu(F.linear(xb, gg)) * F.linear(xb, uu)
        ys = F.linear(h, dd)
        loss = F.mse_loss(ys, yt) / yt.pow(2).mean().clamp_min(1e-12)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gg, uu, dd], 1.0)
        opt.step()
    with torch.no_grad():
        h = F.silu(F.linear(x, gg)) * F.linear(x, uu)
        got = F.linear(h, dd)
        serr = float((ref - got).norm() / ref.norm().clamp_min(1e-12))
    rep["student_err"] = serr
    rep["student_steps"] = args.student_steps
    rep["student_width_frac"] = 0.25
    print(f"  student err {serr:.4f} in {time.perf_counter()-t0:.0f}s "
          f"(vs slice oracle {oracle:.4f})", flush=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
