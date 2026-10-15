#!/usr/bin/env python3
"""mocm_depth.py -- per-sequence depth routing probe (mixture-of-depths).

No slicing, no training: if a domain only needs a subset of MLP layers, skip
the rest (identity) for that domain. Active layers stay exact dense.

  1. contribution: ||mlp(x)|| / ||x|| per layer per domain (one forward each)
  2. skip: bypass bottom-k MLPs (uniform vs domain-specific ranking), ppl cost
  3. verdict: does domain-specific skipping beat uniform skipping?

  python3 experiments/mocm_depth.py --out results/mocm_depth.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _load_sibling(name: str):
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_dep_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
DOM = _load_sibling("mocm_domains")


class SkipMLP(torch.nn.Module):
    """Identity stand-in for an MLP (residual stream passes through)."""

    def forward(self, x):
        return torch.zeros_like(x)


def block_passthrough_hook(_m, args, _out):
    """Skip a whole decoder block: output hidden = input hidden, keep rest."""
    inp = args[0]
    if torch.is_tensor(_out):
        return inp
    return (inp,) + tuple(_out[1:])


def zero_output_hook(_m, _args, _out):
    """Zero a submodule's output (skip attention, keep residual path)."""
    if torch.is_tensor(_out):
        return torch.zeros_like(_out)
    return (torch.zeros_like(_out[0]),) + tuple(_out[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--unit", default="mlp", choices=["mlp", "block", "attn"],
                    help="mlp: bypass MLP only; block: bypass whole decoder block; "
                         "attn: zero attention output (keep norms+MLP)")
    ap.add_argument("--out", default="results/mocm_depth.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = POC.pick_amp_dtype(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16).to(device)
    model.eval()
    layers = list(range(len(model.model.layers)))
    eval5 = {d: [(x.to(device), y.to(device)) for x, y in DOM.domain_blocks(tok, d)]
             for d in DOM.DOMAINS}

    # 1. contribution per layer per domain (reuse cache for block runs)
    contrib = None
    if args.unit == "block" and os.path.exists(".cache/moeme/depth_contrib.pt"):
        contrib = torch.load(".cache/moeme/depth_contrib.pt", weights_only=True)
    if contrib is None:
        contrib = {}
        for dom, blocks in eval5.items():
            acc = {i: [] for i in layers}
            for x, y in blocks:
                caps = {}
                hs = [model.model.layers[i].mlp.register_forward_pre_hook(
                    lambda _m, a, i=i: caps.setdefault(i, a[0].detach())) for i in layers]
                hs += [model.model.layers[i].mlp.register_forward_hook(
                    lambda _m, _a, o, i=i: caps.setdefault(f"y{i}", o.detach())) for i in layers]
                with torch.no_grad():
                    model(x)
                for h in hs:
                    h.remove()
                for i in layers:
                    r = float(caps[i].float().norm() / caps[f"y{i}"].float().norm().clamp_min(1e-12))
                    r = float((caps[f"y{i}"].float().norm() / caps[i].float().norm().clamp_min(1e-12)))
                    acc[i].append(r)
            contrib[dom] = {i: sum(v) / len(v) for i, v in acc.items()}
            order = sorted(layers, key=lambda i: contrib[dom][i])
            print(f"{dom}: bottom-6 {order[:6]} top-3 {order[-3:]}", flush=True)

    torch.save(contrib, ".cache/moeme/depth_contrib.pt")

    # 2. skip evals: uniform bottom-k (by mean rank) vs domain bottom-k
    mean_rank = sorted(layers, key=lambda i: sum(contrib[d][i] for d in contrib) / len(contrib))
    orig = [(i, model.model.layers[i].mlp) for i in layers]
    results = {"contrib": {d: {str(i): v for i, v in c.items()} for d, c in contrib.items()},
               "runs": []}
    base = DOM.domain_ppls(model, eval5, device, amp_dtype)
    results["dense"] = base
    print("dense: " + " ".join(f"{d}:{base[d]:.1f}" for d in DOM.DOMAINS), flush=True)

    patterns = {}
    for k in (4, 8, 12):
        patterns[f"uni-k{k}"] = ("all", mean_rank[:k])
        for d in DOM.DOMAINS:
            own = sorted(layers, key=lambda i: contrib[d][i])[:k]
            patterns[f"{d}-k{k}"] = (d, own)
    for name, (scope, skip) in patterns.items():
        hdl = []
        if args.unit == "mlp":
            for i in skip:
                model.model.layers[i].mlp = SkipMLP().to(device)
        elif args.unit == "attn":
            for i in skip:
                blk = model.model.layers[i]
                attn = getattr(blk, "self_attn", None) or \
                    getattr(blk, "linear_attn", None)
                hdl.append(attn.register_forward_hook(zero_output_hook))
        else:
            for i in skip:
                hdl.append(model.model.layers[i].register_forward_hook(
                    block_passthrough_hook))
        doms = [scope] if scope != "all" else DOM.DOMAINS
        row = {}
        for d in (doms if scope != "all" else DOM.DOMAINS):
            row[d] = DOM.domain_ppls(model, {d: eval5[d]}, device, amp_dtype)[d]
        for i, m in orig:
            model.model.layers[i].mlp = m
        for h in hdl:
            h.remove()
        samb = {d: round(row[d] / base[d], 3) for d in row}
        print(f"{name} skip{skip}: " + " ".join(f"{d}:{samb[d]:.2f}x" for d in sorted(row)),
              flush=True)
        results["runs"].append({"pattern": name, "scope": scope, "skip": skip,
                                "ppl": row, "xdense": samb})
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
