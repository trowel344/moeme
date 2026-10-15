#!/usr/bin/env python3
"""mocm_distill.py -- distill skippable MLP layers into narrow students.

No partition, no router, no switching: each target layer's dense MLP (teacher,
frozen) trains a small random-init swiGLU student (default 1/4 width) on MSE
against the teacher's own outputs, on fresh tokens. Then plug students in and
measure domain ppl.

Why this instead of self-teach (which managed 9% transfer)? Self-teach used
197 calibration tokens for ~44M params, a sliced student, and no schedule.
This uses ~1M fresh tokens, a dense-narrow student, and the evo recipe.

  python3 experiments/mocm_distill.py --layers 9 10 8 12 --frac 0.25 \
      --out results/mocm_distill.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _load_sibling(name: str):
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_dst_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
DOM = _load_sibling("mocm_domains")


class NarrowStudent(nn.Module):
    """Dense-narrow swiGLU with HF weight layout (gate/up: I,H; down: H,I)."""

    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate = nn.Parameter(torch.randn(inter, hidden) * 0.02)
        self.up = nn.Parameter(torch.randn(inter, hidden) * 0.02)
        self.down = nn.Parameter(torch.randn(hidden, inter) * 0.02)

    def forward(self, x):
        h = F.silu(F.linear(x, self.gate)) * F.linear(x, self.up)
        return F.linear(h, self.down)


class ThreshMLP(nn.Module):
    """Zero the smallest-|h| fraction q of FFN intermediate activations.

    TEAL-style training-free sparsity: quality-preserving to q~0.4 on SwiGLU,
    no router, no switching. Needs sparse kernels for wall-time; here it
    measures the quality side of the byte tradeoff.
    """

    def __init__(self, mlp, q: float):
        super().__init__()
        self.m = mlp
        self.q = q

    def forward(self, x):
        g = F.silu(F.linear(x, self.m.gate_proj.weight))
        u = F.linear(x, self.m.up_proj.weight)
        h = g * u
        with torch.no_grad():
            thr = torch.quantile(h.abs().float().reshape(-1), self.q)
            mask = (h.abs() >= thr).to(h.dtype)
        return F.linear(h * mask, self.m.down_proj.weight)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layers", nargs="+", default=["9", "10", "8", "12"])
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--train-file", default=".cache/moeme/evo_fresh.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sequential", action="store_true",
                    help="train layers bottom-up one at a time (frozen below), "
                         "splitting --steps evenly: tests whether compounding "
                         "is order-dependent at identical token budget")
    ap.add_argument("--save", default=None,
                    help="save student state dicts here for reuse")
    ap.add_argument("--load", default=None,
                    help="eval-only: load students here, skip training")
    ap.add_argument("--out", default="results/mocm_distill.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    amp_dtype = POC.pick_amp_dtype(device)
    layers = [int(v) for v in args.layers]
    wdtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=wdtype).to(device)
    model.eval()

    fargs = argparse.Namespace(text_file=args.train_file, offline=False,
                               dataset="x", corpus_chars=0)
    ids, _ = POC.load_token_stream(fargs, tok)
    blocks = POC.make_blocks(ids, args.seq_len)
    train_blocks = blocks[:-4]
    eval5 = {d: [(x.to(device), y.to(device)) for x, y in DOM.domain_blocks(tok, d)]
             for d in DOM.DOMAINS}
    dense_dom = DOM.domain_ppls(model, eval5, device, amp_dtype)
    print("dense: " + " ".join(f"{d}:{dense_dom[d]:.1f}" for d in DOM.DOMAINS),
          flush=True)

    hidden = int(model.model.layers[layers[0]].mlp.gate_proj.weight.shape[1])
    inter_full = int(model.model.layers[layers[0]].mlp.gate_proj.weight.shape[0])
    inter = max(64, int(inter_full * args.frac))
    print(f"hidden {hidden} teacher-inter {inter_full} student-inter {inter}",
          flush=True)

    # teachers: frozen originals; students: fresh narrow nets (or loaded)
    teachers, students = {}, {}
    for i in layers:
        mlp = model.model.layers[i].mlp
        teachers[i] = {"gate": mlp.gate_proj.weight.detach(),
                       "up": mlp.up_proj.weight.detach(),
                       "down": mlp.down_proj.weight.detach()}
        st = NarrowStudent(hidden, inter).to(device=device, dtype=wdtype)
        students[i] = st
    if args.load:
        ck = torch.load(args.load, map_location=device, weights_only=True)
        assert ck["layers"] == layers and ck["inter"] == inter, \
            f"ckpt {ck['layers']}/{ck['inter']} != {layers}/{inter}"
        for i in layers:
            students[i].load_state_dict(ck["students"][str(i)])
        print(f"loaded {args.load}, skipping training", flush=True)
        args.steps = 0

    # capture inputs per target layer (model keeps ORIGINAL mlps throughout
    # training; students train on captured (input, teacher-output) pairs)
    caps = {}
    for i in layers:
        model.model.layers[i].mlp.register_forward_pre_hook(
            lambda _m, a, i=i: caps.setdefault(i, []).append(a[0].detach())
            if len(caps.setdefault(i, [])) < 1 else None)

    params = [p for s in students.values() for p in s.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))
    steps = args.steps

    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.perf_counter()
    order = sorted(layers) if args.sequential else [None]
    per = max(1, steps // len(order)) if args.sequential else steps
    # schedule horizon: per-block when sequential (each layer gets its own
    # warmup+decay), global when joint -- otherwise early layers train
    # entirely inside warmup and underfit catastrophically
    horizon = per if args.sequential else steps
    sched_warm = max(1, horizon // 10)

    def lr_at(s):
        if s < sched_warm:
            return args.lr * (s + 1) / sched_warm
        t = (s - sched_warm) / max(1, horizon - sched_warm)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
    global_step = 0
    for block, target in enumerate(order):
        active_ids = [target] if target is not None else layers
        for p in params:
            p.requires_grad_(False)
        for i in active_ids:
            for p in students[i].parameters():
                p.requires_grad_(True)
        current = [p for s in [students[i] for i in active_ids]
                   for p in s.parameters()]
        for step_in in range(per):
            step = global_step
            global_step += 1
            # schedule position: per-block when sequential, global when joint
            spos = step_in if args.sequential else min(step, steps - 1)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(spos)
            opt.zero_grad(set_to_none=True)
            j = int(torch.randint(len(train_blocks), (1,), generator=gen).item())
            x, _ = train_blocks[j]
            x = x.to(device)
            for v in caps.values():
                v.clear()
            with torch.no_grad():
                model.eval()
                model(x)
            loss = 0.0
            for i in active_ids:
                xin = caps[i][0].reshape(-1, hidden).to(wdtype)
                t = teachers[i]
                with torch.no_grad():
                    h = F.silu(F.linear(xin.float(), t["gate"].float())) \
                        * F.linear(xin.float(), t["up"].float())
                    yt = F.linear(h, t["down"].float())
                ys = students[i](xin).float()
                loss = loss + F.mse_loss(ys, yt) / yt.detach().pow(2).mean().clamp_min(1e-12)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(current, 1.0)
            opt.step()
            if step % max(1, min(steps // 5, 25)) == 0 or step == steps - 1:
                tag = f"L{target}" if target is not None else "joint"
                print(f"  step {step:4d}[{tag}] loss {float(loss):.4f} "
                      f"elapsed {time.perf_counter()-t0:5.0f}s", flush=True)

    # plug students in and score
    originals = [(i, model.model.layers[i].mlp) for i in layers]
    for i in layers:
        model.model.layers[i].mlp = students[i].to(device)
    model.eval()
    got = DOM.domain_ppls(model, eval5, device, amp_dtype)
    mx = sum(got[d] / dense_dom[d] for d in DOM.DOMAINS) / len(DOM.DOMAINS)
    print("distilled: " + " ".join(f"{d}:{got[d]:.1f}" for d in DOM.DOMAINS),
          flush=True)
    print(f"mean_xdense {mx:.3f} at {args.frac:.0%} width, {len(layers)} layers",
          flush=True)
    for i, m in originals:
        model.model.layers[i].mlp = m
    if args.save:
        torch.save({"layers": layers, "hidden": hidden, "inter": inter,
                    "students": {str(i): students[i].state_dict() for i in layers}},
                   args.save)
        print(f"saved {args.save}", flush=True)
    json.dump({"layers": layers, "frac": args.frac, "steps": steps,
               "dense_dom": dense_dom, "distilled_dom": got,
               "mean_xdense": mx}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
