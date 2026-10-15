#!/usr/bin/env python3
"""mocm_lora.py -- frozen-backbone LoRA expert-choice MoE prototype (v-next).

Per converted layer (easy layers only):
  * base dense MLP frozen, never partitioned -- the basis stays intact
  * E LoRA experts (rank r) on gate/up/down; B=0 init => exact dense at step 0
  * expert-choice: each expert claims top-C tokens by affinity (C = T*k/E);
    unclaimed tokens fall back to exact dense output (quality floor)
  * NO shared dense path (the 95% attractor), NO aux-loss
  * loss = 0.5 * CE + 0.5 * KL(student || frozen dense teacher)

Success: domain ppl ~= distilled-4L (<=1.15x) AND affinity skews by domain
(specialization), with perfect load balance by construction.

  python3 experiments/mocm_lora.py --out results/mocm_lora.json
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
    spec = importlib.util.spec_from_file_location(f"_lx_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
DOM = _load_sibling("mocm_domains")


class LoRAExpertFFN(nn.Module):
    """One dense MLP + E LoRA experts with expert-choice dispatch."""

    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                 n_experts=8, top_k=2, rank=16, seed=0, b_init=0.0):
        super().__init__()
        I, H = gate.shape
        self.register_buffer("Wg", gate.detach().clone())
        self.register_buffer("Wu", up.detach().clone())
        self.register_buffer("Wd", down.detach().clone())
        self.E, self.k = n_experts, top_k
        g = torch.Generator().manual_seed(seed)
        self.Ag = nn.Parameter(torch.randn(n_experts, rank, H, generator=g) * 0.02)
        self.Bg = nn.Parameter(torch.randn(n_experts, I, rank, generator=g) * b_init)
        self.Au = nn.Parameter(torch.randn(n_experts, rank, H, generator=g) * 0.02)
        self.Bu = nn.Parameter(torch.randn(n_experts, I, rank, generator=g) * b_init)
        self.Ad = nn.Parameter(torch.randn(n_experts, rank, I, generator=g) * 0.02)
        self.Bd = nn.Parameter(torch.randn(n_experts, H, rank, generator=g) * b_init)
        self.router = nn.Linear(H, n_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.force_expert: int | None = None
        self.last_affinity = None

    def set_force_expert(self, e: int | None) -> None:
        self.force_expert = e

    def base_forward(self, x):
        h = F.silu(F.linear(x, self.Wg)) * F.linear(x, self.Wu)
        return F.linear(h, self.Wd)

    def expert_forward(self, x, e: int):
        Wg = self.Wg + self.Bg[e] @ self.Ag[e]
        Wu = self.Wu + self.Bu[e] @ self.Au[e]
        Wd = self.Wd + self.Bd[e] @ self.Ad[e]
        h = F.silu(F.linear(x, Wg)) * F.linear(x, Wu)
        return F.linear(h, Wd)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        T = x2.shape[0]
        with torch.no_grad():
            base = self.base_forward(x2.to(self.Wg.dtype)).float()
        if self.force_expert is not None:
            ye = self.expert_forward(x2.to(self.Wg.dtype),
                                     self.force_expert).float()
            self.last_affinity = torch.zeros(T, self.E)
            return ye.to(x.dtype).reshape(shape)
        s = self.router(x2.float())                      # (T, E) affinity
        C = max(1, T * self.k // self.E)
        top = torch.topk(s, C, dim=0).indices            # (C, E) claimed tokens
        claim = torch.zeros(T, self.E, dtype=torch.bool, device=x2.device)
        claim.scatter_(0, top, True)
        w = torch.softmax(s, dim=1)                      # unbiased weights
        y = base.clone()
        claimed = claim.any(dim=1)
        # expert outputs only where claimed; unclaimed keep exact dense
        for e in range(self.E):
            tok = claim[:, e]
            if not bool(tok.any()):
                continue
            ye = self.expert_forward(x2[tok].to(self.Wg.dtype), e).float()
            wsum = w[tok][:, e].clamp_min(1e-6)
            nclaim = claim[tok].sum(dim=1).clamp_min(1).to(w.dtype)
            y[tok] += (ye - base[tok]) * (wsum / nclaim).unsqueeze(1)
        self.last_affinity = s.detach()
        self.last_claim = claim.detach()
        return y.to(x.dtype).reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layers", nargs="+", default=["9", "10", "8", "12"])
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kl", type=float, default=0.5)
    ap.add_argument("--domain-loss", type=float, default=0.0,
                    help="supervised domain-routing CE weight: batches come from "
                         "labeled BTX sources, mean router affinity must classify "
                         "the domain (MoELoRA-style contrastive pressure)")
    ap.add_argument("--b-init", type=float, default=0.01,
                    help="B init std (0 = exact dense at step 0 but router/A "
                         "get zero grad until B moves; small nonzero wakes all)")
    ap.add_argument("--btx", action="store_true",
                    help="BTX stage-2 first: each expert alone on its own domain "
                         "(LM only, no KL anchor) to force divergence, then joint")
    ap.add_argument("--bsteps", type=int, default=40)
    ap.add_argument("--load", default=None,
                    help="start from ckpt states instead of init")
    ap.add_argument("--router-only", action="store_true",
                    help="train router (+domain clf) only, experts frozen: "
                         "tests whether routing is learnable given diverged experts")
    ap.add_argument("--save", default=None)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--train-file", default=".cache/moeme/evo_fresh.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mocm_lora.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    amp_dtype = POC.pick_amp_dtype(device)
    layers = [int(v) for v in args.layers]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    model.eval()

    fargs = argparse.Namespace(text_file=args.train_file, offline=False,
                               dataset="x", corpus_chars=0)
    dom_names = None
    if args.domain_loss > 0:
        # labeled domain-balanced batches from the BTX sources
        import glob as _glob
        dom_names = ["code1", "code2", "math1", "math2",
                     "chat1", "chat2", "wiki1", "web1"]
        dom_blocks = {}
        for n in dom_names:
            with open(f".cache/moeme/btx_src/{n}.txt") as fh:
                ids_d = tok(fh.read(), return_tensors="pt").input_ids[0]
            dom_blocks[n] = POC.make_blocks(ids_d, args.seq_len)
        train_pick = [(n, b) for n in dom_names for b in dom_blocks[n]]
    else:
        ids, _ = POC.load_token_stream(fargs, tok)
        blocks = POC.make_blocks(ids, args.seq_len)
        train_pick = [(None, b) for b in blocks[:-4]]
        train_blocks = blocks[:-4]
    eval5 = {d: [(x.to(device), y.to(device)) for x, y in DOM.domain_blocks(tok, d)]
             for d in DOM.DOMAINS}
    dense_dom = DOM.domain_ppls(teacher, eval5, device, amp_dtype)
    print("dense: " + " ".join(f"{d}:{dense_dom[d]:.1f}" for d in DOM.DOMAINS),
          flush=True)

    moes = []
    for i in layers:
        mlp = model.model.layers[i].mlp
        m = LoRAExpertFFN(mlp.gate_proj.weight.detach(),
                          mlp.up_proj.weight.detach(),
                          mlp.down_proj.weight.detach(),
                          n_experts=args.experts, top_k=args.topk,
                          rank=args.rank, seed=args.seed + i,
                          b_init=args.b_init).to(device)
        model.model.layers[i].mlp = m
        moes.append((i, m))
    if args.load:
        ck = torch.load(args.load, map_location=device, weights_only=False)
        for i, m in moes:
            m.load_state_dict(ck["moes"][str(i)], strict=True)
        print(f"loaded {args.load}", flush=True)
    # exact-dense check at init (B=0, but router may claim: deltas are 0 anyway)
    with torch.no_grad():
        x0, y0 = eval5["code"][0]
        ppl0 = DOM.domain_ppls(model, {"code": [(x0.to(device), y0.to(device))]},
                               device, amp_dtype)["code"]
    print(f"init ppl {ppl0:.2f} vs dense {dense_dom['code']:.2f} "
          f"(must match: B=0 init)", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for _, m in moes:
        only_router = ["router.weight", "router.bias"] if args.router_only else None
        for n, p in [("Ag", m.Ag), ("Bg", m.Bg), ("Au", m.Au), ("Bu", m.Bu),
                     ("Ad", m.Ad), ("Bd", m.Bd),
                     ("router.weight", m.router.weight),
                     ("router.bias", m.router.bias)]:
            if only_router is not None and n not in only_router:
                p.requires_grad_(False)
                continue
            p.requires_grad_(True)
            trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    scaler = POC.make_grad_scaler(amp_dtype)
    warm = max(1, args.steps // 10)

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        t = (s - warm) / max(1, args.steps - warm)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.perf_counter()
    model.train()
    if args.btx:
        # stage-2: diverge experts apart, LM-only, no anchor, router uninvolved
        btx8 = ["code1", "code2", "math1", "math2",
                "chat1", "chat2", "wiki1", "web1"]
        assert args.experts == len(btx8), "btx needs one domain per expert"
        ids16 = torch.load(".cache/moeme/btx_blocks16.pt", weights_only=True)
        for e, name in enumerate(btx8):
            dblocks = POC.make_blocks(ids16[name], args.seq_len)
            for _, m in moes:
                m.set_force_expert(e)
            eparams = []
            # Optimizer needs leaves: pass whole LoRA tensors. With
            # force_expert set, only expert e is in the graph, so only its
            # rows receive grads; the rest are skipped (grad None).
            for _, m in moes:
                for p in [m.Ag, m.Bg, m.Au, m.Bu, m.Ad, m.Bd]:
                    eparams.append(p)
            eopt = torch.optim.AdamW(eparams, lr=args.lr, betas=(0.9, 0.95))
            escaler = POC.make_grad_scaler(amp_dtype)
            for st in range(args.bsteps):
                eopt.zero_grad(set_to_none=True)
                jj = int(torch.randint(len(dblocks), (1,),
                                       generator=gen).item())
                xx, yy = dblocks[jj]
                xx, yy = xx.to(device), yy.to(device)
                with torch.autocast("cuda", dtype=amp_dtype):
                    o2 = model(xx, labels=yy)
                    l2 = o2.loss
                escaler.scale(l2).backward()
                escaler.unscale_(eopt)
                torch.nn.utils.clip_grad_norm_(eparams, 1.0)
                escaler.step(eopt)
                escaler.update()
            for _, m in moes:
                m.set_force_expert(None)
            print(f"  stage2 expert {e} ({name}) done", flush=True)
    dom_clf = None
    if args.domain_loss > 0:
        dom_clf = nn.Linear(args.experts, len(dom_names)).to(device)
        trainable.append(dom_clf.weight)
        trainable.append(dom_clf.bias)
    for step in range(args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        j = int(torch.randint(len(train_pick), (1,), generator=gen).item())
        dname, (x, y) = train_pick[j][0], train_pick[j][1]
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=amp_dtype):
            out = model(x, labels=y)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=amp_dtype):
                    ref = teacher(x, labels=y)
            ce = out.loss
            kl = F.kl_div(F.log_softmax(out.logits.float(), -1),
                          F.softmax(ref.logits.float(), -1),
                          reduction="batchmean")
            loss = (1 - args.kl) * ce + args.kl * kl
            if dom_clf is not None:
                aff = torch.stack([m.last_affinity.float().mean(0)
                                   for _, m in moes]).mean(0)
                dlogits = dom_clf(aff)
                dt = torch.tensor([dom_names.index(dname)],
                                  device=device)
                dloss = F.cross_entropy(dlogits.unsqueeze(0), dt)
                loss = loss + args.domain_loss * dloss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % max(1, min(args.steps // 5, 25)) == 0 or step == args.steps - 1:
            print(f"  step {step:4d} loss {float(loss):.4f} ce {float(ce):.3f} "
                  f"kl {float(kl):.3f} elapsed {time.perf_counter()-t0:5.0f}s",
                  flush=True)
    model.eval()
    got = DOM.domain_ppls(model, eval5, device, amp_dtype)
    mx = sum(got[d] / dense_dom[d] for d in DOM.DOMAINS) / len(DOM.DOMAINS)
    ntr = sum(p.numel() for p in trainable)
    print("lora-moe: " + " ".join(f"{d}:{got[d]:.1f}" for d in DOM.DOMAINS),
          flush=True)
    print(f"mean_xdense {mx:.3f}, trainable {ntr/1e6:.1f}M", flush=True)
    if args.save:
        torch.save({"layers": layers, "E": args.experts, "k": args.topk,
                    "rank": args.rank,
                    "moes": {str(i): {k: v.detach().cpu() for k, v in
                                       m.state_dict().items()}
                             for i, m in moes}}, args.save)
        print(f"saved {args.save}", flush=True)
    # affinity specialization: mean affinity per domain per expert
    spec = {}
    with torch.no_grad():
        for d, bl in eval5.items():
            affs = []
            for x, y in bl:
                with torch.autocast("cuda", dtype=amp_dtype):
                    model(x.to(device))
                for _, m in moes:
                    affs.append(m.last_affinity.float().mean(0))
            A = torch.stack(affs).mean(0)
            spec[d] = [round(float(v), 3) for v in A]
    print("mean affinity per expert (avg layers): " +
          " ".join(f"{d[:2]}:[{','.join(f'{v:.2f}' for v in spec[d])}]" for d in spec),
          flush=True)
    json.dump({"layers": layers, "dense_dom": dense_dom, "got": got,
               "mean_xdense": mx, "trainable_M": ntr / 1e6,
               "affinity": spec}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
