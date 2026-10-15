#!/usr/bin/env python3
"""mocm_seq.py -- per-sequence domain MoCM prototype.

One routing decision per *prompt*, not per token: each domain owns an expert
(trained alone on its domain, forced, never mixed). At inference the whole
sequence uses one expert -- no per-token switching, no dispatch overhead,
no compounding from switching.

Test (no stage-3, frozen after stage-2):
  1. Train E experts, each alone on its own domain (BTX stage-2 only).
  2. Score the E x D matrix: every expert forced on every domain's held-out
     tail. Diagonal dominance = specialization happened.
  3. Sequence-routed model (oracle domain pick) vs dense on the fixed suite.

  python3 experiments/mocm_seq.py --out results/mocm_seq.json
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
    spec = importlib.util.spec_from_file_location(f"_seq_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
BR = _load_sibling("branch_poc")
EVO = _load_sibling("mocm_evo")
DOM = _load_sibling("mocm_domains")

SEQ_DOMAINS = ["code1", "code2", "math1", "math2",
               "chat1", "chat2", "wiki1", "web1"]


@torch.no_grad()
def forced_ppl(model, moes, blocks, device, amp_dtype, expert: int) -> float:
    import math
    for _, m in moes:
        m.set_force_expert(expert)
    model.eval()
    total, ntok = 0.0, 0
    for x, y in blocks:
        x, y = x.to(device), y.to(device)
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(x, labels=y).loss
        else:
            loss = model(x, labels=y).loss
        total += loss.item() * y.numel()
        ntok += y.numel()
    for _, m in moes:
        m.set_force_expert(None)
    return math.exp(total / max(ntok, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layers", nargs="+", default=["20", "21", "22", "23"])
    ap.add_argument("--branch-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--domains", nargs="+", default=None,
                    help="subset of SEQ_DOMAINS (default all); E = len")
    ap.add_argument("--out", default="results/mocm_seq.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from moeme.branch import BranchConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    amp_dtype = POC.pick_amp_dtype(device)
    scaler = POC.make_grad_scaler(amp_dtype)
    layers = [int(v) for v in args.layers]
    dom_names = args.domains or SEQ_DOMAINS
    E = len(dom_names)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16).to(device)
    model.eval()

    # per-domain train blocks (head) + held-out tails, from fixed BTX files
    ids_map = torch.load(EVO.BTX_BLOCKS_CACHE, weights_only=True)
    train_dom, tail_dom = [], {}
    for n in dom_names:
        bl = POC.make_blocks(ids_map[n], args.seq_len)
        train_dom.append(bl[:-2])
        tail_dom[n] = [(x.to(device), y.to(device)) for x, y in bl[-2:]]
    print(f"domains: {len(train_dom)}, "
          f"blocks/expert ~{sum(len(b) for b in train_dom)//len(train_dom)}",
          flush=True)

    inter = int(model.model.layers[layers[0]].mlp.gate_proj.weight.shape[0])
    cfg = BranchConfig(n_experts=E, top_k=1, init="tile",
                       expert_units=inter // E, combine="softmax",
                       n_shared=1, shared_units=inter // 8, seed=args.seed)
    moes = BR.branchify(model, layers, cfg)

    # stage-2 only: each expert alone on its domain, then FREEZE (no mixing)
    bargs = argparse.Namespace(branch_steps=args.branch_steps, lr=args.lr,
                               weight_decay=0.0, grad_accum=4, clip=1.0,
                               seed=args.seed)
    BR.branch_train_experts(model, moes, bargs, train_dom, device, amp_dtype,
                            scaler)

    # E x D specialization matrix on held-out tails
    matrix = {}
    for e, name in enumerate(dom_names):
        matrix[name] = {}
        for d, tbl in tail_dom.items():
            matrix[name][d] = round(forced_ppl(model, moes, tbl, device,
                                               amp_dtype, e), 2)
        print(f"  expert {name}: " + " ".join(f"{d}:{matrix[name][d]:.1f}"
                                              for d in dom_names), flush=True)
    diag = sum(matrix[n][n] for n in dom_names) / E
    off = (sum(v for row in matrix.values() for v in row.values())
           - sum(matrix[n][n] for n in dom_names)) / (E * (E - 1))
    print(f"diagonal mean {diag:.2f} vs off-diagonal {off:.2f}", flush=True)

    # sequence-routed (oracle pick) vs dense on the fixed 5-domain suite
    eval5 = {d: [(x, y) for x, y in DOM.domain_blocks(tok, d)] for d in DOM.DOMAINS}
    eval5 = {d: [(x.to(device), y.to(device)) for x, y in b]
             for d, b in eval5.items()}
    # map fixed domains to nearest expert: code->code1, math->math1, chat->chat1,
    # wiki->wiki1, web->web1
    pick = {"code": 0, "math": 2, "chat": 4, "wiki": 6, "web": 7}
    pick = {d: p % E for d, p in pick.items()}
    routed = {d: forced_ppl(model, moes, b, device, amp_dtype, pick[d])
              for d, b in eval5.items()}
    for _, m in moes:
        m.set_force_expert(None)
    # dense reference: fresh reload (this model's mlps are specialized now)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dense = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16).to(device)
    dense.eval()
    dense5 = DOM.domain_ppls(dense, eval5, device, amp_dtype)
    print("seq-routed vs dense: " + " ".join(
        f"{d}:{routed[d]:.1f}/{dense5[d]:.1f}" for d in DOM.DOMAINS), flush=True)

    json.dump({"experts": dom_names, "layers": layers,
               "matrix": matrix, "diag_mean": diag, "offdiag_mean": off,
               "seq_routed": routed, "dense": dense5}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
