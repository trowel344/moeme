#!/usr/bin/env python3
"""
branch_poc.py -- does *branching* into complete modules beat *slicing* them?

The question
------------
E14/E15 "upcycled" a dense FFN by **slicing** the intermediate axis into E
disjoint experts and keeping k. That loses most of the layer output, and the
loss only vanishes as k -> E. The standing objection is that real MoE models do
not do that: their experts are *complete* feed-forward networks, so activating
one performs a whole layer's maths and there is no partial-sum deficit.

This driver measures both constructions on the same model, the same activations
and the same metric, and separates two things that are easy to conflate:

    "slicing hurts"        (a claim about tiles)
    "reading less hurts"   (a claim about width)

The separation is done by comparing a **tile** against a **replica at the same
active width**. A replica has one distinct block repeated, so the router has
nothing to choose and its residual is the residual of that one block; a tile can
pick the *best* block, so the oracle does at least as well. If both improve with
width in the same way, the governing variable is width, not tiling -- which is
the whole argument, and it is measurable rather than assertable.

What is reported
----------------
All residuals are against the **original dense** output, which is the only
reference that means the same thing for a tile and a replica. (`oracle_topk` in
`moeme.ffn` compares against the sum of all experts, which is the dense layer
for a tile but E x the dense layer for a replica -- so it cannot be used here.)

* `residual_best_single_expert` -- best possible single expert, per token
* `residual_oracle_topk`       -- best possible top-k set, per token
* `residual_router`            -- what the router actually picks
* `residual_all_experts_sum`   -- faithfulness of keeping every expert

Usage
-----
    python3 experiments/branch_poc.py --layers 0 3 8 23 --steps 150 \
        --out results/branch_poc.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from moeme.branch import BranchConfig, BranchMoEFFN          # noqa: E402


def _load_sibling(name: str):
    """Import `experiments/<name>.py` by path; `experiments/` is not a package."""
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_poc_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")


# ---------------------------------------------------------------------------
# domains: the data each *branch* is trained on independently (stage 2)
# ---------------------------------------------------------------------------

# Branch-Train-MiX trains each branch on a different slice of the data before
# the branches are combined. Doing that on genuinely different *domains* is what
# gives the router something to learn: if every expert sees the same mixture,
# they converge to the same function and the router has nothing to choose. Each
# entry is (label, dataset, config or None, split, text field).
#
# Deliberately not `bigcode/the-stack-smol` for the code branch: it is gated, and
# a gated dataset silently costs you a branch instead of raising where it hurts.
DOMAINS = [
    ("code", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", "problem"),
    ("math", "meta-math/MetaMathQA-40K", None, "train", "query"),
    ("chat", "HuggingFaceH4/ultrachat_200k", None, "train_sft", "prompt"),
    ("wiki", "Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
]


def load_domain_text(label, dataset, config, split, field, want_chars) -> str:
    """Up to `want_chars` characters of one domain, or "" if it cannot load."""
    from datasets import load_dataset
    try:
        if config:
            ds = load_dataset(dataset, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset, split=split, streaming=True)
    except Exception as e:                                      # noqa: BLE001
        print(f"  domain {label}: {dataset} unavailable ({type(e).__name__})")
        return ""
    chunks, have = [], 0
    try:
        for row in ds:
            t = row.get(field)
            if not isinstance(t, str) or len(t) < 200:
                continue
            chunks.append(t)
            have += len(t)
            if have >= want_chars:
                break
    except Exception as e:                                      # noqa: BLE001
        print(f"  domain {label}: stream ended early ({type(e).__name__})")
    print(f"  domain {label}: {have:,} chars from {len(chunks)} docs")
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------

def config_table(inter: int) -> list[dict]:
    """The measurement grid.

    Two ladders plus the decisive comparison:

    * tile ladder   -- E in {8,4,2,1} at k=1 gives active widths 12.5/25/50/100%,
                       and reproduces the shape of the pilot's curve.
    * replica ladder -- the *same* widths built from a single repeated block, so
                       the router has no choice to make.
    * replica_full  -- complete modules, k=1: the "cut it into real sub-models"
                       proposal. Exact by construction, at E x the storage.
    """
    u8, u4, u2 = inter // 8, inter // 4, inter // 2
    return [
        # --- tile ladder (the E15 construction) ---------------------------
        dict(name="tile_E8_k1_12pct", combine="sum", train=False,
             cfg=dict(n_experts=8, top_k=1, init="tile", expert_units=u8)),
        dict(name="tile_E4_k1_25pct", combine="sum", train=False,
             cfg=dict(n_experts=4, top_k=1, init="tile", expert_units=u4)),
        dict(name="tile_E2_k1_50pct", combine="sum", train=False,
             cfg=dict(n_experts=2, top_k=1, init="tile", expert_units=u2)),
        dict(name="tile_E1_k1_100pct", combine="sum", train=False,
             cfg=dict(n_experts=1, top_k=1, init="tile", expert_units=inter)),
        dict(name="tile_E4_k2_50pct", combine="sum", train=False,
             cfg=dict(n_experts=4, top_k=2, init="tile", expert_units=u4)),
        dict(name="tile_E4_k4_100pct", combine="sum", train=False,
             cfg=dict(n_experts=4, top_k=4, init="tile", expert_units=u4)),
        # --- replica ladder: same widths, one repeated block --------------
        dict(name="replica_E4_k1_12pct", combine="sum", train=False,
             cfg=dict(n_experts=4, top_k=1, init="replica", expert_units=u8)),
        dict(name="replica_E4_k1_25pct", combine="sum", train=False,
             cfg=dict(n_experts=4, top_k=1, init="replica", expert_units=u4)),
        dict(name="replica_E4_k1_50pct", combine="sum", train=False,
             cfg=dict(n_experts=4, top_k=1, init="replica", expert_units=u2)),
        # --- the proposal: complete modules ------------------------------
        dict(name="replica_E4_k1_full", combine="sum", train=False,
             cfg=dict(n_experts=4, top_k=1, init="replica", expert_units=inter)),
        # --- trainee set: softmax, because `sum` gives the router no
        #     gradient from the output at all (see branch.py's docstring) ---
        dict(name="trained_tile_E4_k1_25pct", combine="softmax", train=True,
             cfg=dict(n_experts=4, top_k=1, init="tile", expert_units=u4)),
        dict(name="trained_replica_E4_k1_full", combine="softmax", train=True,
             cfg=dict(n_experts=4, top_k=1, init="replica", expert_units=inter)),
        dict(name="trained_replica_E4_k1_full_shared", combine="softmax",
             train=True,
             cfg=dict(n_experts=4, top_k=1, init="replica", expert_units=inter,
                      n_shared=1)),
    ]


def btx_table(inter: int, n_domains: int) -> list[dict]:
    """The branch-train-miX configs: complete modules, trained apart, then mixed.

    Two widths, because the whole argument of this study is about width:

    * `full`  -- every expert a complete FFN. `k=1` reads full width, so the
                 result should match the dense model. Cost: `E` x the storage.
    * `half`  -- each expert half width. `k=1` reads 50% of the layer, the same
                 active width as `tile_E4_k1_25pct`/`replica_E4_k1_50pct`, so
                 the three are directly comparable at equal active compute.

    Both train their experts independently on different domains first, then get
    the same short cooperation finetune. That is the comparison that says
    whether specialisation buys back what width costs.
    """
    return [
        dict(name=f"btx_replica_k1_full", combine="softmax", train=True, btx=True,
             cfg=dict(n_experts=n_domains, top_k=1, init="replica",
                      expert_units=inter)),
        dict(name=f"btx_replica_k1_half", combine="softmax", train=True, btx=True,
             cfg=dict(n_experts=n_domains, top_k=1, init="replica",
                      expert_units=inter // 2)),
        dict(name=f"btx_tile_k1_25pct", combine="softmax", train=True, btx=True,
             cfg=dict(n_experts=n_domains, top_k=1, init="tile",
                      expert_units=inter // n_domains)),
    ]


# ---------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------

def branchify(model, layers, cfg: BranchConfig):
    """Replace `layers[i].mlp` with a `BranchMoEFFN` built from it."""
    blocks = model.model.layers
    out = []
    for i in layers:
        moe = BranchMoEFFN.from_hf_mlp(getattr(blocks[i], "mlp"), cfg)
        setattr(blocks[i], "mlp", moe)
        out.append((i, moe))
    return out


def restore(blocks, originals) -> None:
    for i, m in originals:
        setattr(blocks[i], "mlp", m)


@torch.no_grad()
def residual_across_layers(moes, reference, device) -> dict:
    """Mean residual_report over the upcycled layers."""
    acc: dict[str, list[float]] = {}
    per_layer = {}
    for i, m in moes:
        ref = reference.get(i)
        if ref is None:
            continue
        rep = m.residual_report(ref["x"].to(device))
        per_layer[i] = rep
        for k, v in rep.items():
            acc.setdefault(k, []).append(float(v))
    mean = {k: sum(v) / len(v) for k, v in acc.items()}
    mean["per_layer"] = per_layer
    return mean


def accounting(moes) -> dict:
    """What the sparse layer costs to run versus to store."""
    return {
        "active_params": sum(m.active_params for _, m in moes),
        "stored_params": sum(m.stored_params for _, m in moes),
        "active_width_fraction": sum(m.active_width_fraction for _, m in moes)
                                 / max(len(moes), 1),
        "stored_width_fraction": sum(m.total_width_fraction for _, m in moes)
                                 / max(len(moes), 1),
        "n_layers": len(moes),
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def branch_train_experts(model, moes, args, domain_blocks, device, amp_dtype,
                         scaler) -> dict:
    """Stage 2 of branch-train-miX: train each expert alone on its own domain.

    Every expert in every upcycled layer is forced to serve the *whole* batch
    (router bypassed, `set_force_expert`) and trained only on its own domain's
    tokens. The experts never see each other here, so nothing makes them
    similar except their shared initialisation -- which is exactly the point:
    they start identical and are pushed apart by the data.

    No router is trained in this stage, because the router is not in the
    forward path and so receives no gradient. It is learned in stage 3.
    """
    E = moes[0][1].E
    if len(domain_blocks) != E:
        raise ValueError(f"btx needs one domain per expert: {len(domain_blocks)}"
                         f" shards for E={E}")
    per_expert, totals = [], {"steps": 0, "wall_seconds": 0.0}
    for e in range(E):
        blocks = domain_blocks[e]
        if not blocks:
            print(f"    expert {e}: no data, skipped")
            continue
        for _, m in moes:
            m.set_force_expert(e)
        print(f"    -- expert {e} alone on {len(blocks)} own-domain blocks --")
        st = train_model(model, moes, args, blocks, device, amp_dtype, scaler,
                         steps=args.branch_steps)
        per_expert.append({"expert": e, "blocks": len(blocks), **st})
        totals["steps"] += st["steps_completed"]
        totals["wall_seconds"] += st["wall_seconds"]
    for _, m in moes:
        m.set_force_expert(None)
    return {"per_expert": per_expert, **totals}


def train_model(model, moes, args, train_blocks, device, amp_dtype, scaler,
                steps: int | None = None):
    """Short continued-pretraining run. Only the MoE modules move."""
    steps = args.steps if steps is None else steps
    if getattr(args, "grad_ckpt", False):
        # trade ~25% step time for ~5-10x less activation memory: required
        # for 4B+ training on 8GB cards (151k-vocab logits + 36-layer
        # activations otherwise OOM by ~150MB with no headroom left).
        model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for _, m in moes:
        for p in m.parameters():
            p.requires_grad_(True)
            trainable.append(p)

    if getattr(args, "adam8", False):
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    curve, t0 = [], time.perf_counter()
    accum = max(1, args.grad_accum)
    done = 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            j = int(torch.randint(len(train_blocks), (1,), generator=gen).item())
            x, y = train_blocks[j]
            x, y = x.to(device), y.to(device)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    out = model(x, labels=y)
                    # `total_aux_loss` already applies the config's coefficient;
                    # scaling it again here would silently halve the effective
                    # weighting relative to the E14/E15 runs.
                    aux = sum(m.total_aux_loss() for _, m in moes)
                    loss = (out.loss + aux) / accum
                scaler.scale(loss).backward()
            else:
                out = model(x, labels=y)
                aux = sum(m.total_aux_loss() for _, m in moes)
                loss = (out.loss + aux) / accum
                loss.backward()
            total += float(loss.item())
        if scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step()
        done = step + 1
        every = max(1, min(steps // 5, 25))
        if step % every == 0 or step == steps - 1:
            el = time.perf_counter() - t0
            curve.append({"step": step, "loss": total, "elapsed": el})
            print(f"      step {step:4d}  loss {total:.4f}  elapsed {el:6.0f}s",
                  flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"steps_completed": done, "loss_curve": curve,
            "wall_seconds": time.perf_counter() - t0,
            "trainable_params": sum(p.numel() for p in trainable)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def self_test() -> int:
    """Offline check of the module and the driver's own plumbing.

    Runs in seconds on CPU with no model download, so a dtype or shape
    regression is found here rather than 20 minutes into a GPU run. Each
    assertion is an identity the construction is supposed to guarantee.
    """
    torch.manual_seed(0)
    H, I = 32, 64
    gate = torch.randn(I, H) * 0.5
    up = torch.randn(I, H) * 0.5
    down = torch.randn(H, I) * 0.5
    x = torch.randn(7, H)

    def build(**kw):
        cfg = BranchConfig(seed=0, **kw)
        return BranchMoEFFN(gate.clone(), up.clone(), down.clone(), cfg)

    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1. A tiled cut that keeps every tile IS the dense layer.
    m = build(n_experts=4, top_k=4, init="tile", expert_units=I // 4,
              combine="sum")
    check("tile k=E is exact", m.residual_report(x)["residual_all_experts_sum"]
          < 1e-5)
    check("tile k=1 loses output",
          m.residual_report(x)["residual_best_single_expert"] > 0.5)

    # 2. A full-width replica at k=1 reproduces the dense layer exactly, while
    #    a narrow replica cannot -- the router has one block and no choice.
    full = build(n_experts=3, top_k=1, init="replica", expert_units=I,
                 combine="sum")
    rep = full.residual_report(x)
    check("replica k=1 full width is exact", rep["residual_router"] < 1e-5,
          f"router={rep['residual_router']:.2e}")
    check("replica full is not sparse", full.active_width_fraction == 1.0)
    narrow = build(n_experts=3, top_k=1, init="replica", expert_units=I // 4,
                   combine="sum")
    check("narrow replica loses output",
          narrow.residual_report(x)["residual_router"] > 0.5)
    check("narrow replica matches its width",
          abs(narrow.active_width_fraction - 0.25) < 1e-9)

    # 3. Storage scales with E; active width does not.
    check("stored grows with E", build(n_experts=8, top_k=1, init="replica",
                                       expert_units=I).stored_params
          > build(n_experts=2, top_k=1, init="replica",
                  expert_units=I).stored_params)

    # 4. A shared expert adds width without taking a routing slot.
    sh = build(n_experts=3, top_k=1, init="replica", expert_units=I // 2,
               n_shared=1, shared_units=I // 2, combine="softmax")
    check("shared expert contributes", "residual_shared_only"
          in sh.residual_report(x))

    # 5. The residual_report keys are all present and finite for both inits.
    for name, mod in (("tile", m), ("replica", full)):
        r = mod.residual_report(x)
        need = {"residual_best_single_expert", "residual_oracle_topk",
                "residual_router", "residual_all_experts_sum",
                "active_width_fraction", "stored_width_fraction"}
        check(f"{name} report complete", need <= set(r))
        check(f"{name} report finite",
              all(v == v and abs(v) < 1e6 for v in r.values()))

    # 6. `force_expert` bypasses the router (the stage-2 mechanism). At full
    #    width it must reproduce the dense layer, and it must be clearable.
    fr = build(n_experts=4, top_k=1, init="replica", expert_units=I,
               combine="softmax")
    fr.set_force_expert(2)
    check("force_expert reproduces dense",
          float((fr(x) - fr.dense_forward(x)).norm()) < 1e-5)
    check("force_expert is recorded", fr.last_stats.get("forced_expert") == 2)
    check("force_expert suppresses aux loss",
          float(fr.last_aux_loss) == 0.0)
    fr.set_force_expert(None)
    check("force_expert clears", fr.force_expert is None)
    try:
        fr.set_force_expert(9)
        check("force_expert range-checked", False)
    except ValueError:
        check("force_expert range-checked", True)

    # 7. A short training loop runs end to end on synthetic tokens.
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            lay = nn.Module()
            lay.mlp = BranchMoEFFN(gate.clone(), up.clone(), down.clone(),
                                   BranchConfig(n_experts=2, top_k=1,
                                                init="replica",
                                                expert_units=I // 2,
                                                combine="softmax", seed=0))
            self.model.layers = nn.ModuleList([lay])
            self.head = nn.Linear(H, 3)

        def forward(self, idx, labels=None):
            h = self.model.layers[0].mlp(idx.float().mean(-1, keepdim=True)
                                         .expand(-1, H))
            logits = self.head(h)
            loss = F.cross_entropy(logits, labels) if labels is not None else None
            return type("O", (), {"loss": loss})()

    tiny = _Tiny()
    moe = tiny.model.layers[0].mlp
    before = moe.router.weight.detach().clone()
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    blocks = [(torch.randint(0, 3, (4, 8)), torch.randint(0, 3, (4,)))]
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out = tiny(blocks[0][0], labels=blocks[0][1])
        (out.loss + moe.total_aux_loss()).backward()
        opt.step()
    moved = not torch.equal(before, moe.router.weight.detach())
    check("router receives gradient under softmax", moved)

    fails = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""))
    print(f"self-test: {len(checks) - len(fails)}/{len(checks)} passed")
    return 1 if fails else 0


def build_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--load-bit", default="none", choices=["none", "8bit", "4bit"],
                    help="quantized backbone for >8GB models: frozen layers stay "
                         "quantized, converted layers dequantize on build")
    ap.add_argument("--layers", nargs="+", default=["0", "3", "8", "23"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--aux-loss-coef", type=float, default=1e-2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-blocks", type=int, default=4)
    ap.add_argument("--corpus-chars", type=int, default=4_000_000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/branch_poc.json")
    ap.add_argument("--btx", action="store_true",
                    help="add the branch-train-miX configs: each expert is "
                         "trained alone on its own domain, then a router is "
                         "learned on top")
    ap.add_argument("--branch-steps", type=int, default=40,
                    help="stage-2 steps per expert (independent training)")
    ap.add_argument("--domain-chars", type=int, default=600_000,
                    help="characters of each domain stream for stage 2")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="weight dtype. float32 makes the exactness identities "
                         "(tile k=E, replica k=1) hold to ~1e-6 instead of the "
                         "bf16 rounding floor; use it on a model that fits.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only the configs whose name contains one of these "
                         "substrings; everything else is left untouched. Use "
                         "with --resume to finish a run that died part-way "
                         "without recomputing the ladder that already passed.")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the configs already present in --out, skipping "
                         "the ones that completed cleanly")
    ap.add_argument("--self-test", action="store_true",
                    help="offline CPU check of the module and the plumbing")
    return ap.parse_args(argv)


def main() -> int:
    args = build_args()
    if args.self_test:
        return self_test()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    # The whole study compares residuals of a *sum* against the sum itself. On
    # CUDA, TF32 leaves a ~5e-3 relative floor on every matmul, which is the
    # same size as the differences being reported (tile vs replica at equal
    # width differs by ~0.03). Turn it off so an "exact" identity reads as
    # exact and a small difference is architectural rather than arithmetic.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    amp_dtype = POC.pick_amp_dtype(device)
    scaler = POC.make_grad_scaler(amp_dtype)
    print(f"device={device} amp={amp_dtype} model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    wdtype = getattr(torch, args.dtype)
    if args.load_bit in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        qconf = BitsAndBytesConfig(
            load_in_8bit=(args.load_bit == "8bit"),
            load_in_4bit=(args.load_bit == "4bit"),
            bnb_4bit_compute_dtype=wdtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=qconf, device_map="auto",
            low_cpu_mem_usage=True)
        device = torch.device("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=wdtype).to(device)
    model.eval()

    ids, corpus = POC.load_token_stream(args, tokenizer)
    blocks = POC.make_blocks(ids, args.seq_len)
    n_val = max(args.eval_blocks, 2)
    train_blocks = blocks[:-n_val]
    val_blocks = blocks[-n_val:]
    print(f"blocks: {len(train_blocks)} train / {len(val_blocks)} val")

    # Stage-2 data: one token stream per domain. Held-out blocks from each
    # domain are kept so specialisation can be *measured* per domain rather
    # than only inferred from the pooled validation loss.
    domain_train, domain_val = [], []
    if args.btx:
        print("domains (one per branch):")
        for label, ds, cfg_name, split, field in DOMAINS:
            text = load_domain_text(label, ds, cfg_name, split, field,
                                    args.domain_chars)
            if not text:
                continue
            dids = tokenizer(text, return_tensors="pt").input_ids[0]
            dblocks = POC.make_blocks(dids, args.seq_len)
            if len(dblocks) < 3:
                print(f"  domain {label}: only {len(dblocks)} blocks, skipped")
                continue
            domain_train.append(dblocks[:-2])
            domain_val.append({"domain": label, "blocks": dblocks[-2:]})
        print(f"domains usable: {len(domain_train)}")

    layers = [int(v) for v in args.layers]
    reference = POC.capture_reference(model, val_blocks[:1], layers, device)
    dense_ppl = POC.eval_perplexity(model, val_blocks, device, amp_dtype,
                                    args.eval_blocks)
    print(f"dense val ppl {dense_ppl:.4f}")

    inter = int(model.model.layers[layers[0]].mlp.gate_proj.weight.shape[0])
    print(f"intermediate width {inter}\n")

    originals = [(i, model.model.layers[i].mlp) for i in layers]
    blocks_mod = model.model.layers

    runs, report = [], {
        "model": args.model, "device": str(device),
        "weight_dtype": args.dtype,
        "amp_dtype": str(amp_dtype), "corpus": corpus,
        "corpus_tokens": int(ids.numel()), "seq_len": args.seq_len,
        "dense_val_ppl": dense_ppl, "layers": layers,
        "intermediate": inter,
    }

    specs = config_table(inter)
    if args.btx and domain_train:
        specs += btx_table(inter, len(domain_train))

    # Resume: a run that is killed part-way (OOM, session cap) has usually
    # finished the cheap configurations and died on an expensive one. Reloading
    # the completed entries means the expensive part can be retried alone.
    done: dict[str, dict] = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out) as fh:
            prev = json.load(fh)
        for e in prev.get("runs", []):
            if e.get("name") and not e.get("error"):
                done[e["name"]] = e
        runs = [done[s["name"]] for s in specs if s["name"] in done]
        report = {**report, **{k: v for k, v in prev.items()
                               if k not in ("runs",)}}
        print(f"resume: {len(done)} configs already complete "
              f"({', '.join(sorted(done))})")
    if args.only:
        specs = [s for s in specs
                 if any(sub in s["name"] for sub in args.only)]
        print(f"only: {len(specs)} configs selected")

    for spec in specs:
        if args.resume and spec["name"] in done:
            print(f"=== {spec['name']}: already complete, skipped ===")
            continue
        name = spec["name"]
        cfg = BranchConfig(combine=spec["combine"], seed=args.seed, **spec["cfg"])
        print(f"=== {name}  combine={cfg.combine} "
              f"E={cfg.n_experts} k={cfg.top_k} "
              f"units={cfg.expert_units} shared={cfg.n_shared} ===")
        restore(blocks_mod, originals)
        try:
            moes = branchify(model, layers, cfg)
            before = residual_across_layers(moes, reference, device)
            train_stats = None
            btx_stats = None
            after_stage2 = None
            if spec.get("btx"):
                btx_stats = branch_train_experts(model, moes, args, domain_train,
                                                 device, amp_dtype, scaler)
                after_stage2 = residual_across_layers(moes, reference, device)
            if spec["train"]:
                train_stats = train_model(model, moes, args, train_blocks,
                                          device, amp_dtype, scaler)
                after = residual_across_layers(moes, reference, device)
            else:
                after = before
            ppl = POC.eval_perplexity(model, val_blocks, device, amp_dtype,
                                     args.eval_blocks)
            entry = {
                "name": name,
                "config": {"n_experts": cfg.n_experts, "top_k": cfg.top_k,
                           "init": cfg.init, "expert_units": cfg.expert_units,
                           "n_shared": cfg.n_shared, "combine": cfg.combine},
                "accounting": accounting(moes),
                "untrained": before,
                "trained": after if train_stats else None,
                "train": train_stats,
                "val_ppl": ppl,
                "routing": {i: m.last_stats for i, m in moes},
                "error": None,
            }
            if btx_stats is not None:
                entry["btx"] = btx_stats
                entry["btx"]["residual_after_stage2"] = after_stage2
                entry["domain_ppl"] = {
                    d["domain"]: POC.eval_perplexity(
                        model, d["blocks"], device, amp_dtype, len(d["blocks"]))
                    for d in domain_val
                }
                # Per-expert usage on a shared batch: a specialised router
                # should spread across experts, not collapse onto one.
                entry["expert_usage_on_val"] = {
                    i: m.last_stats.get("usage_fraction")
                    for i, m in moes
                }
            print(f"    width {before['active_width_fraction']:.3f} "
                  f"stored {before['stored_width_fraction']:.2f}x | "
                  f"best1 {before['residual_best_single_expert']:.4f} "
                  f"oracle {before['residual_oracle_topk']:.4f} "
                  f"router {before['residual_router']:.4f} "
                  f"all {before['residual_all_experts_sum']:.4f} "
                  f"| ppl {ppl:.2f}")
            if train_stats:
                print(f"    after {train_stats['steps_completed']} steps: "
                      f"best1 {after['residual_best_single_expert']:.4f} "
                      f"router {after['residual_router']:.4f} | ppl {ppl:.2f}")
        except Exception as e:                                  # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}")
            entry = {"name": name, "error": f"{type(e).__name__}: {e}"}
        runs.append(entry)
        report["runs"] = runs
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    restore(blocks_mod, originals)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
