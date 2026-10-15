#!/usr/bin/env python3
"""
mocm_poc.py -- MoCM v1 stubborn-path driver: shared + activation-cut + softmax.

What it does, per cell (mocm25 = 25% active, mocm12 = 12.5% active):
  1. Capture dense MLP inputs on a small calibration batch (no training).
  2. Rank intermediate units by activation frequency -> unit_order.
  3. Build BranchMoEFFN tiles with 1 always-on shared expert (head of order).
  4. Report frozen residual_report + val ppl (the no-training baseline).
  5. Joint-train the MoE modules only, report trained residual + ppl.

Memory safety (the user asked not to OOM):
  * default 4 late layers only (20-23), 44M trainable, bf16, batch 1 x accum 4
  * small corpus default (2M chars), seq 512, eval 4 blocks
  * torch.cuda.empty_cache() between cells, TF32 off, incremental JSON writes
  * --frozen-only runs zero training steps (calibration + eval only)

Usage:
    python3 experiments/mocm_poc.py --frozen-only --out results/mocm_poc.json
    python3 experiments/mocm_poc.py --steps 80 --out results/mocm_poc.json
    python3 experiments/mocm_poc.py --steps 80 --btx --out results/mocm_poc.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _load_sibling(name: str):
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_mocm_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
BR = _load_sibling("branch_poc")

from moeme.mocm import MoCMv1Config, activation_order, build_mocm_v1_layer  # noqa: E402


def cells_for(inter: int, which: str) -> list[tuple[str, MoCMv1Config]]:
    out = []
    if which in ("mocm25", "both"):
        out.append(("mocm25", MoCMv1Config(n_routed=8, top_k=1,
                                           shared_units=inter // 8)))
    if which in ("mocm12", "both"):
        out.append(("mocm12", MoCMv1Config(n_routed=16, top_k=1,
                                           shared_units=inter // 16)))
    return out


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--load-bit", default="none", choices=["none", "8bit", "4bit"],
                    help="quantized backbone for >8GB models (converted layers "
                         "dequantize on build via BENCH.build_champ)")
    ap.add_argument("--ckpt-out", default=None,
                    help="save trained MoE weights here (.pt, emitter-ready)")
    ap.add_argument("--adam8", action="store_true",
                    help="bitsandbytes 8-bit Adam states (fits big-model "
                         "training on 8GB cards)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="HF activation checkpointing: trades step time for "
                         "activation memory (needed at 4B+ on 8GB)")
    ap.add_argument("--gpu-mem", default=None,
                    help="e.g. 7000MB: cap cuda:0 in device_map to keep "
                         "embeddings on GPU with quantized backbones")
    ap.add_argument("--layers", nargs="+", default=["20", "21", "22", "23"])
    ap.add_argument("--cell", default="both", choices=["mocm25", "mocm12", "both"])
    ap.add_argument("--tag", default=None,
                    help="custom single-cell run: --tag NAME --n-routed E "
                         "--topk K --shared-units S --router-hidden H --aux-coef C")
    ap.add_argument("--n-routed", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--shared-units", type=int, default=None)
    ap.add_argument("--router-hidden", type=int, default=None)
    ap.add_argument("--aux-coef", type=float, default=None)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--aux-loss-coef", type=float, default=1e-2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-blocks", type=int, default=4)
    ap.add_argument("--corpus-chars", type=int, default=2_000_000)
    ap.add_argument("--calib-blocks", type=int, default=4)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frozen-only", action="store_true")
    ap.add_argument("--router-only", action="store_true",
                    help="train the router only (experts frozen): tests whether "
                         "routing is the blocker before spending joint steps")
    ap.add_argument("--btx", action="store_true")
    ap.add_argument("--branch-steps", type=int, default=40)
    ap.add_argument("--domain-chars", type=int, default=600_000)
    ap.add_argument("--out", default="results/mocm_poc.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="offline pipeline check on a tiny random model")
    return ap.parse_args(argv)


def self_test() -> int:
    """Offline check: build v1 layers on a tiny random stack, verify the
    dense limit (all experts = dense) and a short training loop runs."""
    import torch.nn as nn
    import torch.nn.functional as F
    from moeme.mocm import MoCMv1Config, activation_order, build_mocm_v1_layer
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, I = 32, 64
    g = torch.randn(I, H) * 0.5
    u = torch.randn(I, H) * 0.5
    d = torch.randn(H, I) * 0.5
    x = torch.randn(16, H)
    order = activation_order(g, u, x)
    assert sorted(order) == list(range(I)), "order must permute all units"
    m = build_mocm_v1_layer(g, u, d, MoCMv1Config(n_routed=4, top_k=1,
                                                 shared_units=8,
                                                 unit_order=order))
    m = m.to(device)
    from moeme.mocm import align_moe_dtype
    align_moe_dtype(m, torch.bfloat16 if device.type == "cuda"
                    else torch.float32)
    rep = m.residual_report(x.to(m.gate.dtype).to(device))
    assert rep["residual_all_experts_sum"] < 1e-2, rep
    assert rep["residual_best_single_expert"] > 0.1, rep
    # a few training steps move the loss without error
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=1e-3)
    xb = x.to(device).to(m.gate.dtype)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = m(xb).pow(2).mean() + m.total_aux_loss()
        loss.backward()
        opt.step()
    assert math.isfinite(float(loss))
    print("mocm self-test OK "
          f"(all-sum {rep['residual_all_experts_sum']:.2e}, "
          f"best1 {rep['residual_best_single_expert']:.3f})")
    return 0


def train_router_only(model, moes, args, train_blocks, device, amp_dtype, scaler):
    """Train only router params (experts frozen). Router-blocker test."""
    import time
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for _, m in moes:
        for p in m.router.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    curve, t0, done = [], time.perf_counter(), 0
    accum = max(1, args.grad_accum)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            j = int(torch.randint(len(train_blocks), (1,), generator=gen).item())
            x, y = train_blocks[j]
            x, y = x.to(device), y.to(device)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    out = model(x, labels=y)
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
        if step % max(1, min(args.steps // 5, 25)) == 0 or step == args.steps - 1:
            el = time.perf_counter() - t0
            curve.append({"step": step, "loss": total, "elapsed": el})
            print(f"      step {step:4d}  loss {total:.4f}  elapsed {el:6.0f}s",
                  flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"steps_completed": done, "loss_curve": curve,
            "wall_seconds": time.perf_counter() - t0, "mode": "router-only",
            "trainable_params": sum(p.numel() for p in trainable)}


def main() -> int:
    args = build_args()
    if args.self_test:
        return self_test()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA; Qwen3.5 Gated DeltaNet needs CUDA causal_conv1d.")
    amp_dtype = POC.pick_amp_dtype(device)
    scaler = POC.make_grad_scaler(amp_dtype)
    print(f"device={device} amp={amp_dtype} model={args.model}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"gpu free {free/2**30:.2f} / total {total/2**30:.2f} GiB", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.load_bit in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        wdtype = torch.bfloat16
        qconf = BitsAndBytesConfig(
            load_in_8bit=(args.load_bit == "8bit"),
            load_in_4bit=(args.load_bit == "4bit"),
            bnb_4bit_compute_dtype=wdtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=qconf, device_map="auto",
            max_memory={0: args.gpu_mem, "cpu": "20GiB"} if args.gpu_mem else None,
            low_cpu_mem_usage=True)
        device = torch.device("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        model = model.to(device)
    model.eval()
    print(f"params {sum(p.numel() for p in model.parameters())/1e9:.3f}B", flush=True)

    ids, corpus = POC.load_token_stream(args, tok)
    blocks = POC.make_blocks(ids, args.seq_len)
    n_val = max(args.eval_blocks, 2)
    train_blocks, val_blocks = blocks[:-n_val], blocks[-n_val:]
    print(f"blocks: {len(train_blocks)} train / {len(val_blocks)} val", flush=True)

    domain_train = []
    if args.btx:
        for label, ds, cfg_name, split, field in BR.DOMAINS:
            text = BR.load_domain_text(label, ds, cfg_name, split, field,
                                        args.domain_chars)
            if not text:
                continue
            dids = tok(text, return_tensors="pt").input_ids[0]
            dblocks = POC.make_blocks(dids, args.seq_len)
            if len(dblocks) >= 3:
                domain_train.append(dblocks[:-2])
        print(f"domains usable: {len(domain_train)}", flush=True)

    layers = [int(v) for v in args.layers]
    inter = int(model.model.layers[layers[0]].mlp.gate_proj.weight.shape[0])
    print(f"intermediate {inter}, layers {layers}", flush=True)

    # Dense reference + calibration inputs (captured once, before surgery).
    reference = POC.capture_reference(model, val_blocks[:1], layers, device)
    calib = POC.make_blocks(ids, args.seq_len)[:args.calib_blocks]
    calib_x = calib[0][0].to(device) if calib else val_blocks[0][0].to(device)
    dense_ppl = POC.eval_perplexity(model, val_blocks, device, amp_dtype,
                                    args.eval_blocks)
    print(f"dense val ppl {dense_ppl:.4f}", flush=True)

    originals = [(i, model.model.layers[i].mlp) for i in layers]
    blocks_mod = model.model.layers

    specs = cells_for(inter, args.cell)
    if args.tag:
        base = MoCMv1Config(
            n_routed=args.n_routed or 8, top_k=args.topk or 1,
            shared_units=(inter // 8 if args.shared_units is None
                          else args.shared_units),
            router_hidden=args.router_hidden,
            aux_loss_coef=(args.aux_coef if args.aux_coef is not None else 1e-2),
            seed=args.seed)
        specs = [(args.tag, base)]
    if args.only:
        specs = [(n, c) for n, c in specs if any(s in n for s in args.only)]

    report = {"model": args.model, "device": str(device), "corpus": corpus,
              "corpus_tokens": int(ids.numel()), "seq_len": args.seq_len,
              "dense_val_ppl": dense_ppl, "layers": layers,
              "intermediate": inter, "runs": []}
    done: dict[str, dict] = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out) as fh:
            prev = json.load(fh)
        for e in prev.get("runs", []):
            if e.get("name") and not e.get("error"):
                done[e["name"]] = e
        report["runs"] = list(done.values())
        print(f"resume: {len(done)} complete", flush=True)

    for name, base in specs:
        if args.resume and name in done:
            print(f"=== {name}: skipped (resume) ===", flush=True)
            continue
        print(f"=== {name} E={base.n_routed} k={base.top_k} "
              f"shared={base.resolved_shared(inter)} ===", flush=True)
        BR.restore(blocks_mod, originals)
        try:
            # Per-layer activation order from the calibration batch through
            # the *dense* mlps (captured before any surgery on this cell).
            moes = []
            for i in layers:
                mlp = getattr(blocks_mod[i], "mlp")
                with torch.no_grad():
                    # dense mlp input for the calib batch via pre-hook capture
                    cap = {}
                    h = mlp.register_forward_pre_hook(
                        lambda _m, a, c=cap: c.setdefault("x", a[0].detach()))
                    model.eval()
                    _ = model(calib_x)
                    h.remove()
                    xin = cap["x"].reshape(-1, cap["x"].shape[-1])
                from moeme.upcycle import _dequant
                _gw, _uw, _dw = (_dequant(mlp.gate_proj.weight),
                                 _dequant(mlp.up_proj.weight),
                                 _dequant(mlp.down_proj.weight))
                order = activation_order(_gw.detach(), _uw.detach(), xin[:4096])
                cfg = MoCMv1Config(n_routed=base.n_routed, top_k=base.top_k,
                                   shared_units=base.resolved_shared(inter),
                                   aux_loss_coef=args.aux_loss_coef,
                                   seed=args.seed, unit_order=order)
                m = build_mocm_v1_layer(_gw.detach(), _uw.detach(), _dw.detach(),
                                        cfg)
                # Experts inherit the dense dtype; the router must stay float32
                # (it takes x2.float()). A blanket .to() breaks it; dequant
                # builds land in fp32 and need aligning to activations.
                from moeme.mocm import align_moe_dtype
                m = m.to(device)
                align_moe_dtype(m, torch.bfloat16 if device.type == "cuda"
                                else torch.float32)
                setattr(blocks_mod[i], "mlp", m)
                moes.append((i, m))

            frozen = BR.residual_across_layers(moes, reference, device)
            frozen_ppl = POC.eval_perplexity(model, val_blocks, device,
                                             amp_dtype, args.eval_blocks)
            print(f"  frozen: best1 {frozen['residual_best_single_expert']:.4f} "
                  f"router {frozen['residual_router']:.4f} "
                  f"active {frozen['active_width_fraction']:.3f}(routed-only) "
                  f"| ppl {frozen_ppl:.2f}", flush=True)

            trained, train_stats, btx_stats = None, None, None
            if not args.frozen_only:
                if args.btx and len(domain_train) == moes[0][1].E:
                    btx_stats = BR.branch_train_experts(
                        model, moes, args, domain_train, device, amp_dtype, scaler)
                elif args.btx:
                    print(f"  btx skipped: {len(domain_train)} domains "
                          f"!= E={moes[0][1].E} (needs one domain per expert)",
                          flush=True)
                pseudo = argparse.Namespace(steps=args.steps, lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            grad_accum=args.grad_accum,
                                            clip=args.clip, seed=args.seed,
                                            adam8=args.adam8,
                                            grad_ckpt=args.grad_ckpt)
                if args.router_only:
                    train_stats = train_router_only(
                        model, moes, pseudo, train_blocks, device, amp_dtype, scaler)
                else:
                    train_stats = BR.train_model(model, moes, pseudo, train_blocks,
                                                 device, amp_dtype, scaler)
                trained = BR.residual_across_layers(moes, reference, device)
                print(f"  trained {train_stats['steps_completed']} steps: "
                      f"router {trained['residual_router']:.4f} "
                      f"| ppl check below", flush=True)
            ppl = POC.eval_perplexity(model, val_blocks, device, amp_dtype,
                                      args.eval_blocks)
            print(f"  val ppl {ppl:.2f} (dense {dense_ppl:.2f})", flush=True)
            entry = {"name": name,
                     "config": {"n_routed": base.n_routed, "top_k": base.top_k,
                                "shared_units": base.resolved_shared(inter),
                                "router_hidden": base.router_hidden,
                                "aux_loss_coef": base.aux_loss_coef,
                                "combine": "softmax", "init": "tile",
                                "activation_order": True},
                     "accounting": BR.accounting(moes),
                     "active_width_true": (base.resolved_shared(inter)
                                           + base.top_k * (inter // base.n_routed)) / inter,
                     "frozen": frozen, "frozen_ppl": frozen_ppl,
                     "trained": trained, "train": train_stats,
                     "btx": btx_stats, "val_ppl": ppl,
                     "routing": {i: m.last_stats for i, m in moes},
                     "error": None}
            if args.ckpt_out and not args.frozen_only:
                os.makedirs(os.path.dirname(args.ckpt_out) or ".", exist_ok=True)
                torch.save({"tag": name, "layers": layers,
                            "spec": {"E": base.n_routed, "k": base.top_k,
                                     "sh": base.resolved_shared(inter)},
                            "moes": {str(i): {k: v.detach().cpu() for k, v in
                                               m.state_dict().items()}
                                     for i, m in moes}}, args.ckpt_out)
                print(f"  saved {args.ckpt_out}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            entry = {"name": name, "error": f"{type(e).__name__}: {e}"}
        report["runs"] = [r for r in report["runs"] if r.get("name") != name] + [entry]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  wrote {args.out}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    BR.restore(blocks_mod, originals)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
