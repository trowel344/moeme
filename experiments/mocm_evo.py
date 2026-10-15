#!/usr/bin/env python3
"""mocm_evo.py -- evolution harness: train children, score multi-domain.

One generation = several children trained identically, scored on the fixed
5-domain suite (code/math/chat/wiki/web) + frozen baseline. Checkpoints are
saved so the rotating spotlight (mbpp exec / gsm8k grade) can run later on
finalists without retraining.

  Train:
    python3 experiments/mocm_evo.py --gen 4 --out results/mocm_gen4.json \\
      --child g4-s500:E=8:k=1:sh=896:steps=500:lr=3e-5 \\
      --child g4-e4big:E=4:k=1:sh=448:steps=300:lr=3e-5
  Spotlight (top-2 by domain score):
    python3 experiments/mocm_evo.py --spot mbpp --gen 4 --finalists g4-s500,g4-e4big
    python3 experiments/mocm_evo.py --spot gsm8k --gen 4 --finalists g4-s500,g4-e4big
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
    spec = importlib.util.spec_from_file_location(f"_evo_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
BR = _load_sibling("branch_poc")
BENCH = _load_sibling("mocm_bench")
DOM = _load_sibling("mocm_domains")

CKPT_DIR = os.path.join(HERE, "..", ".cache", "moeme", "evo_ckpts")


def parse_child(spec: str) -> dict:
    """tag:E=8:k=1:sh=896:steps=300:lr=3e-5"""
    parts = spec.split(":")
    d = {"tag": parts[0]}
    for p in parts[1:]:
        k, v = p.split("=")
        d[k] = float(v) if "." in v or "e" in v.lower() else int(v)
    d.setdefault("E", 8)
    d.setdefault("k", 1)
    d.setdefault("sh", 448)
    d.setdefault("steps", 300)
    d.setdefault("lr", 3e-5)
    return d


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--child", action="append", default=[])
    ap.add_argument("--layers", nargs="+", default=["20", "21", "22", "23"])
    ap.add_argument("--train-file", default=".cache/moeme/mocm_corpus.txt")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--spot", choices=["mbpp", "gsm8k"], default=None)
    ap.add_argument("--finalists", default=None)
    ap.add_argument("--spot-k", type=int, default=0)  # 0 => 16 mbpp / 50 gsm8k
    ap.add_argument("--seq-len", type=int, default=512)
    return ap.parse_args(argv)


def probe_layer_residual(model, layers, calib_x, device, n_routed=16):
    """Frozen best1 residual per layer at E=n_routed tiles (the need metric)."""
    from moeme.branch import BranchConfig, BranchMoEFFN
    from moeme.mocm import activation_order
    blocks_mod = model.model.layers
    inter = int(blocks_mod[layers[0]].mlp.gate_proj.weight.shape[0])
    caps = {}
    handles = [blocks_mod[i].mlp.register_forward_pre_hook(
        lambda _m, a, i=i: caps.setdefault(i, a[0].detach())) for i in layers]
    model.eval()
    with torch.no_grad():
        _ = model(calib_x)
    for h in handles:
        h.remove()
    res = {}
    with torch.no_grad():
        for i in layers:
            mlp = getattr(blocks_mod[i], "mlp")
            g, u, d = (mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach(),
                       mlp.down_proj.weight.detach())
            xin = caps[i].reshape(-1, caps[i].shape[-1])
            order = activation_order(g, u, xin[:4096])
            from moeme.mocm import MoCMv1Config, build_mocm_v1_layer
            m = build_mocm_v1_layer(
                g, u, d, MoCMv1Config(n_routed=n_routed, top_k=1,
                                      shared_units=0, unit_order=order))
            m = m.to(device)
            m.router.to(dtype=torch.float32)
            rep = m.residual_report(xin[:2048].to(m.gate.dtype))
            res[i] = rep["residual_best_single_expert"]
            del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def allocate_shared(resid: dict, mean_sh: int, lo=224, hi=2400) -> dict:
    mean_r = sum(resid.values()) / len(resid)
    return {i: int(min(hi, max(lo, round(mean_sh * r / mean_r))))
            for i, r in resid.items()}


BTX_SRC = os.path.join(HERE, "..", ".cache", "moeme", "btx_src")
# 16 slots: 13 distinct source-views + 3 disjoint-halves (code4, chat4, web3
# reuse a fetched file's second half: disjoint tokens, same distribution).
BTX_SOURCES = {
    "code1": ("code1.txt", 0, None), "code2": ("code2.txt", 0, None),
    "code4": ("code1.txt", 500_000, None),
    "math1": ("math1.txt", 0, None), "math2": ("math2.txt", 0, None),
    "math3": ("math3.txt", 0, None), "math4": ("math4.txt", 0, None),
    "chat1": ("chat1.txt", 0, None), "chat2": ("chat2.txt", 0, None),
    "chat3": ("chat3.txt", 0, None),
    "wiki1": ("wiki1.txt", 0, None), "wiki2": ("wiki2.txt", 0, None),
    "web1": ("web1.txt", 0, None),
    "mix1": (None, 0, 1_000_000), "mix2": (None, None, None),
    "web3": ("web1.txt", 500_000, None),
}
BTX_ORDER16 = ["code1", "code2", "code4", "math1", "math2", "math3", "math4",
               "chat1", "chat2", "chat3", "wiki1", "wiki2", "web1", "web3",
               "mix1", "mix2"]
BTX_ORDER8 = ["code1", "code2", "math1", "math2",
              "chat1", "chat2", "wiki1", "web1"]
BTX_BLOCKS_CACHE = os.path.join(HERE, "..", ".cache", "moeme", "btx_blocks16.pt")


def btx_domain_blocks(tok, train_file, seq_len=256, order=None):
    """Expert-domain block lists (cached tokenization)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_btx_poc", os.path.join(HERE, "upcycle_poc.py"))
    poc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = poc
    spec.loader.exec_module(poc)
    order = order or BTX_ORDER16
    if os.path.exists(BTX_BLOCKS_CACHE):
        ids_map = torch.load(BTX_BLOCKS_CACHE, weights_only=True)
        if all(n in ids_map for n in order):
            return [poc.make_blocks(ids_map[n], seq_len) for n in order]
    ids_map = {}
    if os.path.exists(BTX_BLOCKS_CACHE):
        ids_map = torch.load(BTX_BLOCKS_CACHE, weights_only=True)
    fresh = open(train_file).read()
    for n in order:
        if n in ids_map:
            continue
        fn, a, b = BTX_SOURCES[n]
        if fn is None:
            text = fresh[:1_000_000] if n == "mix1" else None
            if n == "mix2":
                text = open(".cache/moeme/mocm_corpus.txt").read()[1_000_000:2_000_000]
        else:
            with open(os.path.join(BTX_SRC, fn)) as fh:
                text = fh.read()[a:b]
        ids_map[n] = tok(text, return_tensors="pt").input_ids[0]
    torch.save(ids_map, BTX_BLOCKS_CACHE)
    return [poc.make_blocks(ids_map[n], seq_len) for n in order]


def train_sched(model, moes, args, train_blocks, device, amp_dtype, scaler,
                steps=None, warmup_frac=0.1, frozen=()):
    """Joint training with linear warmup + cosine decay to 10% (the schedule
    the evo runs lacked: constant LR all the way). Params whose name starts
    with any of `frozen` (e.g. shared experts) stay frozen throughout."""
    import math
    import time
    steps = args.steps if steps is None else steps
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for _, m in moes:
        for name, p in m.named_parameters():
            if any(name.startswith(fx) for fx in frozen):
                p.requires_grad_(False)
                continue
            p.requires_grad_(True)
            trainable.append(p)
    base_lr = args.lr
    opt = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    warm = max(1, int(steps * warmup_frac))

    def lr_at(s):
        if s < warm:
            return base_lr * (s + 1) / warm
        t = (s - warm) / max(1, steps - warm)
        return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    curve, t0, done = [], time.perf_counter(), 0
    accum = max(1, args.grad_accum)
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
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
        if step % max(1, min(steps // 5, 25)) == 0 or step == steps - 1:
            el = time.perf_counter() - t0
            curve.append({"step": step, "loss": total, "elapsed": el,
                          "lr": lr_at(step)})
            print(f"      step {step:4d}  loss {total:.4f}  "
                  f"lr {lr_at(step):.1e}  elapsed {el:6.0f}s", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"steps_completed": done, "loss_curve": curve,
            "wall_seconds": time.perf_counter() - t0, "schedule": "warmup-cosine",
            "trainable_params": sum(p.numel() for p in trainable)}


@torch.no_grad()
def measure_specialisation(model, moes, layers, calib_x):
    """Post-train specialization probe: shared share, eff experts, load."""
    out = {}
    for i, m in moes:
        mlp = getattr(model.model.layers[i], "mlp")
        cap = {}
        h = mlp.register_forward_pre_hook(
            lambda _m, a, c=cap: c.setdefault("x", a[0].detach()))
        model.eval()
        _ = model(calib_x)
        h.remove()
        xin = cap["x"].reshape(-1, cap["x"].shape[-1]).to(m.gate.dtype)
        sh = m._shared_output(xin).detach().float()
        tot = m.forward(xin).detach().float()
        share = float(sh.norm() / tot.norm().clamp_min(1e-12))
        st = m.last_stats
        out[i] = {"shared_share": share,
                  "effective_experts": st.get("effective_experts"),
                  "load_max_over_mean": st.get("load_max_over_mean")}
    return out


def load_model_tokenizer(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16).to(device)
    model.eval()
    return model, tok


def train_flow(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    amp_dtype = POC.pick_amp_dtype(device)
    scaler = POC.make_grad_scaler(amp_dtype)
    out = args.out or f"results/mocm_gen{args.gen}.json"
    layers = [int(v) for v in args.layers]

    model, tok = load_model_tokenizer(args, device)
    fargs = argparse.Namespace(text_file=args.train_file, offline=False,
                               dataset="x", corpus_chars=0)
    ids, _ = POC.load_token_stream(fargs, tok)
    blocks = POC.make_blocks(ids, args.seq_len)
    train_blocks = blocks[:-4]
    calib_x = blocks[0][0].to(device)

    dom_blocks = {d: DOM.domain_blocks(tok, d) for d in DOM.DOMAINS}
    dom_blocks = {d: [(x.to(device), y.to(device)) for x, y in b]
                  for d, b in dom_blocks.items()}
    dense_dom = DOM.domain_ppls(model, dom_blocks, device, amp_dtype)
    print(f"dense domains: { {d: round(v, 2) for d, v in dense_dom.items()} }",
          flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    report = {"gen": args.gen, "layers": layers, "dense_dom": dense_dom,
              "runs": [], "spotlight": {}}
    if os.path.exists(out):
        try:
            prev = json.load(open(out))
            report["spotlight"] = prev.get("spotlight", {})
            if prev.get("gen") == args.gen:
                report["runs"] = prev.get("runs", [])
                report["dense_dom"] = prev.get("dense_dom", report["dense_dom"])
        except (OSError, ValueError):
            pass

    originals = [(i, model.model.layers[i].mlp) for i in layers]
    for spec in args.child:
        c = parse_child(spec)
        tag = c["tag"]
        print(f"=== {tag} E={c['E']} k={c['k']} sh={c['sh']} "
              f"steps={c['steps']} lr={c['lr']} ===", flush=True)
        BR.restore(model.model.layers, originals)
        try:
            shs = None
            if c.get("adaptive"):
                resid = probe_layer_residual(model, layers, calib_x, device,
                                             n_routed=c["E"])
                shs = allocate_shared(resid, c["sh"])
                print(f"  oracle residuals: " +
                      " ".join(f"{i}:{resid[i]:.2f}" for i in layers), flush=True)
                print(f"  allocated shared: " +
                      " ".join(f"{i}:{shs[i]}" for i in layers), flush=True)
            moes = BENCH.build_champ(model, layers, device, c["E"], c["k"],
                                     shs or c["sh"], calib_x, seed=args.seed,
                                     noise=float(c.get("noise", 0.0)))
            frozen_dom = DOM.domain_ppls(model, dom_blocks, device, amp_dtype)
            btx_stats = None
            if c.get("btx"):
                E0 = moes[0][1].E
                bord = BTX_ORDER8 if E0 == 8 else BTX_ORDER16
                dom_train = btx_domain_blocks(tok, args.train_file, args.seq_len,
                                              order=bord)
                E = moes[0][1].E
                if len(dom_train) != E:
                    if E % len(dom_train) == 0:
                        # split each domain into equal shards (same-distribution
                        # halves: less specialization per pair, noted, but keeps
                        # one-domain-per-expert with distinct initializations)
                        n = E // len(dom_train)
                        split = []
                        for bl in dom_train:
                            k = len(bl) // n
                            split += [bl[i * k:(i + 1) * k] for i in range(n)]
                        dom_train = split
                    else:
                        print(f"  btx skipped: {len(dom_train)} domains "
                              f"indivisible into E={E}", flush=True)
                        dom_train = []
                if dom_train:
                    bargs = argparse.Namespace(
                        branch_steps=int(c.get("bsteps", 40)), lr=c["lr"],
                        weight_decay=0.0, grad_accum=4, clip=1.0, seed=args.seed)
                    btx_stats = BR.branch_train_experts(
                        model, moes, bargs, dom_train, device, amp_dtype, scaler)
                    print(f"  stage2 done: {btx_stats.get('steps', '?')} "
                          f"expert-steps", flush=True)
            pseudo = argparse.Namespace(steps=c["steps"], lr=c["lr"],
                                        weight_decay=0.0, grad_accum=4,
                                        clip=1.0, seed=args.seed)
            if c.get("sched") or c.get("fsh"):
                frozen = ("s_gate", "s_up", "s_down") if c.get("fsh") else ()
                train_stats = train_sched(model, moes, pseudo, train_blocks,
                                          device, amp_dtype, scaler,
                                          frozen=frozen)
            else:
                train_stats = BR.train_model(model, moes, pseudo, train_blocks,
                                             device, amp_dtype, scaler)
            trained_dom = DOM.domain_ppls(model, dom_blocks, device, amp_dtype)
            mx = sum(trained_dom[d] / dense_dom[d] for d in DOM.DOMAINS) / len(DOM.DOMAINS)
            inter = 3584
            if shs:
                active = sum(shs[i] + c["k"] * (inter // c["E"])
                             for i in layers) / (len(layers) * inter)
            else:
                active = (c["sh"] + c["k"] * (inter // c["E"])) / inter
            print(f"  mean_xdense {mx:.3f} active {active:.3f} | " +
                  " ".join(f"{d[:2]}:{trained_dom[d]:.1f}" for d in DOM.DOMAINS),
                  flush=True)
            spec_probe = measure_specialisation(model, moes, layers, calib_x)
            mean_share = sum(v["shared_share"] for v in spec_probe.values()
                             ) / max(len(spec_probe), 1)
            mean_eff = sum(v["effective_experts"] or 0 for v in spec_probe.values()
                           ) / max(len(spec_probe), 1)
            print(f"  shared_share {mean_share:.2f} eff {mean_eff:.1f}", flush=True)
            ckpt = {"tag": tag, "layers": layers,
                    "spec": {"E": c["E"], "k": c["k"], "sh": c["sh"],
                             "shs": ({str(i): shs[i] for i in layers}
                                     if shs else None)},
                    "moes": {str(i): m.state_dict() for i, m in moes}}
            torch.save(ckpt, os.path.join(CKPT_DIR, f"gen{args.gen}_{tag}.pt"))
            entry = {"name": tag, "spec": c, "active": active,
                     "layers": layers,
                     "frozen_dom": frozen_dom, "trained_dom": trained_dom,
                     "mean_xdense": mx, "train": train_stats,
                     "btx": btx_stats,
                     "shared_share": mean_share, "eff_experts": mean_eff,
                     "error": None}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            entry = {"name": tag, "error": f"{type(e).__name__}: {e}"}
        report["runs"] = [r for r in report["runs"] if r.get("name") != tag] + [entry]
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    BR.restore(model.model.layers, originals)
    print(f"wrote {out}")
    return 0


@torch.no_grad()
def gen_text(model, tok, prompt: str, n=128) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def apply_ckpt(model, layers, ckpt_path, device):
    """Swap trained MoCM weights into the model from a checkpoint.

    The unit order is only used at *build* time (which dense rows initialise
    each expert). After training, the function lives entirely in the stacked
    expert tensors, which forward uses directly -- so rebuilding with
    order=None and loading the state is exactly the trained layer.
    """
    from moeme.mocm import MoCMv1Config
    from moeme.mocm import build_mocm_v1_layer
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    blocks_mod = model.model.layers
    moes = []
    for i in layers:
        mlp = getattr(blocks_mod[i], "mlp")
        shs = ckpt["spec"].get("shs")
        sh = int(shs[str(i)]) if shs else ckpt["spec"]["sh"]
        cfg = MoCMv1Config(n_routed=ckpt["spec"]["E"],
                           top_k=ckpt["spec"]["k"],
                           shared_units=sh)
        m = build_mocm_v1_layer(mlp.gate_proj.weight.detach(),
                                mlp.up_proj.weight.detach(),
                                mlp.down_proj.weight.detach(), cfg)
        m = m.to(device)
        m.router.to(dtype=torch.float32)
        m.load_state_dict(ckpt["moes"][str(i)], strict=True)
        setattr(blocks_mod[i], "mlp", m)
        moes.append((i, m))
    return moes


def spot_flow(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    amp_dtype = POC.pick_amp_dtype(device)
    out = args.out or f"results/mocm_gen{args.gen}.json"
    report = json.load(open(out))
    tags = args.finalists.split(",")
    layers = report["layers"]
    model, tok = load_model_tokenizer(args, device)
    originals = [(i, model.model.layers[i].mlp) for i in layers]

    # dense reference outputs once
    results = {"dense": None, "tags": {}}
    if args.spot == "mbpp":
        k = args.spot_k or 16
        tasks = DOM.mbpp_subset(k)
        dense_codes = [DOM.extract_code(gen_text(model, tok, DOM.mbpp_prompt(t)))
                       for t in tasks]
        results["dense"] = sum(DOM.mbpp_pass(t, c) for t, c in zip(tasks, dense_codes))
        print(f"dense mbpp: {results['dense']}/{k}", flush=True)
        for tag in tags:
            BR.restore(model.model.layers, originals)
            apply_ckpt(model, layers, os.path.join(CKPT_DIR, f"gen{args.gen}_{tag}.pt"), device)
            model.eval()
            passes = 0
            for t in tasks:
                code = DOM.extract_code(gen_text(model, tok, DOM.mbpp_prompt(t)))
                passes += DOM.mbpp_pass(t, code)
            results["tags"][tag] = passes
            print(f"{tag} mbpp: {passes}/{k}", flush=True)
    else:
        k = args.spot_k or 50
        tasks = DOM.gsm8k_subset(k)
        dense_ok = sum(DOM.gsm8k_grade(t, gen_text(model, tok, DOM.gsm8k_prompt(t), n=64))
                       for t in tasks)
        results["dense"] = dense_ok
        print(f"dense gsm8k: {dense_ok}/{k}", flush=True)
        for tag in tags:
            BR.restore(model.model.layers, originals)
            apply_ckpt(model, layers, os.path.join(CKPT_DIR, f"gen{args.gen}_{tag}.pt"), device)
            model.eval()
            ok = sum(DOM.gsm8k_grade(t, gen_text(model, tok, DOM.gsm8k_prompt(t), n=64))
                     for t in tasks)
            results["tags"][tag] = ok
            print(f"{tag} gsm8k: {ok}/{k}", flush=True)
    results["k"] = k
    report["spotlight"][args.spot] = results
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}")
    return 0


def main() -> int:
    args = build_args()
    if args.spot:
        return spot_flow(args)
    if not args.child:
        print("no --child given")
        return 2
    return train_flow(args)


if __name__ == "__main__":
    sys.exit(main())
