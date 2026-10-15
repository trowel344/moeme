#!/usr/bin/env python3
"""
mocm_bench.py -- benchmarks for the evolved MoCM: fitness table, latency,
and text-level face-off (dense vs trained MoCM greedy continuations).

  python3 experiments/mocm_bench.py --table-only
  python3 experiments/mocm_bench.py --champs 25,50 --steps 300
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _load_sibling(name: str):
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
BR = _load_sibling("branch_poc")

from moeme.mocm import MoCMv1Config, activation_order, build_mocm_v1_layer  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "17 * 23 =",
    "Write a fibonacci function in Python:",
    "The quick brown fox",
    "Explain photosynthesis in one sentence:",
    "def gcd(a, b):",
]

# champ key -> (tag label, n_routed, top_k, shared_units)
CHAMPS = {
    "25": ("bench25", 8, 1, 448),     # g1-s500 lineage: 25% active
    "37": ("bench37", 8, 1, 896),     # g2-s500 lineage: 37.5% active
    "50": ("bench50", 8, 1, 1344),    # g2-shared1344 lineage: 50% active
    "50t": ("bench50t", 16, 2, 1344),  # g3-topk2: two narrow picks, 50% active
    "62": ("bench62", 8, 1, 1792),    # g3-shared1792 lineage: 62.5% active
    "12": ("bench12", 16, 1, 224),    # exp4 lineage: 12.5% active
}


def token_agreement(tok, a: str, b: str, n: int = 48) -> float:
    """Mean positional token agreement over the first n continuation tokens."""
    ta, tb = (tok.encode(a)[:n] + [None] * n)[:n], (tok.encode(b)[:n] + [None] * n)[:n]
    return sum(1 for u, v in zip(ta, tb) if u == v) / n


@torch.no_grad()
def bench_table() -> int:
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "results", "mocm_*.json"))):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        dense = r.get("dense_val_ppl")
        for e in r.get("runs", []):
            if e.get("error") or "val_ppl" not in e:
                continue
            cfg = e.get("config", {})
            rows.append({
                "run": e["name"],
                "E": cfg.get("n_routed", cfg.get("n_experts", "?")),
                "k": cfg.get("top_k", "?"),
                "shared": cfg.get("shared_units", "?"),
                "active": round(e.get("active_width_true", float("nan")), 3),
                "frozen": round(e.get("frozen_ppl", float("nan")), 2),
                "trained": round(e["val_ppl"], 2),
                "xdense": round(e["val_ppl"] / dense, 2) if dense else "?",
                "file": os.path.basename(f),
            })
    rows.sort(key=lambda d: (d["trained"] if isinstance(d["trained"], float) else 1e9))
    print(f"{'run':22s} {'E':>3s} {'k':>2s} {'shared':>6s} {'active':>6s} "
          f"{'frozen':>7s} {'trained':>8s} {'xdense':>6s}")
    for d in rows:
        print(f"{d['run']:22s} {str(d['E']):>3s} {str(d['k']):>2s} "
              f"{str(d['shared']):>6s} {d['active']:>6.3f} "
              f"{d['frozen']:>7.2f} {d['trained']:>8.2f} {d['xdense']:>6.2f}")
    print(f"\n{len(rows)} runs, dense 23.42 throughout")
    return 0


@torch.no_grad()
def time_layer(layer, x, iters=100) -> float:
    for _ in range(10):
        layer(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        layer(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def build_champ(model, layers, device, n_routed, top_k, shared, calib_x, seed=0,
                noise=0.0):
    from moeme.mocm import MoCMv1Config
    from moeme.upcycle import _dequant
    blocks_mod = model.model.layers
    moes = []
    for i in layers:
        sh = shared[i] if isinstance(shared, dict) else shared
        mlp = getattr(blocks_mod[i], "mlp")
        gw, uw, dw = (_dequant(mlp.gate_proj.weight),
                      _dequant(mlp.up_proj.weight),
                      _dequant(mlp.down_proj.weight))
        with torch.no_grad():
            cap = {}
            h = mlp.register_forward_pre_hook(
                lambda _m, a, c=cap: c.setdefault("x", a[0].detach()))
            model.eval()
            _ = model(calib_x)
            h.remove()
            xin = cap["x"].reshape(-1, cap["x"].shape[-1])
        order = activation_order(gw.detach(), uw.detach(), xin[:4096])
        cfg = MoCMv1Config(n_routed=n_routed, top_k=top_k, shared_units=sh,
                           seed=seed, unit_order=order, init_noise=noise)
        m = build_mocm_v1_layer(gw.detach(), uw.detach(), dw.detach(), cfg)
        m = m.to(device)
        from moeme.mocm import align_moe_dtype
        align_moe_dtype(m, torch.bfloat16 if device.type == "cuda"
                        else torch.float32)
        setattr(blocks_mod[i], "mlp", m)
        moes.append((i, m))
    return moes


@torch.no_grad()
def greedy_cont(model, tok, prompt: str, n=48) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--champs", default="25,50")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--layers", nargs="+", default=["20", "21", "22", "23"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--text-file", default=".cache/moeme/mocm_corpus.txt")
    ap.add_argument("--out", default="results/mocm_bench.json")
    args = ap.parse_args()

    if args.table_only:
        return bench_table()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = POC.pick_amp_dtype(device)
    scaler = POC.make_grad_scaler(amp_dtype)
    layers = [int(v) for v in args.layers]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    fargs = argparse.Namespace(text_file=args.text_file, offline=False,
                               dataset="x", corpus_chars=0)
    ids, _ = POC.load_token_stream(fargs, tok)
    import torch as _t
    blocks = POC.make_blocks(ids, 512)
    train_blocks, val_blocks = blocks[:-4], blocks[-4:]
    calib_x = blocks[0][0]

    report = {"champs": {}}
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            report["champs"] = prev.get("champs", {})
        except (OSError, ValueError):
            pass
    for key in args.champs.split(","):
        label, E, k, shared = CHAMPS[key.strip()]
        active = (shared + k * (3584 // E)) / 3584
        print(f"\n===== champ {label}: E={E} k={k} shared={shared} "
              f"({active:.1%} active) =====", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                     dtype=torch.bfloat16).to(device)
        model.eval()
        originals = [(i, model.model.layers[i].mlp) for i in layers]
        dense_ppl = POC.eval_perplexity(model, val_blocks, device, amp_dtype, 4)

        # latency: dense mlp vs MoCM layer on one representative layer
        H = 1024
        x1 = torch.randn(1, H, device=device, dtype=torch.bfloat16)  # decode-like
        dense_mlp = model.model.layers[layers[0]].mlp
        t_dense = time_layer(dense_mlp, x1)
        moes = build_champ(model, layers, device, E, k, shared,
                           calib_x.to(device))
        m0 = dict(moes)[layers[0]]
        t_mocm = time_layer(m0, x1)
        dense_bytes = sum(p.numel() for p in dense_mlp.parameters()) * 2
        mocm_bytes = (m0.active_params) * 2
        print(f"  latency/forward b1s1: dense {t_dense:.3f}ms "
              f"({dense_bytes/1e6:.1f}MB) vs MoCM {t_mocm:.3f}ms "
              f"({mocm_bytes/1e6:.1f}MB active)", flush=True)

        frozen_ppl = POC.eval_perplexity(model, val_blocks, device, amp_dtype, 4)
        pseudo = argparse.Namespace(steps=args.steps, lr=args.lr,
                                    weight_decay=0.0, grad_accum=4,
                                    clip=1.0, seed=0)
        BR.train_model(model, moes, pseudo, train_blocks, device, amp_dtype,
                       scaler)
        trained_ppl = POC.eval_perplexity(model, val_blocks, device, amp_dtype, 4)
        print(f"  ppl: dense {dense_ppl:.2f} frozen {frozen_ppl:.2f} "
              f"trained {trained_ppl:.2f} ({trained_ppl/dense_ppl:.2f}x)",
              flush=True)

        # text face-off on identical prompts
        dense_texts = None
        BR.restore(model.model.layers, originals)
        model.eval()
        with torch.no_grad():
            dense_texts = [greedy_cont(model, tok, p) for p in PROMPTS]
        # re-apply trained moes (still in memory under `moes`)
        for i, m in moes:
            setattr(model.model.layers[i], "mlp", m)
        model.eval()
        with torch.no_grad():
            mocm_texts = [greedy_cont(model, tok, p) for p in PROMPTS]
        same = sum(1 for a, b in zip(dense_texts, mocm_texts) if a == b)
        agree = [token_agreement(tok, a, b) for a, b in zip(dense_texts, mocm_texts)]
        mean_agree = sum(agree) / len(agree)
        div_at = []
        for a, b in zip(dense_texts, mocm_texts):
            ta, tb = tok.encode(a), tok.encode(b)
            d = next((j for j, (u, v) in enumerate(zip(ta, tb)) if u != v),
                     min(len(ta), len(tb)))
            div_at.append(d)
        print(f"  continuations identical: {same}/{len(PROMPTS)}, "
              f"first-divergence tokens: {div_at}", flush=True)
        for p, a, b in zip(PROMPTS, dense_texts, mocm_texts):
            print(f"    | {p!r}\n      dense: {a[:100]!r}\n      mocm : {b[:100]!r}")
        report["champs"][label] = {
            "E": E, "k": k, "shared": shared, "active": active,
            "dense_ppl": dense_ppl, "frozen_ppl": frozen_ppl,
            "trained_ppl": trained_ppl, "xdense": trained_ppl / dense_ppl,
            "lat_ms_dense": t_dense, "lat_ms_mocm": t_mocm,
            "identical": same, "n_prompts": len(PROMPTS),
            "mean_agreement": mean_agree, "agreement": agree,
            "first_divergence": div_at,
            "dense_texts": dense_texts, "mocm_texts": mocm_texts,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
