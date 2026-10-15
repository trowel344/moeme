#!/usr/bin/env python3
"""mocm_distill_eval.py -- speed + behavior of a distilled checkpoint vs dense.

  python3 experiments/mocm_distill_eval.py \
      --ckpt .cache/moeme/distill_4L25.pt --out results/mocm_distill_eval.json
"""

from __future__ import annotations

import argparse
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
    spec = importlib.util.spec_from_file_location(f"_dev_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
DOM = _load_sibling("mocm_domains")
DST = _load_sibling("mocm_distill")
BENCH = _load_sibling("mocm_bench")

PROMPTS = BENCH.PROMPTS


@torch.no_grad()
def greedy_cont(model, tok, prompt: str, n=48) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def gen_speed(model, tok, n=64) -> float:
    ids = tok("The capital of France is",
              return_tensors="pt").input_ids.to(model.device)
    for _ in range(2):
        model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--thresh", type=float, default=0.0,
                    help="wrap remaining dense MLPs with ThreshMLP(q): the "
                         "new-arch combination (distilled easy layers + "
                         "threshold sparsity elsewhere)")
    ap.add_argument("--out", default="results/mocm_distill_eval.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    layers = ck["layers"]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = {"ckpt": args.ckpt, "layers": layers}
    texts = {}
    for label in ("dense", "distilled"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(device)
        model.eval()
        if label == "distilled":
            for i in layers:
                st = DST.NarrowStudent(ck["hidden"], ck["inter"]).to(device)
                st.load_state_dict(ck["students"][str(i)])
                st = st.to(dtype=torch.bfloat16)
                model.model.layers[i].mlp = st
            if args.thresh > 0:
                have = set(layers)
                for i in range(len(model.model.layers)):
                    if i not in have:
                        model.model.layers[i].mlp = DST.ThreshMLP(
                            model.model.layers[i].mlp, args.thresh).to(device)
            model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        spd = gen_speed(model, tok)
        cont = [greedy_cont(model, tok, p) for p in PROMPTS]
        texts[label] = cont
        ev = {d: [(x.to(device), y.to(device)) for x, y in DOM.domain_blocks(tok, d)]
              for d in DOM.DOMAINS}
        import math
        dom = {}
        with torch.no_grad():
            for d, bl in ev.items():
                t, n = 0.0, 0
                for x, y in bl:
                    l = model(x, labels=y).loss
                    t += l.item() * y.numel()
                    n += y.numel()
                dom[d] = math.exp(t / n)
        print(f"{label} domains: " + " ".join(f"{d}:{dom[d]:.1f}" for d in dom),
              flush=True)
        mbpp_tasks = DOM.mbpp_subset(16)
        mbpp_ok = sum(DOM.mbpp_pass(t, DOM.extract_code(
            greedy_cont(model, tok, DOM.mbpp_prompt(t), n=128))) for t in mbpp_tasks)
        gsm_tasks = DOM.gsm8k_subset(50)
        gsm_ok = sum(DOM.gsm8k_grade(t, greedy_cont(model, tok, DOM.gsm8k_prompt(t), n=64))
                     for t in gsm_tasks)
        print(f"{label}: {spd:.1f} tok/s mbpp {mbpp_ok}/16 gsm8k {gsm_ok}/50 "
              f"params {n_params/1e9:.3f}B", flush=True)
        out[label] = {"tok_s": spd, "mbpp": mbpp_ok, "gsm8k": gsm_ok,
                      "params_B": n_params / 1e9, "domains": dom,
                      "thresh": args.thresh if label == "distilled" else 0.0}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    agree = [BENCH.token_agreement(tok, a, b)
             for a, b in zip(texts["dense"], texts["distilled"])]
    same = sum(1 for a, b in zip(texts["dense"], texts["distilled"]) if a == b)
    out["text"] = {"identical": same, "n": len(PROMPTS),
                   "mean_agreement": sum(agree) / len(agree),
                   "dense": texts["dense"], "distilled": texts["distilled"]}
    print(f"text: {same}/{len(PROMPTS)} identical, "
          f"agreement {sum(agree)/len(agree):.3f}", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
