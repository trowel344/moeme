#!/usr/bin/env python3
"""mocm_final.py -- clean final benchmark: best full-model MoCM vs dense.

Same tokens (fixed domain blocks), same prompts (greedy), same tasks
(mbpp exec, gsm8k grade), same hardware. One number per dimension.

  python3 experiments/mocm_final.py --ckpt .cache/moeme/evo_ckpts/gen15_g15-long.pt
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
    spec = importlib.util.spec_from_file_location(f"_fin_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
BR = _load_sibling("branch_poc")
EVO = _load_sibling("mocm_evo")
DOM = _load_sibling("mocm_domains")
BENCH = _load_sibling("mocm_bench")

PROMPTS = BENCH.PROMPTS + [
    "Write a Python function that returns the sum of a list:",
    "A train travels 120 km in 2 hours. Its speed is",
]


@torch.no_grad()
def greedy_cont(model, tok, prompt: str, n=48) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def gen_speed(model, tok, n=48) -> float:
    """ Greedy tok/s on a fixed prompt (batch 1, includes prefill)."""
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(model.device)
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
    ap.add_argument("--out", default="results/mocm_final.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = POC.pick_amp_dtype(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    layers = ckpt["layers"]
    spec = ckpt["spec"]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dom_blocks = {d: [(x, y) for x, y in DOM.domain_blocks(tok, d)]
                  for d in DOM.DOMAINS}

    report = {"ckpt": args.ckpt, "spec": spec, "layers": layers}
    texts = {}

    for label in ("dense", "mocm"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(device)
        model.eval()
        if label == "mocm":
            EVO.apply_ckpt(model, layers, args.ckpt, device)
            model.eval()
        db = {d: [(x.to(device), y.to(device)) for x, y in b]
              for d, b in dom_blocks.items()}
        dom = DOM.domain_ppls(model, db, device, amp_dtype)
        mx = sum(dom.values()) / len(dom)
        spd = gen_speed(model, tok)
        cont = [greedy_cont(model, tok, p) for p in PROMPTS]
        texts[label] = cont
        # spotlights
        mbpp_tasks = DOM.mbpp_subset(16)
        mbpp_ok = sum(DOM.mbpp_pass(t, DOM.extract_code(greedy_cont(model, tok, DOM.mbpp_prompt(t), n=128)))
                      for t in mbpp_tasks)
        gsm_tasks = DOM.gsm8k_subset(50)
        gsm_ok = sum(DOM.gsm8k_grade(t, greedy_cont(model, tok, DOM.gsm8k_prompt(t), n=64))
                     for t in gsm_tasks)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{label}: mean-dom-ppl {mx:.2f} | {spd:.1f} tok/s | "
              f"mbpp {mbpp_ok}/16 gsm8k {gsm_ok}/50 | params {n_params/1e9:.2f}B",
              flush=True)
        report[label] = {"domain_ppl": dom, "mean_dom_ppl": mx, "tok_s": spd,
                         "mbpp": mbpp_ok, "gsm8k": gsm_ok,
                         "total_params_B": n_params / 1e9}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # text agreement (dense as reference)
    agree = [BENCH.token_agreement(tok, a, b)
             for a, b in zip(texts["dense"], texts["mocm"])]
    same = sum(1 for a, b in zip(texts["dense"], texts["mocm"]) if a == b)
    report["text"] = {"identical": same, "n": len(PROMPTS),
                      "mean_agreement": sum(agree) / len(agree),
                      "agreement": agree,
                      "dense": texts["dense"], "mocm": texts["mocm"]}
    print(f"text: {same}/{len(PROMPTS)} identical, "
          f"agreement {sum(agree)/len(agree):.3f}", flush=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
