#!/usr/bin/env python3
"""mocm_decode.py -- static-KV manual decode loop + compile/graphs.

Compares end-to-end batch-1 decode: HF generate (eager) vs manual loop
(dynamic cache) vs manual loop (static cache) vs compiled forward.
The 2-4x claim from the literature lives or dies here.

  python3 experiments/mocm_decode.py --model Qwen/Qwen3.5-0.8B --n 64
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.join(
    __import__("os").path.dirname(__file__), "..", "src"))


def bench_fn(fn, n=64, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device).eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    N = args.n

    # 1. eager generate baseline
    def run_gen():
        with torch.no_grad():
            return model.generate(ids, max_new_tokens=N, do_sample=False,
                                  use_cache=True)
    spd, _ = bench_fn(run_gen, N)
    print(f"eager generate: {spd:.1f} tok/s", flush=True)

    # 2. manual loop, dynamic cache
    def run_manual():
        with torch.no_grad():
            out = model(ids, use_cache=True)
            cache = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            toks = [nxt]
            for _ in range(N - 1):
                out = model(nxt, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
                toks.append(nxt)
            return torch.cat(toks, dim=1)
    try:
        spd, t2 = bench_fn(run_manual, N)
        print(f"manual dynamic: {spd:.1f} tok/s", flush=True)
    except Exception as e:
        print(f"manual dynamic FAILED: {type(e).__name__}: {str(e)[:150]}",
              flush=True)
        t2 = None

    # 3. manual loop, static cache
    t3 = None
    try:
        static = StaticCache(config=model.config, max_batch_size=1,
                             max_cache_len=ids.shape[1] + N + 16,
                             device=device, dtype=torch.bfloat16)
        model2 = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(device).eval()

        def run_static():
            cache = StaticCache(config=model2.config, max_batch_size=1,
                                max_cache_len=ids.shape[1] + N + 16,
                                device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                out = model2(ids, past_key_values=cache, use_cache=True,
                             cache_position=torch.arange(ids.shape[1],
                                                         device=device))
                pos = ids.shape[1]
                nxt = out.logits[:, -1:].argmax(-1)
                toks = [nxt]
                for _ in range(N - 1):
                    out = model2(nxt, past_key_values=cache, use_cache=True,
                                 cache_position=torch.arange(
                                     pos, pos + 1, device=device))
                    pos += 1
                    nxt = out.logits[:, -1:].argmax(-1)
                    toks.append(nxt)
                return torch.cat(toks, dim=1)
        spd, t3 = bench_fn(run_static, N)
        print(f"manual static: {spd:.1f} tok/s", flush=True)
    except Exception as e:
        print(f"manual static FAILED: {type(e).__name__}: {str(e)[:150]}",
              flush=True)

    # 4. compiled static loop
    if t3 is not None:
        try:
            fwd = torch.compile(model2.forward, mode="max-autotune-no-cudagraphs",
                                fullgraph=False, dynamic=False)

            def run_compiled():
                cache = StaticCache(config=model2.config, max_batch_size=1,
                                    max_cache_len=ids.shape[1] + N + 16,
                                    device=device, dtype=torch.bfloat16)
                with torch.no_grad():
                    out = fwd(ids, past_key_values=cache, use_cache=True,
                              cache_position=torch.arange(ids.shape[1],
                                                          device=device))
                    pos = ids.shape[1]
                    nxt = out.logits[:, -1:].argmax(-1)
                    toks = [nxt]
                    for _ in range(N - 1):
                        out = fwd(nxt, past_key_values=cache, use_cache=True,
                                  cache_position=torch.arange(
                                      pos, pos + 1, device=device))
                        pos += 1
                        nxt = out.logits[:, -1:].argmax(-1)
                        toks.append(nxt)
                    return torch.cat(toks, dim=1)
            spd, t4 = bench_fn(run_compiled, N)
            print(f"compiled static: {spd:.1f} tok/s", flush=True)
            # correctness: same tokens as eager?
            with torch.no_grad():
                ref = run_gen()
            a4 = t4.reshape(-1).cpu().tolist()
            ar = ref.reshape(-1).cpu().tolist()[-N:]
            div = next((j for j, (u, v) in enumerate(zip(a4, ar)) if u != v), N)
            print(f"token match vs eager: {div}/{N} identical prefix",
                  flush=True)
        except Exception as e:
            print(f"compiled static FAILED: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
