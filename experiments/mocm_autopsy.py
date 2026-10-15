#!/usr/bin/env python3
"""mocm_autopsy.py -- where exactly does the full-model MoCM lose dense?

Loads dense 0.8B + a full-model MoCM ckpt, feeds identical tokens, and per
layer measures:
  * input drift:   ||x_mocm - x_dense|| / ||x_dense||  (compounding?)
  * output error:  ||y_mocm - y_dense|| / ||y_dense||  (local damage)
  * router usage:  effective experts, max/mean load, entropy
  * shared share:  ||shared_out|| / ||total_out||      (who carries the layer)

  python3 experiments/mocm_autopsy.py \
    --ckpt .cache/moeme/evo_ckpts/gen15_g15-long.pt --out results/mocm_autopsy.json
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
    spec = importlib.util.spec_from_file_location(f"_aut_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


POC = _load_sibling("upcycle_poc")
BR = _load_sibling("branch_poc")
EVO = _load_sibling("mocm_evo")


def capture_mlp_io(model, layers, batch, device):
    """{i: {x, y}} dense-or-current MLP in/outputs on one batch (no grad)."""
    blocks = model.model.layers
    caps = {i: {} for i in layers}

    def pre(i):
        def fn(_m, a):
            caps[i]["x"] = a[0].detach()
        return fn

    def post(i):
        def fn(_m, _a, o):
            caps[i]["y"] = o.detach() if torch.is_tensor(o) else o[0].detach()
        return fn

    hs = []
    for i in layers:
        hs.append(blocks[i].mlp.register_forward_pre_hook(pre(i)))
        hs.append(blocks[i].mlp.register_forward_hook(post(i)))
    model.eval()
    with torch.no_grad():
        model(batch)
    for h in hs:
        h.remove()
    return caps


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--train-file", default=".cache/moeme/evo_fresh.txt")
    ap.add_argument("--out", default="results/mocm_autopsy.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    layers = ckpt["layers"]

    tok = AutoTokenizer.from_pretrained(args.model)
    fargs = argparse.Namespace(text_file=args.train_file, offline=False,
                               dataset="x", corpus_chars=0)
    ids, _ = POC.load_token_stream(fargs, tok)
    batch = POC.make_blocks(ids, 256)[5][0].to(device)

    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16).to(device)
    model.eval()
    dense_caps = capture_mlp_io(model, layers, batch, device)
    dense_logits = model(batch).logits.detach().float()

    originals = [(i, model.model.layers[i].mlp) for i in layers]
    EVO.apply_ckpt(model, layers, args.ckpt, device)
    model.eval()
    mocm_caps = capture_mlp_io(model, layers, batch, device)
    moes = [(i, model.model.layers[i].mlp) for i in layers]
    mocm_logits = model(batch).logits.detach().float()

    per_layer = {}
    for i in layers:
        xd, yd = dense_caps[i]["x"].float(), dense_caps[i]["y"].float()
        xm, ym = mocm_caps[i]["x"].float(), mocm_caps[i]["y"].float()
        drift = float((xm - xd).norm() / xd.norm().clamp_min(1e-12))
        err = float((ym - yd).norm() / yd.norm().clamp_min(1e-12))
        m = dict(moes)[i]
        # shared share of the layer output (routed path excluded)
        with torch.no_grad():
            x2 = xm.reshape(-1, xm.shape[-1]).to(m.gate.dtype)
            sh = m._shared_output(x2).detach().float() if m.config.n_shared else None
            tot = m.forward(xm.to(m.gate.dtype)).detach().float()
        share = float(sh.norm() / tot.norm().clamp_min(1e-12)) if sh is not None else 0.0
        st = m.last_stats
        per_layer[i] = {"input_drift": drift, "output_error": err,
                        "shared_share": share,
                        "effective_experts": st.get("effective_experts"),
                        "load_max_over_mean": st.get("load_max_over_mean"),
                        "usage": st.get("usage_fraction")}
        print(f"layer {i:2d} drift {drift:.3f} err {err:.3f} "
              f"shared {share:.2f} eff {st.get('effective_experts', float('nan')):.1f}",
              flush=True)

    import math
    kl = float((torch.softmax(dense_logits, -1) *
                (torch.log_softmax(dense_logits, -1)
                 - torch.log_softmax(mocm_logits, -1))).sum(-1).mean())
    agree = float((dense_logits.argmax(-1) == mocm_logits.argmax(-1)).float().mean())
    print(f"final logit KL {kl:.3f}, argmax agreement {agree:.3f}", flush=True)

    BR.restore(model.model.layers, originals)
    json.dump({"ckpt": args.ckpt, "per_layer": per_layer,
               "final_logit_KL": kl, "argmax_agreement": agree},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
