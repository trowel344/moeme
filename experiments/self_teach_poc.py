#!/usr/bin/env python3
"""
self_teach_poc.py -- can a model teach its own sparse layers, with no corpus?

The question
------------
Every training run in this repo optimises next-token cross-entropy on a streamed
corpus (E14: the same 0.8B, 614k tokens in E15). But for an *upcycled* layer the
teacher already exists inside the model: the dense FFN it was partitioned from.
So the objective does not need labels, or a pretraining corpus, or even much
text -- only enough input to generate activations. This is the corpus-free
version: train the experts (and, where the combination is differentiable, the
router) to reproduce their own dense layer on the model's own activations,
captured from a short fixed list of sentences.

Two things are measured, and the first one is the more interesting.

1. **The magnitude analysis (frozen, no training).** A k-of-E partial sum is
   thought to be "scaled by ~k/E relative to the dense layer", which is the
   stated reason LLaMA-MoE and NVIDIA rescale their output. That claim is only
   true if the experts' contributions add *coherently*. If they are roughly
   orthogonal, every rescaling is a strictly worse approximation and the optimal
   scale is 1. So this measures, per layer, the least-squares optimal scale
   `s* = <ref, kept> / ||kept||^2` and the residual at three scales: 1 (the
   plain sum), E/k (the literature's fix), and s* (the floor for *any* magnitude
   correction). `s*` says which regime a real FFN partition is in -- and
   therefore whether the rescaling can help at all. This alone answers a
   question the frozen "oracle" tables cannot: `ffn.oracle_topk_residual` is a
   ratio of two quantities that scale together, so it is *invariant* to any
   global output scale.

2. **The treatment.** The same layers, trained on nothing but the reconstruction
   loss, in three arms that differ only in how the selected experts are
   combined. Compared against E14's cross-entropy run on the same layers.

Arms
----
  sum            the existing default; k-of-E, unscaled
  sum_rescaled   k-of-E scaled by E/k (= 1.0 at k == E, so the dense limit
                 survives), the fix the field calls essential
  softmax        the only arm where the router receives gradient from the
                 reconstruction loss at all -- under `sum` the selection weights
                 are constant ones, so the router is trained by the aux loss
                 alone. This is the arm that tests "let the modules figure out
                 their own division of labour"

Traps this experiment exists to avoid
-------------------------------------
* The residual metric is *anti-correlated* with quality in the trained regime:
  E16 measured the dense-reproduction residual **rising** while perplexity
  improved, in every trained row. So perplexity and generated text are measured
  beside it; a residual win alone would not mean the model got better.
* Training on activations from the evaluation window would leak. The calibration
  sentences are hand-written and outside it, and the report records them.

Usage:
    python3 experiments/self_teach_poc.py
    python3 experiments/self_teach_poc.py --steps 300 --layers 0 3 8 23
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import torch

import compare_upcycled_08b as cmp08b      # noqa: E402  the prompts + greedy loop
import upcycle_poc as poc                 # noqa: E402  corpus/ppl plumbing
from moeme.upcycle import (               # noqa: E402
    MoEConfig,
    collect_total_aux_loss,
    reconstruction_residual,
    upcycle_model,
)

# Hand-written, and deliberately not the evaluation window: these are the "goal"
# sentences a reader would propose, plus a few ordinary ones so the activations
# are not all from one grammatical shape. Nothing here is sampled from the
# pretraining corpus, which is the point -- the teacher is the model itself.
SENTENCES = [
    "Paris is the capital of France.",
    "The world is round.",
    "The sky is blue.",
    "Water boils at one hundred degrees Celsius.",
    "Two plus two is four.",
    "The Moon orbits the Earth.",
    "Ice is frozen water.",
    "Fish live in water.",
    "The Sun is a star.",
    "Birds have feathers and lay eggs.",
    "A triangle has three sides.",
    "The book is on the table.",
    "She opened the window and looked outside.",
    "The train arrives at nine in the morning.",
    "He wrote a short letter to his friend.",
    "The garden was full of red flowers.",
    "They walked along the river until it got dark.",
    "The meeting was moved to Thursday afternoon.",
    "def add(a, b):\n    return a + b",
    "for i in range(10):\n    print(i)",
    "x = [n ** 2 for n in range(5)]",
    "if temperature > 100:\n    print('boiling')",
    "The quick brown fox jumps over the lazy dog.",
    "A rolling stone gathers no moss.",
]
# 24 sentences of ordinary English and code: ~400 tokens, i.e. about 0.07% of
# the 614k tokens E15 trained on, and none of them labelled or corpus-sampled.


# ---------------------------------------------------------------------------
# activation capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def _capture(model, layers, feeds) -> dict[int, dict]:
    """Per-target-layer (input, dense output) over a list of token-id batches.

    Captured *before* upcycling, so `y` is the pretrained dense function and
    remains a valid teacher after training has moved the MoE weights. This is
    the same contract `upcycle_poc.capture_reference` uses, generalised from one
    batch to an arbitrary feed list.
    """
    caps: dict[int, dict] = {i: {"x": [], "y": []} for i in layers}
    blocks = model.model.layers

    def pre(i):
        def fn(_m, args):
            caps[i]["x"].append(args[0].detach().reshape(-1, args[0].shape[-1]))
        return fn

    def post(i):
        def fn(_m, _args, out):
            caps[i]["y"].append(out.detach().reshape(-1, out.shape[-1]))
        return fn

    handles = []
    for i in layers:
        handles.append(blocks[i].mlp.register_forward_pre_hook(pre(i)))
        handles.append(blocks[i].mlp.register_forward_hook(post(i)))
    model.eval()
    for ids in feeds:
        model(ids)
    for h in handles:
        h.remove()
    return {i: {"x": torch.cat(d["x"]).float(),
                "y": torch.cat(d["y"]).float()} for i, d in caps.items()}


def capture_activations(model, tok, layers, device) -> dict[int, dict]:
    """Activations over the calibration sentences -- the training set."""
    return _capture(model, layers,
                    [tok(s, return_tensors="pt").input_ids.to(device)
                     for s in SENTENCES])


def capture_holdout(model, layers, blocks, device, n_blocks=4) -> dict[int, dict]:
    """Activations over corpus blocks the training never sees.

    Needed because the in-sample residual is not evidence: ~44M trainable
    parameters against ~200 tokens can drive it to near zero by memorising the
    calibration set while the model gets *worse*. These blocks come from the
    front of the token stream and the evaluation window is the last 4 blocks, so
    the two are disjoint.
    """
    return _capture(model, layers, [b[0].to(device) for b in blocks[:n_blocks]])


def n_calibration_tokens(caps: dict[int, dict]) -> int:
    return int(next(iter(caps.values()))["x"].shape[0])


# ---------------------------------------------------------------------------
# the magnitude analysis
# ---------------------------------------------------------------------------

def magnitude_analysis(moe, x: torch.Tensor, ref: torch.Tensor,
                       selection: str) -> dict:
    """How much of the deficit is *magnitude* and how much is *direction*?

    A k-of-E partial sum can be corrected by any scalar. Writing
    `s* = <ref, kept> / ||kept||^2` (the per-token least-squares scale), the
    residual at `s*` is the best any magnitude correction can do, and it is
    recovered exactly when the deficit is purely one of scale.

    `s*` also identifies the regime. For experts whose contributions are
    orthogonal, `s* == 1` and rescaling *hurts*; for contributions that add
    coherently (each ~1/E of the output in the same direction), `s* == E/k`,
    which is exactly the literature's fix. A real partition sits somewhere
    between, and that is what makes this worth measuring rather than assuming.
    """
    eo = moe.expert_outputs(x).float()                 # (E, T, H)
    total = eo.sum(dim=0)                              # == the dense layer
    if selection == "oracle":
        k = moe.top_k
        keep = torch.topk(eo.norm(dim=2), k, dim=0).indices
        mask = torch.zeros(eo.shape[:2], dtype=torch.bool, device=eo.device)
        mask.scatter_(0, keep, True)
    else:
        scores = moe.router_scores(x)                  # (T, E)
        k = moe.top_k
        idx = torch.topk(scores, k, dim=1).indices     # (T, k)
        mask = torch.zeros(x.shape[0], moe.n_experts, dtype=torch.bool,
                           device=eo.device)
        mask.scatter_(1, idx, True)
        mask = mask.t()
    kept = (eo * mask.unsqueeze(-1)).sum(dim=0)        # (T, H)

    num = (ref * kept).sum(dim=1)
    den = kept.pow(2).sum(dim=1).clamp_min(1e-12)
    s_best = num / den                                 # (T,)

    def resid(scale):
        sc = scale if torch.is_tensor(scale) else torch.full_like(den, scale)
        err = (ref - sc.unsqueeze(1) * kept).norm(dim=1)
        return float((err / ref.norm(dim=1).clamp_min(1e-12)).mean())

    E_over_k = moe.n_experts / moe.top_k
    return {
        "selection": selection,
        "k": moe.top_k,
        "n_experts": moe.n_experts,
        "best_scale_mean": float(s_best.mean()),
        "best_scale_median": float(s_best.median()),
        "E_over_k": float(E_over_k),
        "residual_at_scale_1": resid(1.0),
        "residual_at_E_over_k": resid(E_over_k),
        "residual_at_best_scale": resid(s_best),
        # what a full identity split would give, for reference
        "residual_at_all_experts": float(
            ((ref - total).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)).mean()
        ),
    }


# ---------------------------------------------------------------------------
# the treatment
# ---------------------------------------------------------------------------

def frozen_scale_scan(model, caps, layers, n_experts, ks, plan,
                      originals) -> dict:
    """The magnitude analysis across k, on untrained weights -- no GPU training.

    This is the table that decides whether the field's output rescaling applies
    to a *partition*. `s*` is the least-squares optimal scale; if it stays near 1
    while E/k grows (8, 4, 2 for E=8), then the rescaling is correcting a scale
    error that does not exist here, and applying it can only hurt.
    """
    out = {}
    for k in ks:
        # every iteration starts from the dense MLPs again: upcycling a layer
        # that is already a MoEFFN raises, which is the intended guard
        for i, mlp in originals:
            setattr(model.model.layers[i], "mlp", mlp)
        moe = upcycle_model(model, MoEConfig(n_experts=n_experts, top_k=k,
                                             combine="sum", plan=plan,
                                             seed=0), layers)
        per_layer = {}
        with torch.no_grad():
            for i, m in moe:
                dt = next(m.parameters()).dtype
                x = caps[i]["x"].to(device=next(m.parameters()).device, dtype=dt)
                y = caps[i]["y"].to(device=x.device)
                per_layer[i] = {
                    "oracle": magnitude_analysis(m, x, y, "oracle"),
                    "router": magnitude_analysis(m, x, y, "router"),
                }
        out[k] = per_layer
    return out


def reconstruction_loss(moes, caps, dtype) -> torch.Tensor:
    """Mean relative-L2 reconstruction error over the upcycled layers.

    Relative rather than absolute so layers with different output magnitudes
    contribute comparably; float32 so the loss is not quantised by bf16.
    """
    total = None
    for i, m in moes:
        x = caps[i]["x"].to(device=next(m.parameters()).device, dtype=dtype)
        y = caps[i]["y"].to(device=x.device)
        out = m(x).float()
        err = (out - y).norm() / y.norm().clamp_min(1e-12)
        total = err if total is None else total + err
    return total / max(len(moes), 1)


def self_teach(model, caps, holdout, config, layers, args, device, dtype) -> dict:
    """Upcycle `layers`, then train on nothing but the reconstruction loss."""
    moes = upcycle_model(model, config, layers)
    trainable = [p for _, m in moes for p in list(m.parameters())]
    n_train = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)

    @torch.no_grad()
    def evaluate_moes(tag, source=None):
        """Per-layer residuals at the current weights, both selections.

        `source` defaults to the calibration set; pass the held-out set to get
        the only number that says whether anything was *learned*.
        """
        src = caps if source is None else source
        out = {}
        for i, m in moes:
            # both the device AND the dtype: the captured activations are float32
            # while the module's weights are bf16, and F.linear rejects the mix
            dt = next(m.parameters()).dtype
            x = src[i]["x"].to(device=device, dtype=dt)
            y = src[i]["y"].to(device=device)
            out[i] = {
                "oracle": magnitude_analysis(m, x, y, "oracle"),
                "router": magnitude_analysis(m, x, y, "router"),
                "vs_original_dense": reconstruction_residual(y, m(x)),
            }
        return out

    frozen = evaluate_moes("frozen")
    frozen_hold = evaluate_moes("frozen", holdout)
    model.train()
    curve = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = reconstruction_loss(moes, caps, dtype)
        aux = collect_total_aux_loss(model)
        total = loss + aux
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            curve.append({"step": step, "recon": float(loss.item()),
                          "aux": float(aux.item()),
                          "elapsed": time.perf_counter() - t0})
    trained_s = time.perf_counter() - t0
    model.eval()
    trained = evaluate_moes("trained")
    trained_hold = evaluate_moes("trained", holdout)

    hold_frozen = {i: float(frozen_hold[i]["vs_original_dense"]) for i, _ in moes}
    hold_trained = {i: float(trained_hold[i]["vs_original_dense"]) for i, _ in moes}

    by_layer = {i: float(trained[i]["vs_original_dense"]) for i, _ in moes}
    frozen_by_layer = {i: float(frozen[i]["vs_original_dense"]) for i, _ in moes}
    return {
        "config": {"n_experts": config.n_experts, "top_k": config.top_k,
                   "combine": config.combine, "plan": config.plan,
                   "aux_loss_coef": config.aux_loss_coef, "seed": config.seed},
        "layers": [i for i, _ in moes],
        "trainable_params": n_train,
        "steps": args.steps,
        "lr": args.lr,
        "seconds": trained_s,
        "loss_curve": curve,
        "vs_original_dense_frozen": frozen_by_layer,
        "vs_original_dense_trained": by_layer,
        "vs_original_dense_frozen_mean": sum(frozen_by_layer.values())
        / len(frozen_by_layer),
        "vs_original_dense_trained_mean": sum(by_layer.values()) / len(by_layer),
        # the held-out pair: `frozen` vs `trained` here is the difference between
        # memorising 197 tokens and actually learning the layer's function
        "vs_original_dense_holdout_frozen": hold_frozen,
        "vs_original_dense_holdout_trained": hold_trained,
        "vs_original_dense_holdout_frozen_mean": sum(hold_frozen.values())
        / len(hold_frozen),
        "vs_original_dense_holdout_trained_mean": sum(hold_trained.values())
        / len(hold_trained),
        "magnitude_frozen": frozen,
        "magnitude_trained": trained,
        "magnitude_holdout_trained": trained_hold,
        "_moes": moes,
        "_opt": opt,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layers", nargs="+", default=["0", "3", "8", "23"])
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--plan", default="random")
    ap.add_argument("--arms", nargs="+", default=["sum", "sum_rescaled", "softmax"])
    ap.add_argument("--steps", type=int, default=300)
    # 1e-4, not 1e-3: the relative-L2 reconstruction loss is scale-sensitive and
    # these layers hold ~44M trainable parameters against a few hundred tokens.
    # At 1e-3 the first attempt diverged (loss 0.92 -> 9.01, one layer's residual
    # to 28.8) -- which measures the learning rate, not the method.
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scan-topks", type=int, nargs="+", default=[1, 2, 4],
                    help="k values for the frozen magnitude scan")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-blocks", type=int, default=4)
    ap.add_argument("--corpus-chars", type=int, default=4_000_000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--holdout-blocks", type=int, default=4,
                    help="corpus blocks used only to measure held-out "
                         "reconstruction; never trained on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=os.path.join(".cache", "moeme",
                                                       "selfteach_ckpt"))
    ap.add_argument("--out", default="results/self_teach_08b.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = poc.pick_amp_dtype(device)
    torch.manual_seed(args.seed)
    layers = [int(v) for v in args.layers]
    print(f"device={device} amp={amp_dtype} model={args.model} layers={layers}")

    model, tok = cmp08b.load_model(args.model, device, amp_dtype)
    print(f"params {sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    # ---- calibration activations: the only "data", and it is 24 sentences
    caps = capture_activations(model, tok, layers, device)
    n_cal = n_calibration_tokens(caps)
    print(f"calibration: {len(SENTENCES)} sentences, {n_cal} tokens "
          f"(teacher = the model's own dense layers 0/3/8/23)")

    # ---- the evaluation side, identical to the other runs so numbers compare
    corp = SimpleNamespace(text_file=None, offline=False, dataset=args.dataset,
                           corpus_chars=args.corpus_chars)
    ids, corpus_source = poc.load_token_stream(corp, tok)
    blocks = poc.make_blocks(ids, args.seq_len)
    n_val = max(1, len(blocks) // 20)
    val_blocks = blocks[-n_val:]
    model.eval()
    # Held-out activations for the residual: blocks from the front of the stream,
    # disjoint from both the calibration sentences and the evaluation window (the
    # last n_val blocks). Without these the reconstruction number is in-sample.
    holdout = capture_holdout(model, layers, blocks, device,
                              n_blocks=args.holdout_blocks)
    holdout_tokens = n_calibration_tokens(holdout)
    print(f"held-out activations: {args.holdout_blocks} blocks, "
          f"{holdout_tokens} tokens (disjoint from the eval window)")
    dense_ppl = poc.eval_perplexity(model, val_blocks, device, amp_dtype,
                                    args.eval_blocks)
    prompt_ids = [tok(p, return_tensors="pt").input_ids.to(device)
                  for p in cmp08b.PROMPTS]
    dense_out = {p: cmp08b.greedy_ids(model, pi, args.new_tokens)
                 for p, pi in zip(cmp08b.PROMPTS, prompt_ids)}
    print(f"dense val ppl {dense_ppl:.4f} (same {args.eval_blocks}-block window "
          f"as E14/E20)")

    report = {
        "model": args.model,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "corpus": corpus_source,
        "corpus_tokens": int(ids.numel()),
        "eval_blocks": args.eval_blocks,
        "seq_len": args.seq_len,
        "dense_val_ppl": dense_ppl,
        "calibration_sentences": SENTENCES,
        "calibration_tokens": n_cal,
        # the held-out column is quoted against the in-sample one everywhere, so
        # what it was measured on has to be in the artifact: a residual without
        # its window is the defect this repo already had to fix once
        "holdout_blocks": args.holdout_blocks,
        "holdout_tokens": holdout_tokens,
        "calibration_note": "hand-written, outside the evaluation window; the "
                            "teacher is the model's own dense FFN, so no labels "
                            "and no pretraining corpus are used",
        "new_tokens": args.new_tokens,
        "arms": [],
    }

    # The dense MLPs, kept so every arm starts from the *same* weights and
    # differs only in how the selected experts are combined. Without this the
    # second arm would inherit the first arm's trained weights and the arms
    # would not be comparable at all.
    originals = [(i, model.model.layers[i].mlp) for i in layers]

    # ---- the frozen magnitude scan, before any training touches the weights
    scan = frozen_scale_scan(model, caps, layers, args.experts,
                             args.scan_topks, args.plan, originals)
    for i, mlp in originals:
        setattr(model.model.layers[i], "mlp", mlp)
    report["frozen_scale_scan"] = scan
    print(f"\nfrozen magnitude scan (E={args.experts}, oracle selection, "
          f"mean over layers {layers}):")
    print(f"  {'k':>2} {'E/k':>6} {'s* oracle':>10} {'s* router':>10} "
          f"{'resid@1':>9} {'resid@s*':>9}")
    for k in args.scan_topks:
        o = [scan[k][i]["oracle"] for i in layers]
        r = [scan[k][i]["router"] for i in layers]
        mean = lambda vs, f: sum(v[f] for v in vs) / len(vs)  # noqa: E731
        print(f"  {k:>2} {args.experts/k:>6.1f} {mean(o,'best_scale_mean'):>10.3f} "
              f"{mean(r,'best_scale_mean'):>10.3f} "
              f"{mean(o,'residual_at_scale_1'):>9.4f} "
              f"{mean(o,'residual_at_best_scale'):>9.4f}")

    failed = False
    for combine in args.arms:
        print(f"\n=== arm combine={combine} (E={args.experts} k={args.topk}) ===",
              flush=True)
        for i, mlp in originals:
            setattr(model.model.layers[i], "mlp", mlp)
        try:
            res = None
            config = MoEConfig(n_experts=args.experts, top_k=args.topk,
                               combine=combine, plan=args.plan,
                               aux_loss_coef=1e-2, seed=0)
            res = self_teach(model, caps, holdout, config, layers, args, device,
                             amp_dtype)
            moes, opt = res.pop("_moes"), res.pop("_opt")

            # perplexity and text: the axes E16 showed can move *against* the
            # residual, so a residual win alone would not be a result
            res["val_ppl"] = poc.eval_perplexity(model, val_blocks, device,
                                                 amp_dtype, args.eval_blocks)
            res["ppl_ratio_vs_dense"] = res["val_ppl"] / dense_ppl
            outs = {p: cmp08b.greedy_ids(model, pi, args.new_tokens)
                    for p, pi in zip(cmp08b.PROMPTS, prompt_ids)}
            res["prompts_identical"] = sum(1 for p in cmp08b.PROMPTS
                                           if outs[p] == dense_out[p])
            res["prompts"] = [{
                "prompt": p,
                "identical": outs[p] == dense_out[p],
                "first_divergence_index": cmp08b.first_divergence(dense_out[p],
                                                                 outs[p]),
                "token_match_fraction": cmp08b.match_fraction(dense_out[p],
                                                              outs[p]),
                "dense_text": tok.decode(dense_out[p], skip_special_tokens=True),
                "selfteach_text": tok.decode(outs[p], skip_special_tokens=True),
            } for p in cmp08b.PROMPTS]

            ckpt = os.path.join(args.ckpt_dir,
                                f"selfteach_E{args.experts}_k{args.topk}_"
                                f"{combine}_{args.plan}_s0_"
                                f"{'-'.join(str(i) for i in layers)}.pt")
            poc._save(ckpt, moes, opt, args.steps)
            res["checkpoint"] = ckpt
            print(f"  vs_original_dense  in-sample {res['vs_original_dense_frozen_mean']:.4f}"
                  f" -> {res['vs_original_dense_trained_mean']:.4f}")
            print(f"  vs_original_dense  HELD-OUT  "
                  f"{res['vs_original_dense_holdout_frozen_mean']:.4f}"
                  f" -> {res['vs_original_dense_holdout_trained_mean']:.4f}")
            print(f"  magnitude ceiling: best scale (oracle) "
                  f"{res['magnitude_trained'][layers[0]]['oracle']['best_scale_mean']:.3f}"
                  f" vs E/k {args.experts/args.topk:.1f}; residual at best scale "
                  f"{res['magnitude_trained'][layers[0]]['oracle']['residual_at_best_scale']:.4f}"
                  f" vs at scale 1 {res['magnitude_trained'][layers[0]]['oracle']['residual_at_scale_1']:.4f}")
            print(f"  val ppl {res['val_ppl']:.4f} ({res['ppl_ratio_vs_dense']:.2f}x dense)"
                  f"  |  {res['prompts_identical']}/{len(cmp08b.PROMPTS)} continuations"
                  f" identical to dense")
            print(f"  wrote {ckpt}")
        except Exception as e:                                   # noqa: BLE001
            import traceback
            failed = True
            traceback.print_exc()
            res = {"config": {"combine": combine}, "error": f"{type(e).__name__}: {e}"}
        report["arms"].append(res)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)

    print(f"\nwrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
