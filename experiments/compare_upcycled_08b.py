#!/usr/bin/env python3
"""
compare_upcycled_08b.py -- does the upcycled 0.8B still behave like the dense 0.8B?

The upcycling PoC reports perplexity and weight-space residuals. Neither answers
the question a reader actually asks: if the dense FFN is replaced by this, does
the model still answer the same? E18 answered that for the pretrained 35B-A3B by
comparing generated text (`docs/quality_vs_speed.md`); nothing has done it for
the upcycled 0.8B, whose only recorded evidence is a perplexity ratio.

So this loads the dense Qwen3.5-0.8B, upcycles the same layers with the same
configuration the checkpoint was trained under, loads the trained experts, and
compares on *identical* tokens:

  1. perplexity, both models, same val window. The dense figure reproduces the
     recorded 19.5707, which is what makes every other number comparable.
  2. greedy continuations on a fixed prompt set: exact match, first divergent
     token, and the fraction of generated tokens that agree. A model that
     "works like the default" continues the same text; one that does not
     diverges, and *where* it diverges says how much of the network moved.
  3. a dense-limit control (`top_k = n_experts`) measured on the *fresh*
     partition, before the trained experts are loaded. A fresh partition with
     every expert active is the dense layer exactly, so this must return the
     dense perplexity; if it does not, the comparison above is measuring a bug.
     The same figure on the *trained* partition is reported separately and is
     not a control -- it is ~80x dense, which is the clearest statement of what
     training did to the experts (they no longer sum to the function they were
     sliced from).

Usage:
    python3 experiments/compare_upcycled_08b.py
    python3 experiments/compare_upcycled_08b.py --ckpt .cache/moeme/poc_ckpt/E8_k1_sum_random_s0_0-3-8-23.pt
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

import upcycle_poc as poc  # noqa: E402  reuse the driver's corpus/ppl plumbing
from moeme.upcycle import MoEConfig, upcycle_model  # noqa: E402

DEFAULT_CKPT = os.path.join(".cache", "moeme", "poc_ckpt",
                            "E8_k1_sum_random_s0_0-3-8-23.pt")

# Fixed prompts, not sampled: the point is comparability between the two models
# and between runs, so a prompt set that moves cannot be compared to anything.
PROMPTS = [
    "The capital of France is",
    "Q: What is 17 * 23?\nA:",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a small village,",
    "The three largest planets in the solar system, in order, are",
    "A recipe for bread needs flour, water, yeast and",
]


# ---------------------------------------------------------------------------
# pure comparison helpers (tested offline in tests/test_compare_upcycled_08b.py)
# ---------------------------------------------------------------------------

def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first token the two sequences disagree on, else None.

    Returns None only when one is a prefix of the other (or they are equal):
    a shorter sequence that agrees everywhere it exists has not diverged yet,
    and reporting the end of it as a divergence would overstate the difference.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def match_fraction(a: list[int], b: list[int]) -> float:
    """Fraction of positions (over the shorter length) where tokens agree.

    Empty input is 1.0: two empty sequences are identical, and returning 0.0
    would make a model that generated nothing look maximally different.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n


def summarise_prompt(prompt: str, dense_ids: list[int], moe_ids: list[int],
                     text: dict[str, str]) -> dict:
    d = first_divergence(dense_ids, moe_ids)
    return {
        "prompt": prompt,
        "identical": dense_ids == moe_ids,
        "first_divergence_index": d,
        "tokens_compared": min(len(dense_ids), len(moe_ids)),
        "token_match_fraction": match_fraction(dense_ids, moe_ids),
        "dense_text": text["dense"],
        "moe_text": text["moe"],
    }


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_ids(model, ids: torch.Tensor, n_new: int) -> list[int]:
    """Greedy decode n_new tokens. Manual rather than .generate(): no sampling
    config, no stopping criteria, and identical code for both models."""
    out = ids
    for _ in range(n_new):
        logits = model(out).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    return out[0, ids.shape[1]:].tolist()


def load_model(name: str, device, amp_dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    dtype = amp_dtype or torch.float32
    last = None
    for kw in ({"dtype": dtype}, {"torch_dtype": dtype}):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, low_cpu_mem_usage=True, **kw)
            break
        except TypeError as e:
            last = e
    else:
        raise last
    return model.to(device), tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--layers", nargs="+", default=["0", "3", "8", "23"])
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--combine", default="sum")
    ap.add_argument("--plan", default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-blocks", type=int, default=4)
    ap.add_argument("--corpus-chars", type=int, default=4_000_000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--out", default="results/upcycled_08b_vs_dense.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = poc.pick_amp_dtype(device)
    torch.manual_seed(args.seed)
    print(f"device={device} amp={amp_dtype} model={args.model}")

    model, tok = load_model(args.model, device, amp_dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params/1e9:.3f}B")

    # ---- corpus: same streamer and flags the recorded run used, so the dense
    # perplexity below is directly comparable to results/upcycle_poc_rerun.json
    corp = SimpleNamespace(text_file=None, offline=args.offline,
                           dataset=args.dataset, corpus_chars=args.corpus_chars)
    ids, corpus_source = poc.load_token_stream(corp, tok)
    blocks = poc.make_blocks(ids, args.seq_len)
    if len(blocks) < 4:
        raise SystemExit("corpus too small for a train/val split")
    n_val = max(1, len(blocks) // 20)
    val_blocks = blocks[-n_val:]
    print(f"blocks: {len(blocks) - n_val} train, {n_val} val")

    # ---- dense baseline
    model.eval()
    t0 = time.perf_counter()
    dense_ppl = poc.eval_perplexity(model, val_blocks, device, amp_dtype,
                                    args.eval_blocks)
    print(f"dense val ppl {dense_ppl:.4f}  ({time.perf_counter() - t0:.1f}s)")

    # The dense FFN's own (input, output) on a held-out block, captured before
    # upcycling. This is the reference the driver's per-layer residuals are
    # defined against, and reproducing its recorded values is how the checkpoint
    # is confirmed to be the weights the report quotes rather than some other
    # run of the same tag.
    reference = poc.capture_reference(model, val_blocks,
                                      [int(v) for v in args.layers], device)

    prompt_ids = [tok(p, return_tensors="pt").input_ids.to(device) for p in PROMPTS]
    dense_out = {}
    for p, pi in zip(PROMPTS, prompt_ids):
        dense_out[p] = greedy_ids(model, pi, args.new_tokens)
    print(f"dense: generated {args.new_tokens} tokens on {len(PROMPTS)} prompts")

    # ---- upcycle the same layers with the config the checkpoint carries
    layers = [int(v) for v in args.layers]
    config = MoEConfig(n_experts=args.experts, top_k=args.topk,
                       combine=args.combine, plan=args.plan,
                       aux_loss_coef=1e-2, seed=args.seed)
    moes = upcycle_model(model, config, layers)

    def set_topk(k):
        """Context manager-ish: set top_k on every MoE, return what to restore."""
        saved = []
        for _, m in moes:
            saved.append((m, m.config))
            m.config = m.config.with_(top_k=k)
        return saved

    def restore(saved):
        for m, cfg in saved:
            m.config = cfg

    # ---- the dense-limit control, measured on the FRESH partition, before the
    # trained experts are loaded. This ordering is the whole point and getting it
    # backwards is a trap: a fresh partition with every expert active *is* the
    # dense layer (the experts are disjoint slices of it), so this must return
    # the dense perplexity. Measure it after loading trained weights and it
    # returns something enormous -- not because the plumbing broke, but because
    # training has rescaled the experts away from the slices they were cut from.
    # Both numbers are informative; only the fresh one is a correctness check.
    model.eval()
    fresh_ppl = poc.eval_perplexity(model, val_blocks, device, amp_dtype,
                                    args.eval_blocks)
    saved = set_topk(args.experts)
    limit_ppl = poc.eval_perplexity(model, val_blocks, device, amp_dtype,
                                    args.eval_blocks)
    limit_out = {p: greedy_ids(model, pi, args.new_tokens)
                 for p, pi in zip(PROMPTS, prompt_ids)}
    restore(saved)
    limit_rel = abs(limit_ppl - dense_ppl) / dense_ppl
    print(f"fresh (post-hoc, k={args.topk}) ppl {fresh_ppl:.4f}")
    print(f"dense-limit ppl {limit_ppl:.4f}  (dense {dense_ppl:.4f}, "
          f"rel {limit_rel:.2e})")
    if limit_rel > 1e-3:
        print("  WARNING: the dense-limit control is off by more than 0.1%. A fresh "
              "partition with all experts active is the dense layer exactly, so "
              "this means the partition plumbing is wrong -- read no other "
              "number below until it is fixed.")

    # ---- load the trained experts over the fresh initialisation
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing = [i for i, _ in moes if i not in state["moes"]]
    if missing:
        raise SystemExit(f"checkpoint {args.ckpt} has no experts for layers "
                         f"{missing}; it holds {sorted(state['moes'])}")
    for i, m in moes:
        m.load_state_dict(state["moes"][i])
    print(f"loaded {args.ckpt} (step {state['step']}) into layers {layers} "
          f"at E={args.experts} k={args.topk}")

    model.eval()
    moe_ppl = poc.eval_perplexity(model, val_blocks, device, amp_dtype,
                                  args.eval_blocks)
    moe_out = {p: greedy_ids(model, pi, args.new_tokens)
               for p, pi in zip(PROMPTS, prompt_ids)}

    # Per-layer residuals on the LOADED weights, same definitions the driver
    # records. `vs_original_dense` measures against the dense FFN captured above;
    # `oracle_topk`/`router_topk` measure against this module's own all-experts
    # sum. If these reproduce the recorded run's numbers, the checkpoint is that
    # run's weights and any residual perplexity difference is evaluation
    # arithmetic rather than a different model.
    trained_m = poc.evaluate_layers(moes, reference, device,
                                    next(model.parameters()).dtype, args.topk)
    trained_summary = poc.summarize_layers(trained_m)
    print(f"per-layer residuals on the loaded weights: "
          f"vs_original_dense {trained_summary['vs_original_dense_mean']:.6f} | "
          f"oracle {trained_summary['oracle_topk_mean']:.6f} | "
          f"router {trained_summary['router_topk_mean']:.6f}")

    # Cross-check against the run that produced the checkpoint, when its report
    # is present. A mismatch here is worth knowing before quoting any of the
    # numbers above; it is recorded rather than raised, because the comparison
    # against dense is valid either way.
    recorded = None
    rec_path = os.path.join("results", "upcycle_poc_rerun.json")
    if os.path.exists(rec_path):
        with open(rec_path) as fh:
            rec = json.load(fh)
        tag = lambda c: (c["n_experts"], c["top_k"], c["combine"], c["plan"])  # noqa: E731
        want = (args.experts, args.topk, args.combine, args.plan)
        for r in rec.get("runs", []):
            if tag(r["config"]) != want or sorted(r.get("layers", [])) != sorted(layers):
                continue
            if r.get("steps_completed") != state["step"]:
                continue
            recorded = {
                "source": rec_path,
                "trained_val_ppl": r.get("trained_val_ppl"),
                "metrics_trained": r.get("metrics_trained"),
                "steps": r.get("steps_completed"),
            }
            break
    if recorded:
        for k, v in trained_summary.items():
            ref = recorded["metrics_trained"].get(k)
            if ref is None:
                continue
            print(f"  {k:24s} loaded {v:.6f}  recorded {ref:.6f}  "
                  f"diff {abs(v - ref):.2e}")
        print(f"  perplexity               loaded {moe_ppl:.4f}  "
              f"recorded {recorded['trained_val_ppl']:.4f}  "
              f"diff {abs(moe_ppl - recorded['trained_val_ppl']):.4f}")

    # The TRAINED model with every expert active. Not a control -- there is no
    # reason it should equal dense, and it does not, by ~80x. It is the sharpest
    # single statement of what training did: it re-purposed the experts away from
    # the dense function they were partitioned from, so the whole partition
    # (all experts, nothing sparsified) is *far* worse than dense while the
    # sparse top-k version is only 1.5x worse. That is a model whose FFN has
    # been reshaped, not one with a cheaper approximation of the same function.
    saved = set_topk(args.experts)
    trained_all_ppl = poc.eval_perplexity(model, val_blocks, device, amp_dtype,
                                          args.eval_blocks)
    restore(saved)
    print(f"trained all-experts ppl {trained_all_ppl:.4f}  "
          f"({trained_all_ppl/dense_ppl:.0f}x dense)")

    # ---- report
    rows = []
    for p in PROMPTS:
        text = {
            "dense": tok.decode(dense_out[p], skip_special_tokens=True),
            "moe": tok.decode(moe_out[p], skip_special_tokens=True),
        }
        rows.append(summarise_prompt(p, dense_out[p], moe_out[p], text))
    n_identical = sum(r["identical"] for r in rows)
    mean_match = sum(r["token_match_fraction"] for r in rows) / len(rows)
    limit_identical = sum(1 for p in PROMPTS if limit_out[p] == dense_out[p])

    report = {
        "model": args.model,
        "checkpoint": args.ckpt,
        "checkpoint_step": state["step"],
        "layers_upcycled": layers,
        "n_layers_total": len(model.model.layers),
        "config": {"n_experts": args.experts, "top_k": args.topk,
                   "combine": args.combine, "plan": args.plan, "seed": args.seed},
        "corpus": corpus_source,
        "corpus_tokens": int(ids.numel()),
        "eval_blocks": args.eval_blocks,
        "seq_len": args.seq_len,
        "new_tokens": args.new_tokens,
        "dense_val_ppl": dense_ppl,
        "fresh_val_ppl": fresh_ppl,
        "moe_val_ppl": moe_ppl,
        "ppl_ratio_moe_over_dense": moe_ppl / dense_ppl,
        "dense_limit_val_ppl": limit_ppl,
        "dense_limit_rel_error": limit_rel,
        "trained_all_experts_val_ppl": trained_all_ppl,
        "metrics_trained": trained_summary,
        "metrics_trained_per_layer": {str(k): v for k, v in trained_m.items()},
        "recorded_run": recorded,
        "prompts": rows,
        "prompts_identical": n_identical,
        "prompts_total": len(rows),
        "mean_token_match_fraction": mean_match,
        "dense_limit_prompts_identical": limit_identical,
        "n_params": n_params,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n=== dense vs upcycled 0.8B "
          f"({len(layers)}/{report['n_layers_total']} layers converted) ===")
    print(f"perplexity     dense {dense_ppl:.4f} | fresh {fresh_ppl:.4f} | "
          f"trained {moe_ppl:.4f}  ({moe_ppl/dense_ppl:.2f}x dense)")
    print(f"dense-limit    {limit_ppl:.4f} on the fresh partition "
          f"(control: must match dense; rel err {limit_rel:.2e})")
    print(f"               {trained_all_ppl:.4f} on the trained one "
          f"({trained_all_ppl/dense_ppl:.0f}x dense -- not a control, see docstring)")
    print(f"continuations  {n_identical}/{len(rows)} byte-identical, "
          f"mean token agreement {mean_match:.3f}")
    for r in rows:
        mark = "same" if r["identical"] else f"diverges at token {r['first_divergence_index']}"
        print(f"\n  [{mark}] {r['prompt']!r}")
        print(f"    dense: {r['dense_text']!r}")
        if not r["identical"]:
            print(f"    moe  : {r['moe_text']!r}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
