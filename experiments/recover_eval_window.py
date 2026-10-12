#!/usr/bin/env python3
"""
recover_eval_window.py -- recover the `eval_blocks` a report did not record.

The problem
-----------
`experiments/upcycle_poc.py` scores perplexity over `val_blocks[:k]` with
`k = --eval-blocks`. On this model and this corpus that number is **19.57 at k=4
and 23.42 at k=8**, so a report that omits `k` cannot be reproduced -- and E14's
report predates the field, which is why its 21.736 could not be matched against
E15's 19.562. That gap started this tool.

Why it is recoverable
---------------------
Perplexity over the first k val blocks is a *running* statistic:

    ppl(k) = exp( sum_{i<=k} loss_i * n_i  /  sum_{i<=k} n_i )

so it is a deterministic function of k alone, given the corpus. Measure the
per-block losses once, and every perplexity that function can produce becomes
computable -- including the ones produced before anyone thought to write k down.
A reported perplexity is therefore an invertible fingerprint of the window that
produced it.

What this does
--------------
1. rebuilds the same corpus, the same val split and the same per-block losses the
   driver uses (reusing its functions, so there is no second implementation to
   drift);
2. builds `ppl(k)` for every k the driver could have used;
3. for each `--target` perplexity, reports the k that reproduces it, and the
   residual -- so a target that no window can produce is *disproved* rather than
   left ambiguous. That negative case is the point: it distinguishes "the window
   differed" from "the data or the code differed".

A caution that this tool exists to end: a target that matches nothing within
tolerance means the difference was NOT the window, and no amount of curve-fitting
should be read as a match. The residual column is printed for every k considered.

Usage
-----
    python3 experiments/recover_eval_window.py \
        --target 21.7359 --target 21.7396 \
        --out results/eval_window_recovery.json

`--self-test` checks the curve inversion offline, with no model and no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import upcycle_poc as poc  # noqa: E402


# ---------------------------------------------------------------------------
# the arithmetic (offline, testable)
# ---------------------------------------------------------------------------

def ppl_curve(losses, weights) -> list[float]:
    """Running perplexity over the first k blocks, exactly as `eval_perplexity`.

    `eval_perplexity` accumulates `out.loss.item() * y.numel()` and divides by the
    token count, so this must weight by tokens and not simply average the losses --
    the two agree only while every block is full, and the last val block is
    allowed to be short.
    """
    out: list[float] = []
    total, ntok = 0.0, 0
    for loss, n in zip(losses, weights):
        total += loss * n
        ntok += n
        out.append(math.exp(total / ntok))
    return out


def match_window(curve, target, rel_tol: float = 1e-4) -> list[int]:
    """Every k (1-based) whose running perplexity reproduces `target`.

    Returned as a list, not a single value: the curve is not monotonic, so two
    different windows can land on the same perplexity and the honest answer is
    then "one of these", not a guess.
    """
    return [k for k, v in enumerate(curve, start=1)
            if abs(v - target) <= rel_tol * target]


def nearest_window(curve, target) -> tuple[int, float]:
    """The closest k and its *relative* residual, for reporting a near miss."""
    best_k, best_rel = None, float("inf")
    for k, v in enumerate(curve, start=1):
        rel = abs(v - target) / target
        if rel < best_rel:
            best_k, best_rel = k, rel
    return best_k, best_rel


def verdict(curve, target, rel_tol: float = 1e-4,
            near_tol: float = 1e-3) -> tuple[str, list[int], int, float]:
    """Recovered / near / none -- and never a bare guess.

    Three outcomes, because the middle one is real and mistaking it for either
    neighbour misleads in opposite directions:

    * `recovered` -- a window reproduces the target to within `rel_tol`. The curve
      is reproducible to ~1e-7 on one machine, so this is an identification.
    * `near` -- the closest window is within `near_tol` but not `rel_tol`. This is
      what a target from an *equivalent but differently rounded* path looks like:
      the driver's `dense_limit_val_ppl` re-runs the same evaluation through the
      sum-of-all-experts MoE, which reproduces the dense FFN only to ~2e-4, so its
      perplexity lands a hair off the dense curve. Reporting that as "no window"
      would refute a window hypothesis on evidence that actually supports it.
    * `none` -- outside `near_tol`. The window did not produce this number.
    """
    hits = match_window(curve, target, rel_tol)
    if hits:
        k, rel = nearest_window(curve, target)
        return "recovered", hits, k, rel
    k, rel = nearest_window(curve, target)
    return ("near" if rel <= near_tol else "none"), [], k, rel


def self_test() -> int:
    """Offline: the curve and the inversion, no model and no GPU."""
    # Equal weights: ppl(k) is exp(mean of the first k losses), so it is analytic.
    losses = [math.log(4.0), math.log(8.0), math.log(16.0), math.log(32.0)]
    curve = ppl_curve(losses, [1, 1, 1, 1])
    expected = [4.0, math.sqrt(4 * 8), (4 * 8 * 16) ** (1 / 3), (4 * 8 * 16 * 32) ** 0.25]
    assert all(abs(a - b) < 1e-9 for a, b in zip(curve, expected)), (curve, expected)

    # A perplexity in the curve is recovered to its own k...
    for k in (1, 2, 3, 4):
        assert match_window(curve, expected[k - 1]) == [k], (k, curve)
        assert verdict(curve, expected[k - 1])[0] == "recovered", k
    # ...a target that is not on the curve is never reported as a match...
    assert match_window(curve, 5.0) == []
    assert verdict(curve, 5.0)[0] in ("near", "none")
    # ...and a target that is not the window at all is explicitly `none`, not a
    # near miss. This is the guard against fitting a window to a non-window effect.
    assert verdict(curve, 500.0)[0] == "none", verdict(curve, 500.0)
    # The middle branch is load-bearing: the driver's dense_limit check is the same
    # evaluation through the sum-of-all-experts path, which reproduces the dense
    # number only to ~2e-4, so its perplexity must read as `near` -- pointing at
    # the window -- and not as `none`, which would refute the right answer.
    path_rounded = expected[-1] * (1 + 2e-4)
    assert verdict(curve, path_rounded)[0] == "near", verdict(curve, path_rounded)
    assert verdict(curve, path_rounded)[1] == []

    # Non-monotonic curves can alias: losses (9, 1, 9, 1) give running perplexities
    # (9, 3, 81**1/3, 3), so the target 3.0 has two valid windows and the honest
    # answer is both rather than the first.
    aliased = ppl_curve([math.log(9), math.log(1), math.log(9), math.log(1)], [1] * 4)
    assert match_window(aliased, 3.0) == [2, 4], (aliased, match_window(aliased, 3.0))

    # Weighting is by tokens, so a short final block must not be treated as a full
    # one -- v1 of the curve divided by block count and would drift here.
    weighted = ppl_curve([math.log(100.0), math.log(2.0)], [100, 1])
    assert abs(weighted[-1] - math.exp((100 * math.log(100) + 1 * math.log(2)) / 101)) < 1e-9
    assert abs(weighted[-1] - math.exp((math.log(100) + math.log(2)) / 2)) > 1.0
    print("recover_eval_window self-test: ok")
    return 0


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def measure_block_losses(model, blocks, device, amp_dtype) -> list[tuple[float, int]]:
    """Per-block (mean NLL, tokens), computed the way `eval_perplexity` does."""
    import torch

    model.eval()
    out: list[tuple[float, int]] = []
    with torch.no_grad():
        for x, y in blocks:
            x, y = x.to(device), y.to(device)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    o = model(x, labels=y)
            else:
                o = model(x, labels=y)
            out.append((o.loss.item(), y.numel()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--corpus-chars", type=int, default=4_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--target", action="append", type=float, default=[],
                    metavar="PPL", help="a perplexity whose window you want back")
    ap.add_argument("--out", default="results/eval_window_recovery.json")
    ap.add_argument("--rel-tol", type=float, default=1e-4,
                    help="relative tolerance for calling a window a match; the "
                         "curve is reproducible to ~1e-7 on one machine, so a "
                         "loose tolerance is what would manufacture a false match")
    ap.add_argument("--near-tol", type=float, default=1e-3,
                    help="a closest-window residual inside this reads as `near`: "
                         "the window is supported, the target was rounded by an "
                         "equivalent path. Outside it, the window is excluded.")
    ap.add_argument("--curve-from", default=None, metavar="REPORT.json",
                    help="re-decide targets against a previously measured curve "
                         "instead of re-measuring it on a GPU")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.target:
        raise SystemExit("give at least one --target PPL")

    if args.curve_from:
        return redecide(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = poc.pick_amp_dtype(device)
    print(f"device={device} amp={amp_dtype} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = amp_dtype or torch.float32
    last = None
    for kw in ({"dtype": dtype}, {"torch_dtype": dtype}):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, low_cpu_mem_usage=True, **kw)
            break
        except TypeError as e:
            last = e
    else:
        raise last
    model = model.to(device)

    corpus_args = SimpleNamespace(text_file=None, dataset=args.dataset,
                                 corpus_chars=args.corpus_chars, offline=False)
    ids, source = poc.load_token_stream(corpus_args, tok)
    blocks = poc.make_blocks(ids, args.seq_len)
    n_val = max(1, len(blocks) // 20)
    val_blocks = blocks[-n_val:]
    print(f"corpus {source}, {ids.numel():,} tokens; val blocks {len(val_blocks)}")

    measured = measure_block_losses(model, val_blocks, device, amp_dtype)
    losses = [l for l, _ in measured]
    weights = [n for _, n in measured]
    curve = ppl_curve(losses, weights)

    # The curve must reproduce the windows that ARE on record, or it is measuring a
    # different quantity and every match below would be meaningless.
    checks = {4: 19.5707, 8: 23.4212}
    print()
    for k, recorded in checks.items():
        got = curve[k - 1] if k <= len(curve) else float("nan")
        flag = "OK" if abs(got - recorded) < 1e-3 else "MISMATCH"
        print(f"  calibration ppl({k}) = {got:.4f}  vs recorded {recorded}  [{flag}]")
    calibrated = all(abs((curve[k - 1] if k <= len(curve) else float("nan")) - v) < 1e-3
                     for k, v in checks.items())

    print(f"\n  ppl(k) over k=1..{len(curve)}: min {min(curve):.4f} "
          f"max {max(curve):.4f}")

    report = {
        "model": args.model, "device": str(device), "amp_dtype": str(amp_dtype),
        "corpus": source, "corpus_tokens": int(ids.numel()),
        "seq_len": args.seq_len, "n_val_blocks": len(val_blocks),
        "per_block_loss": losses, "per_block_tokens": weights, "ppl_curve": curve,
        "calibration": {str(k): {"measured": curve[k - 1], "recorded": v}
                        for k, v in checks.items()},
        "calibrated": calibrated,
        "targets": {},
    }
    print()
    for target in args.target:
        report["targets"][repr(target)] = describe(curve, target, args.rel_tol,
                                                   args.near_tol)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out}")
    return 0


def describe(curve, target, rel_tol: float, near_tol: float) -> dict:
    """Print and return one target's verdict, with its ambiguity kept visible."""
    kind, hits, k, rel = verdict(curve, target, rel_tol, near_tol)
    if kind == "recovered":
        print(f"  target {target}: RECOVERED -- window = {hits} "
              f"(residual {rel:.2e}; nearest k={k})")
    elif kind == "near":
        print(f"  target {target}: LIKELY window = {k} -- closest by {rel:.2e} "
              f"({100 * rel:.3f}%), just outside the {rel_tol:.0e} match tolerance")
        print("      This is the signature of an equivalent-but-differently-rounded "
              "path")
        print("      (e.g. the sum-of-all-experts dense-limit check), not a "
              "different window.")
    else:
        print(f"  target {target}: NO window -- closest k={k} is off by {rel:.2e} "
              f"({100 * rel:.2f}%)")
        print(f"      -> not the window: curve range {min(curve):.4f}-{max(curve):.4f}. "
              f"Do not read k={k} as an answer.")
    return {"verdict": kind, "matching_windows": hits, "nearest_window": k,
            "nearest_relative_residual": rel,
            "within_range": min(curve) <= target <= max(curve)}


def redecide(args) -> int:
    """Re-decide targets against a saved curve -- no model, no GPU.

    The curve is the measurement; a verdict about it is cheap to redo and should
    not cost a second forward pass, least of all when the thing being re-examined
    is whether a verdict was too strict.
    """
    with open(args.curve_from) as fh:
        prior = json.load(fh)
    losses, weights = prior["per_block_loss"], prior["per_block_tokens"]
    curve = ppl_curve(losses, weights)
    report = {k: prior[k] for k in
              ("model", "device", "amp_dtype", "corpus", "corpus_tokens",
               "seq_len", "n_val_blocks", "calibration", "calibrated")
              if k in prior}
    report.update({"per_block_loss": losses, "per_block_tokens": weights,
                   "ppl_curve": curve, "targets": {}})
    # Recorded only when the source is a *different* file: a self-reference reads
    # as though this curve came from somewhere else, which is the opposite of what
    # a provenance field is for.
    if os.path.abspath(args.curve_from) != os.path.abspath(args.out):
        report["redecided_from"] = args.curve_from
    print(f"re-deciding {len(args.target)} target(s) against the curve in "
          f"{args.curve_from} (calibrated: {prior.get('calibrated')})")
    print()
    for target in args.target:
        report["targets"][repr(target)] = describe(curve, target, args.rel_tol,
                                                   args.near_tol)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
