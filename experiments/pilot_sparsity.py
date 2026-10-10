#!/usr/bin/env python3
"""
pilot_sparsity.py -- pilot sparsity study: quality proxy vs active compute.

Question this answers
---------------------
If you take a REAL trained FFN and slice its intermediate axis into E experts,
how much of the layer's output can you recover by activating only k of them --
with a *perfect* router (oracle) and with a *cheap learnable* router?

If the oracle curve is bad, no router can fix it and the design must change
before any training compute is spent. That is the checkpoint this pilot exists
to be.

Model choice
------------
Qwen/Qwen3.5-2B. Its text config is `qwen3_5_text` with the same
3x(GatedDeltaNet) + 1x(GatedAttention) layer pattern as Qwen3.8-27B, so its FFN
blocks are architecturally the same family. Caveats, stated plainly:
  * it is ~14x smaller, and its FFN expansion ratio is 3.0 vs the 27B's 3.4;
  * activations are real (captured from forward passes on real text), but the
    calibration corpus is small and not a benchmark.

Metric
------
Relative L2 error of the k-expert reconstruction against the true dense layer
output. 0.0 = exact, 1.0 = no better than predicting zero.

Usage:
    python3 experiments/pilot_sparsity.py --out results/pilot_sparsity.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from moeme.ffn import (
    DenseFFN,
    PartitionedFFN,
    clustered_plan,
    contiguous_plan,
    contribution_concentration,
    interleaved_plan,
    oracle_topk_residual,
    random_plan,
    router_topk_residual,
    train_linear_router,
)

# Chosen because it shares the 27B's architecture exactly:
#   model_type qwen3_5_text, layer pattern 3x(GatedDeltaNet)+1x(GatedAttention),
#   and FFN expansion 3584/1024 = 3.5 vs the 27B's 17408/5120 = 3.4.
# It also fits entirely in the free VRAM, which the 2B sibling does not.
MODEL_ID = os.environ.get("MOEME_PILOT_MODEL", "Qwen/Qwen3.5-0.8B")
TARGET_LAYERS = [0, 3, 8, 23]     # mix of DeltaNet and full-attention layers
EXPERTS = [8, 32, 64]
TOPKS = [1, 2, 4, 8]
MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# calibration corpus
# ---------------------------------------------------------------------------

def build_calibration_text() -> str:
    """Small, reproducible, dependency-free mixed corpus.

    Code comes from this repository's own sources (real code, no network), plus
    hand-written maths/instruction/chat passages. Deliberately NOT a benchmark:
    the pilot measures routing structure, not task quality.
    """
    here = os.path.dirname(__file__)
    parts = []
    for rel in ("src/moeme/ffn.py", "src/moeme/accounting.py",
                "tools/verify_model_geometry.py"):
        p = os.path.join(here, "..", rel)
        if os.path.exists(p):
            with open(p) as fh:
                parts.append(f"# source file {rel}\n" + fh.read())
    parts.append("""
    Problem. A train leaves at 14:20 and travels 210 km at 84 km/h. When does it
    arrive? Working: the travel time is 210 / 84 = 2.5 hours, so the arrival time
    is 14:20 + 2:30 = 16:50. Check: 84 * 2.5 = 210, which matches the distance.
    Therefore the train arrives at 16:50.

    Theorem. For any finite set S, the power set of S has exactly 2**|S| elements,
    because each element is independently either present or absent, giving 2
    choices per element and hence 2**|S| distinct subsets. This generalises to
    the cardinal arithmetic identity 2**n = sum over k of the binomial
    coefficients C(n, k).

    Instruction. Summarise the following text in one sentence. Then list three
    action items as a JSON array of strings. Respond only with valid JSON.
    {"summary": "...", "actions": ["...", "...", "..."]}

    Conversation. A user asks how to reduce memory usage when loading a large
    quantised model on a laptop. A helpful answer explains memory mapping, the
    difference between resident and active parameters, why decoding is
    bandwidth bound rather than compute bound, and why swapping to a solid state
    drive makes token generation slow but not impossible.

    Code review. The function below is correct but allocates a new list on every
    call. Consider using a generator, or appending into a preallocated buffer, to
    avoid the repeated allocation cost in the hot loop.
    """)
    return "\n\n".join(parts)


def capture_activations(model_id: str, layers: list[int], text: str):
    """Run a real forward pass and return {layer: (tokens, hidden)} MLP inputs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16
    last_err = None
    for kwargs in ({"dtype": dtype}, {"torch_dtype": dtype}):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, low_cpu_mem_usage=True, **kwargs
            )
            break
        except TypeError as e:      # transformers renamed torch_dtype -> dtype
            last_err = e
    else:
        raise last_err
    model.eval()
    print(f"  loaded, dtype={next(model.parameters()).dtype}, "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # The qwen3_5 Gated DeltaNet path dispatches to causal_conv1d, whose kernel
    # in this environment is CUDA-only (`Expected x.is_cuda() to be true`).
    # So the pilot must run on the GPU, not CPU.
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"  moved to cuda; vram allocated "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
    else:
        raise SystemExit(
            "no CUDA device: the qwen3_5 GatedDeltaNet path needs causal_conv1d, "
            "which is CUDA-only here. Run on the GPU, or use a qwen3 model "
            "(standard attention) instead."
        )

    blocks = model.model.layers
    caps: dict[int, list] = {i: [] for i in layers}

    def hook(i):
        def fn(_module, args):
            caps[i].append(args[0].detach().reshape(-1, args[0].shape[-1]).float())
        return fn

    handles = [blocks[i].mlp.register_forward_pre_hook(hook(i)) for i in layers
               if i < len(blocks)]
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=MAX_TOKENS).input_ids.to(model.device)
    print(f"  forward pass over {ids.shape[1]} tokens ...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(ids)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    for h in handles:
        h.remove()

    return {i: torch.cat(v, dim=0) for i, v in caps.items() if v}, model


def ffn_from_module(model, layer: int) -> DenseFFN:
    m = model.model.layers[layer].mlp
    return DenseFFN(
        gate=m.gate_proj.weight.detach().float(),
        up=m.up_proj.weight.detach().float(),
        down=m.down_proj.weight.detach().float(),
    )


# ---------------------------------------------------------------------------

PLANS = {
    # name: (factory, expert counts)
    "contiguous": (lambda ffn, E: contiguous_plan(ffn.intermediate, E), EXPERTS),
    "interleaved": (lambda ffn, E: interleaved_plan(ffn.intermediate, E), EXPERTS),
    "random": (lambda ffn, E: random_plan(ffn.intermediate, E, seed=0), EXPERTS),
    "clustered": (lambda ffn, E: clustered_plan(ffn.gate, ffn.up, E, seed=0), [8, 32]),
}


def study_layer(ffn: DenseFFN, x: torch.Tensor, router_steps: int) -> dict:
    """All plans / expert counts for one layer, on real activations x."""
    n = x.shape[0]
    n_train = n // 2
    x_train, x_test = x[:n_train], x[n_train:]
    out: dict = {"n_tokens": n, "n_train": n_train}

    # sanity: partitioning must be exact on these real activations
    p1 = PartitionedFFN(ffn, contiguous_plan(ffn.intermediate, 8))
    out["exactness_rel_error"] = p1.dense_equivalent_error(x_test)
    assert out["exactness_rel_error"] < 1e-4, out["exactness_rel_error"]

    for pname, (factory, es) in PLANS.items():
        for E in es:
            plan = factory(ffn, E)
            p = PartitionedFFN(ffn, plan)
            sizes = torch.tensor(plan.sizes(), dtype=torch.float)
            outs = p.expert_outputs(x_test)                    # (E, T, hidden)
            conc = contribution_concentration(outs)

            oracle = {}
            for k in TOPKS:
                if k <= E:
                    oracle[k] = oracle_topk_residual(outs, k)

            # cheap learnable router, trained on the other half of the tokens
            outs_train = p.expert_outputs(x_train)
            train_targets = outs_train.norm(dim=2).argmax(dim=0)   # oracle top-1
            wb = train_linear_router(x_train, train_targets, E, steps=router_steps)
            feats = torch.cat(
                [x_test, torch.ones(x_test.shape[0], 1, device=x_test.device)], dim=1
            )
            scores = feats @ wb
            test_targets = outs.norm(dim=2).argmax(dim=0)
            learned = {}
            for k in TOPKS:
                if k <= E:
                    learned[k] = router_topk_residual(outs, scores, k)

            out[f"{pname}_E{E}"] = {
                "expert_sizes": {
                    "min": int(sizes.min()), "max": int(sizes.max()),
                    "mean": float(sizes.mean()),
                    "max_over_mean": float(sizes.max() / sizes.mean()),
                    "largest_share": float(sizes.max() / sizes.sum()),
                },
                "concentration": conc,
                "oracle": oracle,
                "learned_router": learned,
                # how often the cheap router agrees with the best expert
                "router_top1_accuracy": float(
                    (scores.argmax(dim=1) == test_targets).float().mean()
                ),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilot_sparsity.json")
    ap.add_argument("--layers", type=int, nargs="*", default=TARGET_LAYERS)
    ap.add_argument("--router-steps", type=int, default=300)
    args = ap.parse_args()

    text = build_calibration_text()
    print(f"calibration corpus: {len(text):,} characters")

    caps, model = capture_activations(MODEL_ID, args.layers, text)
    print("captured MLP inputs for layers:",
          {k: tuple(v.shape) for k, v in caps.items()})

    report = {"model": MODEL_ID, "layers": {}, "calibration_chars": len(text)}
    for layer, x in sorted(caps.items()):
        print(f"\n=== layer {layer} · {x.shape[0]} tokens · hidden {x.shape[1]} ===",
              flush=True)
        ffn = ffn_from_module(model, layer)
        print(f"  FFN: intermediate={ffn.intermediate} "
              f"(expansion {ffn.intermediate/ffn.hidden:.2f})")
        res = study_layer(ffn, x, args.router_steps)
        report["layers"][str(layer)] = {"geometry": {
            "hidden": ffn.hidden, "intermediate": ffn.intermediate},
            **res}
        for key, v in res.items():
            if not isinstance(v, dict) or "oracle" not in v:
                continue
            best_k = min(v["oracle"], key=lambda k: v["oracle"][k]["mean_rel_residual"])
            print(f"    {key:16s} largest_share={v['expert_sizes']['largest_share']:.3f}"
                  f"  PR={v['concentration']['participation_ratio']:.1f}"
                  f"  oracle(k={best_k})={v['oracle'][best_k]['mean_rel_residual']:.4f}"
                  f"  learned(k={best_k})="
                  f"{v['learned_router'][best_k]['mean_rel_residual']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
