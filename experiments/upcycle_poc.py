#!/usr/bin/env python3
"""
upcycle_poc.py -- does *training* beat the frozen ceiling?

The question
------------
`experiments/pilot_sparsity.py` measured the no-retrain ceiling: slicing a
trained dense FFN into E experts and keeping k of them loses most of the layer
output (Qwen3.5-0.8B layer 0, E=8, k=1: mean relative residual 0.746). That
experiment touched no weights, so it says nothing about gradient descent.

This driver upcycles the FFN with `moeme.upcycle.MoEFFN` and *trains* the
experts and router, then measures three things:

  1. `val_ppl` dense vs untrained-MoE vs trained-MoE. The untrained number is
     the post-hoc path; the trained number is the upcycling path.
  2. `residual` = how well the MoE reproduces the *original* dense layer output
     on held-out activations. Same metric as the pilot's oracle, so the trained
     number can be compared to the frozen ceiling directly.
     **If trained residual < frozen-oracle residual, training created structure
     that a frozen partition did not have.**
  3. routing diagnostics: does the router specialise (effective experts) or
     collapse (all mass on one expert)?

Design notes
------------
* Only the upcycled layers' parameters are trained; the rest of the model is
  frozen. That keeps the run affordable (0.8B fully trainable would still fit a
  T4, but this makes the experiment cheap and the effect attributable).
* `--self-test` exercises the whole pipeline end to end on a tiny random model
  with no download, so the training/eval/residual plumbing is verifiable
  offline. Run that before spending a Kaggle session.
* Checkpoints are written after every configuration, so a 12-hour session cap
  cannot lose a finished config.
* `--max-seconds` caps the *training* time of each configuration. On a rented
  device whose speed is unknown ahead of time (a free Kaggle T4, say) the
  honest failure mode is an unbounded loop that hits the session cap with
  nothing written. With a budget the step is stopped early and the report is
  still written, so a slow device returns fewer steps of evidence rather than
  no evidence. `steps_completed` and `stopped_early` record what happened.
* A configuration that raises is recorded in the report as an `error` entry and
  the remaining configurations still run; the process exits non-zero at the
  end, so the failure is not swallowed.

Usage
-----
    python3 experiments/upcycle_poc.py --self-test
    python3 experiments/upcycle_poc.py --experts 8 --topks 1 2 4 \
        --steps 400 --layers all --out results/upcycle_poc.json
    # an unattended cloud run: never exceed ~4 h of training per config
    python3 experiments/upcycle_poc.py --experts 64 --topks 1 --layers all \
        --max-seconds 14400 --out results/upcycle_poc.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

from moeme.upcycle import (
    MoEConfig,
    collect_total_aux_loss,
    reconstruction_residual,
    upcycle_model,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT_JSON = os.path.join(HERE, "..", "results", "pilot_sparsity.json")


# ---------------------------------------------------------------------------
# device / dtype
# ---------------------------------------------------------------------------

def pick_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_grad_scaler(amp_dtype):
    enabled = amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)   # torch >= 2.4
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

BUILTIN_TEXT = """\
A mixture of experts layer replaces one feed forward network with several and a
router that selects a small subset per token. Training such a layer from scratch
is expensive, so upcycling initialises the experts from an existing dense model
and continues pretraining, letting the router and the experts specialise.
The load balancing auxiliary loss discourages the router from collapsing onto a
few experts. A collapsed router wastes capacity: most experts receive no
gradient and the effective parameter count falls.
Inference cost is set by the active parameters, one expert per token, while the
storage cost is set by the total parameters, every expert on disk. Those two
numbers diverge exactly in proportion to the number of experts.
A laptop with eight gigabytes of video memory cannot hold a twenty seven billion
parameter model, so decoding streams weights from a solid state drive and the
achievable rate is set by storage throughput rather than by arithmetic.
""" * 40


def load_token_stream(args, tokenizer) -> tuple[torch.Tensor, str]:
    """A flat token stream, plus a description of where it came from.

    The description is returned rather than only printed because it lands in the
    report: a run that silently fell back to the built-in corpus still produces
    perfectly finite perplexities, and on a repeated 1 kB text those numbers
    mean nothing. An unattended run must be able to record which corpus it used.
    """
    text = None
    source = None
    if args.text_file:
        with open(args.text_file) as fh:
            text = fh.read()
        source = f"text-file:{args.text_file}"
        print(f"corpus: {args.text_file} ({len(text):,} chars)")
    elif not args.offline:
        try:
            from datasets import load_dataset
            ds = load_dataset(args.dataset, split="train", streaming=True)
            want = args.corpus_chars
            chunks, have = [], 0
            for row in ds:
                t = row.get("text")
                if not t:
                    continue
                chunks.append(t)
                have += len(t)
                if have >= want:
                    break
            text = "\n".join(chunks)
            source = f"dataset:{args.dataset}"
            print(f"corpus: {args.dataset} streaming, {have:,} chars "
                  f"from {len(chunks)} documents")
        except Exception as e:                                  # noqa: BLE001
            print(f"corpus: dataset load failed ({type(e).__name__}: {e}); "
                  f"falling back to the built-in corpus")
    if text is None:
        text = BUILTIN_TEXT
        source = "built-in"
        print(f"corpus: built-in ({len(text):,} chars)")

    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    print(f"corpus: {ids.numel():,} tokens")
    return ids, source


def make_blocks(ids: torch.Tensor, seq_len: int, limit: int | None = None):
    """Non-overlapping (input_ids, labels) blocks.

    `labels` is deliberately the **same tensor** as `input_ids`: Hugging Face
    causal LM heads shift internally (`logits[..., :-1]` against
    `labels[..., 1:]`). Pre-shifting here shifts twice, giving a misaligned,
    near-random loss -- which is exactly what happened, and it read as a dense
    perplexity of 20,664 where the true figure is ~13. If this ever looks wrong
    again, check this first.
    """
    n = ids.numel() // seq_len
    if limit:
        n = min(n, limit)
    blocks = []
    for i in range(n):
        chunk = ids[i * seq_len:(i + 1) * seq_len]
        if chunk.numel() < seq_len:
            break
        blocks.append((chunk.unsqueeze(0), chunk.unsqueeze(0)))
    return blocks


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def make_report(args, *, device, amp_dtype, corpus_source, corpus_tokens,
                dense_ppl, n_train_blocks, n_val_blocks) -> dict:
    """The run header: everything needed to reproduce `dense_val_ppl`.

    Two of these fields were missing once, and each cost a re-run rather than a
    re-read:

    * `corpus` -- which text was streamed. The corpus is read with
      `streaming=True`, so a report without it cannot say what its perplexities
      were measured on.
    * `eval_blocks` -- how many val blocks the perplexity averages over. This one
      is not a nicety: on this model and this text, perplexity is **19.57 over 4
      val blocks and 23.42 over 8**, a 20% swing from a single flag that changes
      no weights and no data. A head line that omits it makes its own baseline
      unreproducible, which is what happened between E14 and E15.

    So: a perplexity without its eval window is not a measurement, for the same
    reason a perplexity without its corpus is not.
    """
    return {
        "model": args.model,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "corpus": corpus_source,
        "corpus_tokens": int(corpus_tokens),
        "eval_blocks": args.eval_blocks,
        "dense_val_ppl": dense_ppl,
        "seq_len": args.seq_len,
        "train_blocks": n_train_blocks,
        "val_blocks": n_val_blocks,
        "runs": [],
    }


@torch.no_grad()
def eval_perplexity(model, blocks, device, amp_dtype, max_blocks=8) -> float:
    model.eval()
    total, ntok = 0.0, 0
    for x, y in blocks[:max_blocks]:
        x, y = x.to(device), y.to(device)
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(x, labels=y)
        else:
            out = model(x, labels=y)
        total += out.loss.item() * y.numel()
        ntok += y.numel()
    return math.exp(total / max(ntok, 1))


@torch.no_grad()
def capture_reference(model, batches, layers, device) -> dict:
    """Per-layer (mlp input, dense mlp output) on a fixed held-out batch.

    Captured *before* upcycling, so it is the original dense function and stays
    a valid reference after training has moved the MoE weights.
    """
    layer_mods = model.model.layers
    caps: dict[int, dict] = {i: {} for i in layers}

    def pre(i):
        def fn(_m, args):
            caps[i]["x"] = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        return fn

    def post(i):
        def fn(_m, _args, out):
            caps[i]["y"] = out.detach().reshape(-1, out.shape[-1]).float()
        return fn

    handles = []
    for i in layers:
        handles.append(layer_mods[i].mlp.register_forward_pre_hook(pre(i)))
        handles.append(layer_mods[i].mlp.register_forward_hook(post(i)))

    model.eval()
    x = batches[0][0].to(device)
    model(x)
    for h in handles:
        h.remove()
    return {i: {k: v.cpu() for k, v in d.items() if v is not None}
            for i, d in caps.items() if "x" in d and "y" in d}


@torch.no_grad()
def evaluate_layers(moes, reference, device, dtype, top_k: int) -> dict:
    """Per-layer sparsity/preservation metrics for the *current* weights.

    * `vs_original_dense` -- how well the sparse layer still reproduces the
      pretrained dense function it replaced. This one **can exceed 1.0**: once
      training rescales experts to compensate for the experts that are never
      selected, the output can end up farther from the original than predicting
      zero would be. That is a real property of the trained model, not a bug.
      ("1.0 = no better than predicting zero" is the same scale the pilot used.)
    * `oracle_topk` -- best-case top-k against the module's **own** all-experts
      sum. This is exactly what the pilot measured, but on the *trained*
      weights, which is what makes trained-vs-frozen a fair comparison.
    * `router_topk` -- what the learned router actually achieves against that
      same all-experts sum. The oracle gap bounds what a better router could buy.
    """
    from moeme.ffn import oracle_topk_residual, router_topk_residual

    out: dict[int, dict] = {}
    by_layer = {i: m for i, m in moes}
    for i, ref in reference.items():
        m = by_layer.get(i)
        if m is None:
            continue
        x = ref["x"].to(device=device, dtype=dtype)
        outs = m.expert_outputs(x)                        # (E, T, hidden)
        scores = m.router_scores(x)                       # (T, E)
        k = min(top_k, m.n_experts)
        out[i] = {
            "vs_original_dense": reconstruction_residual(ref["y"].to(device), m(x)),
            "oracle_topk": oracle_topk_residual(outs, k)["mean_rel_residual"],
            "router_topk": router_topk_residual(outs, scores, k)["mean_rel_residual"],
        }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def summarize_layers(metrics: dict) -> dict:
    def mean(field):
        vals = [m[field] for m in metrics.values()]
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "vs_original_dense_mean": mean("vs_original_dense"),
        "oracle_topk_mean": mean("oracle_topk"),
        "router_topk_mean": mean("router_topk"),
    }


def routing_summary(moes) -> dict:
    per_layer, effs, loads, ents = {}, [], [], []
    for i, m in moes:
        s = m.last_stats
        per_layer[i] = s
        effs.append(s.get("effective_experts", float("nan")))
        loads.append(s.get("load_max_over_mean", float("nan")))
        ents.append(s.get("usage_entropy_normalised", float("nan")))

    def mean(v):
        vals = [x for x in v if x == x]                     # drop NaNs
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "per_layer": per_layer,
        "effective_experts_mean": mean(effs),
        "load_max_over_mean": mean(loads),
        "usage_entropy_mean": mean(ents),
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def load_pilot_oracle(experts: int, topk: int, layers) -> dict | None:
    """The pilot's frozen-oracle number for the same (E, k), if recorded."""
    try:
        with open(PILOT_JSON) as fh:
            pilot = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    out = {}
    for layer_str, data in pilot.get("layers", {}).items():
        if data.get("geometry", {}).get("intermediate") is None:
            continue
        key = f"random_E{experts}"
        block = data.get(key) or data.get(f"contiguous_E{experts}")
        if not block:
            continue
        oracle = block.get("oracle", {}).get(str(topk))
        if oracle:
            out[int(layer_str)] = {
                "mean_rel_residual": oracle["mean_rel_residual"],
                "plan_key": key,
            }
    return {"model": pilot.get("model"), "per_layer": out} if out else None


def train_one_config(*, model, originals, blocks_mod, config,
                     args, train_blocks, val_blocks, reference, device,
                     amp_dtype, scaler, dense_ppl: float) -> dict:
    """Upcycle, measure untrained, train, measure trained. One (E, k)."""
    layers = list(range(len(blocks_mod))) if args.layers == ["all"] \
        else [int(v) for v in args.layers]

    # restore the original dense MLPs, then upcycle afresh (identical init)
    for i, orig in originals:
        setattr(blocks_mod[i], "mlp", orig)
    for m in model.modules():
        for p in m.parameters(recurse=False):
            p.requires_grad_(False)

    t0 = time.perf_counter()
    moes = upcycle_model(model, config, layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"  upcycled {len(moes)} layers, trainable params {n_train/1e6:.1f}M")

    # --- untrained (the post-hoc path) ---
    untrained_ppl = eval_perplexity(model, val_blocks, device, amp_dtype,
                                    args.eval_blocks)
    untrained_m = evaluate_layers(moes, reference, device,
                                  next(model.parameters()).dtype, config.top_k)

    # --- dense-equivalence integration check ---
    saved_k = [(m, m.config) for _, m in moes]
    for _, m in moes:
        m.config = m.config.with_(top_k=m.n_experts)
    dense_limit_ppl = eval_perplexity(model, val_blocks, device, amp_dtype,
                                      args.eval_blocks)
    for m, cfg in saved_k:
        m.config = cfg

    # --- train ---
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)
    ckpt_path = os.path.join(args.ckpt_dir, f"{config_tag(config, args)}.pt")
    start_step = 0
    if args.resume and ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        for (i, m) in moes:
            m.load_state_dict(state["moes"][i])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        print(f"  resumed from {ckpt_path} at step {start_step}")

    model.train()
    curve = []
    t_train = time.perf_counter()
    accum = max(1, args.grad_accum)
    budget = float(getattr(args, "max_seconds", 0) or 0)
    stopped_early = False
    steps_done = start_step
    for step in range(start_step, args.steps):
        if budget and (time.perf_counter() - t_train) > budget:
            print(f"  wall-clock budget {budget:.0f}s reached before step {step}; "
                  f"stopping training and reporting what was measured",
                  flush=True)
            stopped_early = True
            break
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accum):
            k = int(torch.randint(len(train_blocks), (1,), generator=gen).item())
            x, y = train_blocks[k]
            x, y = x.to(device), y.to(device)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    out = model(x, labels=y)
                    aux = collect_total_aux_loss(model)
                    loss = (out.loss + aux) / accum
                scaler.scale(loss).backward()
            else:
                out = model(x, labels=y)
                aux = collect_total_aux_loss(model)
                loss = (out.loss + aux) / accum
                loss.backward()
            total_loss += float(loss.item())
        if scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step()
        # progress at most every 25 steps, and always report elapsed time so the
        # per-step cost is visible immediately. That matters: cost scales with
        # (upcycled layers x experts), so a subset at E=8 and all layers at E=64
        # differ by ~48x, which is the difference between minutes and a session.
        every = max(1, min(args.steps // 10, 25))
        if step % every == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - t_train
            # `elapsed` is stored, not merely printed: it is the only way to
            # recover the measured step cost from the report afterwards, which
            # is what sizing a later run on a different device depends on.
            curve.append({"step": step, "loss": total_loss, "elapsed": elapsed})
            print(f"    step {step:5d}  loss {total_loss:.4f}  "
                  f"elapsed {elapsed:6.0f}s", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            _save(ckpt_path, moes, opt, step + 1)
        steps_done = step + 1

    trained_ppl = eval_perplexity(model, val_blocks, device, amp_dtype,
                                  args.eval_blocks)
    # evaluate_layers runs every MoE layer forward, which also refreshes the
    # per-layer routing statistics used below
    trained_m = evaluate_layers(moes, reference, device,
                                next(model.parameters()).dtype, config.top_k)
    stats = routing_summary(moes)

    def mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else float("nan")

    result = {
        "config": {
            "n_experts": config.n_experts, "top_k": config.top_k,
            "combine": config.combine, "plan": config.plan,
            "aux_loss_coef": config.aux_loss_coef, "router_hidden": config.router_hidden,
            "expert_noise": config.expert_noise,
        },
        "layers": layers,
        "trainable_params": n_train,
        "steps": args.steps,
        "steps_completed": steps_done,
        "stopped_early": stopped_early,
        "max_seconds": budget,
        "dense_val_ppl": dense_ppl,
        "dense_limit_val_ppl": dense_limit_ppl,
        "untrained_val_ppl": untrained_ppl,
        "trained_val_ppl": trained_ppl,
        "metrics_untrained": summarize_layers(untrained_m),
        "metrics_trained": summarize_layers(trained_m),
        "metrics_per_layer": {
            "untrained": {str(k): v for k, v in untrained_m.items()},
            "trained": {str(k): v for k, v in trained_m.items()},
        },
        "routing": stats,
        "loss_curve": curve,
        "wall_seconds": time.perf_counter() - t0,
    }
    frozen = load_pilot_oracle(config.n_experts, config.top_k, layers)
    if frozen:
        result["frozen_oracle_reference"] = frozen
        common = set(frozen["per_layer"]) & set(trained_m)
        if common:
            f_mean = mean(frozen["per_layer"][i]["mean_rel_residual"] for i in common)
            o_mean = mean(trained_m[i]["oracle_topk"] for i in common)
            r_mean = mean(trained_m[i]["router_topk"] for i in common)
            result["beats_frozen_oracle"] = bool(o_mean < f_mean)
            result["frozen_oracle_mean_on_common_layers"] = f_mean
            result["trained_oracle_mean_on_common_layers"] = o_mean
            result["trained_router_mean_on_common_layers"] = r_mean
    return result


def config_tag(config, args) -> str:
    return (f"E{config.n_experts}_k{config.top_k}_{config.combine}_"
            f"{config.plan}_s{args.seed}_{'-'.join(args.layers)}")


def _save(path, moes, opt, step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "moes": {i: m.state_dict() for i, m in moes},
        "opt": opt.state_dict(),
        "step": step,
    }, path)


# ---------------------------------------------------------------------------
# self-test: the whole pipeline on a tiny random model, no download
# ---------------------------------------------------------------------------

class _TinyMLP(torch.nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = torch.nn.Linear(h, i, bias=False)
        self.up_proj = torch.nn.Linear(h, i, bias=False)
        self.down_proj = torch.nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyBlock(torch.nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.mlp = _TinyMLP(h, i)

    def forward(self, x):
        return x + self.mlp(x)


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab=256, h=64, i=128, n=3):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, h)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyBlock(h, i) for _ in range(n)])
        self.head = torch.nn.Linear(h, vocab, bias=False)

    def forward(self, ids, labels=None):
        x = self.embed(ids)
        for b in self.model.layers:
            x = b(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   labels.reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)


def self_test() -> int:
    """Run upcycle -> untrained eval -> train -> residual on a tiny model."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _TinyLM().to(device)
    # run the self-test in the same dtype the real run uses, so dtype bugs in the
    # forward path surface here rather than on a rented GPU (this one did)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
    blocks_mod = model.model.layers
    originals = [(i, blocks_mod[i].mlp) for i in range(len(blocks_mod))]

    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (4096,), generator=g)
    blocks = make_blocks(ids, seq_len=128)
    train_blocks, val_blocks = blocks[:-2], blocks[-2:]

    model.eval()
    dense_ppl = eval_perplexity(model, val_blocks, device, None, max_blocks=2)
    reference = capture_reference(model, val_blocks, [0, 1, 2], device)
    print(f"dense ppl {dense_ppl:.3f}; reference layers {sorted(reference)}")

    config = MoEConfig(n_experts=4, top_k=2, plan="random", aux_loss_coef=0.01)
    args = SimpleNamespace(layers=["all"], lr=3e-3, weight_decay=0.0,
                           grad_accum=2, steps=40, clip=1.0, seed=0, resume=False,
                           max_seconds=0.0,
                           ckpt_dir=os.path.join(HERE, "..", ".cache", "moeme",
                                                 "selftest_ckpt"),
                           eval_blocks=2, save_every=0)
    scaler = make_grad_scaler(None)
    res = train_one_config(model=model, originals=originals, blocks_mod=blocks_mod,
                           config=config, args=args,
                           train_blocks=train_blocks, val_blocks=val_blocks,
                           reference=reference, device=device, amp_dtype=None,
                           scaler=scaler, dense_ppl=dense_ppl)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("routing", "loss_curve", "metrics_per_layer",
                                   "config", "layers")}, indent=2))
    print("routing:", json.dumps(res["routing"]["per_layer"][0], indent=2))
    tr = res["metrics_trained"]
    assert res["dense_limit_val_ppl"] < dense_ppl * 1.5, res["dense_limit_val_ppl"]
    assert math.isfinite(res["trained_val_ppl"])
    assert math.isfinite(tr["oracle_topk_mean"]) and tr["oracle_topk_mean"] >= 0
    # the oracle is a ceiling: a learned router cannot beat the best-possible k
    assert tr["oracle_topk_mean"] <= tr["router_topk_mean"] + 1e-5, tr
    print("\nself-test OK")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MOEME_POC_MODEL",
                                                      "Qwen/Qwen3.5-0.8B"))
    ap.add_argument("--experts", type=int, nargs="+", default=[8])
    ap.add_argument("--topks", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--combine", default="sum", choices=["sum", "mean", "softmax"])
    ap.add_argument("--plan", default="random",
                    choices=["contiguous", "interleaved", "random", "clustered"])
    ap.add_argument("--layers", nargs="+", default=["all"],
                    help='layer indices, or the single token "all"')
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--aux-loss-coef", type=float, default=1e-2)
    ap.add_argument("--expert-noise", type=float, default=0.0)
    ap.add_argument("--router-hidden", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-blocks", type=int, default=4)
    ap.add_argument("--corpus-chars", type=int, default=4_000_000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="wall-clock budget for each configuration's training "
                         "loop; 0 = unlimited. Set this on an unattended or "
                         "rented device: without it a run slower than expected "
                         "is killed by the session cap having written nothing.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-dir", default=os.path.join(".cache", "moeme", "poc_ckpt"))
    ap.add_argument("--out", default="results/upcycle_poc.json")
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args(argv)


def main() -> int:
    args = build_args()
    if args.self_test:
        return self_test()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device. Qwen3.5's Gated DeltaNet path needs "
              "causal_conv1d, which is CUDA-only in this environment.")
    amp_dtype = pick_amp_dtype(device)
    scaler = make_grad_scaler(amp_dtype)
    torch.manual_seed(args.seed)
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
    print(f"params {sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    ids, corpus_source = load_token_stream(args, tok)
    if corpus_source == "built-in" and not args.offline:
        print("  WARNING: the requested dataset did not load and the built-in "
              "corpus was used instead. Perplexities from a repeated 1 kB text "
              "are not comparable to a real corpus -- check network access "
              "before reading any ppl number below.")
    blocks = make_blocks(ids, args.seq_len)
    if len(blocks) < 4:
        raise SystemExit("corpus too small for train/val split; raise --corpus-chars")
    n_val = max(1, len(blocks) // 20)
    train_blocks, val_blocks = blocks[:-n_val], blocks[-n_val:]
    print(f"blocks: {len(train_blocks)} train, {len(val_blocks)} val")

    blocks_mod = model.model.layers
    originals = [(i, blocks_mod[i].mlp) for i in range(len(blocks_mod))]
    layers = list(range(len(blocks_mod))) if args.layers == ["all"] \
        else [int(v) for v in args.layers]

    model.eval()
    dense_ppl = eval_perplexity(model, val_blocks, device, amp_dtype,
                                args.eval_blocks)
    print(f"dense val ppl {dense_ppl:.4f}")
    if dense_ppl > 200:
        print("  WARNING: dense perplexity is implausibly high. For a pretrained "
              "model on natural text expect ~5-40. Usual cause: label "
              "misalignment in make_blocks, or the wrong corpus/tokenizer.")
    reference = capture_reference(model, val_blocks, layers, device)

    report = make_report(args, device=device, amp_dtype=amp_dtype,
                         corpus_source=corpus_source, corpus_tokens=ids.numel(),
                         dense_ppl=dense_ppl, n_train_blocks=len(train_blocks),
                         n_val_blocks=len(val_blocks))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    failed = False
    for E in args.experts:
        for k in args.topks:
            if k > E:
                continue
            config = MoEConfig(
                n_experts=E, top_k=k, combine=args.combine, plan=args.plan,
                aux_loss_coef=args.aux_loss_coef, router_hidden=args.router_hidden,
                seed=args.seed, expert_noise=args.expert_noise,
            )
            print(f"\n=== E={E} k={k} combine={args.combine} plan={args.plan} ===",
                  flush=True)
            try:
                res = train_one_config(
                    model=model, originals=originals, blocks_mod=blocks_mod,
                    config=config, args=args,
                    train_blocks=train_blocks, val_blocks=val_blocks,
                    reference=reference, device=device, amp_dtype=amp_dtype,
                    scaler=scaler, dense_ppl=dense_ppl,
                )
            except Exception as e:                              # noqa: BLE001
                # record it, keep the other configurations runnable, and exit
                # non-zero at the end -- the failure stays visible, but a run
                # that already produced results is not thrown away with it
                failed = True
                traceback.print_exc()
                report["runs"].append({
                    "config": {"n_experts": E, "top_k": k,
                               "combine": args.combine, "plan": args.plan},
                    "layers": layers,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                with open(args.out, "w") as fh:
                    json.dump(report, fh, indent=2)
                print(f"  FAILED: {type(e).__name__}: {e} -- recorded in "
                      f"{args.out}", flush=True)
                continue
            report["runs"].append(res)
            tag = "  [EARLY-STOPPED]" if res["stopped_early"] else ""
            print(f"  steps: {res['steps_completed']}/{res['steps']}{tag}")
            print(f"  ppl: dense {res['dense_val_ppl']:.4f} | "
                  f"untrained {res['untrained_val_ppl']:.4f} | "
                  f"trained {res['trained_val_ppl']:.4f} | "
                  f"dense-limit {res['dense_limit_val_ppl']:.4f}")
            ut, tr = res["metrics_untrained"], res["metrics_trained"]
            print(f"  top-k vs all-experts: untrained-oracle "
                  f"{ut['oracle_topk_mean']:.4f} -> trained-oracle "
                  f"{tr['oracle_topk_mean']:.4f} | trained-router "
                  f"{tr['router_topk_mean']:.4f} | effective_experts "
                  f"{res['routing']['effective_experts_mean']:.2f}")
            if "beats_frozen_oracle" in res:
                print(f"  frozen pilot oracle "
                      f"{res['frozen_oracle_mean_on_common_layers']:.4f} vs trained "
                      f"{res['trained_oracle_mean_on_common_layers']:.4f} "
                      f"-> beats_frozen_oracle={res['beats_frozen_oracle']}")
            with open(args.out, "w") as fh:          # write after every config
                json.dump(report, fh, indent=2)
            print(f"  wrote {args.out}")

    if failed:
        print("\ndone WITH FAILURES -- see the \"error\" entries in the report")
        return 1
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
