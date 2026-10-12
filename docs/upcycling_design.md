# Design: turning Qwen3.8-27B into a ~9B-active MoE

**Status:** design, and its gate has been run and **failed**. The PoC was executed
on the real Qwen3.5-0.8B model — E14 locally on 4 layers, E15 on all 24 layers at
E=8/32/64 on free Kaggle T4s. Training beat the frozen ceiling only at the mildest
ratio (E=8, 12.5% active, by 0.006) and **lost at the ratios this design needs**,
E=32 and E=64, where the residual rose instead of falling (§9).
**Recommendation: do not build this.** Cost is deliberately ignored below (per the
brief).

This document exists because the earlier feasibility work answered a narrower
question than it looked like it did. It showed that **post-hoc** FFN
sparsification has no usable operating point -- but that was measured with *no
weight updates*, so it bounds the frozen partition, not the trained one. This is
the plan for the trained version, and the honest arithmetic of what it can reach.

---

## 1. What "9B active" can and cannot mean

Verified geometry of the Qwen3.8-27B text tower (`src/moeme/accounting.py`,
checked against the real safetensors headers):

| Component | Params | Active every token? |
|---|---:|---|
| FFN (64 layers, swiGLU) | 17,112,760,320 | only if dense |
| Gated DeltaNet (48 layers) | 5,562,044,928 | **yes** |
| Full attention (16 layers) | 1,677,721,600 | **yes** |
| LM head | 1,271,398,400 | **yes** |
| Embedding | 1,271,398,400 | a gather, not a matmul |
| Norms | 674,816 | **yes** |
| **backbone total** | **9,783,238,144** | |

Two accounting conventions give two different answers, and the design has to
pick one out loud:

* **Strict (the embedding table counts as active).** The floor is **9.78B**.
  Routing the FFN to *zero* experts cannot get under it. The goal as originally
  stated is unreachable by any FFN-only change -- this is arithmetic, not a
  modelling difficulty.
* **Compute-bearing (the embedding gather is excluded).** The floor is
  **8,511,839,744**, leaving **0.49B** for active FFN. Reachable.

Under the second convention the target needs `E/k ≈ 35`:

| Config | Active FFN | Total active | |
|---|---:|---:|---|
| 32 experts, top-1 | 0.535B | 9.04B | over |
| **64 experts, top-1** | **0.267B** | **8.78B** | reachable |
| **128 experts, top-2** | **0.267B** | **8.78B** | reachable |
| 64 experts, top-2 | 0.535B | 9.04B | over |

Note that the model **stays ~27B stored**. This shrinks FLOPs per token; it does
not shrink the footprint. On the target laptop, where the measured bottleneck is
*storage* (0.561 tok/s streaming from an NVMe that delivers 0.8-1.9 GB/s), that
is the crux: see §7.

## 2. Why partitioning, not replication

Standard sparse upcycling (Komatsuzaki et al., 2022) replicates the dense FFN
into E full-width copies, which multiplies *total* parameters by roughly E.
Mixtral-style models are built that way.

That is the wrong primitive here, for two reasons:

1. It does not hold total parameters constant, so it does not answer a "27B
   total / 9B active" question at all.
2. The repository already has a **partition** implementation that is proven
   *exactly* dense-equivalent when all experts are active
   (`src/moeme/ffn.py`, worst relative error 5.1e-07 on real 27B weights).

So this design partitions the intermediate axis: expert `e` owns a disjoint
slice of the 17,408 intermediate units. Total parameters are unchanged, and
`top_k = E` with `combine="sum"` **is** the dense layer -- the dense model is a
strict special case, which is what makes the PoC's dense-limit check meaningful
rather than decorative.

## 3. Expert configuration

**Recommendation: few large experts, top-1** (64 experts, ~272 intermediate
units each), not many small ones with top-2. On this hardware at batch 1 you are
memory-bandwidth-bound, and top-1 gives the smallest per-token read with the
least routing overhead. `128×2` equals it on active params but costs more
dispatch and more small GEMMs; `src/moeme/ffn.py` measured a 2.1-3.9× small-GEMM
penalty at batch 1 for exactly this reason (two runs; the ratio is noisy). Note
that top-k recovers none of it -- top-2 of 32 timed the same as all-32.

Concretely, per configuration to run:

| | experts | top-k | active FFN | total active |
|---|---:|---:|---:|---:|
| primary | 64 | 1 | 0.267B | 8.78B |
| fallback | 128 | 2 | 0.267B | 8.78B |
| control | 32 | 1 | 0.535B | 9.04B |

## 4. Router initialisation

A fresh linear router with default init routes near-uniformly, which wastes the
first part of the run. Two concrete steps, in order of value:

1. **Warm-start the router from the frozen model.** The intermediate units of a
   trained dense FFN already have different output norms, and
   `ffn.train_linear_router` can be fit to the *oracle* top-1 labels (the
   largest-norm expert per token) in a few hundred steps. This costs almost
   nothing and starts the model from "route where the signal already is"
   instead of from noise. `ffn.oracle_topk_residual` supplies the labels.
2. **Small-init router plus a load-balancing loss.** Auxiliary loss with
   coefficient ~1e-2 (Switch-style) and a router z-loss ~1e-3, as implemented in
   `src/moeme/upcycle.py`. Watch `effective_experts`; if it collapses toward 1,
   raise the coefficient.

## 5. The output-scale problem, stated plainly

A **partition with a sparse top-k is a partial sum**: with `combine="sum"`, k of
E slices sum to roughly `k/E` of the dense layer's output. So the sparse layer
starts with a systematic magnitude deficit that training has to absorb, and one
place it can go wrong is experts scaling up so far that the layer saturates.

**Measured (E21), and the `k/E` premise above is wrong for a partition.** That
framing assumes the expert contributions add *coherently*. They do not: the experts
are disjoint slices of one layer, so the selected expert's output is a *component*
of the dense output, and for roughly orthogonal components the least-squares optimal
output scale is exactly **1**. Fitting `s* = <ref, kept> / ||kept||^2` on frozen
weights at E=8, mean over four layers, gives `s*` = 1.406, 1.257 and 1.114 at k = 1,
2 and 4 -- against `E/k` of 8, 4 and 2. The whole magnitude correction is worth
0.8493 -> 0.8163 at k=1, so **about 96% of the deficit is directional, not a scale
error**: no output rescaling, imposed or learned, can recover it. Applying `E/k`
makes the frozen residual worse (0.9249 -> 2.4461) and the trained model much worse
(29.35x dense against 4.15x). `E/k` is the right correction for *replicated* experts,
which is not what §2 builds.

This is a real design risk, not a footnote. Three mitigations, in the order I
would try them:

1. **Scale expert init by `E/k`**, so the sparse sum starts at the right
   magnitude. Cheap to test; may be unstable for large E/k. **Tested (E21) and it
   is the wrong scale for a partition** -- `s*` is ~1.1-1.4, not `E/k` = 8/4/2, and
   imposing it degrades both the frozen residual and the trained model. Kept here
   as the hypothesis it was, not as a recommendation.
2. **Let the router carry a learned gate**, `combine="softmax"` -- but note this
   *forfeits* the exact dense limit at `top_k=E` (softmax weights sum to 1, so a
   full-k combination is a weighted mean, not the sum). `tests/test_upcycle.py`
   pins this difference deliberately. **Trained once (E21)**, at k=1 where the
   weights are 1.0 and so the forward is identical to `sum` -- the arms differ
   only in whether the *router* receives gradient from the reconstruction loss.
   It was the best of the three arms and still worse than token training, which
   is the interesting part: letting the router learn is the one version of this
   mitigation that does something, and it is worth ~0.08 of a 0.84 deficit.
3. **An explicit layer scale** (a per-layer multiplier on the MoE output),
   which is the most flexible and the least principled. **E21 bounds what it can
   recover**: the least-squares optimal scale is ~1.1-1.4 rather than 1, so a
   learned layer scale has at most that much to find, and the deficit it leaves
   behind is directional.

The PoC reports `vs_original_dense`, which can exceed 1.0 precisely because of
this effect -- that is a signal, not a bug.

## 6. Continued-pretraining recipe

Upcycling does not work as a pure architecture change; the experts must be
trained. The pilot's most damning result -- the trained dense FFN has **no
latent expert structure** (`participation_ratio ≈ E`; top-3 experts hold 91% of
the energy) -- is exactly why: there is nothing to *extract*, so the structure
has to be *created*.

* **Objective:** next-token cross-entropy + aux loss + z-loss. The router needs
  its own, higher learning rate (10× the expert LR is a reasonable start).
* **Data:** the model's original pretraining mixture if available, otherwise a
  general corpus (FineWeb-Edu / DCLM class). Sequence length 4096-8192.
* **Schedule:** warmup then cosine decay; watch the load-balance statistics for
  the first few thousand steps before trusting the run.
* **On the pilot's caveat:** the pilot's activations were a small,
  non-benchmark corpus. That limitation carries over here -- a real run needs a
  real corpus and a real evaluation set, which is what
  `experiments/eval_gate.py` exists to provide.

## 7. Checkpointing, sharding, and why this is not a laptop job

* **Training** shards experts across devices with expert parallelism and an
  all-to-all dispatch. Checkpoint frequently: the recipe below is a multi-day
  job on rented hardware.
* **Inference at batch 1** is the regime this whole study is about, and there
  the win is different from the training win: only the *selected* expert's
  weights must be read per token. If the active set is resident, throughput
  scales with `active_bytes`, not total bytes. That is the same mechanism as the
  resident-quantisation recommendation, applied to a sparser model.
* **Locality matters more than the textbook design.** A router free to pick any
  of 64 experts defeats prefetching. A hashed or otherwise locality-constrained
  router trades a little quality for a large bandwidth win -- worth prototyping.

## 8. Cost

Compute is approximately `6 × N_active × D` FLOPs. With `N_active ≈ 8.8B`:

| Continued-pretrain budget | FLOPs | ≈ A100-hours | ≈ cost at $2/hr |
|---|---:|---:|---:|
| 50B tokens (partial recovery) | 2.6e21 | ~4,900 | ~$10k |
| 500B tokens | 2.6e22 | ~49,000 | ~$100k |
| 2T tokens (~15% of a full pretrain) | 1.1e23 | ~196,000 | ~$390k |

These assume ~150 TFLOP/s effective per A100 and will move with model,
parallelism efficiency, and how much quality you insist on. The point is the
order of magnitude: **the real job is 10⁴-10⁵ GPU-hours.** A free tier is ~30
GPU-hours/week, so this is not a quota problem you can engineer around.

For contrast, the measured recommendation in `docs/decision_memo.md` -- quantise
below ~3.4 bpw so the model is fully resident -- was measured at **~8×** with
**zero** training. Any MoE retrofit has to beat that, not merely exist.

## 9. The gate: what would make me abandon this

The PoC decides it on free hardware. Kill criteria:

1. **At 0.8B, trained `oracle_topk` does not fall below the frozen pilot oracle**
   for the same (E, k). If training cannot create structure at small scale, there
   is no reason to believe 27B's continued pretraining will.
2. **The gap fails to shrink as E grows.** The mechanism should show up more
   strongly with more experts; if it does not, it is noise.
3. **`effective_experts` collapses** and no aux-loss coefficient fixes it.

If any of those hold, the retrofit route is dead and the answer is the one the
earlier work already found: **quantise resident** (free, measured ~8×), and if a
sparse model is genuinely wanted, **pretrain it MoE natively** rather than
converting this bottleneck.

### What the PoC actually found

**E14 — local RTX 4060, 4 layers, E=8 (the mild-sparsity control).**

| k | active | oracle untrained | oracle trained | frozen pilot | ppl untrained | ppl trained |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12.5% | 0.8584 | **0.8501** | 0.8624 | 112.0 | 33.8 |
| 2 | 25% | 0.7520 | **0.7383** | 0.7578 | 73.2 | 30.6 |
| 4 | 50% | 0.5638 | **0.5368** | 0.5710 | 48.9 | 28.4 |

Dense ppl 21.7. `beats_frozen_oracle = True` at every fraction, by 1.0-4.8%.

**E15 — free Kaggle T4s, all 24 layers, k=1, out to the ratio this design needs.**
This is the run that decides it, because §1 requires `E/k ≈ 35` and k=1 is the
config §3 recommends.

| E | active | frozen pilot oracle | oracle trained | router trained | beats frozen | ppl trained |
|---|---:|---:|---:|---:|:--:|---:|
| 8 | 12.5% | 0.8624 | **0.8560** | 0.8951 | yes, by 0.006 | 109.5 |
| 32 | 3.13% | 0.9188 | **0.9355** | 0.9653 | **no** (+0.017) | 159.3 |
| 64 | 1.56% | 0.9277 | **0.9549** | 0.9806 | **no** (+0.027) | 202.5 |

Dense ppl 19.562 (dense-limit check reproduces it at 19.573/19.592/19.587, so the
plumbing is exact). Seed 1 at E=64 gives oracle 0.9551 and router 0.9802 against
seed 0's 0.9549/0.9806 — the inversion is not noise.

**The gate fails as tested.** Criterion 1 holds at E=32 and E=64: trained `oracle_topk`
*rises* above the frozen pilot oracle, and does so by more as E grows — criterion 2
also points the wrong way. The router never gets close to the oracle, and the
oracle itself loses to simply keeping the frozen geometric split. The residual is
dominated by partial-sum scaling, exactly as §5 predicted: with `combine="sum"`,
keeping k of E roughly equal slices leaves about (1 − k/E) of the output no matter
how long you train. Training **adapts the function** rather than creating
extractable sparsity — `vs_original_dense`, the residual against the dense FFN the
experts were partitioned from, actually **rises** at every E (0.9324 → 0.9480 at
E=8; 0.9919 → 0.9974 at E=64) while perplexity falls three orders of magnitude.
E14's "residual ~0.96 at 1.6% active" was an extrapolation across k; E15 **measured
0.9549** there.

End to end, the best trained MoE (E=8, 109.5 ppl) is 5.6x dense and the target
ratio (E=64, 202.5) is 10.4x dense.

**E20 — what the trained 0.8B actually says.** Everything above is perplexity and
weight-space residuals. `experiments/compare_upcycled_08b.py` adds the one a reader
asks about a converted model -- does it still answer the same -- by loading the
4-of-24-layer checkpoint (E=8, k=1) and generating 48 greedy tokens per prompt from
both it and the stock dense model on identical prompts. **0 of 6 continuations are
byte-identical**, the first divergence is at token **0, 1, 2, 3, 0, 3**, and mean
positional token agreement is **0.073**. It is not gibberish: both models answer
`17 * 23` with 391 and both write fluent English, but on the first prompt dense says
"The capital of France is Paris." where the converted model says "...the capital of
the world." On 4 of 24 converted layers the difference is already visible in the
text, not only in the statistic; E15's all-24-layer configurations are the ones a
27B conversion would actually need.

**The dense limit, measured over the whole model.** The caution in §5 -- that
`vs_original_dense` can exceed 1.0 -- has a model-level form. With the *fresh*
partition and every expert active the perplexity is 19.5635 against dense 19.5707
(relative error 3.69e-04), so the partition is the dense layer exactly, which
confirms §2 on the full model rather than on one layer. The same measurement on the
*trained* partition is **1597.7289, 82x dense**, while its sparse top-1 version is
**1.52x dense** (29.7929). Training did not produce a cheaper approximation of the
dense function: it moved the experts to a different function, one that only becomes
usable when most of them are switched off. That is the mechanism §5 predicts, in one
number. (The dense 19.5707 and the fresh post-hoc 99.6753 reproduce
`results/upcycle_poc_rerun.json` exactly; the trained figure is 0.35% from its
recorded 29.6900, and the loaded weights reproduce that run's per-layer residuals
to <=7e-04, so this is the same model evaluated twice rather than two models.)

**E21 -- the corpus-free arm, and what it settles.** An upcycled layer already has a
teacher inside the model (the dense FFN it was partitioned from), so the objective
needs no labels and no pretraining corpus. `experiments/self_teach_poc.py` trains on
that signal alone, from 24 hand-written sentences (197 tokens, against E15's 614,400
corpus tokens). The dense-reconstruction residual goes 0.9249 -> **0.0196** in-sample
and 0.9239 -> **0.8398** held out: a ~98% in-sample improvement transfers as ~9%,
because ~44M trainable parameters are fitting 197 tokens. Quality agrees with the
held-out number rather than the in-sample one -- 4.15x, 29.35x and 3.12x dense
perplexity for the `sum`, `sum_rescaled` and `softmax` arms, against E14's
cross-entropy-trained **1.52x** on the same four layers, with 0 of 6 continuations
identical in every arm. `softmax` is the best of the three and the only arm where the
router receives gradient from the reconstruction loss (at k=1 softmax weights are
1.0, so the arms differ only in whether the router learns). Self-organisation
re-arranges; it does not create the knowledge that is missing.

**Recommendation for *this project* stands: quantise resident; do not retrofit.**
The remaining looseness is the aux loss (33.2 of 64 effective experts,
`load_max_over_mean` up to 9.25 at E=64) and `combine="softmax"` — but neither can
explain a deficit that the *oracle* route also has, since the oracle needs no
router at all.

**However, E15 is not a fair test of the method this document designs, and the
recommendation above is about budget, not about feasibility.** `docs/prior_work.md`
reviews what the field actually did: upcycling works, at ~50% of the original
pretraining compute (Sparse Upcycling); LLaMA-MoE partitioned the FFN exactly as
§2 does and needed **200B tokens** where E15 used **614k**; NVIDIA got +2.3 MMLU
over continued dense training on 1T tokens. E15 also used `combine="sum"` without
the N/k rescaling §5 calls mitigation #1, and partitioned by weights where CMoE
shows the axis should be the **activation pattern** with a shared expert — which is
the published explanation for E15's router collapse. **One of those three items no
longer stands as a defect: E21 measured the optimal output scale for a partition at
~1.1-1.4, not `E/k`, so omitting the rescaling cost E15 nothing — and applying it
would have made the result worse.** That leaves the token budget and the partition
axis as the real objections. Treat §9's gate as satisfied
for *an untrained-partition initialisation*, not for upcycling.

## 10. Limitations of this design

* The expert counts assume the *compute-bearing* definition of active params;
  under strict accounting the goal is impossible and that should be said plainly
  to anyone who asks for "9B active".
* The 0.8B PoC trains on a small corpus (16.4 KB/step x 300 steps) and, in E15,
  all 24 layers of a 0.8B model. It tests the mechanism, not a finished
  conversion, and it cannot speak to whether recovery *improves* with scale —
  though it does remove the reason to assume it would, since the deficit grows
  with E rather than shrinking.
* Only `plan="random"`, `combine="sum"` was trained at the decisive ratios. The
  `clustered` and `interleaved` plans are frozen-only. `combine="softmax"` has since
  been trained once (E21) but only at the mildest cell -- 4 layers, E=8, k=1 -- where
  its forward is identical to `sum`'s and the arms differ only in whether the router
  learns, so it is still untested at the ratios a conversion needs.
* The corpus-free arm (E21) was tested at **one** calibration size, 197 tokens. That
  shows ~44M trainable parameters fit 197 tokens; it does not locate the crossover to
  generalisation, so it closes the idea as *tested* rather than as *possible*.
* Cost figures are FLOP-based estimates with an assumed efficiency, not quotes.
* No benchmark suite or contamination analysis is included; quality claims would
  need `experiments/eval_gate.py` run at scale. A partial, text-level answer now
  exists (E20: 6 fixed prompts, 48 greedy tokens each, 0/6 byte-identical against
  dense, mean positional token agreement 0.073 -- `results/upcycled_08b_vs_dense.json`),
  but six prompts and one perplexity window are not a benchmark, and the gate
  *cannot* be pointed at this model: `experiments/eval_gate.py` scores over HTTP
  (ollama, llama.cpp) while the converted model is a PyTorch state dict with a
  custom partitioned FFN and no GGUF converter. A task-level evaluation of the
  trained MoE needs either a converter or a gate that speaks PyTorch; neither
  exists today.
