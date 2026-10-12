# The solution space: every route to a fast, near-27B-quality model on 14.28 GB

This is a survey, not a result. It exists because the repo has accumulated a lot
of *closed* findings and a few *unrun* ones, and the difference was getting hard
to see. Every route below carries its status and its evidence:

* **MEASURED** — an artifact in this repo, cited.
* **PUBLISHED** — someone else's result, cited, not replicated here.
* **PREDICTED** — this repo's tier model (validated within ~2x, a ceiling
  estimator, not a forecast).
* **UNTESTED** — nobody has run it here. These are the only interesting rows.

The headline: **the original goal is reachable as a bytes-per-token target and
unreachable as a parameter-count target, and the repo has been arguing about the
second one.** §1 reframes it. §2-§5 are the route families. §6 is what to run
next, cheapest first.

---

## 1. First, the constraint, stated so it can be attacked

Decode on this machine is bandwidth-bound. One equation governs everything:

```
   tokens/second  =  bandwidth of the tier holding the weights
                     -------------------------------------------
                          bytes that must be read per token
```

Capacities and bandwidths (**MEASURED**, `results/hardware.json`,
`configs/tiers.json`): GPU 246.8 GB/s, RAM 40.7 GB/s, NVMe **1.5 GB/s**.
Usable fast memory = 3.8 GiB VRAM + 10.2 GB RAM = **14.28 GB**. That is the
whole budget; the disk is ~27x slower than RAM, so *being resident is worth
more than any other single factor* — measured at **3.0x** on identical weights
(19.356 vs 6.450 tok/s, same 35.5B MoE at two sizes).

Two independent problems hide in that equation, and they are routinely conflated:

| | problem | attacked by |
|---|---|---|
| **(a) Residency** | do the weights fit 14.28 GB? | quantisation, tier placement, dropping unused heads |
| **(b) Bytes per token** | how many must be read? | sparsity (active width), speculative decoding, skipping work |

The FFN-partition programme attacks **(b)** and *cannot* attack **(a)**: a
converted model stays ~27B stored (`docs/upcycling_design.md` §2). Quantisation
attacks **(a)** and only incidentally **(b)**. Nothing in the repo's survey
attacks both at once — which is where several unexplored routes live.

### The arithmetic that reframes the goal

**MEASURED/derived** (`src/moeme/accounting.py`, verified against the safetensors
headers by `tools/verify_model_geometry.py`):

| quantity | value |
|---|---:|
| total parameters | 27,781,427,952 |
| FFN | 17,112,760,320 (61.6%) |
| non-FFN text backbone ("the floor") | 9,783,238,144 |
| — Gated DeltaNet | 5,562,044,928 |
| — attention | 1,677,721,600 |
| — embedding + LM head | 2,542,796,800 |
| — norms | 674,816 |
| floor excluding the embedding gather | 8,511,839,744 |

So: **9B active is impossible** by routing (it needs −0.78B of FFN) and 12B active
needs 2.22B of FFN = **13.0%** of the FFN width. That is the number that kills
route C1 (§4), and it is a *parameter* number.

The byte version of the same goal is gentler. Run the same targets through the
tier model and they are all resident and all fast (**PREDICTED**, using the
tested estimator):

| active | at 4.9 bpw (Q4-class) | at 3.4 bpw | at 2.8 bpw |
|---|---:|---:|---:|
| 9B | 5.51 GB/token → **19.3 tok/s** | 3.83 GB/token → 64.5 | 3.15 GB/token → 78.3 |
| 12B | 7.35 GB/token → **10.3 tok/s** | 5.10 GB/token → 24.0 | 4.20 GB/token → 51.3 |

(Both bolded rows are RAM-bound, which is why the choice of bpw matters so much
there; the 2.8 bpw rows are GPU-bound because 3-4 GB nearly fits the 3.8 GiB
VRAM. These are the same numbers the decision memo quotes as "~19 tok/s" and
"~10 tok/s" — they are at Q4-class bits, not at 2.8 bpw.)

So nothing about the 9.78B floor prevents a 12B-active model at usable speed. What
prevents it is *quality at 13% FFN width* — a different problem, with a different
set of routes (families C and D).

---

## 2. Family A — make *this* 27B resident and fast (no training)

### A1. Aggressive uniform quantisation — **half-measured, and the other half is the critical path**

| quant | bpw | size | streamed/token | predicted | regime | quality measured? |
|---|---:|---:|---:|---:|---|---|
| Q2_K | 2.8 | 9.7 GB | 0.00 | 6.44 tok/s | resident | **no** |
| Q3_K_M | 3.4 | 11.8 GB | 0.00 | 4.85 tok/s | resident | **no** |
| Q4_K_S | 4.4 | 15.3 GB | 1.00 GB | 1.07 tok/s | disk-bound | no |
| Q4_K_M | 4.9 | 17.0 GB | 2.74 GB | 0.48 (measured **0.561**) | disk-bound | yes (gate + ppl) |

The speed side is settled: below **4.112 bpw** the model stops streaming and the
prediction is ~9x the measured Q4 rate. The quality side is **empty**. The
feasibility report's own §8.1 named this as the first experiment ("*This is the
whole decision*") and `kaggle/README.md` still lists "the resident-quantisation
check on the real 27B" as worth running next. **UNTESTED.**

Why it dominates every other cheap option: it needs one download and no code, and
it decides whether "resident 27B" is viable at all. Related cost already measured
elsewhere: Q2_K on a *matched 9B pair* cost **+10.1% perplexity**
(7.7128 vs 7.0029; ~1.1 sigma, and **that pair is not backed by a committed
artifact** — the committed number is the 35B's own Q2_K at 12.7113 ± 0.809),
while a *pass/fail gate* could not see the difference at all (17/19 vs 16/19, with
**13 of 19 responses differing** between the quants). So the ladder must be run
with perplexity on identical tokens *and* a task gate, and the gate alone must not
be trusted to arbitrate.

### A2. Sensitivity-aware mixed quantisation — **PREDICTED only**

The page's own envelope view shows this: a per-module assignment that lands
between Q3_K_M (fits, 3.4 bpw) and Q4_K_S (misses, 4.4 bpw) — a ~1.0 bpw hole
exactly where the threshold sits. The atlas predicts three recipes resident
(75/15/10 at 4/3/2 bpw → 12.68 GB; IQ-style 4/2b 70/30 → 11.81 GB). None has ever
been built or scored. The unmeasured input that would make it principled is
**per-module sensitivity**: which projections tolerate 2 bits (candidates: the
FFN's `gate`/`up`, the GDN `in_proj_a`/`b`, the vision tower) and which do not
(candidates: `down_proj`, attention `o_proj`, the embedding). **UNTESTED.**

### A3. Quantisation-aware training at 2–3 bits — **PUBLISHED, affordable-ish, untested here**

QAT is the standard answer to "PTQ falls apart below 3 bits". Recent results:
2-bit QAT recovers **3.5–4.3 percentage points** over strong QAT/PTQ baselines
(QUASAR, [2608.13966](https://arxiv.org/html/2608.13966v1)); 2-bit Qwen3-8B QAT
beats PTQ baselines by ~50% on average (ICLR 2026); QAT in general recovers up to
96% of accuracy degradation and 68% of perplexity degradation (PyTorch blog,
2024). This is *not* pretraining — it is a fine-tune — so it is orders of
magnitude cheaper than family D, though still well beyond a free tier for a 27B.
It is the only route that could make **Q2_K-class size with Q4-class quality**,
which is exactly the (a)+(b) combination nothing else offers. **UNTESTED, cost
unknown — and worth costing, because it is the highest-ceiling affordable idea
in the survey.**

### A4. MTP self-speculative decoding — **UNTESTED here, and the cheapest large win**

The released checkpoint **ships a multi-token-prediction head**: 424,699,392
parameters, measured and accounted for in `src/moeme/accounting.py`, and the atlas
labels it "draft". Speculative decoding is the right tool for a *bandwidth-bound*
decode for a structural reason: verifying k drafted tokens takes a single forward
pass over the weights, so the bytes-per-token of §1 buy k tokens instead of one.
Drafting from the model's own MTP head avoids what usually kills speculation on a
small machine — a second model competing for the 3.8 GiB of VRAM.

Support state, **checked rather than assumed**:

- llama.cpp has MTP implemented behind `--spec-type draft-mtp`, with the
draft/verify loop living in `common/speculative.cpp`
([issue #27469](https://github.com/ggml-org/llama.cpp/issues/27469), Aug 2026 —
opened precisely *because* the feature is reachable from the CLI but not from the
exported C API that bindings use).
- Reported speedups cluster at **~1.5-2.9x**, not one number: ~2.9x on the
Qwen3.5/3.6 family (community report), ~1.5-2x for Qwen3.5-9B-MTP (unsloth model
card), +30-50% throughput (NVIDIA developer forum), and 67→120 tok/s on a 27B
combining MTP with n-gram speculation.
- There is a llama.cpp discussion about **this exact model** (#27164), whose
finding is a caveat worth having before spending an evening: the base
Qwen3.8-27B was *already* broken on CUDA there before MTP was involved.

Two practical caveats. (1) **Every speed number in this repo goes through
ollama**, and MTP lives in llama.cpp's `common/` layer; the open question is not
"does llama.cpp support MTP" (it does) but "can this repo's runtime turn it on"
— check that first, because it decides whether the experiment is an afternoon or
a port. (2) The draft need not be the MTP head: issue #20039 notes that
Qwen3.5-0.8B is "a strong draft model that can greatly accelerate
Qwen3.5-27B", and this repo already has that model and its artifacts.

Note how it composes: MTP multiplies whatever rate you already have, so it is
nearly worthless on the streaming Q4 model (0.561 → ~1 tok/s, still unusable) and
decisive on a resident one (4.85 tok/s **PREDICTED** → order 7-14 tok/s). That
combination is an *inference*, not a prediction — the tier model has no term for
speculation — and it needs measuring. A4 is only interesting *after* A1, and
together they are the whole "usable resident 27B" story. This is the largest
untried idea in the repo.

### A5. Free parameter removal and KV budget

The vision tower (**460,730,096**) and the MTP head (**424,699,392**) are 885M
parameters ≈ **0.5 GB at 4.9 bpw** — worth having when the threshold is 4.112 bpw,
and the vision tower is dead weight for a text-only workload (keep the MTP head if
A4 works; it pays for itself). The tier model already takes `kv_bytes`, and
`results/moe_placement.json` benches with q8_0 KV, so KV is partly handled — but
nothing in the repo has priced *context length* against residency, which is the
difference between a model that fits at 4k and one that fits at 32k. **UNTESTED as
a budget line.**

### A6. Explicit placement beats the runtime default — **MEASURED on the MoE only**

`ollama`'s `num_gpu = -1` offloads by *whole layer*. `-ncmoe` splits at *expert*
granularity and keeps every layer's attention on the GPU: **1.42x faster decode
and ~10x faster prompt processing on identical weights** (`results/moe_placement.json`),
with `--n-cpu-moe 36` freeing ~4 GiB. The repo draws the right moral itself:

> "Any conclusion of the form 'this machine cannot run X faster' has to be checked
> against an explicit placement first."

That check has been run for the 35B MoE and **never for the dense 27B**, whose
0.561 tok/s figure comes from a run with `num_gpu = -1`. Part of that 0.561 may be
placement, not physics. **UNTESTED.**

### A7. The SSD is anomalously slow — **flagged, unresolved**

NVMe measured at **0.82-1.9 GB/s**, which is low for the class. Both the
feasibility report (risk table) and `make doctor` flag it. Every streaming design
in the survey is hostage to this number; a throttled PCIe link would change the
calculus for the whole disk-bound region. Cheap to check, never resolved.

---

## 3. Family B — run a model that *already* has sparse activation

### B1. The 35.5B-A3B at Q2_K — **MEASURED, and the best current answer**

13.0 GB, resident, **19.356 tok/s** — ~35x the dense 27B, and a *larger* model.
Quality on the repo's own gate: **17/19 vs the dense 27B's 13/19** (dense had 4
capped rows; MoE wall 62.7 s vs 3958 s). The honest caveats are already recorded:
the gate cannot see quantisation damage, the Q4_K_M half of the same model could
not even be scored on this host (OOM), and sparsity at equal bits is not free of
cost. But this is the only configuration in the repo that is simultaneously
resident, fast, and measurably competent.

### B2. Selectivity-weighted expert quantisation — **PUBLISHED, untested here, and B1's obvious upgrade**

If experts are not used equally, they should not be quantised equally. Assigning
precision by *how often an expert is actually selected* is reported to cut
effective VRAM **30–50% without retraining** (vendor write-up,
[Spheron](https://www.spheron.network/blog/dynamic-expert-quantization-moe-offloading-gpu-cloud/) —
treat as a claim, not a measurement). Applied here it would mean: take the MoE
whose Q4_K_M is 22.0 GB (does **not** fit, 6.450 tok/s) and shrink it *below 14.28
GB* by quantising the rarely-selected experts harder — buying Q4-class quality on
the experts that matter at Q2-ish total size. The input is an **expert-usage
histogram**, which costs one eval pass with the harness already in
`experiments/eval_gate.py`. **UNTESTED.** This is the most promising route that
needs no training at all.

### B3. Expert caching and prefetching — **PUBLISHED; the repo diagnosed it and stopped**

The repo's own analysis nails the mechanism: with 256 experts and top-8, "the
working set over a generation is spread across a large fraction of the experts,
so the hit rate is far from 100%… **offloading is therefore not a switch that
makes the cold part free; it is a hit-rate problem**" — and it measured the
consequence (~11.8 GB/s effective on the non-resident MoE, i.e. a RAM/disk mix).

That is precisely the problem the published systems solve, and none of them was
tried: **MoE-Infinity** (sparsity-aware expert cache on personal machines),
**Mixtral-offloading** (LRU expert cache + speculative prefetch of the next
layer's experts), **Pre-gated MoE** (predict the next layer's expert set and
prefetch), **fMoE**/**CommitMoE** (finer-grained offload). Upstream llama.cpp now
has an **expert cache** discussion in flight (hot experts pinned in VRAM). A
hit-rate-aware cache is what turns the 22 GB / 6.45 tok/s configuration from
"does not fit" into "mostly fits", and it stacks with B2. **UNTESTED here.**

---

## 4. Family C — reduce active width at inference, without training

### C1. Post-hoc FFN top-k selection — **DISPROVEN as a speed win**

**MEASURED** (`results/branch_poc_08b.json`, k=1 tile family, 4 layers of
Qwen3.5-0.8B, dense val ppl 15.4803):

| active FFN width | oracle residual | ppl |
|---|---:|---:|
| 12.5% | 0.7104 | 54.00 |
| 25% | 0.6339 | 45.63 |
| 50% | 0.4975 | 26.02 |
| 100% | 0.0000 | 15.48 |

At the only width that buys the speed (13% → 2.22B active for a 12B target), the
*best possible* router leaves ~73% of the layer output wrong. Training does not
rescue it — E15 measured the residual *rising* under training at E=64/k=1
(**0.9277 frozen → 0.9549 trained**), and the free-tier Kaggle rerun reproduced
the direction (0.9335 → 0.9663, `effective_experts` 33.16 of 64). The router is
not the problem: the oracle-vs-learned gap is ≤ 0.138 anywhere sparse. Two further
measured obstacles: partitioning costs 2.1-2.4x (E=32) to 2.9-3.9x (E=128) at
batch 1, and top-2-of-32 costs the same as all-32, so sparsity recovers none of it.

**PUBLISHED diagnosis for why:** post-hoc MoE success stories (MoEfication: 10-30%
of FFN params at >95% performance, 2x speedup at 25%, no retraining) were built on
**ReLU** FFNs, where inputs naturally activate a tiny fraction of neurons. SwiGLU
"does not bring natural sparsity" — so this model has far less to exploit. **C1 is
closed.** Do not reopen it without a new mechanism.

### C2. Activation-pattern restructuring with a shared expert — **UNTESTED, cheap, modest ceiling**

CMoE ([2502.04416](https://arxiv.org/abs/2502.04416)) partitions by **activation
pattern** rather than weight similarity, puts always-active neurons in an
always-on *shared expert*, and builds the router analytically from neuron
statistics — "minutes of processing and 2k-sample fine-tuning", and it directly
answers the failure E15 recorded (`eff_experts` 33 of 64, `load_max_over_mean`
9.25 — exactly the router-collapse symptom CMoE predicts from weight-space
clustering). E15's own outlier finding (layer 23: participation ratio 0.185 vs
0.6-0.9 for layers 0/3/8) is independently confirmed by LLaMA-MoE ("expert
partition on the latter layers' FFNs may bring further improvements").

Be clear-eyed about the ceiling: CMoE's own headline is **up to 1.17x**. A 17%
speedup does not close a 9x bytes-per-token gap. This route is worth running on
the 0.8B **to test the mechanism** (the machinery is already in
`src/moeme/ffn.py`) and to decide whether *any* post-hoc route retains value — not
because it will solve the problem.

### C3. Depth allocation / layer skipping — **UNTESTED, and the only conditional-computation family that attacks the real bottleneck**

Mixture-of-Depths ([2404.02258](https://arxiv.org/html/2404.02258v1)) learns to
route *tokens* around whole blocks, reaching equal-or-better log-probability at a
smaller FLOP budget. Most conditional-computation work optimises FLOPs, which is
the wrong axis here (decode is bandwidth-bound). But **skipping a layer skips its
bytes**, so depth allocation is the one member of this family that moves the
numerator in §1. The repo has a lead: sparsity concentrates in late layers (layer
23's participation ratio 0.185; LLaMA-MoE's deep-layer routing preferences). The
question to ask is narrow and cheap: *does skipping or thinning a few late layers
cost measurable quality?* If yes for some subset, that is a direct bytes-per-token
win that stacks with A1 and A4. **UNTESTED, and nobody in the repo has framed it
as a bytes question.**

### C4. Effective-cost reductions (orthogonal, all cheap)

Speculative decoding (see **A4** — the MTP head is already in the checkpoint), KV
cache quantisation (q8_0 already used in benches; more is available), and shorter
contexts. These multiply or divide the §1 equation without touching weights, and
they stack with every route above.

---

## 5. Family D — change the weights with training

**The compute wall, stated once.** Upcycling costs **+10% to +60% of the original
pretraining budget**; LLaMA-MoE's *3.5B-active* run was ~112 A100-80G for ~3 days
≈ **10⁴ A100-hours ≈ $20k**; the honest order of magnitude for the real job is
**10⁴-10⁵ GPU-hours**. A free tier is ≈30 GPU-hours/week of T4s. That is ~50
years, and the project's budget is £20.

### D1. Upcycle this 27B into an MoE — **CLOSED on budget, not on merit**

The literature says it works and is modest: Sparse Upcycling beats dense given
~half a pretraining budget; NVIDIA's upcycling beats continued dense training on
the *same* 1T tokens by **+2.3 MMLU** (67.6 vs 65.3); Drop-Upcycling gets 5.9B-active
≈ 13B dense at ~1/4 the FLOPs. None of it is free, and none of it is startable
here. **CLOSED** — and importantly, closing it does *not* close families A, B or C.

### D2. The repo's own retrofit — **FAILED at its gate, but it was not a fair test**

`docs/prior_work.md` shows four concrete mismatches: 614k training tokens against
a stated 15-20B-token floor to draw any conclusion (**~30,000x short**); the
E/k output-rescaling fix named in the design doc and required by the literature
**never implemented**; `combine="sum"` where NVIDIA found softmax-then-topK
consistently better; and partitioning by weights/random rather than by activation
pattern. Its own conclusion should read "this initialisation, this gating, this
budget do not beat the frozen ceiling" — not "upcycling cannot work". The fair
version is still unaffordable, so this changes the *interpretation*, not the
decision.

### D3. Task-scoped distillation — **UNTESTED, and the honest way to buy the goal**

Nobody in the repo's survey priced distillation of the 27B into a ~3B student for
*a specific set of tasks* rather than for general capability. A general student
needs pretraining-scale data and compute; a *scoped* one (this repo's own 19-task
gate, coding, formats, instruction-following on a fixed corpus) is a fine-tune-scale
project that a free tier can plausibly attempt. It would not be "a 9B-active 27B";
it would be a small fast model that scores well on the tasks you actually run —
which, given the dense 27B scores 13/19 at 0.561 tok/s, may be the real win
available. **UNTESTED.**

### D4. Continue-train a small model on your own corpus — **UNTESTED, cheapest training**

The lowest-risk training route: take the 9B (24.6 tok/s, 6.6 GB) or the 0.8B, and
continue-train on the domain you care about. It cannot reach 27B generality, and
it is the only training family with a plausible path to completion at this budget.

---

## 6. What to run next, cheapest first

| # | experiment | cost | decides |
|---|---|---|---|
| 1 | **Quant-quality ladder on the real 27B**: Q3_K_M / IQ3 / Q2_K vs Q4_K_M, perplexity on identical tokens *and* the 19-task gate | one download | whether "resident 27B" exists at all (§2 A1) |
| 2 | **MTP self-speculative decode** on the resident model | check the runtime can enable it, then one bench | 1.5-2.9x on top of #1 — largest untried win; support state now verified (§2 A4) |
| 3 | **Explicit placement sweep** (`-ngl`/`-ncmoe`) on the dense 27B | one bench | how much of 0.561 tok/s was the runtime's fault (§2 A6) |
| 4 | **Expert-usage histogram → selectivity-weighted quant** for the 35B-A3B | one eval pass + a quant | Q4-class quality under 14.28 GB (§3 B2) |
| 5 | **Sensitivity map → mixed-quant recipe** (build the predicted 75/15/10) | one quant + eval | whether the 3.4-4.4 bpw hole can be split (§2 A2) |
| 6 | **Depth/thinning probe**: does skipping late layers cost quality? | few evals | the only conditional route that cuts bytes (§4 C3) |
| 7 | **Expert cache** (LRU + prefetch) over the 22 GB MoE | implementation | whether hit-rate engineering beats a smaller quant (§3 B3) |
| 8 | **CMoE-style restructuring on the 0.8B** | minutes of compute | whether any post-hoc route survives SwiGLU (§4 C2) |
| 9 | **Cost a 2-bit QAT fine-tune** | analysis, then maybe | the only (a)+(b) route with a real ceiling (§2 A3) |
| 10 | **Task-scoped distillation** | free tier, weeks | a fast model that scores on *your* tasks (§5 D3) |

**Closed, and should not consume more effort:** post-hoc FFN top-k as a speed win
(C1); full upcycling on this budget (D1); "9B active with near-27B quality" read as
a *parameter count* (§1).

---

## 7. Where this survey is weak

* **Provenance is mixed.** Everything marked MEASURED is an artifact in this repo;
  everything under PUBLISHED is someone else's number and none of it is
  replicated here — the dynamic-expert-quantisation 30-50% figure is a vendor
  blog, and the MTP 1.5-2.9x range is community reports and a developer forum,
  not papers.
* **The FFN oracle curve is from 0.8B tile runs** (4 layers), not the 27B. The
  repo flags that its *structure* transfers while the magnitudes may not. C1's
  closure is solid in direction; the exact 0.71 at 12.5% could move with scale.
* **Two ideas here were not in any repo document before this one**: MTP
  self-speculation using the checkpoint's own head, and depth/layer skipping
  reframed as a *bytes* question. They are inferences from the artifacts plus
  published work, not measurements, so treat them as leads.
* **Costs for A3/D3/D4 are order-of-magnitude reasoning**, not quotes.
* The survey was assembled with web search available; the rest of the repo was
  built without it, so this is the second document in `docs/` (after
  `prior_work.md`) whose claims partly rest on secondary sources.
