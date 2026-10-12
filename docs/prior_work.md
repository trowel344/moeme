# Prior work: dense → MoE conversion, and what it says about E15

**Why this document exists.** E15 concluded that "the retrofit route is dead on its
own stated gate." That conclusion is **too strong**, and this is the literature
that shows why. The repo's own experiment was not a fair test of upcycling: it
trained for 614k tokens, used the one gating scheme the literature identifies as
wrong, and never implemented the output-rescaling fix that both the paper trail
and this repo's own design doc say is essential.

Everything below is cited. Read `docs/experiment_log.md` E15 first for what was
actually measured.

---

## 1. The result the handoff was remembered for: upcycling works, modestly

| Work | Method | Continued-pretrain budget | Outcome |
|---|---|---|---|
| **Sparse Upcycling** (Komatsuzaki et al., 2022, [2212.05055](https://arxiv.org/abs/2212.05055)) | Duplicate the dense FFN into E experts, add a router | **~50% of the original dense pretraining cost** | "significantly outperform their dense counterparts on SuperGLUE and ImageNet"; also beat from-scratch MoE given 100% of the dense pretraining budget |
| **LLaMA-MoE** (Zhu et al., EMNLP 2024, [2406.16554](https://arxiv.org/abs/2406.16554)) | **Partition** the FFN into E experts (this repo's method) + continual pretrain | **200B tokens**, 112 × A100-80G | LLaMA-MoE-3.5B (4/16): avg **57.7** vs Sheared-LLaMA-2.7B 56.4, OpenLLaMA-3B 55.6 |
| **NVIDIA upcycling** (He et al., 2024, [2410.07524](https://arxiv.org/abs/2410.07524)) | Virtual-group init + weight scaling | **1T tokens** | Nemotron-4 15B: MMLU **67.6%** upcycled vs **65.3%** continued dense on the *same* 1T tokens |
| **Drop-Upcycling** (Nakamura et al., ICLR 2025, [2502.19261](https://arxiv.org/abs/2502.19261)) | Upcycle + statistically re-initialise part of the weights | Hundreds of billions+ | 5.9B-active ≈ **13B dense** at ~1/4 the training FLOPs |

So "someone did it and it was kind of ok" is exactly right. The wins are real but
**modest**: LLaMA-MoE beats dense models of *similar active parameters*, not the
original 7B dense model it came from (their final loss is explicitly higher than
LLaMA-2-7B's). NVIDIA's headline is **+2.3 MMLU points** over continued dense
training. Sparse Upcycling's is a SuperGLUE gain for ~half a pretraining budget.
Nobody gets a free lunch; they get a real but bounded one, and they pay for it.

**The cost, in one line:** Sparse Upcycling puts it at **+10% to +60% of the cost
of training the original dense network**. For a 27B-class model that is 10⁴–10⁵
GPU-hours. LLaMA-MoE's own 3.5B-active run is ≈112 A100s × ~3 days ≈ **10⁴
A100-hours** (~$20k). Not a laptop job, but two orders of magnitude cheaper than
pretraining a new 27B — which is the actual argument for upcycling.

## 2. The cheap version exists, and it is closer to this repo than anything else

**CMoE — "Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis"**
(Pei et al., Huawei + CUHK, [2502.04416](https://arxiv.org/abs/2502.04416)) is the
direct answer to "can this be done *without* hundreds of billions of tokens."
It restructures a dense FFN into a MoE using **only a small calibration set**,
with **no router training at all**:

- Neuron activation frequency is **bimodal** on LLaMA-2-7B: most neurons activate
  conditionally (rate ≈ 0.07), while a subset activates on **nearly every input**
  (rate → 1.0).
- The method puts the always-active neurons into an always-on **shared expert**
  and clusters the conditional ones into routed experts **by activation pattern**,
  not by weight similarity. The router is then built analytically from
  representative-neuron statistics.
- Result: **up to 1.17× speedup** with "minutes of processing and 2k-sample
  fine-tuning."

Their critique of prior work is a precise description of what this repo did:

> existing methods either rely on **weight-based clustering** that treats all
> neurons uniformly regardless of their activation behavior

and their prediction of the failure mode is exactly the symptom E15 recorded:

> high-frequency neurons may be scattered across different routed experts. This
> forces the router to activate most experts regardless of input, undermining the
> sparsity that MoE relies on for efficiency.

E15 measured `eff_experts = 33.2 of 64` with `load_max_over_mean` up to **9.25**.
That is not a mysterious training failure — it is the predicted consequence of
partitioning by weights while ignoring activation frequency.

## 3. What this repo did that the literature says is wrong

Four concrete mismatches, in order of how much they matter:

1. **The token budget.** E15 trained 300 steps × 4 × 512 = **614k tokens**.
   LLaMA-MoE states: *"models should be trained for at least 15–20B tokens to
   properly conclude"* which partition method is better — they used **200B**.
   E15 is **~30,000× below** even the "enough to form an opinion" floor. This is
   the single biggest problem: E15 is not a measurement of upcycling, it is a
   measurement of an *initialisation*.
2. **No output rescaling.** With `combine="sum"`, keeping k of E equal slices
   leaves ≈(1 − k/E) of the output. Both fixes in the literature target exactly
   this: LLaMA-MoE rescale expert outputs by N/k and report it *"provides
   significantly better performance"*; NVIDIA scale expert weights by
   ∛(E·G²/T) for **1.5% better loss**, and note that without it the naive
   fine-grained result *"did not converge to the original loss."* This repo's
   `docs/upcycling_design.md` §5 listed "scale expert init by E/k" as mitigation
   #1 and **never implemented it** in E14 or E15.
3. **The gating scheme.** E15 used `combine="sum"` (an unweighted partial sum).
   NVIDIA found **softmax-then-topK beats topK-then-softmax** consistently, and
   NLP MoE practice is a *weighted* sum with probabilities. The repo tested a
   configuration the literature says is the weaker one.
4. **Partitioning by weights, not activations.** Contiguous / interleaved /
   random / weight-k-means. Per CMoE, the informative axis is the activation
   pattern, with always-active neurons separated out. The repo's finding that
   `clustered` (weight k-means) degenerates into one giant expert is consistent
   with CMoE's point that weight-space clustering is the wrong tool — and E15
   then trained only on `plan=random`.

**One thing E15 got right, independently confirmed.** E15 noted that layer 23
behaves completely differently from layers 0/3/8 — participation ratio 0.185 vs
0.6–0.9, top-1 energy share 0.82, oracle residual 0.431 vs 0.68–0.85. LLaMA-MoE
found the same thing from the routing side:

> deep layers have more routing preferences than shallow layers … Based on this
> finding, **expert partition on the latter layers' FFNs may bring further
> improvements.**

Two independent pipelines, same conclusion: sparsity lives in the late layers.
That is now a supported lead rather than a hunch from one model.

## 4. What this does *not* change

- **The 9B-active arithmetic still holds.** The non-FFN backbone is 9,783,238,144
  params read every token. No amount of upcycling quality touches it, so
  "~27B total, ~9B active" remains unreachable *from this backbone*. Upcycling
  changes how good a converted model can be, not what active-parameter floor is
  reachable.
- **The cost is still 10⁴–10⁵ GPU-hours.** A free tier (≈30 GPU-hours/week, and
  T4s rather than A100s) is still ~50 years. Better recipes do not make it free.
- **MoE-isn't-automatically-faster-at-batch-1** was overstated (see §5).
- **The resident-quantisation path is unaffected** and remains the cheapest real
  win. It is not in competition with upcycling: one is a download, the other is
  a training campaign.

## 5. Correction: the batch-1 claim was too strong

E15's `.`/`real_ffn.json` microbenchmark showed top-2 of 32 timing the same as
all-32, and this was reported as "MoE is a datacenter-batch technique, not a
laptop-latency technique." That reading is wrong, and the measurement itself
explains why.

Dense reads 535 MB of layer-1 weights in 2.443 ms = **219 GB/s**, ≈89% of this
machine's measured 246.8 GB/s — so the dense case *is* bandwidth-bound. Top-2
should therefore read ~33 MB (≈0.15 ms) and instead took 5.93 ms. The gap is
~5.8 ms of **dispatch overhead from naive per-expert `index_select` + 32 small
GEMMs**, not a property of MoE. NVIDIA's own guidance says the opposite of what
was claimed: *"At batch size 1 decoding is bound by available memory rather than
compute, and this is where MoE performs well since it reads fewer weight bytes."*

So the honest statement is: **at batch 1 MoE should save bytes read, and this
repo's unoptimised dispatch destroyed that saving.** A real implementation
(grouped/fused GEMM) is required before any claim about MoE latency on this
hardware. The open question for the laptop is different and narrower — whether 64
scattered expert reads per token can be served under a 0.8–1.9 GB/s SSD without
destroying the win, which is a *locality* problem, not a FLOPs problem.

Related: **MoEfication** (Zhang et al., ACL Findings 2022,
[2110.01786](https://arxiv.org/abs/2110.01786)) "can conditionally use **10% to
30% of FFN parameters while maintaining over 95% original performance**", with a
**2x speedup at 25% of FFN parameters** — *without* retraining. But it works on
**ReLU** FFNs, where *"most inputs only activate a tiny ratio of neurons"*. LLaMA-MoE
note plainly that *"the Swish-based FFN does not bring natural sparsity as ReLU in
T5 and BERT."* CMoE's activation histograms show SwiGLU is not sparsity-free
either, just weaker — which is why the shared/routed split matters there and
weight-clustering does not work here. This is the sharpest available explanation
for why E15's oracle residual could not beat 0.86 at 12.5% active: a SwiGLU FFN
has far less to exploit than the ReLU models every post-hoc MoE success was built
on.

## 6. Revised verdict

**E15's conclusion should be read as: "this initialisation, this gating, and this
budget do not beat the frozen ceiling."** Not as "upcycling cannot work." The
literature says it can, at a price this project cannot pay, and the specific
failures E15 recorded have published, named causes.

The cheapest genuinely-untested experiment is **CMoE-style restructuring**: an
activation-pattern partition with a shared expert and an analytical router needs
only a calibration set and minutes of compute, and it can be run on the 0.8B model
*locally* with the machinery already in `src/moeme/ffn.py`. It targets the one
thing E15 left unexplained — the router collapse.

## 7. Sources

- Komatsuzaki et al., *Sparse Upcycling*, [arXiv:2212.05055](https://arxiv.org/abs/2212.05055)
- Zhu et al., *LLaMA-MoE*, EMNLP 2024, [arXiv:2406.16554](https://arxiv.org/abs/2406.16554)
- He et al., *Upcycling Large Language Models into Mixture of Experts*, [arXiv:2410.07524](https://arxiv.org/abs/2410.07524)
- Nakamura et al., *Drop-Upcycling*, ICLR 2025, [arXiv:2502.19261](https://arxiv.org/abs/2502.19261)
- Pei et al., *Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis* (CMoE), [arXiv:2502.04416](https://arxiv.org/abs/2502.04416)
- Zhang et al., *MoEfication*, ACL Findings 2022, [arXiv:2110.01786](https://arxiv.org/abs/2110.01786)
- Liew et al., *Scaling Laws for Upcycling MoE Language Models*, ICML 2025
- NVIDIA, *Dense vs. MoE Models: Active Parameters, Throughput, and When to Choose Each* (developer blog, Sep 2026)

**Limitations of this review.** Not every source was read in full: LLaMA-MoE,
NVIDIA and CMoE were read as full HTML text (LLaMA-MoE and NVIDIA truncated at the
tail), while Sparse Upcycling, Drop-Upcycling, MoEfication, the upcycling scaling
laws and the NVIDIA developer blog were read at **abstract level only**, with the
"+10% to +60%" figure for Sparse Upcycling coming from a search snippet of the
paper body rather than text I read. Everything cited is a primary source or the
publisher's own abstract; nothing is a replication. The GPU-hour figures in §1 are
my arithmetic from the papers' stated token counts and GPU counts, not quotes. Web search was available for this
review (unlike the rest of the repository, which was built without it), so this is
the one document in `docs/` whose claims rest on secondary sources rather than
artefacts measured here.
