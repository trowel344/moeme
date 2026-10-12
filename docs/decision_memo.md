# Decision Memo

**Date:** 2026-10-10 · **Hardware:** RTX 4060 Laptop 8 GB / i7-13620H / 15.4 GiB /
Micron 3400 1 TB NVMe, Ubuntu 24.04.4
**Spend to date:** £0.00 · **Compute used:** the target laptop, plus one 3.25 h
session of free Kaggle T4 GPU for the trained-MoE experiment (E15).

---

## 1. Recommendation in one paragraph

**Do not convert Qwen3.8-27B into a sparse MoE.** It is proven impossible to hit
9B active that way (the untouched backbone is already 9.78B params), and it is
separately proven that post-hoc FFN sparsification of this dense model has no
usable operating point (a *perfect oracle* router loses 42–81% of the layer
output at 12.5% active). **Instead, run the existing dense model at aggressive
quantisation so it fits fully resident**, which requires no training, no
architecture change and no money, and should move decode from the measured
0.561 tok/s to roughly 4–6 tok/s. Then measure whether the quality holds up.
That is one download and one afternoon.

## 2. The evidence behind that

| Finding | Number | Where |
|---|---|---|
| 27B backbone (non-FFN) is already larger than the 9B-active target | 9,783,238,144 params | `docs/source_validation.md`, executable test |
| FFN share of the model | 61.6% (17,112,760,320) | safetensors headers |
| Post-hoc sparsification operating point | oracle residual 0.42–0.81 at 12.5% active | `results/pilot_curve.txt` |
| Is the router the bottleneck? No | oracle vs learned gap <= 0.033 | same |
| Does **training** the MoE reach the ceiling? No | at E=64/k=1 the residual *rises*: 0.9277 frozen → 0.9549 trained | `results/kaggle_upcycle_poc.json` (E15) |
| ...and it is not seed noise | seed 1 gives 0.9551 / router 0.9802 | `results/kaggle_upcycle_poc_seed1.json` |
| Trained MoE at the target ratio is 10.4x dense ppl | 202.5 vs 19.6 | same |
| Does latent expert structure exist? No | participation ratio ≈ E (127 of 128) | `results/real_ffn.json` |
| Partitioning is exact on real weights | worst rel. L2 error 5.1e-07 | `results/real_ffn.json` |
| Partitioning is *free* however? No | 2.1-2.4x (E=32) to 2.9-3.9x (E=128) slower at batch 1 across two runs; top-2 of 32 costs the same as all-32 | `results/real_ffn_rerun.json` |
| Target model today | **0.561 tok/s**, prefill 1.6 tok/s | `results/inference_27b.json` |
| Usable fast memory | 3.8 GiB VRAM + 10.2 GB RAM = 14.3 GB | `configs/tiers.json` |
| ...predicted independently from hardware constants | 0.48 tok/s | `tools/residency_table.py` |
| Residency makes it viable | Q3_K_M 11.8 GB resident -> ~4.9 tok/s predicted | `results/residency_table.txt` |

## 3. Three paths, ranked

### Path A — aggressive quantisation to full residency (**recommended**)

*What:* download Qwen3.8-27B at Q3_K_M / IQ3 / Q2_K; measure decode rate, peak
VRAM/RAM, and held-out task scores against the existing Q4_K_M baseline.

*Why it is the right first move:* the measured bottleneck is **storage
streaming, not compute**. At Q4_K_M the model is 17.0 GB against ~14.3 GB of
usable fast memory (3.8 GiB VRAM + 10.2 GB RAM), so 2.74 GB/token comes off an
SSD that delivers 1.5 GB/s. Below ~3.4 bpw the model fits entirely in VRAM+RAM
and that cost disappears.

*Cost:* £0, ~10–15 GB download, a few hours of work. *Expected outcome:*
~4–6 tok/s if the tier model is right, at some quality cost that must be
measured rather than assumed.

*Gate (decide with this, don't hope):* >= 4 tok/s **and** >= 95% of the Q4_K_M
baseline on coding and reasoning held-out tasks. Fail either and go to Path C.

*Risk:* Q3/Q2 can hurt coding and logic disproportionately. This is the real
risk of the whole project and the reason step 3 below (build the eval set) is a
prerequisite, not a nice-to-have.

### Path B — post-hoc MoE conversion of Qwen3.8-27B (**declined, on evidence**)

*Why declined:* two independent blockers, both measured.
1. **Arithmetic:** 9B active is unreachable because the non-FFN backbone is
   9.78B active params. Reaching it means retraining attention, Gated DeltaNet
   and the embedding/head — i.e. a new model, not a conversion.
2. **Quality:** the oracle ceiling for post-hoc FFN top-k is bad, and the
   router is *not* the limiting factor (gap <= 0.033), so training a better
   router cannot rescue it.
   Additionally, the one plan that looked promising (k-means weight clustering)
   is degenerate: for E=128 one cluster absorbs 52% of the units.
3. **And training does not create the missing structure** — this was the open
   objection to blocker 2, and it is now measured rather than assumed. On real
   Qwen3.5-0.8B over all 24 layers (E15, free Kaggle T4s), training the experts
   and router beat the frozen partition only at the mildest ratio tested (E=8,
   12.5% active, by 0.006) and **made the reconstruction worse** at the ratios a
   conversion needs: E=32 (3.1% active) 0.9188 → 0.9355, E=64 (1.6%) 0.9277 →
   0.9549. The deficit grows with E, and a second seed reproduces it. Training
   re-purposes experts away from the split they were partitioned from — the
   residual against the original dense FFN actually rises while perplexity falls
   three orders of magnitude. The learned router loses to the oracle at every
   ratio, so better routing is not the fix either.

   *Scope, after a literature review (`docs/prior_work.md`):* E15 trained **614k
tokens**; LLaMA-MoE, partitioning the FFN the same way, used **200B on 112 A100s**
   and states 15-20B are needed just to compare partition methods. E15 also omitted
   the output rescaling both LLaMA-MoE (N/k) and NVIDIA (∛(E·G²/T)) call essential,
   and partitioned by weights where CMoE shows the axis should be the activation
   pattern with a shared expert. So the blocker is **budget, not feasibility**:
   upcycling demonstrably works at ~50% of original pretraining compute (Sparse
   Upcycling, +10-60%), which is 10⁴-10⁵ GPU-hours — four orders of magnitude beyond
   a free tier. The arithmetic in blocker 1 is unaffected by any of this.

*This is exactly the checkpoint the handoff asked for:* "Stop or adjust
architecture if aggressive sparsity fails oracle recovery." It failed — in both
forms. The frozen partition has no operating point (§4 of the report), and the
trained partition is worse than the frozen one where it matters (E14/E15).
Stopping.

### Path C — train a small MoE from scratch (**blocked on compute; not now**)

*What the goal actually requires:* a model with ~9B active *and* near-27B
quality. Section 4 of the feasibility report shows the active-parameter side of
that is speed-feasible — a fully resident 9B-active model is predicted at
~19 tok/s here. The blocker is that it needs far more than 9B *total*, and the
total must fit in ~14.3 GB, which forces exceptionally aggressive quantisation
of a model whose experts were never trained to tolerate it.

*Why it is blocked:* training a ~20B-A9B MoE is a pretraining-scale project
(10^21–10^22 FLOPs order). Free tiers are hours of a single mid-range GPU. That
is several orders of magnitude short, and no scheduling of free tiers closes
that gap. **The handoff's own framing is right: this is a major retraining
problem, not an FFN routing tweak.** With a £20 ceiling and no ongoing budget,
it is not startable.

## 4. Verdict on "9–12B active with near-27B quality"

- **Reachable by converting Qwen3.8-27B? No.** 9B active is below the 9.78B
  irreducible backbone; 12B active is arithmetically possible (2.2B active FFN)
  but only with *trained* experts, and the pilot shows post-hoc splits have no
  operating point.
- **Reachable as a design point on this hardware? Yes, in principle.** A
  fully resident ~12B-active model is predicted at ~10 tok/s here
  (bandwidth-bound on RAM), and a 9B-active one at ~19 tok/s. The hardware is
  not the obstacle.
- **Reachable within this budget? No.** It requires pretraining compute that
  free tiers cannot supply by orders of magnitude.

So: the hardware can host such a model; the budget cannot produce one. Path A is
how you get the best available assistant on this machine in the meantime.

## 5. Status of each deliverable

| # | Deliverable | Status |
|---|---|---|
| 1 | Feasibility report | **Done** — `docs/feasibility_report.md` |
| 2 | Project repository | **Done** — this repo, README, tests, config, results; git-committed locally. Not pushed to GitHub (no authorisation requested) |
| 3 | Dense-equivalent FFN partitioning + verification | **Done and verified** — real 27B weights, 5.1e-07 worst error, 243 tests pass |
| 4 | Pilot sparsity study | **Done** — `results/pilot_sparsity.json`; quality-vs-active-compute curve with oracle and learned routers |
| 4b | Trained-MoE PoC (the "does training change the answer?" gate) | **Done** — `results/upcycle_poc.json` (E14, local), `results/kaggle_upcycle_poc.json` + `_seed1.json` (E15, 24 layers on free Kaggle T4s). Answer: no; worse than frozen at E=32/64 |
| 5 | Edge runtime experiment | **Partly done** — hardware characterised and three quantised baselines measured end-to-end on the target machine; residency table computed. **Not done:** llama.cpp/GGUF build, `io_uring` vs mmap comparison, prefetch policy, pinned buffers, speculative decoding / native MTP experiments |
| 6 | Decision memo | **Done** — this document |
| 4c | Branching vs slicing (E16): are experts complete modules or slices? | **Done** — `results/branch_poc_08b.json`. Answer: complete modules *are* lossless (ppl 15.49 vs dense 15.48) but cost 4.00x the FFN; width, not tiling, sets the deficit; domain-specialised experts help only where a division of labour exists |

**Executed:** all hardware measurements; three end-to-end inference baselines;
exact parameter accounting from real weight headers; FFN partitioning verified
against real 27B weights; sparsity pilot on real activations; the trained-MoE
PoC at the target ratio over all 24 layers on free Kaggle GPUs, with a second
seed; a branching-vs-slicing experiment separating "slicing hurts" from "reading
less hurts", with independent per-domain expert training; a validated memory-tier
throughput model; 243 passing unit tests.

**Designed but not executed:** the evaluation harness (30/30/20/20 task sets
with executable checks); the quantisation quality ladder; all of workstream B's
async-I/O and prefetch work; speculative decoding against the native MTP head.

**Blocked:** any pretraining-scale work (no budget); CPU-only runs of the
qwen3_5 Gated DeltaNet path (the installed `causal_conv1d` is CUDA-only, which
is a fixable environment issue and worth fixing before CPU-offload design).

**Not attempted, by choice:** SSH or any access to machines other than the one
this session runs on; no paid resources; no irreversible changes. Free Kaggle GPU
compute *was* used for E15 and is not a paid resource.

## 6. Next three actions

1. **Build the held-out evaluation set first** (coding / reasoning / instruction
   / chat, with executable checks). Path A's gate is meaningless without it, and
   it is the only way to avoid "it feels fine" replacing evidence.
2. **Run the quantisation ladder** (Q3_K_M, IQ3, Q2_K) and apply the gate.
3. **Check the SSD** — `lspci -vv` link width and drive health. A 4x error in the
   storage number is the difference between "streaming is hopeless" and
   "streaming is a real design option", and it is a 10-minute check.
