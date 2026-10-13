# Where this project stands

As of 2026-10-13. This page is a verdict, not a survey. It says what has been
measured, what has been ruled out, and what is still unknown — and it does not
ask you to choose between anything. Every number is checked against the artifact
named beside it.

The other documents in `docs/` are longer and, read in sequence, are confusing:
`solution_space.md` is a menu of untested routes, `decision_memo.md` is a dated
set of three paths, `experiment_log.md` is a running diary. This is the one-page
summary they add up to.

---

## The goal, and what actually happened to it

The goal was to make **Qwen3.8-27B** — a 27.8B *dense* model — run usefully on
this laptop (3.8 GiB usable VRAM + 10.2 GB usable RAM, £0 budget). The plan was
to convert it into a mixture-of-experts so a token would read only part of the
model.

Two things are now true, and they point in opposite directions.

- **The conversion did not work, and it is not going to.** That route is closed
  by measurement.
- **The goal was reached anyway** — by running a model that was *already* sparse,
  not by converting this one. It is measured and committed.

The short version: the project spent its effort trying to give a dense model a
sparse FFN, and the answer is that you cannot retrofit that. You can, however,
rent one. That is the whole story.

---

## What is proven

These are measurements on this machine or from real model artifacts. They are
not going to change.

**The model is dense, and it is mostly FFN.**
27,781,427,952 parameters total; **17,112,760,320 of them (61.6%) are FFN**; the
non-FFN text backbone is **9,783,238,144**.
→ `src/moeme/accounting.py`, `docs/source_validation.md` (verified against the
safetensors headers by `tools/verify_model_geometry.py`)

**It runs at 0.561 tok/s, and prefill is worse at 1.6 tok/s.**
→ `results/inference_27b.json`

**The bottleneck is storage streaming, not compute.**
At Q4_K_M the model is 17.0 GB against 14.28 GB of usable fast memory, so
2.74 GB per token streams off an SSD measured at 0.82–1.9 GB/s.
→ `results/hardware.json`, `tools/residency_table.py`

**Being resident is worth 3.0x, by itself, on identical weights.**
The same 35.5B MoE: 13.0 GB → **19.356 tok/s**; 22.0 GB → 6.450 tok/s.
→ `results/inference_moe_q4.json`, `docs/expert_offloading.md`

**Sparsity is worth 11.5x, by itself, at equal quantisation.**
A 35.5B MoE at Q4_K_M (6.450 tok/s) beats a 27.8B *dense* model at Q4_K_M
(0.561 tok/s). The bigger model wins.
→ same artifact

**A sparse model is not a quality compromise, and it is ~19x faster.**
At equal quantisation — both models at Q4_K_M, so only the architecture differs —
reasoning **ties 4/4** and the non-reasoning half differs by one brittle task
(12/15 for the MoE against 13/15 for the dense model), while decode runs at
**6.14 tok/s against 0.32**. The fast 13 GB Q2_K build scores **17/19** on the
same 19 tasks; the dense model's own 13/19 run had all four reasoning rows
truncated at that token cap, which is why the equal-quantisation pair is the fair
comparison.
→ `results/eval_ornith-1.5-35b_short19_768.json`,
`results/eval_ornith-1.5-35b-a3b-q2_k_short19_768.json`,
`results/eval_qwen3.8-latest_short19.json`,
`results/eval_qwen3.8-latest_reason768.json`, `docs/quality_vs_speed.md`

**The FFN partitioning machinery works exactly.**
Worst relative L2 error **5.1e-07** on real 27B weights, across 4 plans and 2
input distributions. The plumbing is not the problem.
→ `results/real_ffn.json`

**The tier model predicts decode rate within ~2x.**
Predicted 0.48 tok/s where 0.561 was measured. Useful as a ceiling estimate,
not as a forecast.
→ `tools/residency_table.py`, `configs/tiers.json`

---

## What is dead

Each of these was closed by a measurement in this repo, not by an assumption.

**"9B active" as a parameter count.** The non-FFN backbone alone is
9,783,238,144 params, read on every token. Delete the *entire* FFN and that
9.78B is still the floor. There is no router, at any quality, that reaches 9B
active from this model.
→ `tests/test_accounting.py::test_no_ffn_routing_can_reach_9b_active`

**Post-hoc FFN top-k. There is no operating point.** At 12.5% active compute a
*perfect oracle* router still loses **42–81%** of the layer output. The router is
not the bottleneck (it trails the oracle by ≤0.138, and by ≤0.033 with the
coarsest partition), so a better router cannot fix it.
→ `results/pilot_curve.txt`, `results/pilot_sparsity.json`

**The reason it cannot work: there is no latent structure to find.** Energy is
spread almost uniformly across any partition — participation ratio 31.6 of 32,
126.1–127.8 of 128 on real weights. The oracle residual only reaches 0 when nothing
is sparse at all. A trained dense SwiGLU FFN has no experts hiding in it. The
published post-hoc MoE successes were on **ReLU** FFNs, which do have natural
sparsity; SwiGLU does not.
→ `results/real_ffn.json`, `docs/prior_work.md` §5

**Training does not rescue it — at the ratios that matter it makes it worse.**
E=8 (12.5% active): 0.8624 → 0.8560, a win of 0.006. E=32 (3.1%): 0.9188 →
**0.9355**. E=64 (1.6%): 0.9277 → **0.9549**. The deficit grows as the ratio gets
sparser, and a second seed reproduces it.
→ `results/kaggle_upcycle_poc.json`, `results/kaggle_upcycle_poc_seed1.json`

**The literature's output rescaling does not apply here.** The fix the field
calls essential is `E/k` — 8.0 at k=1 of E=8. The measured optimal scale is
**1.406**, and ~96% of the deficit is *directional*, not a magnitude error. So
the omission everyone flagged in the repo's own run cost nothing, and imposing
the fix would have made it worse.
→ `results/self_teach_08b.json`, `docs/upcycling_design.md` §5

**A model cannot teach its own sparse layers.** Training a converted layer
against its own dense FFN drove the error to 0.0196 on the calibration set and
only 0.9239 → 0.8398 on held-out tokens — a 98% in-sample improvement that
transfers as 9%.
→ `make self-teach`, `results/self_teach_08b.json`

**Complete-module experts are lossless but save zero bytes.** A full-width
replica reproduces dense exactly (ppl 15.49 against 15.48) — and reads 100% of
the FFN to do it. **Losslessness and byte savings are the same axis.** You cannot
get one without giving up the other.
→ `results/branch_poc_08b.json`, `docs/branching_vs_slicing.md`

**Offloading cannot rescue a dense model.** Offloading chooses *where a byte
comes from*, never *which bytes are needed*. A dense layer has no cold subset, so
17.0 GB still crosses into compute every token.
→ `docs/expert_offloading.md`

**Full upcycling on this budget.** It costs 10⁴–10⁵ GPU-hours. A free tier is
~30 GPU-hours/week and the budget is £20. That is off by orders of magnitude, not
by a factor you can schedule around.
→ `docs/prior_work.md` §1

**Training a new ~9B-active model from scratch.** Pretraining scale. Not
startable here.

---

## What is still open

These are unknown. They are *not* disproven — but they are also not solutions,
and none of them is known to work. Listed so the state of knowledge is honest.

**Whether a resident quantisation of the real 27B keeps its quality.**
Q3_K_M (11.8 GB) and Q2_K (9.7 GB) both fit fully resident and are predicted at
4.85 and 6.44 tok/s. **Their quality has never been measured on this model.** One
download settles it. This is the single unrun test that decides whether a fast
27B exists here at all.
→ `docs/feasibility_report.md` §8, experiment 1, calls this "the whole decision"

**Whether the 0.561 tok/s figure is partly a runtime artifact.** Every dense 27B
speed number here comes from `num_gpu = -1`, the runtime's offload-what-fits
default. Explicit placement was worth 1.42x on the MoE and has never been tried
on the dense model.
→ `docs/solution_space.md` §2 A6

**Whether the FFN can be curbed by *precision* rather than width.** Nothing in
this repo has asked whether the FFN's projections tolerate different bit widths
from each other, or from the attention layers. No training required. Untested.
→ `docs/solution_space.md` §2 A2

**Whether all 64 FFN layers are equally unsparsifiable.** Every experiment so far
treated them identically. Layer 23's oracle residual at 12.5% active is **0.431**
against 0.68–0.85 for layers 0/3/8 — a 1.6–2x spread nobody has followed up.
→ `docs/prior_work.md` §3, `docs/experiment_log.md` E15

**Whether the checkpoint's own MTP head is usable from this repo's runtime.**
llama.cpp has MTP behind `--spec-type draft-mtp` (reported 1.5–2.9x), but every
speed number here goes through ollama. Whether the runtime can enable it is
unchecked, and MTP only pays off on a *resident* model.
→ `docs/solution_space.md` §2 A4

**Whether the SSD is anomalously slow.** Measured at 0.82–1.9 GB/s, which is low
for the class and is flagged by `make doctor`. A throttled PCIe link would change
the calculus for every streaming design. A ten-minute check, never done.
→ `results/hardware.json`

**The engine work: llama.cpp/GGUF, `io_uring`, prefetch policy, pinned buffers.**
Designed, not executed.
→ `docs/decision_memo.md` §5

---

## The bottom line

The original target — make *this* 27B fast by making it sparse — is unreachable.
Not because the code is wrong, but because a dense SwiGLU model has no sparsity
to expose, and the arithmetic floor forbids the active-parameter goal regardless.

The practical target — a fast model on this laptop that is as good as the 27B on
the tasks this repo runs — is **already met**, and has been for several
experiments. At equal precision it lands within one task of the dense 27B on the
repo's 19-task set and answers every reasoning task identically — while decoding
19 times faster.

So the project's own goal has been achieved. It was achieved by a different means
than the project set out to use, and the documents that proposed the original
means are still on disk saying so, which is why reading them in order is
disorienting.
