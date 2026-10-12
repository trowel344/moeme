# moeme — edge sparsity feasibility for Qwen3.8-27B

A reproducible study of whether Qwen3.8-27B-class quality can be delivered on an
NVIDIA RTX 4060 Laptop (8 GB VRAM) / i7-13620H / 15.4 GiB RAM / 1 TB NVMe
machine, using no paid compute.

Everything in this repository was measured or computed on that machine. Results
are committed under `results/` so the claims can be checked against raw data.

---

## Headline findings

1. **Qwen3.8-27B is dense, not Mixture-of-Experts.** Its non-FFN text backbone
   alone is **9,783,238,144 params**, active on every token. A "9B active per
   token" target is therefore **arithmetically unreachable** from this model, no
   matter how good the router is. This is an executable test, not an opinion.
2. **Post-hoc FFN sparsification has no usable operating point.** At 12.5%
   active compute, a *perfect oracle* router still loses **42–81%** of the layer
   output. The router is not the bottleneck (it trails that oracle by ≤ 0.138,
   and by ≤ 0.033 with the coarsest partition), so better routing cannot fix it.
3. **The target model runs at 0.561 tok/s today** (Q4_K_M), and prefill at
   1.6 tok/s. Independently predicted from hardware constants at 0.48 tok/s.
4. **The bottleneck is storage streaming, not compute.** The SSD measures
   **0.8–1.9 GB/s** against a ~6.6 GB/s datasheet figure, and 2.2 GB/token has
   to come off it at Q4_K_M. *This is the whole problem.*
5. **The fix is aggressive quantisation to full residency, not sparsity.**
   Below ~3.4 bpw the model fits entirely in VRAM+RAM, which predicts
   4.9 tok/s at Q3_K_M and 6.4 tok/s at Q2_K — an ~9x gain with **no training
   and no architecture change**.

6. **Converting the 27B into a MoE does not rescue the goal, and training does
   not rescue the conversion.** Training only beat the frozen sparsity ceiling at
   the *mildest* ratio tested (E=8, 12.5% active, by 0.006). At the ratios a
   conversion actually needs it makes the reconstruction **worse**: E=32 (3.1%
   active) 0.9188 → 0.9355 and E=64 (1.6%) 0.9277 → 0.9549, reproducible across
   seeds. The trained 0.8B candidate is still **5.6–10.4x** worse than dense in
   perplexity. Measured in E14 (locally, 4 layers) and E15 (24 layers, free
   Kaggle T4s), not argued.

7. **And the converted model is not the same model, at the text level.** Four of
   the 0.8B's 24 layers upcycled (E=8, top-1, 300 steps) is **1.52x** dense
   perplexity, and on six fixed prompts **0 of 6** greedy continuations are
   byte-identical: it diverges within its first three tokens, at a mean
   positional agreement of **0.073**. The stock model answers *"The capital of
   France is Paris."*; the converted one answers *"the capital of the world."*
   Both get `17 * 23 = 391` right. So the retrofit costs quality in what the
   model actually says, not only in a perplexity ratio — and with only 4 of its
   24 layers converted.

8. **The converted layers cannot be taught by the model itself, and the output
   rescaling the whole field calls essential does not apply to them.** Training
   the same four layers against *their own dense FFN* — no labels, no pretraining
   corpus, 24 hand-written sentences (**197 tokens**) instead of E15's 614,400 —
   drives the reconstruction residual on that calibration set to **0.0196**, while
   the residual on 2,048 **held-out** tokens only moves 0.9239 → **0.8398**: a 98%
   in-sample improvement transfers as **9%**, because ~44M trainable parameters are
   fitting 197 tokens. Perplexity agrees — 4.15x dense against the
   cross-entropy-trained **1.52x**, and 0 of 6 continuations identical to dense.
   The same run measures the least-squares optimal output scale at **s\* = 1.406**
   for k=1 of E=8, where the literature's fix is **E/k = 8.0**, and s\* stays near
   1 at k=2 (1.257) and k=4 (1.114) too. So ~96% of the deficit is directional
   rather than a matter of scale, and applying E/k makes it *worse*: the frozen
   residual rises 0.9249 → 2.4461 and the trained model lands at **29.35x** dense
   against 4.15x without it. This corrects the repo's own account — E15's omission
   of the N/k rescaling was listed as a defect of that run, and it cost nothing.

Read `docs/feasibility_report.md` then `docs/decision_memo.md`. For the MoE
question specifically, `docs/upcycling_design.md` (the plan) and
`docs/prior_work.md` (what the field actually achieved, and why E15's negative
result is about budget and configuration rather than feasibility).

## Repository layout

```
src/moeme/ffn.py            dense-equivalent FFN partitioning, partition plans,
                            oracle + learned-router diagnostics, concentration
src/moeme/accounting.py     exact parameter accounting, memory-tier throughput model
src/moeme/branch.py         the *branching* alternative: complete experts (tile or
                            replica init), shared experts, forced-expert mode for
                            independent training, residual_report diagnostics
tests/                      371 tests: exactness, shapes, plans, oracle bounds,
                            accounting, upcycling (dense limit, gradients, aux loss,
                            half precision, CUDA autocast, the E/k output rescaling),
                            branching (exact and narrow limits, shared experts,
                            forced-expert training), PoC driver budget and error
                            resilience, the evaluation scorer and runner, the memory
                            watchdog, GGUF metadata reading and perplexity parsing,
                            the eval-window recovery (identification, ambiguity,
                            false-match refusal), the upcycled-vs-dense comparison,
                            and corpus-free self-teaching (best-scale optimality,
                            the fixed calibration set, scale sensitivity)
tools/verify_model_geometry.py   exact param accounting from HF safetensors headers
tools/hf_tensor.py               fetch single tensors via HTTP range requests
tools/bench_hardware.py          non-destructive hardware characterisation
tools/bench_inference.py         end-to-end quantised inference benchmark (ollama)
tools/residency_table.py         what quantisation fits resident, and the tok/s that follows
tools/summarise_results.py       re-print every headline number from results/*.json
tools/model_atlas.py             derive every number from the artifacts and write
                                 docs/model_atlas.html: a dependency-free 3D view of
                                 the layer geometry, the active-parameter arithmetic
                                 and the residency quantisation envelope (`make atlas`)
tools/mem_watchdog.py            armed memory floor for a run that may not fit;
                                 terminates the target before the OOM killer does
experiments/verify_real_ffn.py   partitioning verified on REAL 27B weights
experiments/pilot_sparsity.py    sparsity pilot with real activations
experiments/upcycle_poc.py       upcycling PoC: does TRAINING beat the frozen ceiling?
experiments/branch_poc.py        branching vs slicing, measured head-to-head, plus
                                 branch-train-miX (experts trained apart, then mixed)
experiments/eval_gate.py         evaluation set with EXECUTABLE code checks
experiments/quant_perplexity.py  what a quantisation costs, on identical tokens:
                                 llama-perplexity per GGUF, with the corpus hash,
                                 the scored window and a guard verdict recorded
experiments/recover_eval_window.py  recovers the --eval-blocks window a report
                                 never recorded, by measuring the val loss curve
                                 and inverting the perplexity it produced
experiments/compare_upcycled_08b.py  does the upcycled 0.8B still *behave* like the
                                 dense one? perplexity plus greedy continuations,
                                 both models on identical tokens, with the
                                 dense-limit control measured on a fresh partition
experiments/self_teach_poc.py    can a layer be taught by its own dense FFN with no
                                 corpus? trains against the model's own dense
                                 output and measures the least-squares optimal
                                 output scale, which decides whether the E/k
                                 rescaling applies to a partition (it does not)
kaggle/upcycle_poc.ipynb         the same PoC, run on free Kaggle GPUs (E15);
kaggle/README.md, kaggle_run.log  how to run it, and the raw session log
configs/tiers.json               measured constants, with provenance and limitations
docs/                            source validation, feasibility report, decision memo,
                                 upcycling design, branching-vs-slicing, expert
                                 offloading, quality-vs-speed (MoE vs dense on a
                                 fixed task set, plus engine placement),
                                 prior-work review of the MoE route, and a 3D model
                                 atlas (model_atlas.html, regenerate with `make atlas`)
results/                         raw evidence: JSON + printed tables
```

## Quickstart

```bash
make venv        # .venv reusing system torch, plus pytest
make test        # 371 tests, offline, ~17 s
make doctor      # one-command local diagnostics (~90 s, writes results/hardware.json)
make bench-inference
make verify-sources      # re-derive parameter counts from the real artefacts
```

Optional, requires network and (for `pilot`) a GPU:

```bash
make verify-real-ffn   # 535 MB ranged download; exactness on real 27B weights
make pilot             # ~1.8 GB download; real activations
make upcycle-selftest  # upcycling PoC pipeline check, offline, ~2 s (no download)
make upcycle           # upcycling PoC on a real model (GPU; or use kaggle/)
make branch-selftest   # branching-vs-slicing check, offline, CPU, ~2 s (no download)
make branch            # branching vs slicing on a real model (GPU)
make compare-upcycled-08b  # upcycled 0.8B vs dense: perplexity + generated text (GPU)
make self-teach        # teach the layers from the model's own dense FFN, no corpus (GPU)
make eval MODEL=qwen3:4b NUM_PREDICT=1536   # 30-task gate, executable code checks

# A model too slow for the full gate: same tasks, fixed settings, resumable.
make eval MODEL=qwen3.8:latest NUM_PREDICT=192 \
  EVAL_EXTRA="--think off --tasks r_lily,f_json_nulls,i_only_word --resume"
```

`--think off` is what makes a thinking model evaluable at a small `num_predict`,
and the gate checkpoints after every task, so an interrupted multi-hour run keeps
what it has and continues with `--resume`.

### The MoE question: does *training* change the answer? — answered, and the answer is no

**Measured (E14 + E15): training buys a small win only at the mildest sparsity,
and loses at the ratio a 27B conversion actually needs.** On real Qwen3.5-0.8B
with `combine=sum`, `plan=random`, k=1:

| E | active | frozen pilot oracle | oracle *trained* | router *trained* | beats frozen |
|---|---:|---:|---:|---:|:--|
| 8 | 12.5% | 0.8624 | **0.8560** | 0.8951 | yes, by 0.006 |
| 32 | 3.13% | 0.9188 | **0.9355** | 0.9653 | **no** (+0.017) |
| 64 | 1.56% | 0.9277 | **0.9549** | 0.9806 | **no** (+0.027) |

(The residual is the relative L2 error of the k-expert reconstruction against the
true dense FFN output — 0 is exact, 1 is as bad as predicting zero. `beats
frozen` uses only the four layers the pilot measured.)

So the deficit **grows with E**: the sparser the ratio, the more training hurts.
Training is not creating extractable sparsity, it is *adapting the function* — the
residual against the original dense FFN actually **rises** under training
(0.9324 → 0.9480 at E=8; 0.9919 → 0.9974 at E=64) while perplexity falls three
orders of magnitude. Re-purposing experts away from the split they were
initialised to be is exactly what costs the model its oracle ceiling.
E15 ran all 24 layers (not 4), on free Kaggle T4s, and a second seed reproduces
the E=64 inversion to three decimals (0.9551 / 0.9802).

**So the retrofit did not beat its frozen ceiling.** `docs/upcycling_design.md` §9
said to abandon the route if trained `oracle_topk` does not fall below the frozen
oracle at the same (E, k). At E=32 and E=64 it does not — it rises.
→ `results/kaggle_upcycle_poc.json`, `results/kaggle_upcycle_poc_seed1.json`,
`kaggle/kaggle_run.log`, `docs/experiment_log.md` E14 and E15.

**Read that as a result about an *initialisation*, not a verdict on upcycling —
which demonstrably works, and which E15 did not fairly test.** LLaMA-MoE partitions
the FFN exactly as E15 does and needed **200B tokens on 112 A100s**, stating that
15-20B are required just to *compare* partition methods; E15 trained **614k**. E15
also used the gating and the partition axis the literature identifies as wrong, and
omitted the output-rescaling fix both LLaMA-MoE and NVIDIA call essential. The
9.78B-backbone arithmetic below is untouched by any of this. Full review with
sources: `docs/prior_work.md`.

The experiment behind those numbers:

`pilot_sparsity.py` measured the ceiling for a **frozen** partition (it updates no
weights). `experiments/upcycle_poc.py` implements the other half: it upcycles the
FFN into a real top-k MoE (`src/moeme/upcycle.py`) and trains the experts and the
router, then compares the top-k-vs-all-experts residual on the **trained** weights
against the pilot's frozen number. If trained residual < frozen oracle, training
created structure a frozen partition never had.

### Does the converted 0.8B still *behave* like the stock one? — no

Perplexity says the retrofit is 5.6-10.4x worse. It does not say what the model
*does*. Loaded with the trained checkpoint over a fresh partition of the same
four layers, and run against the dense model on **identical tokens**:

| | dense 0.8B | upcycled (4 of 24 layers, E=8 top-1, 300 steps) |
|---|---:|---:|
| val perplexity (same 4-block window) | **19.5707** | **29.7929 (1.52x)** |
| post-hoc: fresh partition, k=1 | — | 99.6753 |
| **dense-limit control** (fresh, all 8 experts) | 19.5707 | **19.5635** (rel. err 3.69e-04) |
| trained partition, all 8 experts active | — | 1597.7289 (**82x**) |
| greedy continuations, 6 prompts x 48 tokens | — | **0/6 identical**; first divergence at token 0, 1, 2, 3, 0, 3 |

**The control is the point.** On a *fresh* partition, all-experts-active is the
dense layer by construction — the experts are disjoint slices of it — and it
reproduces dense perplexity to 3.69e-04. So the plumbing is exactly what
`results/real_ffn.json` claims, and everything else in that table is the sparsity
and the training rather than a wiring error. The same figure on the *trained*
partition is **82x dense**, and that is not a control: it says training rescaled
the experts away from the slices they were cut from. It is the text-level
confirmation of the `vs_original_dense > 1.0` result in
`docs/upcycling_design.md`.

**What it actually writes.** Dense: *"The capital of France is Paris."* Upcycled:
*"the capital of the world."* — fluent, on topic, and wrong. Both get
`17 * 23 = 391` right. On a fibonacci prompt the upcycled model writes explicit
`if n == 0: return 0` branches where dense writes `if n <= 1`. Nothing here is
gibberish; it is a different, slightly worse model, which is exactly what a 1.52x
perplexity gap on 4 of 24 layers should look like.

**The honest limit:** this is perplexity plus six prompts of greedy text, not a
task-level evaluation. The gate cannot score it — the gate talks to HTTP engines
(ollama, llama.cpp), and this is a PyTorch state dict with a custom partitioned
FFN that has no GGUF converter. That is also why the branch PoC and the upcycling
PoC have only residuals and perplexities between them.
→ `make compare-upcycled-08b`, `experiments/compare_upcycled_08b.py`,
`results/upcycled_08b_vs_dense.json`, `docs/experiment_log.md`

### Was the partition the problem? — yes. Complete modules are lossless (E16)

**Measured: the objection that ended E15 was correct, and it is now answered.**
Real MoE experts are *complete* feed-forward networks, not disjoint slices. Built
that way on the same model, the same activations and the same metric
(`experiments/branch_poc.py`, `src/moeme/branch.py`):

| active width | stored | best-1 tile | best-1 replica | ppl tile | ppl replica |
|---:|---:|---:|---:|---:|---:|
| 12.5% | 1.00x / 0.50x | 0.7104 | 0.7307 | 54.00 | 33.88 |
| 25% | 1.00x | 0.6339 | 0.6530 | 45.63 | 24.93 |
| 50% | 1.00x / 2.00x | 0.4975 | 0.5134 | 26.02 | 20.11 |
| **100%** | **1.00x / 4.00x** | 0.0000 | **0.0000** | 15.48 | **15.49** |

(dense val ppl **15.4803**; each cell is the best single expert's residual
against the *original dense* layer, which is the only reference that means the
same thing for both constructions.)

1. **Completeness is what buys losslessness.** A full-width replica at k=1
   reproduces dense to within the bf16 floor — ppl 15.49 against 15.48. Slicing
   cannot, at any E unless k=E. E15's degradation was structural.
2. **Width, not tiling, sets the deficit.** Best-tile and best-replica agree to
   ~0.02 at every width, so "slicing hurts" is really "reading 25% hurts".
3. **The big ppl gap at equal width is the router, not the architecture.** An
   untrained tile router picks a random block (0.8878 against a best of 0.6339);
   a replica has one block and nothing to choose. Training the tile router
   recovers most of it (45.63 → 23.19).
4. **Dense residual cannot arbitrate a trained model.** The router residual
   *rises* in every trained row, whichever way ppl moves: 0.8852 → 0.9303 as ppl
   improves 45.63 → 23.19, and 0.0044 → 0.5491 as ppl *worsens* 15.49 → 19.11. A
   rising residual is therefore evidence of nothing.
5. **Giving each module its own domain helps where a division of labour exists.**
   Experts pre-trained apart on code/math/chat/wiki (`--btx`) reach ppl **21.64**
   at 1.00x storage and 25% active width, against **23.19** for the same
   architecture trained on mixed data for the same steps — 1.55 ppl better. At
   full width the effect vanishes, because an exact copy has nothing to specialise
   toward.

**And the wall is unchanged: the lossless case costs 4.00x the dense FFN.** This
does not make a model smaller than dense; it makes the per-token read smaller. On
8 GB the binding constraint is storage, so quantisation remains the route to a
27B here.
→ `docs/branching_vs_slicing.md`, `results/branch_poc_08b.json`,
`docs/experiment_log.md` E16.

### Can we offload the cold experts to RAM and keep the hot ones in VRAM? — yes, but not for this model

**Measured (E17), same machine and runtime, on the 14.3 GB of usable fast memory:**

| model | arch | total / active | quant | file | resident? | decode |
|---|---|---|---:|---:|---:|---:|
| `qwen3.8:latest` | **dense** | 27.8B / 27.8B | Q4_K_M | 17.0 GB | no | 0.561 tok/s |
| `ornith-1.5-35b-a3b:q2_k` | MoE 256×8 | 35.5B / ~3B | Q2_K | 13.0 GB | **yes** | **19.356 tok/s** |
| `ornith-1.5:35b` | MoE 256×8 | 35.5B / ~3B | Q4_K_M | 22.0 GB | no | 6.450 tok/s |

The two `qwen35moe` rows are one architecture at two quantisations, which
separates the two effects: **sparsity is worth 11.5×** (a 35.5B MoE at Q4_K_M
beats a 27.8B *dense* at the same quant) and **residency is worth 3.0× on its own**
(the same MoE at 13.0 GB vs 22.0 GB). Every row fits one equation — decode time =
bytes read per token / the bandwidth of the tiers holding them: 17.0 GB / 1783 ms,
~1.83 GB / 155 ms, ~1.05 GB / 52 ms.

**So why not for the 27B?** Offloading chooses *where each byte comes from*, never
*which bytes are needed* — and in a dense model that set is the entire model.
There is no cold subset to push out; moving layers to RAM re-labels 17.0 GB per
token instead of reducing it, and 17.0 GB does not fit in 14.3 GB at any
placement. Worth noting the idea is not missing from the runtime: these runs used
ollama's default `num_gpu = -1` ("offload what fits") and peaked at 6.7-6.9 GB of
VRAM in every case. **The idea is missing from the model, not the engine.**
The lever that does work remains quantisation — Q3_K_M at 11.8 GB is fully
resident, predicted 4.85 tok/s against the measured 0.561.
→ `docs/expert_offloading.md`, `results/inference_moe*.json`, `docs/experiment_log.md` E17.

### Is the fast MoE as good as the slow dense 27B? — yes, and it is not close on time (E18)

**Measured on this repo's own gate, both models, identical prompts, temperature 0,
fixed seed, identical token cap, all 4 reasoning answers matching exactly:**

| model | reasoning @768 | wall (4 tasks) | 19-task score @768 |
|---|---:|---:|---:|
| `ornith-1.5-35b-a3b:q2_k` (13 GB MoE) | **4/4** | **41 s** | **17/19 = 89.5%** |
| `qwen3.8:latest` (17 GB dense) | **4/4** | 3517 s | not run (0.34 tok/s) |

Both answered `0.05`, `3`, `47`, `42`. **The MoE matches the dense 27B on this
task set while finishing the reasoning work in ~1/86th of the wall time** — and
the two failures it does have (`5 words` asked, 6 given; a forbidden letter) are
the same two the dense model fails at every cap.

**Read the small-cap numbers with care, because the first version of them was
wrong.** At `num_predict=192` the dense model scored 0/4 on reasoning against the
MoE's 2/4. All four of its failures were **truncations** — it was reasoning
correctly and got cut off — and the harness recorded the last number it saw as
the answer. Two grader defects compounded it: `r_bat_ball` expected `0.10` for a
ball that costs `0.05` (failing a correct answer) and `r_modular` expected `1`
where `7**20 % 13 = 3` (passing a wrong one). All three are fixed and pinned by
tests that re-derive the expectations and run a reference solution against every
coding case; `--rescore` re-scored the completed runs instead of re-running them.

**Is the architecture itself the compromise? No — measured at equal quantisation.**
`ornith-1.5:35b` is the *same* `qwen35moe` architecture at Q4_K_M (22 GB), so
against the dense 27B at Q4_K_M (17 GB) it differs in architecture only. Same
runtime, same engine, same 19 tasks: reasoning **ties at 4/4**, the non-reasoning
half differs by **one** brittle task (12/15 vs 13/15), and the sparse model
decodes **6.14 tok/s against 0.32 — ~19×**. Sparsity costs nothing measurable here.
What it does not do is make a model *fit*: at equal bits the 22 GB MoE is 7× slower
than the 13 GB Q2_K one because it misses the fast tiers too — it just reads 8 of
256 experts per token instead of the whole file. On an 8 GB card the 13 GB MoE
stays the one to run, and the reason is memory, not quality.

**Does the smaller quantisation cost quality? Yes, and the pass/fail gate cannot
see it.** The same 35B MoE at Q2_K and at Q4_K_M, same 19 tasks, same 768 cap,
nothing capped on either side: **17/19 against 16/19** — a wash, and the one-task
difference favours Q2. But compare the two runs output by output and **13 of 19
responses are not the same text at all** (6 byte-identical; the median divergence is
at the *first* character). The gate answers "is an easy answer still right?"; it does
not answer "is this the same model?". Perplexity answers that, and it sides with the
intuition rather than the gate: on a matched pair over an identical token stream
(`make quant-ppl`, one corpus built from this repo's own text) Q2_K costs **+0.71 PPL
= +10.1%** against Q4_K_M (7.7128 ± 0.468 vs 7.0029 ± 0.409) — about a 1σ separation,
so a consistent one-directional cost rather than a settled magnitude. The probe
records the corpus hash and the **scored window** beside every number, because "8
chunks" against a 400 KB corpus still means the first 4,096 tokens.

So **Q4_K_M is the closer model, and if you have the RAM, use it.** The reason this
repo runs Q2_K on an 8 GB card is that the 22 GB Q4_K_M costs **7.4× the wall time**
(463.5 s vs 62.7 s for the same 19 tasks) and does not score better — a *memory*
recommendation, not a claim that the two quants are equivalent.
→ `make quant-ppl`, `experiments/quant_perplexity.py`,
`results/quant_perplexity_35b.json`, `docs/quality_vs_speed.md` §4.1

**And the engine question has a measured answer.** The same 19 tasks through
llama.cpp at surplus's planner placement (`-ngl 42 -ncmoe 24`) scored **17/19 —
identical to the ollama tag, same two failures** — at ×1.12 decode, ×1.21 prompt
processing and ×1.75 wall time. A placement changes where bytes come from, never
which weights the model needs, and that is now measured rather than assumed. Two
findings from getting there: a raw `/completion` request applies no chat template,
so the model's `<think>` block landed in the answer and the *same* run scored
13/19 (a 21-point difference from a request setting), and the earlier "×2.59 / ×10"
placements gains compared against ollama's **cold 32-token** baseline — warm, on
the same tasks, ollama runs 43.1 tok/s and the real gain is ~10-20%.
surplus's own `bench` segfaults on the engine it pins, its `shim` returns HTTP 200
with an empty `response` (so it cannot be scored at all), its auto-plan in the
shim chose `-ncmoe 36` for 3.41 tok/s against 48-50 for its own best plan, and its
default 4 GiB RAM cage is smaller than this MoE needs. Its **planner** is the good
part: adopt the placement, verify it with `llama-bench` upstream.
→ `docs/quality_vs_speed.md`, `results/eval_*_reason768.json`,
`results/eval_llama_ngl42_ncmoe24.json`, `docs/experiment_log.md` E18.

Memory safety was sampled, not assumed: over 59 minutes of dense generation RAM
never fell below 5.79 GiB available and peak VRAM was 6811 of 8188 MiB.

`make upcycle-selftest` runs the whole pipeline on a tiny random model with no
download -- do that first, because it is what caught a device bug that would have
failed every CUDA run. The real run is set up for **free Kaggle GPUs** in
`kaggle/upcycle_poc.ipynb` and was executed there: 24 layers at E=8/32/64, plus a
seed-1 replicate, in one 3.25 h session. `kaggle/README.md` records the measured
step cost and the environment gotchas; `docs/upcycling_design.md` is the plan and
the arithmetic for doing it at 27B scale.

`make eval` runs the gate that `docs/decision_memo.md` Path A depends on:
coding tasks are **executed** against fixed cases, not string-matched. Thinking
models need a large `NUM_PREDICT`; rows the cap cut off are labelled `TRUNCATED`,
counted separately, and reported alongside a score that excludes them, rather than
being scored as capability failures. `make eval-llama` runs the same tasks through
llama.cpp at an explicit expert placement (`LLAMA_PLACE`, default surplus's
`-ngl 42 -ncmoe 24`), refusing to start if free VRAM/RAM are below threshold;
`make eval-surplus` does the same through surplus's Ollama-compatible shim.

`make quant-ppl` answers the question the gate cannot: how much a quantisation costs
the model. It builds one corpus from this repo's own prose and code, runs
`llama-perplexity` once per GGUF with every other flag held fixed, and records each
run's estimate, its cumulative series, the window it scored, the corpus sha256 and a
memory-watchdog verdict next to it. It can report *no* estimate for a run the guard
had to end — which is the correct answer for the 21 GB Q4_K_M 35B on this host, and it
is recorded as such rather than as a score. `make quant-ppl-selftest` checks the
parser offline. Run it with a memory guard or not at all: the Q4 attempt drove
MemAvailable to 0.14 GiB in under a minute before the watchdog stopped it.

`make help` lists everything. `make clean-results` removes regenerable outputs.

### One-command diagnostics

If you only run one thing, run this — it is non-destructive, needs no downloads,
and self-deletes its temp files:

```bash
python3 tools/bench_hardware.py --out results/hardware.json
```

It reports usable VRAM/RAM, CPU and GPU bandwidth, and NVMe sequential plus
random throughput at 4K/64K/256K/1M. **It deliberately writes a temp file larger
than RAM**, because otherwise the page cache serves the reads and the numbers
describe DRAM rather than the SSD. It also drains writeback and drops clean
pages with `posix_fadvise` before timing — without that, sequential read is
understated badly. (During development this distinction mattered: a first
attempt reported the 1 MiB cold mmap read at 15.9 GB/s, which was pure cache.
The corrected figure, 0.76 GB/s, agrees with the independent random-1M
measurement.)

## Measurement summary (this machine)

| Quantity | Measured |
|---|---:|
| Usable VRAM (desktop holds ~2.9 GiB) | ~3.8 GiB |
| Usable RAM for weights | ~10.2 GB (9.5 GiB) |
| GPU memory bandwidth | 246.8 GB/s |
| CPU RAM bandwidth (memcpy) | 40.7 GB/s |
| NVMe cold read | 0.82–1.9 GB/s |
| `qwen3:4b` decode | 60.7 tok/s |
| `ornith-1.5:9b` decode | 24.6 tok/s |
| **`qwen3.8:latest` (27B, Q4_K_M) decode** | **0.561 tok/s** |
| **same, prefill** | **1.6 tok/s** |

## Reproducing each claim

| Claim | Command |
|---|---|
| Exact parameter split of Qwen3.8-27B | `make verify-sources` |
| 9.78B irreducible backbone | `make test` (`test_no_ffn_routing_can_reach_9b_active`) |
| Partitioning is exactly dense-equivalent | `make verify-real-ffn` |
| Partitioning is not free | `make verify-real-ffn` (throughput section) |
| No latent expert structure | `make pilot` + `make report` |
| Training does *not* beat the frozen ceiling at the target ratio | `make report` (`results/kaggle_upcycle_poc.json`, E15) |
| Training beats it only at the mildest ratio, by 0.006 | `results/upcycle_poc.json` (E14) |
| The upcycled 0.8B is fluent but not the same model (0/6 continuations identical) | `make compare-upcycled-08b` (`results/upcycled_08b_vs_dense.json`) |
| The partition plumbing is exactly dense-equivalent on real text | same command's dense-limit control: 19.5635 vs 19.5707 |
| The E/k output rescaling does not apply to a partition | `make self-teach` (frozen scan: s\* 1.406 at k=1 where E/k = 8.0) |
| Corpus-free self-teaching memorises instead of generalising | `make self-teach` (in-sample 0.0196 vs held-out 0.8398) |
| Self-teaching is worse than token training on quality | `make self-teach` (4.15x dense vs E14's 1.52x) |
| A perplexity needs its eval window and its corpus | `results/upcycle_poc_rerun.json` vs `results/upcycle_poc.json` (E15 note) |
| An unrecorded eval window is recoverable | `make recover-eval-window` (`results/eval_window_recovery.json`) |
| ...and the recovery logic is sound offline | `make recover-eval-window-selftest` |
| The E=64 inversion is not seed noise | `results/kaggle_upcycle_poc_seed1.json` |
| The upcycling pipeline is sound before spending a GPU | `make upcycle-selftest` |
| 0.561 tok/s and the tier model | `make bench-inference`, `python3 tools/residency_table.py` |
| Evaluation scorer is correct | `make test` (`tests/test_eval_gate.py`, 37 tests) |
| A quantisation costs what it costs | `make quant-ppl` (`results/quant_perplexity_35b.json`) |
| The quant probe reads real GGUFs, not broken synthetic ones | `make quant-ppl-selftest` |

## Scope and limitations

Stated up front, because overclaiming here would be worse than the negative
results: both MoE experiments run at 0.8B scale (that is what fits the free
VRAM, and the qwen3_5 Gated DeltaNet path cannot run on CPU here because the
installed `causal_conv1d` is CUDA-only); the real-weight experiment uses synthetic
inputs, so it probes weight structure rather than end-to-end quality; no benchmark
suite or contamination analysis was run; and the memory-tier model is an
additive ceiling estimator validated within ~2x, not a forecast. The upcycled
0.8B now has a text-level comparison (E20) where it had only a perplexity ratio,
but still no task-level score: the gate speaks HTTP, and the converted model is a
PyTorch state dict whose FFN is a custom partitioned MoE with no GGUF converter. Workstream B's
runtime exploration (llama.cpp/GGUF, `io_uring`, prefetch policy, native-MTP
speculative decoding) is **designed but not executed**. Full list in
`docs/feasibility_report.md` §9.

Web search was unavailable throughout, so **all** source verification used
primary artefacts (Hugging Face API, `config.json`, model cards, safetensors
headers). Free Kaggle GPU compute *was* used for E15 — one 3.25 h session on 2x
Tesla T4, inside the weekly allowance as observed — and the published quota
numbers themselves were never confirmed from a primary source, so nothing here
is sized against them. Lightning AI was not used.

## Licence and provenance

Study code: do what you like with it. Model artefacts remain under their own
licences (Qwen3.8-27B is Apache-2.0, verified from the model card). No model
weights are redistributed here; `tools/hf_tensor.py` fetches them from the
original repositories on demand.
