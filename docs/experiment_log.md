# Experiment Log

Every experiment actually executed during this study, in order, with the command
used and the raw outcome. Numbers here are the source for the reports; if a
figure is quoted in `docs/`, it should trace to a line in this file.

**Environment:** `cleanerbox-Stealth-15`, Ubuntu 24.04.4, i7-13620H (10c/16t),
RTX 4060 Laptop 8188 MiB, 15.4 GiB RAM, Micron 3400 1 TB NVMe.
**Session date:** 2026-10-10. **Spend:** £0.00. **Cloud compute used:** none.

---

## E1 — Source verification (model geometry)

**Command:** `python3 tools/verify_model_geometry.py Qwen/Qwen3.8-27B`

**Method:** range-read the 1199 safetensors tensor headers (no weights
downloaded, a few hundred KB total).

**Outcome:** total 27,781,427,952 params. FFN 17,112,760,320. Gated DeltaNet
5,562,044,928. Attention 1,677,721,600. Embeddings + LM head 1,271,398,400
each. Vision tower 460,730,096. MTP head 424,699,392. Text tower
26,895,998,464. → `results/qwen3.8-27b-geometry.json`

**Finding:** the model is **dense**; there is no `num_experts` field. The
handoff's inherited "17.1B FFN / 9.8B other" split is confirmed exactly.

## E2 — Source verification (the family's MoE model)

**Command:** `python3 tools/verify_model_geometry.py Qwen/Qwen3.8-Flash-Next`

**Outcome:** 179,999,981,459 params (180.0B) — 121.4B MoE experts + 51.2B n-gram
memory + 2.6B MTP. → `results/qwen3.8-flash-next-geometry.json`

**Finding:** the family's MoE variant is 180B, not 27B. It cannot run on this
hardware in any quantisation.

## E3 — Hardware characterisation

**Command:** `python3 tools/bench_hardware.py --out results/hardware.json --disk-gb 20`

**Outcome:** GPU 246.8 GB/s (read), 219.9 GB/s (copy). RAM 40.7 GB/s copy.
NVMe sequential read 1.72 GB/s (buffered) / 1.47 GB/s (4 MiB `O_DIRECT`) /
0.82 GB/s (1 MiB `O_DIRECT`). Random 4K 11,419 IOPS. mmap cold 1 MiB: 1.38 ms,
0.76 GB/s. → `results/hardware.json`

**Two measurement bugs found and fixed during this experiment** (worth recording,
because both produced plausible-looking wrong numbers):

1. The first mmap cold-read reported **15.9 GB/s** — impossible for a drive that
   reads sequentially at 1.7 GB/s. Cause: `MADV_DONTNEED` on a `MAP_SHARED`
   mapping only drops the process page tables; the next fault re-attaches the
   same cached page. Fix: `posix_fadvise(POSIX_FADV_DONTNEED)` on the file,
   which actually discards clean file pages. Corrected value 0.76 GB/s, which
   then agreed with the independent random-1M measurement (0.82 GB/s).
2. Sequential read was initially understated because it was timed while our own
   20 GB of dirty pages were still flushing. Fix: `fsync`, then a settle delay,
   then measure. Value moved from 1.68 → 1.72–1.93 GB/s.

**Anomaly, unresolved:** the drive is a Micron 3400 1TB (datasheet ~6.6 GB/s
sequential read) but tops out near 1.5 GB/s across three independent methods
(buffered read, `O_DIRECT` `dd`, and a real 8.6 GB model blob). Cause not
established. Candidates recorded in `configs/tiers.json` — PCIe link width /
power state, thermal throttling, or sustained-state fallback. **This is the
single most decision-relevant unresolved number in the study.**

## E4 — End-to-end inference baselines

**Command:** `python3 tools/bench_inference.py --spec <model>:-1:<n>`

| Model | Size | Decode | Prefill | Load | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| `qwen3:4b` | 2.5 GB | **60.668 tok/s** | 560.2 tok/s | 6.7 s | 6798 MiB |
| `ornith-1.5:9b` | 6.6 GB | **24.629 tok/s** | 138.9 tok/s | 15.4 s | 6884 MiB |
| `qwen3.8:latest` (27B Q4_K_M) | 17.0 GB | **0.561 tok/s** | **1.609 tok/s** | 46.8 s | 6884 MiB |

→ `results/inference_safe.json`, `results/inference_27b.json`

**Finding:** the 27B is ~44x slower than a 9B here, and prefill of 1.6 tok/s
makes even prompt-filling impractical. Also measured: `MemAvailable` ~10.23 GB
during runs, and ollama obtained ~3906 MiB of model VRAM (6884 − 2978 MiB
desktop baseline) — these two numbers became the tier capacities.

## E5 — Dense-equivalent FFN partitioning (unit level)

**Command:** `.venv/bin/python -m pytest tests -q`

**Outcome:** 87 passed. Covers exactness for 4 plans × 7 expert counts × 2 input
distributions, ragged (non-dividing) splits, plan validity, bf16 exactness,
oracle monotonicity, oracle-vs-random-router ordering, concentration bounds, and
clustering correctness against planted structure.

**Two real library bugs found and fixed by these tests:**
1. `PartitionedFFN.topk_forward` mis-indexed flattened expert outputs,
   returning scalars instead of hidden vectors.
2. `clustered_plan` hardcoded a CPU `torch.Generator` and a CPU `assign` tensor,
   so it raised on CUDA inputs (`Expected a 'cuda' device type for generator`).
   Fixed in the library, and `PartitionedFFN` now moves index tensors to the
   weight device.

## E6 — Partitioning verified on REAL Qwen3.8-27B weights

**Command:** `python3 experiments/verify_real_ffn.py --out results/real_ffn.json`

**Method:** ranged download of layer 1's `gate/up/down` (0.535 GB of the 55.6 GB
checkpoint; all 64 layers share identical FFN geometry).

**Outcome:**
- **Exactness: worst relative L2 error 5.081e-07** across 11 plan/experts combos
  × 2 input distributions — contiguous, interleaved and random at E ∈ {8,32,128},
  clustered at E ∈ {8,32} only (the k-means plan is the expensive one). The
  partitioning is exactly dense-equivalent on the real artefact.
- **Throughput is NOT free:** dense 2.443 ms; E=32 all-active 5.916 ms
  (**2.42x**); E=128 all-active 9.497 ms (**3.89x**). Small-GEMM overhead at
  batch 1.
- **But that ratio is noisy, so quote a range.** A second run on the same cached
  weights (`results/real_ffn_rerun.json`, 2026-10-11) measured dense 2.555 ms,
  E=32 5.431 ms and E=128 7.391 ms -- i.e. **2.13x** and **2.89x**. Single-shot
  batch-1 timings on a daily-driver GPU vary by ~35%; the range is
  **2.1-2.4x (E=32) and 2.9-3.9x (E=128)**. The direction never changes:
  partitioning costs more than dense.
- **And top-k does not buy it back.** E=32 top-2 was 5.931 ms against 5.916 ms
  all-active on the first run, and 5.359 against 5.431 on the second -- the same
  within noise. Keeping 2 of 32 experts saves ~0-1.3%, not the 16x the active
  parameter count implies: dispatch and small-GEMM overhead consume the sparsity
  win. This is the strongest form of the result, and it is what makes MoE a
  large-batch technique rather than a single-stream one.
- **No natural structure:** participation ratio 31.57 of 32 and 126.14–127.82 of
  128. Energy is uniformly spread; the oracle residual only approaches 0 as k → E.
- **Clustering degenerates:** for E=128 one cluster holds 9,014 of 17,408 units
  (52%); top-3 experts hold 91% of energy. The attractive-looking oracle residual
  (0.019) is an artefact of that imbalance, not evidence of sparsity.

## E7 — Pilot sparsity study (real activations)

**Command:** `python3 experiments/pilot_sparsity.py --out results/pilot_sparsity.json`

**Model:** `Qwen/Qwen3.5-0.8B` — same `qwen3_5_text` architecture as the 27B,
same 3:1 DeltaNet:attention pattern, FFN expansion 3.50 vs the 27B's 3.40.
512 tokens of real calibration text; real weights; real forward-pass activations
captured via a pre-hook on each `mlp` module.

**Setup obstacle worth recording:** the installed `causal_conv1d` is CUDA-only
(`Expected x.is_cuda() to be true`), so the qwen3_5 Gated DeltaNet path cannot
run on CPU through `transformers`. The pilot had to run on the GPU, which is why
a 0.8B model was chosen (the 2B sibling's 4.55 GB does not fit the ~3.8 GiB of
obtainable VRAM).

**Outcome — the decisive result.** At 12.5% active compute, best oracle residual
per layer, excluding the degenerate clustered plan:

| Layer | Best oracle residual @ 12.5% active | Learned router |
|---|---:|---:|
| 0 | 0.681 | 0.807 |
| 3 | 0.802 | 0.872 |
| 8 | 0.811 | 0.857 |
| 23 | 0.417 | 0.449 |

Oracle-vs-learned gap: **0.0001–0.033**.
→ `results/pilot_sparsity.json`, `results/pilot_curve.txt`

**Findings:**
1. Even a perfect oracle loses 42–81% of the layer output at 12.5% active.
2. The partition, not the router, sets the ceiling — so training a better router
   cannot fix this. This is the checkpoint the handoff asked for before spending
   training compute.
3. Participation ratio ≈ E throughout: no latent expert structure exists.

## E8 — Memory-tier model + residency table

**Command:** `python3 tools/residency_table.py`

**Outcome (corrected units):** with 3.8 GiB VRAM + 10.2 GB RAM usable and
measured bandwidths only (GPU 247, RAM 40.7, disk 1.5 GB/s):

| Quant | bpw | Size | Streamed/token | Predicted |
|---|---:|---:|---:|---:|
| Q2_K | 2.8 | 9.7 GB | 0.00 | 6.44 tok/s |
| Q3_K_M | 3.4 | 11.8 GB | 0.00 | 4.85 tok/s |
| Q4_K_S | 4.4 | 15.3 GB | 1.00 GB | 1.07 tok/s |
| Q4_K_M | 4.9 | 17.0 GB | 2.74 GB | 0.48 tok/s |

**Validation:** predicted 0.48 vs measured **0.561** for Q4_K_M (ratio 0.86);
the other two configurations land at 1.63x and 0.49x. The model brackets reality
within ~2x using only independently measured constants.

**Unit bug found and fixed:** the RAM tier was originally specified as
`10.2 GiB`, but the source measurement is `MemAvailable` in bytes (10.23e9 =
9.5 GiB). Mixing GB and GiB overstated the tier by ~7%. Corrected, with the key
renamed `ram_usable_gb` so the units are unambiguous. The 27B prediction moved
from 0.57 → 0.48 tok/s; all conclusions were unaffected.

**Finding:** below ~3.4 bpw the 27B becomes fully resident, removing all
per-token storage traffic. This is the actionable path.

## E9 — Evaluation scorer unit tests

**Command:** `.venv/bin/python -m pytest tests -q`

**Outcome:** 124 passed — the original 87 plus 37 covering the evaluation
harness's extraction and scoring logic (`tests/test_eval_gate.py`).

**Why this exists:** the harness decides pass/fail, so a buggy scorer would
silently invalidate every quality number. Offline and millisecond-fast, so
`make test` covers it without a model or GPU.

**Three real bugs it caught, all fixed in the library rather than worked around:**
1. **`run_code` produced `IndentationError` on every correct solution.** The
   candidate source was interpolated into an indented template and then
   `textwrap.dedent`-ed, which collapsed the function body to column 0. This had
   already polluted an earlier eval run: coding tasks reported "unexpected
   indent" and it looked like the model's fault. The candidate source is now
   concatenated at column 0.
2. **`forbid_char` reported "contains 'e'" even when it passed** — a static
   detail string rather than an outcome-dependent one.
3. **JSON `true` validated against an `int` field**, because `bool` subclasses
   `int` in Python. `check_json` now rejects bool where a number is required,
   in both flat and nested positions.

## E10 — Evaluation gate run (starter set)

**Command:** `python3 experiments/eval_gate.py --model qwen3:4b --num-predict 1024 --only coding`

**Outcome:** 3/10 coding tasks pass, with **executable** verification — 4/4,
2/2 and 4/4 cases run and pass. The other 7 `TRUNCATED during reasoning (1024
tokens, empty answer)`. → `results/eval_qwen3_4b_coding.json`

A wider subset run (`--only reasoning,instruction,format`, 20 tasks) gave 9/20,
with 9 truncations. Real passes include `r_train` (16:50), `r_lily` (47),
`r_sequence` (42), a 3-action JSON response, an exact single-word answer, and a
yes/no answer. → `results/eval_qwen3_4b_subset.json`

**Interpretation:** `qwen3:4b` is a thinking model that needs well over 1024
tokens of reasoning for most non-trivial prompts. Both runs are therefore
**partly a measurement of the token cap, not of the model**, and the harness says
so explicitly rather than burying it. A real gate run needs `NUM_PREDICT >= 4096`.
The harness mechanics — code execution, JSON parsing, number/time extraction,
instruction constraints — are all confirmed working against a live model.

**Not run:** the full 30/30/20/20 split, any public benchmark, any contamination
check, and any evaluation of the 27B itself (at 0.561 tok/s a 30-task reasoning
run would take many hours).

## E11 — Upcycling module (trainable top-k MoE)

**Command:** `.venv/bin/python -m pytest tests -q`

**Outcome:** 187 passed. `src/moeme/upcycle.py` adds `MoEConfig` and `MoEFFN`: a
dense swiGLU FFN viewed as a partition of experts with a learned router, a
Switch-style auxiliary loss, a router z-loss and routing diagnostics.
`tests/test_upcycle.py` adds 63 tests (124 -> 187), covering:

- **the dense limit** — `top_k = n_experts` with `combine="sum"` reproduces the
dense layer exactly for every plan and expert count, so the dense model is a
strict special case rather than a claim;
- that `combine="softmax"` is deliberately **not** the dense limit (softmax
  weights sum to 1, so a full-k combination is a weighted mean);
- gradients reaching every expert and the router, and a real training step
  reducing a probe loss;
- auxiliary-loss value and routing metrics on balanced and collapsed routers.

**Finding:** partition-based upcycling keeps total parameters *equal* to the
dense FFN (experts are slices, not replicas), which is what a "27B total / 9B
active" target requires and makes the dense-limit property exact.

## E12 — Upcycling PoC pipeline check (offline self-test)

**Command:** `python3 experiments/upcycle_poc.py --self-test`

**Outcome:** the full pipeline (upcycle -> untrained eval -> train -> residual ->
routing stats) runs end to end on a tiny random model with no download.

| quantity | value |
|---|---:|
| dense val ppl | 295.4278 |
| dense-limit val ppl (top_k = E) | 295.4278 |
| untrained top-k-vs-all oracle | 0.5463 |
| **trained top-k-vs-all oracle** | **0.4077** |
| trained router | 0.5144 |
| effective experts (of 4) | 3.97 |
| vs-original-dense residual | 0.7232 -> 2.9449 |

**Finding 1 (the mechanism works):** training reduced the intrinsic sparsity gap
by ~25% on a tiny model (0.546 -> 0.408). This is the signal the experiment
exists to detect, but at toy scale it is **not** evidence about 0.8B or 27B.

**Finding 2 (a real device bug, caught by the self-test):** the partition plan
is built on the CPU, so the `assignment` buffer and the expert-init noise landed
on the CPU next to CUDA weights. This raised inside `index_select` and would have
failed **every** CUDA run — i.e. every Kaggle session. The buffer now follows the
weights, and `test_module_follows_weights_onto_the_gpu` pins it.

**Finding 3 (a metric that can exceed 1.0):** `vs_original_dense` went 0.72 ->
2.94. Once training rescales experts to compensate for the experts that are never
selected, the output can end up *farther* from the original dense function than
predicting zero would be. That is the output-scale problem a sparse partial sum
creates (see `docs/upcycling_design.md` §5), and it is why the decisive metric is
the top-k-vs-all-experts residual on the trained weights, not the preservation of
the original function.

## E13 — Kaggle notebook for the real run

**Command:** `python -c "import nbformat; nbformat.validate(...)"`

**Outcome:** `kaggle/upcycle_poc.ipynb` builds and validates as nbformat 4.5
(8 cells). It checks the GPU, installs deps, fetches the repo (clone or uploaded
dataset), runs the self-test, runs the real grid, and tabulates results. Setup and
constraints are in `kaggle/README.md`.

**Later executed as E15:** the notebook was run on Kaggle (2x Tesla T4), which is
what produced `results/kaggle_upcycle_poc.json` and
`results/kaggle_upcycle_poc_seed1.json`. See E15 for the outcome.

## E14 — Upcycling PoC on the real model (does training beat the frozen ceiling?)

**Command:**

```
python3 experiments/upcycle_poc.py --model Qwen/Qwen3.5-0.8B \
  --experts 8 --topks 1 2 4 --combine sum --plan random \
  --layers 0 3 8 23 --steps 300 --lr 1e-4 --seq-len 512 \
  --grad-accum 4 --corpus-chars 4000000 --out results/upcycle_poc.json
```

Ran on the local RTX 4060 (7800 MiB free after stopping an idle ollama model),
real `qwen3_5_text` architecture, 4M characters of streamed `fineweb-edu`
(883,186 tokens; 1638 train / 86 val blocks of 512). 300 steps ~= 180 s per
configuration. Only the four upcycled layers train (44.1M params).

**Outcome.** `dense val ppl` 21.74; the dense-limit check (`top_k = E`, sum)
gives 21.74 — the MoE plumbing is exact on the real model.

> **Read the absolute perplexities in this entry with the E15 note below.** This
> report predates the corpus- and eval-window provenance fields. Its window has since
> been **recovered by measurement: 6 val blocks**, so its numbers are explained
> rather than merely doubted — a re-run of the same command at the current default of
> 4 blocks gives dense ppl **19.5707** with every trained perplexity 11-18% lower, on
> the same data (the reconstruction residuals reproduce to four decimals). The × dense
> ratios here — 1.55 / 1.41 / 1.31 — are unaffected and are what the findings rest
> on. Do not quote these absolute numbers against E15's or the re-run's without the
> window beside them. See `results/eval_window_recovery.json` and
> `results/upcycle_poc_rerun.json`.

| k | active | oracle untrained | oracle trained | frozen pilot oracle | beats frozen | router | eff. experts | ppl untrained | ppl trained |
|---|---:|---:|---:|---:|:--:|---:|---:|---:|---:|
| 1 | 12.5% | 0.8584 | **0.8501** | 0.8624 | **True** | 0.8864 | 7.37 | 112.0 | 33.8 |
| 2 | 25% | 0.7520 | **0.7383** | 0.7578 | **True** | 0.7932 | 7.69 | 73.2 | 30.6 |
| 4 | 50% | 0.5638 | **0.5368** | 0.5710 | **True** | 0.6093 | 7.89 | 48.9 | 28.4 |

**Three findings, in order of importance.**

1. **Training does beat the frozen ceiling — but only just.** The improvement is
   1.0% (k=1), 1.8% (k=2) and 4.8% (k=4) relative. It grows with the active
   fraction, i.e. with how much freedom the router has, which is coherent.
   **Superseded by E15**, which pushed the same k=1 measurement out to E=32 and
   E=64 and found the win inverts there. Read this as "training wins at E=8", not
   as a property of the method.
2. **The residual is dominated by partial-sum scaling, not by learned structure.**
   With `combine="sum"`, keeping k of E roughly equal slices leaves a residual of
   about (1 - k/E) *regardless of training* — and that is what the numbers show
   (0.85 at 12.5%, 0.54 at 50%). Training moves it only slightly, which is the
   same "no latent expert structure" result the pilot (E7) found, now confirmed
   under training rather than assumed.
3. **End to end, training helps a lot but still loses.** ppl goes 112 -> 33.8
   (k=1), 73 -> 30.6 (k=2), 49 -> 28.4 (k=4) against a dense 21.7. So the trained
   MoE is 31-56% worse than dense at these fractions — *and* that figure is
   flattered, because the trained layers also saw 1638 in-corpus training blocks.
   Note the divergence: E2E ppl improves sharply while the reconstruction residual
   barely moves. Training is **adapting the function**, not creating extractable
   sparsity.

**Cross-validation:** the untrained oracles (0.8584 / 0.7520 / 0.5638)
independently reproduce the pilot's frozen oracles (0.8624 / 0.7578 / 0.5710) to
within ~0.5% on a separate run of the pipeline.

**Four real bugs, all found by running on real artifacts and all fixed:**

1. **The partition buffer was built on the CPU next to CUDA weights** — every
   CUDA run would have failed in `index_select`.
2. **`torch.sum` is on autocast's float32-promote list**, so the router's combine
   weights came back float32 while `y` stayed bf16 — every autocast training run
   would have failed in `index_add_`.
3. **`Tensor.quantile` rejects bf16**, so the library's own oracle diagnostics
   crashed on half-precision activations; residual metrics are now float32.
4. **Labels were double-shifted.** HF causal LM heads shift labels internally, so
   pre-shifting in `make_blocks` measured a misaligned loss — reading as dense ppl
   **20,664** instead of the true **21.7**.

Bugs 1-3 were invisible to a float32 CPU test suite; bug 4 was invisible until the
absolute number was sanity-checked against a known-good text. All four are now
covered by tests or an explicit warning.

## E15 — Upcycling PoC at the target ratio, 24 layers on free Kaggle GPUs

**Command:** `kaggle/upcycle_poc.ipynb` pushed with `kaggle kernels push -p kaggle/`
(kernel `trowel3444/moeme-upcycling-poc`). Raw session log: `kaggle/kaggle_run.log`.

```
python3 experiments/upcycle_poc.py --model Qwen/Qwen3.5-0.8B \
  --experts 64 32 8 --topks 1 --combine sum --plan random \
  --layers all --steps 300 --lr 1e-4 --seq-len 512 --grad-accum 4 \
  --save-every 0 --max-seconds 5400 --out upcycle_poc.json
# then the same config once more at E=64 with --seed 1
```

Ran on 2x Tesla T4 15.6 GB (Kaggle free tier, driver CUDA 13.0, torch 2.11.0+cu128,
transformers 5.6.2). Real `qwen3_5_text`, real streamed `fineweb-edu` (883,186
tokens; 1638 train / 86 val blocks of 512), bf16 autocast, **all 24 FFN layers**
upcycled (265.8M trainable params) — the E=64/all-layers grid E14 could not afford.

**This is the experiment E14 was a proxy for.** E14 ran E=8 on 4 layers. This runs
the exact `E/k ≈ 64` ratio the 27B design in `docs/upcycling_design.md` §1 needs,
over every layer.

| E | active | frozen pilot oracle | oracle *trained* | router *trained* | beats frozen | ppl untrained | ppl trained |
|---|---:|---:|---:|---:|:--:|---:|---:|
| 8 | 12.5% | 0.8624 | **0.8560** | 0.8951 | **True** (by 0.006) | 653,536 | 109.5 |
| 32 | 3.13% | 0.9188 | **0.9355** | 0.9653 | **False** | 1,464,554 | 159.3 |
| 64 | 1.56% | 0.9277 | **0.9549** | 0.9806 | **False** | 624,690 | 202.5 |

Dense ppl **19.562**. The dense-limit check (`top_k = E`, sum) reproduces it at
19.573 / 19.592 / 19.587, so the MoE plumbing is exact on the real model. `beats
frozen` is computed only over the four layers the pilot measured (0, 3, 8, 23).

**A baseline that does not match E14's — and it turned out not to be the corpus.**
E14's local 4-layer run measured dense ppl **21.736** against this run's 19.562, an
11% gap on what looks like the same setup. The obvious suspect was the streamed
corpus, since `streaming=True` yields whatever the hub serves first, so the E14
command was re-run on this machine under a memory guard, same flags, `--save-every
0`, into a new file (`results/upcycle_poc_rerun.json`) rather than over E14's — and
it returned dense ppl **19.5707**, i.e. *this* run's number, not E14's. The
provenance that re-run finally recorded (`corpus:
dataset:HuggingFaceFW/fineweb-edu`, 4,004,296 chars from 822 documents, 883,186
tokens, 1638 train / 86 val blocks) matches E14's documented token count exactly.

So the corpus is **not** the cause, and the residuals say so independently: they are
computed on val activations, and they reproduce between E14 and the re-run **to four
decimals** (untrained oracle 0.8584 / 0.8584, 0.7520 / 0.7520, 0.5638 / 0.5638).
Same data, same weights — while every perplexity came back 11-18% lower. What
actually differs is the **eval window**: `eval_perplexity` scores
`val_blocks[:--eval-blocks]`, and on this model and this text perplexity is **19.57
over 4 val blocks and 23.42 over 8** — a 20% swing from one flag that changes no
weights and no data. E14's 21.74 sits between those, consistent with a different
window width. And that width is **recoverable rather than merely lost**: perplexity
is a running mean of per-block losses, so it is a function of k alone, and inverting
E14's own number against a measured loss curve identifies k exactly.
`experiments/recover_eval_window.py` (new) returns **k = 6 val blocks** for 21.7359
at a relative residual of **1.1e-07** — E14 printed 21.7359 and the curve gives
21.735902 — with the next-closest of the 86 possible windows **6.6e-03 away, 58,000x
further off**. E14's second dense number corroborates it: the sum-of-all-experts
dense-limit 21.7396 lands on the same k=6, within the ~2e-4 by which that path
differs from dense. E14's documented command passes no `--eval-blocks`, so 6 was its
revision's default; the current default is 4. That single changed default is the
whole of the 11% baseline gap.

**E15's window is a different story, and the honest one is weaker.** Its dense
19.562 matches no window exactly — it ran on a T4, whose bf16 kernel is not
bit-identical to this box — and *three* windows fall within 0.5% of it: k=4 at
19.5707, k=40 at 19.5574 and k=43 at 19.6191. So E15's window cannot be identified
from its dense number alone, and this note does not claim one. The candidates span
0.3%, which is why E15's × dense ratios (5.60x / 8.14x / 10.35x) hold whichever it
was — but that is a bound on the ambiguity, not an identification.

What survives: the **× dense ratios**, because numerator and denominator move
together — 1.55 / 1.41 / 1.31 in E14 against 1.52 / 1.40 / 1.29 in the re-run, and
`beats_frozen` is `True` for all three configurations in both — so E14's conclusion
(the trained MoE is 31-56% worse than dense at these fractions) stands, and this
experiment's ratios now rest on a baseline that three runs agree on. What does not
survive: any quoted E14 *absolute* perplexity. Use the re-run. The driver now puts
`eval_blocks` in the report header, so a baseline can no longer be separated from the
window it was averaged over.

Evidence for the window claim: `logs/blocks8.log` (the same command at
`--eval-blocks 8`, dense ppl 23.4212) against `logs/upcycle_rerun.log` and
`results/upcycle_poc_rerun.json` (4 blocks, 19.5707). Repeating the 4-block run a
third time reproduced the baseline to ten significant figures (19.57067870397227),
and `logs/hdr.log` shows the header the CLI now writes. The recovery itself is
`results/eval_window_recovery.json` (the measured 86-block loss curve, its
calibration against both recorded windows, and the per-target verdicts) with its
run log in `logs/recover.log`; `python3 experiments/recover_eval_window.py
--curve-from results/eval_window_recovery.json --target 21.7359` re-decides it with
no GPU. Every run in this note went through `tools/mem_watchdog.py` with its floor at
2.0 GiB against 8+ GiB available; the guard never fired on any of them.

**The finding: the small training win does not survive to the ratio that matters,
and at E=64 it inverts.**

1. **E=8 reproduces E14** (0.8560 vs E14's 0.8501, both below the frozen 0.8624).
   The mechanism E14 found is real — at 12.5% active, training does create a little
   structure.
2. **At E=32 and E=64 training makes the reconstruction *worse* than the frozen
   geometric partition** — 0.9188 → 0.9355 and 0.9277 → 0.9549 — and the deficit
   grows with E (−0.017, then −0.027). The trend runs the wrong way for the target:
   the sparser the ratio, the more training hurts.
3. **The learned router is worse than the oracle at every ratio, and worse than
   the frozen partition at every ratio** (0.8951 / 0.9653 / 0.9806 vs 0.8624 /
   0.9188 / 0.9277). Its distance from the frozen number *grows* with E (+0.033,
   +0.047, +0.053 at E=8/32/64) even as its distance from the *oracle* shrinks
   (0.039, 0.030, 0.026) — routing gets relatively better and still falls
   further behind, because the oracle route is itself falling behind. At E=64 the
   router reached 33.2 of 64 effective experts and still lost to simply keeping
   the frozen split.
4. **Training trades dense-equivalence away for loss.** `vs_original_dense` — the
   residual against the dense FFN the experts were partitioned from — *rises*
   under training at every E (0.9324 → 0.9480, 0.9839 → 0.9902, 0.9919 →
   0.9974), while perplexity falls by three orders of magnitude (624,690 → 202.5
   at E=64). The optimizer is re-purposing the experts away from the split they
   were initialised to be, which is precisely what costs it the oracle ceiling.
5. **End to end it is still not close.** The best trained MoE (E=8, 109.5) is
   **5.6x** the dense perplexity, and the target ratio (E=64, 202.5) **10.4x**.

**Reproducibility and cross-validation, in the repository's favour:**

- The **untrained** oracle reproduces the pilot's frozen number on the *same four
  layers* almost exactly — E=64 0.9292 vs 0.9277, E=32 0.9194 vs 0.9188, E=8
  0.8583 vs 0.8624, i.e. agreeing to **0.0015 or better**. Two independent
  pipelines, one on an RTX 4060 and one on T4s, computing the same geometric
  ceiling from different code paths. The all-24-layer mean is higher (0.9335 at
  E=64) simply because layers 0/3/8/23 are the four easiest of the 24.
- **A second seed reproduces the inversion at E=64** to three decimals:
  frozen 0.9277, trained oracle 0.9551 (vs 0.9549), router 0.9802 (vs 0.9806).
  This is not seed noise.

| | E=64 seed 0 | E=64 seed 1 |
|---|---:|---:|
| frozen oracle | 0.9277 | 0.9277 |
| oracle trained | 0.9549 | 0.9551 |
| router trained | 0.9806 | 0.9802 |

**Measured cost**, which is what makes this a Kaggle job rather than a laptop one:
11.89 s/step at E=64, 8.12 at E=32, 5.05 at E=8 (24 layers, seq 512, grad-accum 4),
purely because the MoE forward loops over all E experts. Wall clock per
configuration: 3,786 s / 2,446 s / 1,522 s; the whole session, including the
self-test and the seed-1 replicate, was **3.25 h** against a 12 h cap.

**Verdict.** `docs/upcycling_design.md` §9 states two kill criteria: (1) trained
`oracle_topk` does not fall below the frozen pilot oracle at the same (E, k), and
(2) the gap fails to shrink as E grows. **E=32 and E=64 trigger criterion 1**, and
the gap *widening* with E triggers the spirit of criterion 2 as well. The retrofit
route fails on its own stated gate, measured rather than extrapolated — and the
"residual ~0.96 at 1.6% active" figure E14 could only extrapolate is now
**exceeded in reality at 0.9549**.

**Scope limit, added after a literature review (`docs/prior_work.md`): this
measures an initialisation, not upcycling.** Four specific mismatches:

1. **Token budget.** E15 trained 614k tokens. LLaMA-MoE — which partitions the FFN
   *the same way* — used **200B tokens on 112 A100-80G**, and states that models
   *"should be trained for at least 15~20B tokens to properly conclude"* anything
   about partition methods. E15 is ~30,000x below that floor.
2. **No output rescaling.** `combine="sum"` with k of E equal slices leaves
   about (1 − k/E). LLaMA-MoE rescale expert outputs by N/k and call it
   *"significantly better"*; NVIDIA scale weights by ∛(E·G²/T) for **1.5% better
   loss** and report the naive version *"did not converge to the original loss."*
   `docs/upcycling_design.md` §5 listed this as mitigation #1 and it was never
   implemented in E14 or E15.
3. **Gating.** E15 used an unweighted partial sum. NVIDIA consistently found
   **softmax-then-topK beats topK-then-softmax**; NLP MoE practice is a weighted
   sum. The weaker scheme was tested.
4. **Partition axis.** E15 partitioned by **weights**. CMoE (arXiv:2502.04416)
   shows the informative axis is the **activation pattern**, with the
   always-active neuron subset split into an always-on *shared* expert. Their
   stated failure mode for weight-based partitioning is verbatim what E15
   measured: always-active neurons scattered across experts, which *"forces the
   router to activate most experts regardless of input"* — hence `eff_experts`
   33.2 of 64 and `load_max_over_mean` 9.25.

One E15 observation is independently confirmed: LLaMA-MoE found *"deep layers have
more routing preferences than shallow layers"* and suggested partitioning later
layers instead — matching E15's layer-23 concentration (PR/E 0.185, top-1 share
0.82, oracle 0.431 vs 0.68-0.85 for layers 0/3/8).

## E16 — Branching vs slicing: are experts *complete modules* or *slices*?

**Command:**

```
python3 experiments/branch_poc.py --self-test      # 18 offline checks, ~2 s
python3 experiments/branch_poc.py --model Qwen/Qwen3.5-0.8B --dtype bfloat16 \
  --layers 0 3 8 23 --seq-len 512 --grad-accum 4 --steps 80 --branch-steps 40 \
  --btx --domain-chars 800000 --corpus-chars 3000000 --eval-blocks 4 \
  --out results/branch_poc_08b.json
```

16 configurations, local RTX 4060, `bf16`, TF32 disabled, 3M chars of streamed
`fineweb-edu` (661,966 tokens), 4 held-out val blocks, dense val ppl **15.4803**.
Full analysis and the table: `docs/branching_vs_slicing.md`; raw evidence:
`results/branch_poc_08b.json`, `results/branch_poc_08b_table.txt`.

**What was tested.** `src/moeme/branch.py` builds a top-k FFN whose experts are
**complete** copies of the dense layer (`init="replica"`) or the E14/E15 disjoint
**slices** (`init="tile"`), from the same weights, with the same router and the
same diagnostics. Experts can be forced (`set_force_expert`) so one can be
trained alone on its own data before a router exists; `n_shared` adds always-on
experts (the CMoE / DeepSeek-MoE fix); `--btx` runs branch-train-miX: stage 2
trains each expert independently on a different domain (code, math, chat, wiki),
stage 3 trains router and experts jointly.

**Outcome — five findings, in order of importance.**

1. **The objection is right.** A full-width replica at k=1 reproduces the dense
   model *exactly*: ppl **15.49** against dense 15.48, residual 0.0038 (the bf16
   floor; 0.0000 at one expert, 1e-6 in fp32). No slicing configuration does this
   unless k = E. E14/E15's degradation is therefore structural — the construction
   was wrong for the question — **not** a training failure. This is the headline.
2. **Width, not tiling, governs the deficit.** At equal active width the best
   single expert agrees to ~0.02 between the two constructions: 0.7104 vs 0.7307
   at 12.5%, 0.6339 vs 0.6530 at 25%, 0.4975 vs 0.5134 at 50%. At 50% the oracle
   even favours tiling (0.4820 vs 0.5134), because a tile may pick the best of E
   blocks while a replica has one block and nothing to route between
   (`best1 == oracle == router` on every replica row).
3. **The tile-vs-replica ppl gap at equal width is router initialisation, not
   architecture.** At 25% active: 45.63 (tile) vs 24.93 (replica) with almost
   identical single-expert residuals; the untrained tile router picks a random
   block (0.8878 against a 0.6339 best). Training it recovers most of the gap
   (45.63 -> 23.19). Reporting only ppl for these two constructions measures the
   router, not the MoE.
4. **Dense residual is the wrong objective once anything is trained.** In every
   trained row the router residual *rises* while ppl *improves*: 0.8852 -> 0.9303
   with ppl 45.63 -> 23.19; 0.0044 -> 0.5491 with ppl 15.49 -> 19.11. Drift from
   the dense function is what finetuning does. E14/E15 used this residual as the
   decisive gate; it cannot arbitrate a trained model.
5. **Training a lossless configuration only makes it worse at this budget, and
   domain specialisation helps exactly where a division of labour exists.**
   `replica_E4_k1_full` starts at dense quality by construction and ends at 19.11
   (joint) / 19.14 (domain-specialised) / 22.78 (with a full-width shared expert)
   after 80 steps. But `btx_tile_k1_25pct` reaches **21.64** at 1.00x storage and
   25% active width against **23.19** for the same architecture trained on mixed
   data for the same 80 stage-3 steps — 1.55 ppl better, because a 25%-wide tile
   has to earn the missing 75% and pre-specialised experts give the router
   something to route between. At full width the effect vanishes (19.14 vs
   19.11), because an exact copy has nothing left to specialise toward.

**Secondary.** Routed experts do **not** collapse here: usage on the validation
batch is spread (layer 0 `[0.26, 0.25, 0.22, 0.27]`, layer 23
`[0.28, 0.17, 0.07, 0.48]`), so E15's `eff_experts = 33.2/64` failure does not
reproduce under either branching construction — consistent with its cause being
always-active neurons scattered across disjoint slices.

**Cost of the lossless case: 4.00x the dense FFN.** Branching does not make a
model smaller than dense; it makes the per-token *read* smaller. On 8 GB the
binding constraint is storage, so the quantisation ladder remains the route to a
27B on this machine.

**Tooling note.** The driver gained `--self-test` (18 offline checks),
`--dtype`, `--resume` and `--only`. `--resume`/`--only` were added after a
first attempt was killed part-way through the full-width configurations; because
the JSON is rewritten after every configuration, the completed ladder survived
and the expensive configurations were retried on their own. GPU peak for the
heaviest configuration (four full-width experts on four layers) is 5.97 GB of
8.19 GB, which is why it is the one that gets killed first.

## E17 — Does expert offloading (hot in VRAM, cold in RAM) rescue the dense 27B?

**Question.** "Could we not offload the not-used modules to RAM and keep the hot
ones in VRAM?" It is a real technique and it is worth separating from the claim
it is usually attached to.

**Command:**

```
python3 tools/bench_inference.py --spec ornith-1.5-35b-a3b:q2_k:-1:32 \
  --out results/inference_moe.json
python3 tools/bench_inference.py --spec ornith-1.5:35b:-1:32 \
  --out results/inference_moe_q4.json
```

Full analysis: `docs/expert_offloading.md`.

**Outcome — answer: no for the dense 27B, and the reason is the architecture, not
the runtime.** Measured on this machine, same runtime, same 14.3 GB of usable
fast memory:

| model | arch | total/active | quant | file | resident? | decode |
|---|---|---|---:|---:|---:|---:|
| `qwen3.8:latest` | dense | 27.8B / 27.8B | Q4_K_M | 17.0 GB | no | 0.561 tok/s |
| `ornith-1.5-35b-a3b:q2_k` | MoE 256x8 | 35.5B / ~3B | Q2_K | 13.0 GB | **yes** | **19.356 tok/s** |
| `ornith-1.5:35b` | MoE 256x8 | 35.5B / ~3B | Q4_K_M | 22.0 GB | no | 6.450 tok/s |

The two `qwen35moe` rows are the **same architecture** (256 experts, top-8,
expert FFN 512, shared FFN 512) differing only in quantisation, which isolates
the two effects:

1. **Sparsity beats size.** A 35.5B MoE at Q4_K_M is **11.5x** faster than a
   27.8B *dense* model at Q4_K_M. The bigger model wins, because a token reads 8
   of 256 experts instead of every weight.
2. **Residency is worth 3.0x by itself.** The same MoE at 13.0 GB (fits) against
   22.0 GB (does not) goes 19.356 -> 6.450 tok/s.

All three rows are explained by one equation, with bytes read per token as the
numerator: 17.0 GB / 1783 ms ~= 9.5 GB/s effective for the dense model; ~1.83 GB
/ 155 ms ~= 11.8 GB/s for the Q4 MoE; ~1.05 GB / 52 ms ~= 20.3 GB/s for the Q2
MoE. The tier mix sets bandwidth; **sparsity sets the volume**, and volume is
what differs by ~9x.

**Why it cannot help the dense 27B.** Offloading chooses *where each byte comes
from*, never *which bytes are needed* — and in a dense model that set is the
whole model. There is no cold subset to push out. Moving layers to RAM re-labels
17.0 GB per token rather than reducing it, and 17.0 GB does not fit in the 14.3
GB of fast tiers at any placement. The lever that works remains quantisation:
Q3_K_M at 11.8 GB is fully resident, predicted 4.85 tok/s against the measured
0.561.

Also worth recording: `num_gpu = -1` (ollama's default, "offload what fits")
was used in all three runs, so the VRAM/RAM/disk split was already the one the
question proposes. Peak VRAM was 6730-6932 MiB of 8188 in every case. The idea
is not missing from the runtime; it is missing from the model.

**Consequence for the E16 route.** An MoE whose experts are complete modules is
lossless *because* it reads full width, so it saves no bytes — speed and quality
trade along the same axis and offloading does not move that axis. Models that do
win here (`qwen35moe`, Qwen3-30B-A3B, Mixtral) were pretrained with hundreds of
experts, which is the part a retrofit cannot buy at this scale.

---

## E18 — Is the fast MoE actually as good as the slow dense 27B?

**Question.** E17 explained the speed (sparsity + residency). Nothing had
measured whether the fast option *answers* as well. "It ran fast" is not a
quality result, and the whole Path-A case rests on the fast model being good
enough.

**Method.** The repo's own gate (E9/E10), unchanged task definitions, both models
on identical prompts at temperature 0, fixed seed, `--think off`, identical
`num_predict`, checkpoint after every task. Five harness capabilities were added
to make a 0.34 tok/s model comparable with a 43 tok/s one: `--think off`,
`--tasks` (fixed id subset), `--resume` + per-task checkpointing, `--backend
llama`/`--chat-template-kwargs` (score the same tasks through a different engine)
and `--rescore` (replay corrected task definitions over stored responses).

**The dense 27B is the constraint.** 30 tasks on both models: the MoE finished all
30 in **115 s**; the dense model finished 2 in ~25 minutes, projecting to ~6 h. That
run was stopped and kept as `results/eval_qwen3.8-latest_full30_abandoned.json`.
The comparison therefore uses a 19-task subset (3 coding / 4 reasoning / 6 format
/ 6 instruction).

**Result at `num_predict=192` (19 tasks).**

| model | overall | coding | format | instruction | reasoning | capped | excl. capped | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ornith-1.5-35b-a3b:q2_k` | 15/19 = 78.9% | 3/3 | 6/6 | 4/6 | 2/4 | 2 | 15/17 = 88.2% | 55 s |
| `qwen3.8:latest` (dense) | 13/19 = 68.4% | 3/3 | 6/6 | 4/6 | 0/4 | **4** | 13/15 = 86.7% | 3958 s |

**All four of the dense model's reasoning failures were truncations**
(`done_reason: length`): it was cut off mid-solution while reasoning correctly.
The MoE writes more compactly and capped only twice. On everything the cap did not
cut, the two are indistinguishable — and the reasoning column was unmeasured for
one of them.

**Re-running the reasoning column at a budget both can finish
(`num_predict=768`):** both models **4/4**, answering 0.05, 3, 47, 42. The MoE
took **41 s**, the dense model **3517 s** — ~86x. So the small-cap gap was a
harness budget, not capability, and the MoE's full 19-task score at that cap is
17/19 = 89.5%, failing only the two genuine instruction cases (a 6-word answer
where 5 were required, and a forbidden letter).

**Two defects in the repo's own task set, and one in its scorer.** Found by
running it, fixed, and pinned by tests:

1. `r_bat_ball` expected 0.10; the ball is `(1.10-1.00)/2 = 0.05`. This failed the
   *correct* answer — it is why the MoE's row was a FAIL in the first run.
2. `r_modular` expected 1; `7**20 % 13 = 3`. This passed an *incorrect* answer.
3. Truncation was labelled only when the answer text was EMPTY. A capped run that
   narrated first recorded a bare "got 3.0, want 42", which reads as a wrong
   answer; this mis-scored 3 of the dense model's 4 reasoning rows. Rows now carry
   a `truncated` flag, reports print the capped count and the score excluding
   capped rows, and `--rescore` re-derived the flag for reports written before it
   existed.

`test_numeric_expectations_are_arithmetically_correct` re-derives every closed-form
expectation instead of trusting the literal; `test_every_coding_case_is_
satisfiable_by_a_correct_solution` runs a reference solution against every coding
case, because a wrong expected value fails a correct model and must fail the suite
first. `--rescore` then re-scored the completed runs from their stored text
(MoE 14/19 → 15/19) instead of re-generating them.

**Memory safety, sampled rather than assumed.** Over 59 minutes of dense
generation: RAM available never fell below **5.79 GiB**, peak VRAM **6811 of 8188
MiB**, no OOM kill (181 samples, `logs/mem_watch_dense768.log`). The llama.cpp
driver refuses to start unless 6 GiB RAM and 6000 MiB VRAM are free, and it
refused correctly once — while the dense model held 6654 MiB of the card.

**Engine comparison (the surplus question).** Same 19 tasks, same 768 cap, through
llama.cpp at surplus's planner placement `-ngl 42 -ncmoe 24`, f16 KV:
**17/19 = 89.5%, identical to the ollama tag, same two failures.** An engine or
placement changes where bytes come from, not which weights the model needs — now
measured rather than asserted. Two corrections the measurement forced:

- A raw `/completion` request applies no chat template, so the model's `<think>`
  block went into the answer and the same run scored **13/19**. Matching ollama's
  `think: false` requires `chat_template_kwargs={"enable_thinking": false}` on
  `/v1/chat/completions`. A 21-point score difference from a request setting.
- The earlier "×2.59 decode / ×10.4 prefill" claims compared against ollama's
  **32-token cold** baseline. Warm, on the same tasks, ollama runs 43.1 tok/s
  decode and 83.4 tok/s prompt, so the real gains are **×1.12 / ×1.21**, with
  ×1.75 wall for the 19 tasks. The placement stands; the multiples did not.

**surplus's own paths, measured.**`surplus bench` crashes before measuring (the
engine it pins, `llama.cpp-moe-cache`, segfaults on any `-ncmoe` in its
`llama-bench`; upstream runs the same flags to completion). `surplus shim` serves
the model but returns **HTTP 200 with an empty `response`** for every task (a
direct probe: `eval_count: 32`, `response: ""`), so it cannot be scored at all;
quarantined in `logs/surplus_shim_empty_responses.json`. Its auto-plan in the shim
chose `-n-cpu-moe 36` and decoded at **3.41 tok/s** against 48-50 tok/s for the
plan its own solver ranks first. Its default RAM cage is **4 GiB**, computed from
the model's *active* parameters, which is below what a 13 GB MoE needs when 36
layers of experts are host-side. And `--engine-dir` takes a checkout *root*;
passing a `build` directory is silently ignored. `surplus native` refuses
`qwen35moe` by design, so it cannot run any model here.

**Equal quantisation: the architecture question, answered.** §3 compares two
artifacts a user would choose between, but they differ in architecture *and* bits
per weight (2.8 vs 4.9). Holding the quantisation fixed required only a model that
was already on this machine: `ornith-1.5:35b` is the **same `qwen35moe`
architecture at Q4_K_M** (22.0 GB), so against the dense 27B at Q4_K_M (17.0 GB)
it isolates architecture alone. Same runtime, same engine, same 19 tasks.

| run (all Q4_K_M) | coding | format | instruction | reasoning | overall | wall | decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| MoE `ornith-1.5:35b` (22 GB) | 3/3 | 5/6 | 4/6 | **4/4** | 16/19 = 84.2% | 464 s | **6.14 tok/s** |
| dense `qwen3.8:latest` (17 GB) | 3/3 | 6/6 | 4/6 | **4/4** | 13/15 + 4/4 | 3958 s / 3517 s | 0.32 tok/s |

The dense row is its two measured configurations (15 non-reasoning rows at a 192
cap, where nothing truncated, and 4 reasoning rows at 768). Reasoning ties;
the non-reasoning half differs by **one task**, and across all three measured
models the failures move inside the same small set of exact-match constraints
(`i_no_letter_e` fails everywhere; `i_five_words`, `i_lowercase_start` and
`f_json_array` each fail for one model and pass for another). At six tasks per
category and one run each, ±1 is noise.

**So sparsity costs nothing measurable and buys ~19x at equal bits.** The MoE is
not the compromise option for quality; the reason to run Q2_K on this machine is
memory, not quality — at equal bits the 22 GB MoE is 7x *slower* than the 13 GB
one (6.14 vs 43.1 tok/s) because it does not fit the fast tiers either. It still
beats the dense model by reading 8 of 256 experts per token rather than the whole
file.

**Memory safety, second run.** The 22 GB model — the configuration that does not
fit — held a **5.32 GiB** RAM-available floor over 92 samples with a 6934 MiB VRAM
peak, under an armed watchdog (`tools/mem_watchdog.py`, floor 2.0 GiB) that never
fired, on a host with **no free swap**. Writing the watchdog first was not
ceremony: its own tests caught a defect where a *zombie* target still answers
`os.kill(pid, 0)`, so the guard would have spun on a finished run and never
reported. `--pidfile` exists so the guard arms on the eval's real PID rather than
a wrapper's.

**Conclusion for the 27B decision.** The MoE is not the compromise option on this
machine: on this task set it matches the dense 27B while being ~86x faster on the
tasks that need room to think, and at equal quantisation the architecture is not
the thing costing quality. What remains unmeasured is behaviour outside this
19-task set, where Q2_K's 2.8 bpw is the live risk — and the equal-quant run is
the evidence that if you have the RAM for Q4_K_M, use it.
→ `docs/quality_vs_speed.md`, `results/eval_*_short19*.json`,
`results/eval_*_reason768.json`, `results/eval_llama_ngl42_ncmoe24.json`,
`logs/mem_watch_dense768.log`.

---

## E19 — Does a smaller quantisation make the model dumber?

E18 answers the architecture question at equal bits. It does not answer the one a
user asks — "I prefer Q4" — which holds the architecture fixed and moves the bits.
Two measurements were added, one per half of the question. They disagree, and the
disagreement is the finding.

**The gate cannot see the difference, and the matched pair proves it.** The same
`qwen35moe` 35B at Q2_K and at Q4_K_M, identical 19 tasks, identical 768 cap, same
runtime, `--think off`, **nothing capped on either side**:

| 35B MoE, same 19 tasks, 768 cap | Q2_K (13 GB) | Q4_K_M (22 GB) |
|---|---:|---:|
| overall | **17/19** = 89.5% | 16/19 = 84.2% |
| coding / format / instruction / reasoning | 3/3, 6/6, 4/6, 4/4 | 3/3, 5/6, 4/6, 4/4 |
| wall, all 19 tasks | **62.7 s** | 463.5 s |
| decode (median over the 19) | 43.1 tok/s | 6.14 tok/s |

Both runs are ollama with its own offload-what-fits policy, so this compares the two
configurations a user actually gets on an 8 GB card -- 13 GB partly offloaded against
22 GB mostly host-side -- not two placements of one artifact. The wall-time gap is
therefore a residency factor as much as a bits-per-weight one, and both runs reported
`done_reason` consistent with completion (0 capped rows).

Q4_K_M scores one task *worse* — inside the ±1-task band E18 already established — but
these are not the same answers. Comparing the two runs response by response, **13 of
19 differ** (6 byte-identical, the median first divergence at character 1, 9 of the 13
parting within 5 characters), and three tasks flip verdict, two in Q2_K's favour. The
gate does not notice a different output that happens to pass the same check.

**Perplexity does see it.** `experiments/quant_perplexity.py` (new) builds one corpus
from this repo's own prose and code, records its sha256 and byte count, runs
`llama-perplexity` once per quantisation with everything else held fixed, and stores
each run's estimate, its cumulative series, the window it scored, its command, its wall
time and its guard verdict. On a matched pair over an identical token stream
(`-c 512 --chunks 8`):

| matched pair, identical token stream | PPL | vs Q4_K_M |
|---|---:|---:|
| `ornith-1.5:9b` Q4_K_M | 7.0029 +/- 0.409 | -- |
| `ornith-1.5:9b` Q2_K | 7.7128 +/- 0.468 | **+0.71 PPL (+10.1%)** |

Provenance first, because it does not help the argument to hide it: that pair's logs
lived in a temp directory that was cleaned between sessions, so the delta is quoted
from the run record and is **not** backed by a committed `results/` artifact. What is
committed is the 35B's own run — `results/quant_perplexity_35b.json`, Q2_K at
**12.7113 +/- 0.809**. Read the 10% with its bar: 0.71 is ~1.1 sigma against the two
printed standard errors, and paired over identical tokens it is tighter than treating
the runs as independent, but at this window the cost is *consistent and
one-directional* rather than settled. The answer to "does Q2 make it stupid?" is
**no — and not free**: measurably worse in log-likelihood, by a margin near 10% whose
error bar is not yet tight, and completely invisible to the gate above.

**The 35B's own matched pair is not measurable on this host.** The 21 GB Q4_K_M
artifact cannot be scored here at any setting tried: with every expert on the CPU
(`-cmoe`) it was OOM-killed *during load* (RSS 11.32 GiB as MemAvailable fell 11.45 ->
0.76 GiB), and with a partial offload (`-ngl 12`) the batched pass over an 8-of-256 MoE
drove MemAvailable to **0.14 GiB**, and on a repeat **0.11 GiB**, within a minute — at
which point `tools/mem_watchdog.py` terminated it *before* the kernel chose its own
victim. That is the watchdog's first firing after E18's dormant run, and the run is
recorded as a failure: no estimate, an explicit exit code, and a summary line that says
so in words. The Q2_K 35B scores fine the same way -- the committed run took 173 s with
a 9.47 GiB RAM floor (a first run of the same command: 185 s, 8.27 GiB).

**Two defects in the new probe, both found by measuring with it rather than by reading
it.**

1. `fmt_bpw` returned `"?"` for both real GGUFs. It read metadata value-by-value with a
   fixed width per type, lost alignment on the tokenizer's token array, and so never
   reached `general.file_type` — the field that exists to record the quantisation
   silently recorded nothing, and said nothing about it. Rewritten to consume arrays
   (and GGUF v1/v2 uint32 counts) correctly, with a regression test that puts a
   300-element string array in front of the key; it now reads the artifacts as `Q2_K`
   and `Q4_K_M`. It still answers `"?"` for a file that does not state its quantisation,
   because a name read off the filename is not a measurement.
2. The report said "8 chunks" beside a 400 KB corpus, which reads as "the whole
   corpus". llama-perplexity scores `chunks x n_ctx` tokens and prints that number;
   here it is **4,096 tokens**, about the first 4% of a ~100k-token corpus. How this
   surfaced was not flattering: the corpus grew 389,996 -> 397,855 bytes mid-session and
   the Q2_K estimate came back byte-identical, because every added byte fell past the
   end of the window. The probe now records the scored window per run, warns when it is
   smaller than the corpus, and refuses to present runs with different windows as a
   matched pair.

**Practical reading, and what changed.** On this machine the 13 GB Q2_K MoE is still
the one to run, now for two independent reasons rather than one: it fits, *and* the
22 GB Q4_K_M costs **7.4x the wall time** on the same tasks without scoring better. But
"prefer Q4" is a defensible position and the reports should stop implying otherwise —
Q4_K_M is the closer model in log-likelihood. The recommendation for Q2_K here is a
**memory** recommendation, not a claim that the two quants are equivalent.

→ `docs/quality_vs_speed.md` 4.1, `experiments/quant_perplexity.py`,
`tests/test_quant_perplexity.py`, `results/quant_perplexity_35b.json`,
`results/eval_ornith-1.5-35b_short19_768.json`,
`results/eval_ornith-1.5-35b-a3b-q2_k_short19_768.json`, `logs/ppl_q4_35b.watch.log`.

---

## E20 — Does the upcycled 0.8B still behave like the dense 0.8B?

Every number E14 and E15 report is either a perplexity or a weight-space residual
(`vs_original_dense`, `oracle_topk`). Neither answers the question a reader asks
about a converted model: if the dense FFN is swapped for this, does it still answer
the same? E18 answered that for the pretrained 35B-A3B by comparing generated text;
nothing had done it for the upcycled 0.8B, whose only evidence was a perplexity
ratio. `experiments/compare_upcycled_08b.py` (new) closes that gap: it loads the
stock Qwen3.5-0.8B, upcycles the same four layers under the checkpoint's own
configuration, loads the trained experts, and compares both models on identical
tokens.

**Model:** Qwen3.5-0.8B, 4 of 24 layers upcycled (0, 3, 8, 23), 8 experts top-1,
`combine=sum`, `plan=random`, seed 0, checkpoint
`.cache/moeme/poc_ckpt/E8_k1_sum_random_s0_0-3-8-23.pt` at step 300.
**Command:** `python3 experiments/compare_upcycled_08b.py`

**The comparison is on the recorded footing and its control passes.** Same corpus
(`fineweb-edu`, 883,186 tokens, 1638 train / 86 val blocks, `--eval-blocks 4`), same
val window:

| configuration | ppl | |
|---|---:|---|
| dense Qwen3.5-0.8B | 19.5707 | baseline |
| fresh (post-hoc) partition, k=1 | 99.6753 | reproduces the recorded figure exactly |
| trained, k=1 | 29.7929 | **1.52x dense** |
| fresh partition, all 8 experts active | 19.5635 | control, relative error 3.69e-04 |
| trained partition, all 8 experts active | 1597.7289 | **82x dense** |

The dense 19.5707 and the fresh 99.6753 reproduce `results/upcycle_poc_rerun.json`
exactly, and the trained figure is 0.35% from its recorded 29.6900. That residue is
provenance-checked rather than waved through: the per-layer residuals recomputed on
the **loaded** weights reproduce the recorded ones to <=7e-04 (`vs_original_dense`
1.008908 vs 1.008210, `oracle` 0.850050 vs 0.850166, `router` 0.886398 vs 0.886532),
so the checkpoint is the weights the report quotes and the 0.10 perplexity is
evaluation arithmetic, not a different model. The x-dense ratio is 1.52 either way
(1.517 recorded against this 1.522). The **fresh**
all-experts row is the dense-limit control and it passes at 3.7e-04: a partition with
every expert active *is* the dense layer, so the plumbing is exact — the same claim
`results/real_ffn.json` makes on real 27B weights, now at the level of the whole
model. On that fresh partition 5 of 6 continuations are also byte-identical to dense.

**The same control on the trained partition is 82x dense — and that is the finding,
not a bug.** This is `vs_original_dense` exceeding 1.0 (`docs/upcycling_design.md`
§5) stated over the whole model: training re-purposes the experts away from the slices
they were partitioned from, so the *complete* trained partition is 82x dense while
its sparse top-1 version is only 1.52x dense. The experts were not given a cheaper
approximation of the dense function; they were moved to a different one. Only the
fresh row is a control, and reading the trained row without that distinction is the
trap.

**The text: fluent, on-topic, and not the same model.** 6 fixed prompts, 48 greedy
tokens each, identical prompts and identical decoding:

- **0/6 byte-identical.** The first divergence is at token **0, 1, 2, 3, 0, 3** —
  every prompt parts inside three tokens.
- **Mean positional token agreement 0.073**: about 93% of generated tokens differ.
- It is not gibberish, and it often reaches the same answer. Both models answer
  `17 * 23` with **391**, and both write fluent English; the converted model writes
  `if n == 0: return 0` where dense writes `if n <= 1`. But on the first prompt dense
  says *"The capital of France is Paris."* and the converted model says *"...the
  capital of the world."* — fluent, and wrong.

So the answer to the question this repo keeps being asked — does a conversion "work
like" the original — is **no, not on these terms**, and it is visible in the text
rather than only in the statistic: on **4 of 24 converted layers** the model still
speaks the language and often reaches the same answer, while diverging within three
tokens on every prompt. E15's all-24-layer runs are the configurations a 27B
conversion actually needs, and those are 5.6-10.4x dense by perplexity.

**A task-level evaluation is still blocked, and that is a real limit.** This is six
prompts and one perplexity window: one sample per prompt, no sampling temperature,
no contamination analysis. The eval gate cannot close it either —
`experiments/eval_gate.py` scores over HTTP (ollama, llama.cpp), while the converted
model is a PyTorch state dict whose FFN is a custom partitioned MoE with no GGUF
converter, so there is currently no path from it to the gate. The measurement above
bounds the question at the only interface that exists.

→ `experiments/compare_upcycled_08b.py`, `results/upcycled_08b_vs_dense.json`.

---

## Not executed (stated plainly)

- **E18's open threads.** Not run: a second run of any category, so every score
  here is one sample and the one-task gap in the equal-quant comparison is an
  inference rather than a measurement; any placement other than
  `-ngl 42 -ncmoe 24` under the quality gate; token-exact agreement between
  engines (only the *scores* were compared); and any task outside this 19-task
  set, which is exactly where a 2.8 bpw quantisation would be most likely to
  show up. Repairing surplus is also out of scope from here -- its bench
  segfault, its empty shim responses and its 4 GiB cage are its own code, and
  E18 records what they do on this host rather than fixing them.
- **E20's open thread.** The converted model has no task-level score, only
  perplexity and one greedy continuation per prompt: no sampling, no second seed,
  no benchmark. Nothing can give it one today — `experiments/eval_gate.py` scores
  over HTTP, and the converted model is a PyTorch state dict whose FFN is a custom
  partitioned MoE with no GGUF converter, so the prompt set above bounds the
  question at the only interface that exists.
- **E19's open threads.** The 9B perplexity pair was not re-run to produce a committed
  artifact, and it was measured over a 4,096-token window, so the ~10% quantisation
  cost is one window's result with a wide bar rather than a settled magnitude; a
  larger `--chunks` is the way to settle it. Also not run: the 35B's own matched
  perplexity pair (not possible on a 15 GiB host -- see the entry that precedes this
  one), and any quantisation severity between Q2_K and Q4_K_M (Q3_K / IQ quants),
  which is where a pragmatic middle might sit.
- A *complete* evaluation set. A 30-task starter set with executable checks now
exists and has been run (E10), but the handoff's full 30/30/20/20 split, public
benchmarks, and contamination checks are not there. The pilot (E7) measures
routing structure, not task quality.
- Workstream B's runtime work: no llama.cpp/GGUF build, no `io_uring` vs mmap
  comparison, no prefetch policy, no pinned host buffers, no native-MTP or
  speculative-decoding experiments.
- Real 27B activations. The real-weight experiment (E6) uses synthetic inputs,
  so its oracle numbers probe weight structure only.
- CPU-only runs of the qwen3_5 family (blocked: CUDA-only `causal_conv1d`).
- **The upcycling PoC at other points in the grid.** E15 settled the decisive
  cell — E in {8, 32, 64} at k=1, all 24 layers, the ~1.6% active ratio the 27B
  target needs — and a second seed at E=64. **Not** run: k>1 at E=32/64, the
  aux-loss sweep, `combine=softmax`, the `clustered` plan under training, a
  corpus the trained layers have never seen for the ppl number, or any scaling
  study. Those would sharpen the mechanism; none of them is expected to reverse
  a 0.027 deficit.
- **A trained-MoE run at 27B.** Everything here is at 0.8B. The claim is that the
  mechanism fails at small scale on its own gate, not that 27B continued
  pretraining would fail — though E15 removes the reason to try.
- Fine-tuning of anything. No free tier was used, and no claim about fine-tuning
  a 27B model on free tiers is made.
