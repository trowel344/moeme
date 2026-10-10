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
- **Exactness: worst relative L2 error 5.081e-07** across 4 plans × E ∈ {8,32,128}
  × 2 input distributions. The partitioning is exactly dense-equivalent on the
  real artefact.
- **Throughput is NOT free:** dense 2.443 ms; E=32 all-active 5.916 ms
  (**2.42x**); E=128 all-active 9.497 ms (**3.89x**). Small-GEMM overhead at
  batch 1.
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

---

## Not executed (stated plainly)

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
- Fine-tuning of anything. No free tier was used, and no claim about fine-tuning
  a 27B model on free tiers is made.
