# Feasibility Report — Qwen3.8-27B-class assistant on an 8 GB / 16 GB laptop

**Bottom line.** The stated target ("~27B total, ~9B active, near-Qwen3.8-27B
quality, on 8 GB VRAM / 16 GB RAM") is **not reachable by converting
Qwen3.8-27B**, for a reason that is now proven arithmetically, not asserted. A
useful assistant *is* reachable on this hardware via a different route
(aggressive quantisation to full residency), and that route is cheap and
measurable. Details in `docs/decision_memo.md`.

Everything below was measured or computed on the target machine; nothing is
copied from a datasheet. Reproduce with `make doctor`, `make verify-sources`,
`make verify-real-ffn`, `make pilot`.

---

## 1. Verified architecture

Qwen3.8-27B is a **dense** 27.78B-parameter hybrid-attention vision-language
model (48 Gated DeltaNet layers + 16 gated-attention layers), Apache-2.0, with a
native MTP head. Full accounting in `docs/source_validation.md`. Two facts drive
everything else:

- **FFN = 17,112,760,320 params (61.6%).**
- **Non-FFN text backbone = 9,783,238,144 params (9.78B)** and is active on
  every token.

## 2. Verified constraints (all measured on this machine)

| Quantity | Measured | Note |
|---|---:|---|
| VRAM total | 8188 MiB | RTX 4060 Laptop |
| VRAM **usable** | **~3.6 GiB** | the desktop already holds ~2.9 GiB |
| RAM total | 15.4 GiB | |
| RAM **usable** for weights | **~10.2 GiB** | swap was already 3.2/4.0 GiB in use |
| GPU memory bandwidth | 246.8 GB/s | measured with a real kernel |
| CPU RAM bandwidth | 40.7 GB/s | memcpy; naive torch bf16 GEMV only 19.3 GB/s |
| **NVMe cold read** | **0.82–1.9 GB/s** | 3 independent methods; datasheet says ~6.6 |
| NVMe random 4 KiB | 11,419 IOPS, 88 us | |
| NVMe cold 1 MiB touch | 1.38 ms, 0.76 GB/s | mmap, page cache dropped per touch |

**End-to-end baselines** (ollama, real measurement):

| Model | Size | Decode | Prefill | Peak VRAM |
|---|---:|---:|---:|---:|
| `qwen3:4b` | 2.5 GB | **60.7 tok/s** | 560 tok/s | 6798 MiB |
| `ornith-1.5:9b` | 6.6 GB | **24.6 tok/s** | 139 tok/s | 6884 MiB |
| **`qwen3.8:latest` (the target, Q4_K_M)** | 17.0 GB | **0.561 tok/s** | **1.6 tok/s** | 6884 MiB |

The 27B at Q4_K_M is **~44x slower than a 9B** on this machine, and its prefill
of 1.6 tok/s means even filling a modest prompt takes minutes. It is not usable
for interactive work as-is.

## 3. Why the 9B-active target is non-trivial — the arithmetic

This is the load-bearing finding, and it is a proof rather than an opinion.

For any model keeping Qwen3.8-27B's backbone:

```
active_params_per_token  =  irreducible_backbone  +  active_FFN
```

The irreducible backbone — embeddings, LM head, Gated DeltaNet blocks, full
attention, norms — is **9,783,238,144 = 9.78B params** and is read on every
single token. Routing the FFN cannot reduce it. So:

```
to reach 9.0B active:  active_FFN = 9.0B - 9.783B = -0.78B   <-- impossible
```

**A perfect, free, lossless FFN router cannot reach the 9B-active target, because
the untouched backbone already exceeds it.** Reaching 9B active requires
shrinking or routing the *backbone* (attention + Gated DeltaNet + embeddings),
which is a retraining problem, not a routing tweak. This is asserted as an
executable test:
`tests/test_accounting.py::test_no_ffn_routing_can_reach_9b_active`.

Two further consequences of the same accounting:

- **The LM head alone is a real per-token cost.** vocab 248,320 x hidden 5120 =
  1.27B params, read every token. At 4.9 bpw that is ~0.78 GB/token; on CPU at
  40.7 GB/s it costs ~19 ms/token *by itself*, capping decode near 52 tok/s no
  matter how fast everything else is. Large-vocab heads are not free on the
  memory-bound path.
- **There is no natural sparsity to exploit.** See §4.

## 4. What the sparsity pilot actually found (decisive)

Full method in `experiments/pilot_sparsity.py`; raw data in
`results/pilot_sparsity.json`, table in `results/pilot_curve.txt`.

Setup: Qwen3.5-0.8B — the *same* `qwen3_5_text` architecture family as the 27B,
same 3:1 DeltaNet:attention pattern, FFN expansion 3.5 vs the 27B's 3.4 — with
**real weights and real activations** from a real forward pass.

**Result: at 12.5% active compute, even a perfect oracle router loses 42–81% of
the layer output.**

| Layer | Best oracle residual @ 12.5% active | Learned router | Router top-1 acc |
|---|---:|---:|---:|
| 0 | 0.681 | 0.807 | 0.348 |
| 3 | 0.802 | 0.872 | 0.309 |
| 8 | 0.811 | 0.857 | 0.254 |
| 23 | 0.417 | 0.449 | 0.809 |

(residual = relative L2 error of the k-expert reconstruction vs the true dense
layer output; 0 is exact, 1 is as bad as predicting zero)

Three sub-findings, each independently important:

1. **The partition sets the ceiling, not the router.** Oracle vs learned-router
   gap is only 0.0001–0.033. Training a better router cannot fix this — the
   ceiling is what a top-k selection can capture from a post-hoc split, and it
   is bad. This directly answers the handoff's "evaluate oracle selection …
   before spending training compute": the answer is *don't spend it*.
2. **Participation ratio ≈ E.** Energy is spread almost perfectly uniformly
   across partitions (PR = 31.6 of 32, 127 of 128 on the real 27B weights). The
   oracle residual only approaches 0 when k = E, i.e. when nothing is sparse at
   all. A trained dense FFN has no latent expert structure to discover.
3. **Weight-similarity clustering degenerates.** The k-means plan looks
   attractive (oracle residual 0.019 on the 27B) but is an artefact: for E=128 a
   **single cluster holds 9,014 of 17,408 units (52%)** — one giant expert plus
   many singletons. "Top 16 of 128" then means keeping half the parameters.
   Reporting this as a win would have been a mistake.

**Interpretation:** post-hoc FFN→MoE conversion of a trained dense model is not
viable. Any model at 9B active with near-27B quality must have its experts
*trained*, which is a pretraining-scale effort — far beyond free tiers.

## 5. What else is proven, and what is not

| Claim | Status | Evidence |
|---|---|---|
| Dense FFN partitioning is exactly dense-equivalent | **Proven** | worst rel. L2 error **5.1e-07** on real Qwen3.8-27B layer-1 weights, 4 plans x E in {8,32,128} x 2 input distributions; 87 unit tests pass |
| Partitioning is free | **Disproven** | 2.42x slower at E=32, 3.89x at E=128 vs dense, batch-1 GPU (small-GEMM overhead) |
| Post-hoc FFN sparsification can work | **Disproven** | §4 |
| 9B active reachable from this backbone | **Disproven** | §3 |
| 27B Q4_K_M runs at ~0.56 tok/s here | **Proven (measured)** | and the first-principles tier model independently predicted 0.57 |
| Memory-tier model is predictive | **Proven within ~2x** | 3/3 measured configs; 2% error on the 27B case |
| Free tiers allow 27B fine-tuning | **Not credible** | see decision memo |
| Kaggle/Lightning exact quotas | **Unverified** | their pages returned no readable quota text |
| End-to-end task quality of any converted model | **Not measured** | needs real 27B activations; explicitly out of scope here |

## 6. The opening that does exist

Decode is bandwidth-bound, so the only regime with usable speed is **full
residency**. From `tools/residency_table.py` (measured tier constants):

| Quant | bpw | Model size | Streamed/token | Predicted | Regime |
|---|---:|---:|---:|---:|---|
| Q2_K | 2.8 | 9.7 GB | 0.00 | **6.27 tok/s** | resident |
| Q3_K_M | 3.4 | 11.8 GB | 0.00 | **4.74 tok/s** | resident |
| Q4_K_S | 4.4 | 15.3 GB | 0.46 | 1.69 tok/s | disk-bound |
| Q4_K_M | 4.9 | 17.0 GB | 2.20 | **0.57 tok/s** | disk-bound (measured: 0.561) |

Qwen3.8-27B at **Q3_K_M or below becomes fully resident** and should go from
0.56 tok/s to roughly 4–6 tok/s — an ~8x improvement for free, with **no
training and no architecture change**. That is the highest-value experiment on
the table, and it costs one download.

## 7. Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Q3/Q2 quantisation degrades coding/reasoning quality | **High** — this is now the *critical path* | Measure it directly: held-out task set at Q3 vs Q4 vs Q5, on this machine |
| SSD is anomalously slow; a healthy drive changes the design space | Medium | Check PCIe link width + drive health; re-run `make doctor` |
| `causal_conv1d` CUDA-only blocks CPU-offload designs | Medium | Already blocks CPU experimentation on this family; fix or use llama.cpp/GGUF |
| MoE conversion attempted anyway, wasting effort | Medium | §3 and §4 are the written basis for declining; both are executable tests |
| Tier model over-trusts itself | Low | Documented as a ceiling estimator validated within ~2x, not a forecast |
| Machine is a daily driver (desktop already holds 2.9 GiB VRAM) | Low | All tooling is non-destructive; temp files self-delete; GPU allocations <= 1 GiB |

## 8. Concrete initial experiments (in priority order)

1. **Quantisation quality ladder (no training, ~1 download).** Fetch
   Qwen3.8-27B at Q3_K_M / IQ3 / Q2_K; measure decode rate, VRAM/RAM peak, and
   held-out score vs the Q4_K_M baseline. Gate: >=95% of Q4 on coding and
   reasoning tasks, and >=4 tok/s. *This is the whole decision.*
2. **Fix or bound the SSD anomaly.** Check `lspci -vv` link width/state and
   drive health; re-measure. If the drive is being throttled, streaming designs
   become far more attractive than they look today.
3. **Build the held-out evaluation set** the handoff specifies (30 coding, 30
   reasoning/maths, 20 instruction/tool-calling, 20 chat) with executable
   checks, so step 1's gate is real rather than a vibe.
4. **Only if 1 fails:** evaluate a *trained* small MoE. Budget-check first —
   §3 and §4 imply this is a pretraining project, not an afternoon.

## 9. Limitations of this report

- The pilot ran at 0.8B-scale because that is what fits the free VRAM and
  because the 27B's Gated DeltaNet path cannot run on CPU here. The *structure*
  findings (uniform energy, partition-bound ceiling) are the kind that scale
  with width, but they were not re-measured at 27B.
- No real 27B activations were captured; the real-weight experiment uses
  synthetic inputs, so its oracle numbers probe *weight* structure only.
- The calibration corpus is 31.7 k characters of self-authored and repository
  code. It is reproducible and uncontaminated, but it is not a benchmark.
- Only CUDA was tested. No Vulkan/ROCm, no llama.cpp build, no `io_uring` work —
  the runtime-exploration items in the handoff's workstream B are *not* done.
- Memory-tier predictions are additive and single-stream; real runtimes overlap
  CPU and GPU work, which is why the model over-predicts some configurations by
  up to ~1.7x.
