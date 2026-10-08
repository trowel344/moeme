# MoEMe

**An experiment in converting a dense 27B LLM into a sparse mixture-of-experts
model by surgery alone — and an honest record of where it broke.**

MoEMe takes [Qwen3.8-27B](https://huggingface.co/Qwen) and rewrites its dense
SwiGLU feed-forward networks into a sparse MoE layout, with the goal of paging
expert weights on and off a consumer GPU (8 GB laptop card) so a 27B-class model
can run on hardware that has no business running it.

The conversion works and the resulting model runs. **The sparsity does not hold
up.** This repository is the full source of the experiment, released so that
someone with real compute can pick up where we ran out of it.

> Status: **concluded**. This was a solo hobby experiment run on one laptop plus
> free Kaggle/Colab GPUs. It did not reach its goal. The artifacts and the
> negative results are the point.

---

## TL;DR

| Claim | Result |
| --- | --- |
| Can a dense FFN be mechanically partitioned into a sparse MoE? | **Yes** — exact (relative L2 ~`1e-7`) reconstruction with all routed experts active. |
| Does the converted model run on an 8 GB laptop GPU? | **Yes** — Q5_K_M build at ~`3.2–3.3` tok/s, Q4_K_M at ~`3.8` tok/s (RTX 4060 Laptop, 8 GB). |
| Does it give a real expert-paging speedup? | **Barely.** Only `49.4%` of the artifact bytes live in routed experts, so the *theoretical* ceiling is `1.96x` and the *measured* Top-4 speedup is `1.42x`. |
| Can you actually turn experts *off* without wrecking quality? | **No.** The measured quality budget is a per-layer relative-L2 error of ~`0.01`; at Top-4 the static oracle floor has a **median of `0.45`**. |
| Is layer-selective sparsity (sparse only where it's cheap) enough? | **No.** 1 of 64 layers clears even a loose `0.20` gate; **zero** clear the real `0.01` budget at any `K < 12`. |

**Root cause:** the partition is a *disjoint re-slice* of the existing FFN
intermediate dimension (`F/4` shared + 12 × `F/16` routed = the original `F`).
Top-K therefore **deletes network width** instead of selecting redundant,
independently-trained expert capacity. Real MoEs *multiply* the FFN into
full-width experts and gain capability from extra parameters plus training;
this conversion adds nothing and removes channels.

---

## What was built

A complete, tested pipeline from a Hugging Face checkpoint to a runnable
llama.cpp GGUF:

- `src/moeme/` — core library: FFN partition layout, streaming safetensors
  converter, router training, experiment ledger, runtime contract.
- `scripts/` — the pipeline: dense→MoE conversion, MoE-native importance-matrix
  quantization, source-logit parity gates, capability (`chat_gate`) gates, the
  oracle router sweeps, the local runtime benchmark, and the free-tier cloud
  (Kaggle / Lightning) launcher + result sync.
- `configs/` — the model architecture snapshot and the training/cloud recipes.
- `tests/` — 239 tests covering the layout math, converter, ledger, result
  parsing, and cloud tooling.

The `moeme` runtime patches a stock llama.cpp `Qwen35MoE` graph to load
`qwen35moe.expert_weights_scale` and to honour per-layer active-K overrides
(`MOEME_ACTIVE_K`, `MOEME_ACTIVE_K_LAYERS`). The patch is in
`docs/patches/` upstream of this release; a stock build without it will load the
artifact but produce wrong logits.

## The model

Two GGUF builds are published on Hugging Face:

| Build | Size | Speed (RTX 4060 Laptop) | Gate status |
| --- | ---: | ---: | --- |
| `q5_k_m` (recommended) | 18.53 GiB | ~`3.2–3.3` tok/s | 16/16 capability, all parity gates pass |
| `q4_k_m` (faster) | 16.15 GiB | ~`3.8` tok/s | misses the mean-KLD gate by `0.000102` (`0.020102` vs `0.02`) |

Both are **exact Top-12** conversions: all 12 routed experts are active, so the
model is functionally the source dense model, reorganized and quantized — not a
new or "smarter" model. That is the honest description of the deliverable.

- Hugging Face: **https://huggingface.co/bwn2000/moeme-27b**

## Key measured results

Model geometry: 64 layers (48 linear attention, 16 full attention), hidden
`5120`, FFN intermediate `F = 17408`, native context `262144`, KV cache
`34,816` bytes/token.

**Why sparsity can't pay off (byte accounting):**

| Component | Bytes | Share |
| --- | ---: | ---: |
| Routed experts (can be paged out) | 9.09 GiB | 49.4% |
| Always-active weights | 9.44 GiB | 50.6% |
| **Ceiling** | | **1.96x** |
| Measured Top-4 | | **1.42x** |

An MoE is bandwidth-bound on a single device; routing reduces *FLOPs*, not
*bytes moved*. Half the bytes are always resident, so the whole idea is capped
near `2x` before any quality cost is paid.

**Why turning experts off kills quality (all-64-layer oracle sweep):**

| Statistic (Top-4, `relative_l2`) | Value |
| --- | ---: |
| Best layer (L63) | `0.1626` |
| Median layer | `0.454` |
| Worst layer (L29) | `0.534` |
| Layers passing loose `0.20` gate | 1 / 64 |
| Layers passing real `~0.01` budget | 0 / 64 |

A greedy oracle (near the exhaustive-subset upper bound, so no router could do
better) is what produced those numbers — the deficiency is the **partition
geometry**, not the router. Per-channel selection instead of contiguous blocks
does much better (`0.05` vs `0.15` at `0.50F`), but has no contiguous expert
blocks to page, which discards the only mechanism that was supposed to pay off.

The full experiment record, including the dead ends and the three recipe bugs we
found and fixed, is summarized in the Reddit write-up (see below). Detailed
internal docs are intentionally not shipped in this repository.

## Run

Requires a compatible PyTorch installation.

```bash
ruff check .
PYTHONPATH=src python3 -m pytest
PYTHONPATH=src python3 -m moeme.cli selfcheck
PYTHONPATH=src python3 -m moeme.cli inspect-config configs/qwen3.8-27b.config.json
PYTHONPATH=src python3 -m moeme.cli run
PYTHONPATH=src python3 -m moeme.cli status
PYTHONPATH=src python3 -m moeme.cli diagnose
```

Convert and quantize (needs the source checkpoint and a patched llama.cpp):

```bash
PYTHONPATH=src python3 scripts/convert_moeme_to_gguf.py ...
PYTHONPATH=src python3 scripts/quantize_gguf.py ...
PYTHONPATH=src python3 scripts/logit_parity_gate.py ...
```

Serve the validated model (checksums + release receipts verified before the
server opens):

```bash
PYTHONPATH=src python3 scripts/serve_validated_model.py \
  --binary /path/to/llama-server
```

## Hardware

Everything here was developed on a single laptop: **RTX 4060 Laptop (8 GB
VRAM), 15.7 GiB RAM, 16 cores**, plus free-tier Kaggle/Colab GPUs for the
training campaigns. The 64-layer campaign and the oracle sweeps were run on a
Kaggle Tesla T4.

## Credits & license

- Built on **Qwen3.8-27B** (Apache-2.0) and **llama.cpp** (MIT).
- Code in this repository is licensed **Apache-2.0** — see [LICENSE](LICENSE).
- If you're reading this because you have a cluster and want to try the
  co-activation-training / upcycling route that this hardware couldn't reach:
  please do. That's why it's here.
