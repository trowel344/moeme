# Source Validation Notes

**Method.** Everything below was verified against *primary* artefacts: the
Hugging Face model API, official `config.json` files, the official model cards,
and the safetensors weight headers read directly over HTTP range requests
(`tools/verify_model_geometry.py`). The web-search tool was unavailable for the
whole session, so **no secondary sources were used** — which is the stronger
path anyway, since parameter counts read from the real weight headers cannot be
wrong in the way a blog post can.

Every number here is reproducible: `make verify-sources`.

---

## 1. Nomenclature: what actually exists

The handoff referred to "Qwen3.8-27B", "Qwen3.8-Next", and
"DeepSeek V4.1 Flash/Pro" as things to verify rather than assume. They were all
checked. **All three exist**, but one of them is not what the handoff assumed.

| Name | Status | Evidence |
|---|---|---|
| `Qwen/Qwen3.8-27B` | **Real**, official Qwen org, Apache-2.0 | HF API: 6.77M downloads; `config.json`; model card |
| `Qwen/Qwen3.8-Flash-Next` | **Real**, official Qwen org | HF API; `config.json` — **180.0B params, not 27B** |
| `deepseek-ai/DeepSeek-V4.1-Flash` | **Real**, official DeepSeek org | HF API: 1.36M downloads |
| `Qwen/Qwen3-Next-80B-A3B-*` | Real, prior generation | HF API |
| "Qwen3.8-Next" as a ~27B model | **Not verified — no such artefact found** | `Qwen3.8-Flash-Next` is 180B |

### The two framing corrections that matter

**(a) Qwen3.8-27B is DENSE, not a Mixture-of-Experts.** Its `config.json` has no
`num_experts` / `num_experts_per_tok` fields anywhere. All 27.78B parameters are
active on every token. There is no "9B active" variant to discover or extract —
the requested 9B-active property would have to be *created*.

**(b) The family's MoE model is 180B, not 27B.** Verified from its weight
headers: `Qwen/Qwen3.8-Flash-Next` = **179,999,981,459 params**, of which
121.4B is MoE expert FFN and 51.2B is an n-gram memory embedding (20M n-gram
vocab x 2560). It cannot run on 8 GB VRAM in any quantisation.

This means the handoff's target — "~27B total, ~9B active, near-Qwen3.8-27B
quality" — describes a model that **does not exist** in this family. It would
have to be trained.

---

## 2. Exact geometry of Qwen3.8-27B (from the weight headers)

Read from the 1199 safetensors tensor headers, not estimated.

| Component | Parameters | Share |
|---|---:|---:|
| **FFN** (gate + up + down; 17408 intermediate) | **17,112,760,320** | 61.60% |
| Gated DeltaNet block (linear attention) | 5,562,044,928 | 20.02% |
| Full attention block | 1,677,721,600 | 6.04% |
| Token embedding (248,320 x 5120, untied) | 1,271,398,400 | 4.58% |
| LM head (untied) | 1,271,398,400 | 4.58% |
| Vision tower (27 blocks, hidden 1152) | 460,730,096 | 1.66% |
| MTP head | 424,699,392 | 1.53% |
| RMSNorms | 674,816 | 0.002% |
| **Total** | **27,781,427,952** | 100% |

- **Text tower alone: 26,895,998,464** (total minus vision tower and MTP head).
- Per-layer FFN is uniform: **267,386,880** in every one of the 64 layers.
- `hidden_size` 5120, `intermediate_size` 17408, 64 layers, vocab 248,320.
- 48 Gated DeltaNet layers + 16 gated full-attention layers, in a strict
  3:1 repeating pattern (`full_attention_interval` = 4).
- `q_proj` is **fused query+gate** (5120 -> 12288); `k_proj`/`v_proj` are
  4 KV heads x 256.
- **Native MTP is present in the weights** (`mtp.*`: `fc`, one full-attention
  transformer layer, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, 424.7M
  params). The model card states it was "trained with multiple steps".
- Context 262,144 native, extensible to 1,000,000. License Apache-2.0.
- Weights are BF16, 55,562,855,904 bytes.

## 3. Verification of the previous chat's parameter split

The handoff contained second-hand claims of "~17.1B FFN params and ~9.8B other
params" that it asked to be independently verified. **Both are correct**, and
the scope is now pinned down exactly:

| Claim | Verified value | Verdict |
|---|---:|---|
| FFN params ~17.1B | 17,112,760,320 | **Correct** |
| "Other" params ~9.8B | 9,783,238,144 | **Correct**, where "other" = text tower − FFN − vision tower − MTP head |

The "other" 9.783B decomposes as 5.562B Gated DeltaNet + 1.678B attention +
1.271B embedding + 1.271B LM head + norms. This decomposition is the basis of
the central feasibility finding in `docs/feasibility_report.md`.

## 4. Local artefact check

The machine already holds `qwen3.8:latest` in ollama. Interrogating it confirms
the upstream facts independently: `parameters 27.3B`, `quantization Q4_K_M`,
`context length 262144`, `architecture qwen35`, `embedding length 5120`,
capabilities `completion / vision / tools / thinking`, Apache-2.0, 17 GB on
disk. 17.0 GB / 27.78B params = **4.9 bits per parameter**, which is the figure
used throughout the throughput modelling.

## 5. Environment facts discovered while working

- **`causal_conv1d` is CUDA-only here** (`Expected x.is_cuda() to be true`).
  The qwen3_5 Gated DeltaNet path therefore **cannot run on CPU** through
  `transformers` in this environment. This blocks CPU-only experimentation with
  the whole Qwen3.5/3.8 architecture family and is worth fixing before any
  CPU-offload design work.
- The installed stack is unusually complete: torch 2.11.0+cu13, transformers
  5.6.2, flash-attn 3, causal_conv1d, safetensors 0.7, docker, nvcc, ollama.
- Docker is present, so llama.cpp/GGUF experiments can be run in a container
  without touching the host toolchain.

## 6. What could NOT be verified

Stated plainly rather than filled in with plausible-sounding numbers.

1. **Free-tier GPU limits are only partially verified.** The Colab free tier's
   12-hour maximum runtime was confirmed from Google's own FAQ page. The Kaggle
   and Lightning AI pages returned JavaScript application shells with no
   readable quota text, so **their specific GPU models, weekly hour caps and
   verification requirements remain unconfirmed**. No claim about being able to
   fine-tune a 27B model on free tiers is made anywhere in this project — see
   `docs/decision_memo.md` for the arithmetic showing why that is not close.
2. **The NVMe anomaly is unexplained.** The drive is a Micron 3400 1TB (PCIe
   Gen4, datasheet sequential read ~6.6 GB/s) but measures **0.8–1.9 GB/s**
   across three independent methods. The cause was not established; the
   candidate explanations are recorded in `configs/tiers.json`. Checking the
   PCIe link width and drive health is a cheap, high-value follow-up, because a
   healthy drive would roughly quadruple the streaming budget.
3. **No benchmark-suite or contamination analysis was performed.** The pilot
   study measures routing structure, not task quality; its calibration corpus is
   small and self-authored. Scope is stated in each script's docstring.
