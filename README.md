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
   output. The router is not the bottleneck (oracle-vs-learned gap ≤ 0.033), so
   better routing cannot fix it.
3. **The target model runs at 0.561 tok/s today** (Q4_K_M), and prefill at
   1.6 tok/s. Predicted independently from hardware constants at 0.57 tok/s.
4. **The bottleneck is storage streaming, not compute.** The SSD measures
   **0.8–1.9 GB/s** against a ~6.6 GB/s datasheet figure, and 2.2 GB/token has
   to come off it at Q4_K_M. *This is the whole problem.*
5. **The fix is aggressive quantisation to full residency, not sparsity.**
   Below ~3.4 bpw the model fits entirely in VRAM+RAM, which predicts ~4–6 tok/s
   — an ~8x gain with **no training and no architecture change**.

Read `docs/feasibility_report.md` then `docs/decision_memo.md`.

## Repository layout

```
src/moeme/ffn.py            dense-equivalent FFN partitioning, partition plans,
                            oracle + learned-router diagnostics, concentration
src/moeme/accounting.py     exact parameter accounting, memory-tier throughput model
tests/                      87 tests: exactness, shapes, plans, oracle bounds, accounting
tools/verify_model_geometry.py   exact param accounting from HF safetensors headers
tools/hf_tensor.py               fetch single tensors via HTTP range requests
tools/bench_hardware.py          non-destructive hardware characterisation
tools/bench_inference.py         end-to-end quantised inference benchmark (ollama)
tools/residency_table.py         what quantisation fits resident, and the tok/s that follows
tools/summarise_results.py       re-print every headline number from results/*.json
experiments/verify_real_ffn.py   partitioning verified on REAL 27B weights
experiments/pilot_sparsity.py    sparsity pilot with real activations
configs/tiers.json               measured constants, with provenance and limitations
docs/                            source validation, feasibility report, decision memo
results/                         raw evidence: JSON + printed tables
```

## Quickstart

```bash
make venv        # .venv reusing system torch, plus pytest
make test        # 87 tests, offline, ~2 s
make doctor      # one-command local diagnostics (~90 s, writes results/hardware.json)
make bench-inference
make verify-sources      # re-derive parameter counts from the real artefacts
```

Optional, requires network and a GPU:

```bash
make verify-real-ffn   # 535 MB ranged download; exactness on real 27B weights
make pilot             # ~1.8 GB download; real activations
```

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
| Usable VRAM (desktop holds ~2.9 GiB) | ~3.6 GiB |
| Usable RAM for weights | ~10.2 GiB |
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
| 0.561 tok/s and the tier model | `make bench-inference`, `python3 tools/residency_table.py` |

## Scope and limitations

Stated up front, because overclaiming here would be worse than the negative
results: the pilot runs at 0.8B scale (that is what fits the free VRAM, and the
qwen3_5 Gated DeltaNet path cannot run on CPU here because the installed
`causal_conv1d` is CUDA-only); the real-weight experiment uses synthetic inputs,
so it probes weight structure rather than end-to-end quality; no benchmark
suite or contamination analysis was run; and the memory-tier model is an
additive ceiling estimator validated within ~2x, not a forecast. Workstream B's
runtime exploration (llama.cpp/GGUF, `io_uring`, prefetch policy, native-MTP
speculative decoding) is **designed but not executed**. Full list in
`docs/feasibility_report.md` §9.

Web search was unavailable throughout, so **all** source verification used
primary artefacts (Hugging Face API, `config.json`, model cards, safetensors
headers). Kaggle and Lightning AI free-tier quotas could not be confirmed and no
claim depends on them.

## Licence and provenance

Study code: do what you like with it. Model artefacts remain under their own
licences (Qwen3.8-27B is Apache-2.0, verified from the model card). No model
weights are redistributed here; `tools/hf_tensor.py` fetches them from the
original repositories on demand.
