# Expert offloading: keep the hot modules in VRAM, push the cold ones to RAM

**The short answer: yes, and it is already happening — but it cannot rescue the
dense 27B, because a dense layer has no cold modules.** A token reads *every*
FFN weight in a dense model, so "hot" means "all of it". Offloading works by not
reading the unused experts, and that requires the model to have unused experts.

This document is the measurement. All three numbers were taken on this machine
with the same runtime (ollama), one variable at a time.

## The three points, measured

| model | arch | total / active | quant | file | fits 14.3 GB fast memory? | decode |
|---|---|---|---:|---:|---:|---:|
| `qwen3.8:latest` | **dense** | 27.8B / 27.8B | Q4_K_M | 17.0 GB | no (2.7 GB streams) | **0.561 tok/s** |
| `ornith-1.5-35b-a3b:q2_k` | MoE 256×8 | 35.5B / ~3B | Q2_K | 13.0 GB | **yes** | **19.356 tok/s** |
| `ornith-1.5:35b` | MoE 256×8 | 35.5B / ~3B | Q4_K_M | 22.0 GB | no (7.7 GB streams) | **6.450 tok/s** |

`qwen35moe` geometry (from the GGUF headers): `expert_count 256`,
`expert_used_count 8`, `expert_feed_forward_length 512`,
`expert_shared_feed_forward_length 512`.

Raw: `results/inference_moe.json`, `results/inference_moe_q4.json`,
`results/inference_27b.json`.

Two facts fall straight out of the table:

* **Sparsity beats size.** A *35.5B* MoE at Q4_K_M decodes **11.5x** faster than a
  *27.8B* dense model at the same quantisation, on the same machine. The MoE is
  the bigger model and wins.
* **Residency is worth 3x on its own.** The *same* MoE, only 13.0 GB instead of
  22.0 GB, goes from 6.450 to 19.356 tok/s — a **3.0x** gain from nothing but
  fitting in fast memory.

## Why, quantitatively

The whole table is one equation: **decode time = (bytes read per token) / (the
bandwidth of whichever tiers hold those bytes)**.

Measured tiers (`configs/tiers.json`): GPU memory 247 GB/s, RAM 40.7 GB/s
(memcpy; 24.4 GB/s single-thread read), NVMe **0.82-1.9 GB/s**. Usable fast
memory is **3.8 GiB VRAM + 10.2 GB RAM = 14.3 GB**.

| model | bytes read per token | measured | implied effective bandwidth |
|---|---:|---:|---:|
| dense 27.8B Q4_K_M | 17.0 GB (all of it) | 0.561 tok/s = 1783 ms | ~9.5 GB/s (RAM + disk mix) |
| MoE 35.5B Q4_K_M | ~1.83 GB (8 of 256 experts) | 6.450 tok/s = 155 ms | ~11.8 GB/s (RAM + disk mix) |
| MoE 35.5B Q2_K | ~1.05 GB | 19.356 tok/s = 52 ms | ~20.3 GB/s (RAM, resident) |

The MoE rows and the dense row pull bytes at *comparable effective bandwidth*.
The whole difference is the numerator: **the MoE needs ~9x fewer bytes per
token**, and because the 13 GB version also fits entirely in RAM+VRAM it gets the
tier's full speed instead of a disk-blended one.

Note what the `qwen35moe` rows say about caching. With 256 experts and top-8,
the working set over a generation is spread across a large fraction of the
experts, so the hit rate is far from 100% — the Q4 row only reaches ~11.8 GB/s
because a good part of it is landing in RAM rather than disk. Offloading is
therefore not a switch that makes the cold part free; it is a *hit-rate* problem.

This also explains, retroactively, why ollama's default (`num_gpu = -1`, i.e.
"offload what fits") was the right setting in all three runs: it is already
doing the VRAM/RAM/disk split the question asks about. Peak VRAM was 6730-6932
MiB of 8188 in every case, with the remainder in RAM.

## Why this does not rescue Qwen3.8-27B

Offloading chooses *where each byte comes from*. It cannot choose *which bytes
are needed*, and in a dense model that set is the whole model:

```
dense:  bytes per token  =  total parameters        (17.0 GB at Q4_K_M)
MoE:    bytes per token  =  active parameters       (~1.83 GB at Q4_K_M)
```

So there is nothing to declare cold. Moving layers to RAM does not shrink the
per-token read — it re-labels it, and 17.0 GB still has to cross into compute
every token. Worse, the fast tiers total 14.3 GB, so no arrangement of the dense
27B is fully resident at Q4.

What *does* work is making the model smaller, which is the same conclusion the
residency table reached:

| quant | size | fits? | predicted |
|---|---:|---|---:|
| Q4_K_M | 17.0 GB | no | 0.48 tok/s (measured 0.561) |
| Q3_K_M | 11.8 GB | **yes** | **4.85 tok/s** |
| Q2_K | 9.7 GB | yes | 6.44 tok/s |

Q3_K_M puts the entire 27B in the fast tiers, at which point there is again
nothing to offload — and the decode rate improves ~9x over the measured Q4
number purely because the disk leaves the path.

## The consequence for the MoE conversion idea

E16 showed that an MoE whose experts are **complete modules** is lossless at
`top_k = 1` — and measured a price: it reads a *full-width* FFN per token, so it
saves no bytes. That is not an accident of the implementation; it is why the
lossless case is lossless. **A conversion is only a speed win if the resulting
model activates a small fraction of its experts**, i.e. if `top_k * expert_units`
is a small share of the dense FFN width — which is exactly the regime where E16
measured the quality loss (residual 0.63-0.89, ppl 21.6-45.6 against a dense
15.48). Speed and quality trade along the same axis; offloading does not move it.

The models that get away with it (`qwen35moe` here, Qwen3-30B-A3B, Mixtral) do
so because they were **pretrained** with 256 experts and a large total budget, so
each of the 8 active experts is genuinely competent. That is the thing a
retrofit cannot buy at this scale.

## What to do instead

1. **On this laptop, run a model that already has native sparse activation.** At
   19.4 tok/s the 35.5B-A3B at Q2_K is ~35x more usable than the dense 27B, and
   it is a *larger* model. If quality is the concern, spend the budget on a
   better-quantised MoE that still fits 14.3 GB rather than on a bigger dense
   model that does not.
2. **For the dense 27B, quantise to fit.** Q3_K_M (11.8 GB) is the largest
   quantisation that is fully resident here, worth ~4.85 tok/s predicted against
   0.561 measured. That gate is worth running.
3. **If designing a hybrid, allocate by bytes-per-token, not by parameter
   count.** The quantity that predicts decode rate on this machine is
   `active_params * bytes_per_param / mix_bandwidth`. Anything that does not
   reduce the numerator — including a lossless expert conversion — cannot help,
   and the tier split only sets the denominator.

## Limitations

* One runtime (ollama), one machine, one prompt, `num_predict = 32`. Decode rate
  on a longer generation improves slightly as the page cache warms; these are
  near-cold-start figures and are therefore conservative.
* `num_gpu` was `-1` (runtime default) in all three runs, so the VRAM/RAM/disk
  split was chosen by the runtime, not by me. That is deliberate — it measures
  the configuration a user would actually get — but it means the tier split is
  not independently controlled in this comparison.
* The "bytes read per token" column is derived (`active_params * bytes_per_param`
  from the standard bpw of each quantisation), not instrumented. The measured
  decode rates and VRAM are direct.
* Q2_K at 2.8 bpw is an aggressive quantisation; the 19.4 tok/s figure is a
  speed measurement and says nothing about output quality, which was not
  evaluated here.
* Prefill is the other half of the problem and is much worse (3.8-11.6 tok/s
  across these runs). Nothing in this document addresses time-to-first-token,
  which is what makes long prompts painful.
* No swap pressure was observed in these runs (minimum available RAM 8.86-9.29
  GB), so the numbers are not contaminated by thrashing — but they were taken
  with the models the only large consumers of memory.
