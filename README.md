# moeme — Mixture of Cut Modules (MoCM)

A dense model cut into routed modules, trained, emitted as standard GGUF,
and served in a real engine — with every claim measured and committed.

Hardware: NVIDIA RTX 4060 Laptop (8 GB VRAM) / i7-13620H / 15.4 GiB RAM /
1 TB NVMe, plus free-tier cloud GPUs where noted. Nothing here is argued;
everything is in `results/`.

---

## What this is now

The project started as an edge-sparsity feasibility study for Qwen3.8-27B
(`docs/status.md` preserves that verdict). It became an architecture program:

- **MoCM v1** (`src/moeme/mocm.py`): dense FFN cut into disjoint tiles +
  always-on shared expert (activation-ordered) + softmax top-k router.
  Evolved over 10 generations against rotating benchmarks (ppl, mbpp-exec,
  gsm8k-grade, text agreement): quality tracks active width almost linearly,
  seeds reproduce to ±0.01 (`results/mocm_gen{4..10}.json`).
- **Autopsy** (`experiments/mocm_autopsy.py`): shared expert carries
  0.91–0.99 of every layer's output; routers go near-uniform. The failure
  mode is measured, not debated.
- **What actually sparsifies**: narrow distillation of easy layers (4-layer
  25% at 1.15x domain ppl, 60 s training), TEAL-style activation thresholding
  (25% zeros free, zero training), per-prompt static routing (ties per-token
  routing: 21.47 vs 21.51). Full-model conversion fails on every construction
  tried (2–24x) — compounding through depth, measured four ways.
- **Real-27B byte economics** (`experiments/mocm_bytes.py`): participation
  ratio is exactly E on every real layer; oracle damage 0.91 at the mildest
  cut; the 9.78B backbone caps converted speed at ~18 tok/s regardless.
- **Inference efficiency** (`BranchMoEFFN.fast_forward`, `src/moeme/triton_moe.py`):
  batched dispatch is 3.5–7x over the per-expert loop, exact to fp32 1e-7;
  diagnostics cost more than experts at batch 1 (hence the `stats` flag);
  manual static-KV decode +18% end-to-end. Killed with data: naive compile,
  2:4 sparse kernels, threshold kernels, router caching.
- **Surplus integration** (`tools/emit_moe_gguf.py`): converts trained MoCM
  checkpoints into standard `qwen3moe` GGUF (padded tiles, F32/F16/Q8_0/Q4_0,
  tokenizer included) that loads in llama.cpp and surplus native alike.
  Dense-exact validation reproduces dense output token-for-token; the trained
  MoCM-4B (`MoCM-Qwen3-4B-E8K1-F16`) answers Paris identically to dense and
  holds fluent text over 12 tokens where dense itself veers.
- **Resident 27B** (`results/eval_qwen38-q3xl_short19_768.json`): Qwen3.8-27B
  at Q3_K_XL runs resident at 2.35 tok/s decode (4.2x Q4), reasoning and
  coding perfect on the gate; instruction rows show the known think-leakage
  harness artifact.

## Repository layout

```
src/moeme/mocm.py         MoCM v0 (random-cut wrapper) + v1 (shared,
                          activation-ordered, softmax) configs + builders
src/moeme/branch.py       BranchMoEFFN: tiles/replicas, shared experts,
                          forced-expert training, fast_forward dispatch
src/moeme/upcycle.py      trainable partition MoE + quant-aware dequant builds
src/moeme/triton_moe.py   single-launch fused MoE decode kernel (SM89-safe)
src/moeme/ffn.py          dense-equivalent partitioning + oracle diagnostics
src/moeme/accounting.py   parameter accounting + memory-tier throughput model
experiments/mocm_poc.py   MoCM driver: calibrate, build, train, ckpt out
experiments/mocm_evo.py   evolution harness: multi-domain eval, BTX stage,
                          cosine schedule, spotlight graders, checkpoints
experiments/mocm_bench.py fitness table + latency + text face-off harness
experiments/mocm_final.py clean final benchmark (dense vs converted)
experiments/mocm_lora.py  frozen-backbone LoRA expert-choice prototype
experiments/mocm_seq.py   per-sequence domain modules + specialization matrix
experiments/mocm_distill.py  narrow-student distillation (joint + sequential)
experiments/mocm_autopsy.py per-layer drift/error/shared-share measurement
experiments/mocm_domains.py fixed offline domain suite + mbpp/gsm8k graders
experiments/mocm_depth.py MLP/block/attention skip probes
experiments/mocm_decode.py static-KV manual decode + compile comparison
experiments/mocm_bytes.py byte economics on real 27B weights
experiments/mocm_compress.py SVD / pruning / student probes on real weights
tools/emit_moe_gguf.py    MoCM checkpoint -> standard qwen3moe GGUF emitter
tools/                   verify_model_geometry, hf_tensor, bench_hardware,
                         bench_inference, residency_table, model_atlas,
                         mem_watchdog, summarise_results
lightning/               cloud runbooks: README (80GB gate), A100-40-10h,
                         L4-31h, CLI (login -> studio -> stop)
colab/                   mid-scale verification notebook + guide
kaggle/                  batch track (E15 ran here)
docs/                    status verdict, feasibility report, decision memo,
                         upcycling design, branching-vs-slicing, expert
                         offloading, quality-vs-speed, prior work,
                         solution space, experiment log (E1-E28)
results/                 raw evidence: JSON + tables (committed, not generated)
tests/                   399 tests: exactness, MoE math, dispatch equivalence,
                         fast paths, eval harness, atlas, self-teach
```

## Quickstart

```bash
make venv        # .venv reusing system torch, plus pytest
make test        # 399 tests, offline, ~20 s
make doctor      # hardware diagnostics (~90 s, writes results/hardware.json)
make mocm-selftest   # MoCM pipeline check, offline (~2 s, no download)
make upcycle-selftest # upcycling PoC check, offline, tiny model
make branch-selftest  # branching check, offline, CPU
make bench-inference  # quantised inference benchmark (ollama)
make verify-sources   # re-derive parameter counts from real artefacts
```

GPU runs (bitsandbytes needed for quantized backbones):

```bash
# MoCM on 0.8B, late layers, checkpoint out (emitter-ready):
python3 experiments/mocm_poc.py --tag m --n-routed 8 --topk 1 \
  --shared-units 448 --steps 300 --ckpt-out .cache/moeme/m.pt
# Evolution generation with multi-domain scoring:
python3 experiments/mocm_evo.py --gen 4 --out results/gen4.json \
  --child g:E=8:k=1:sh=896:steps=300:lr=3e-5
# Emit a checkpoint to standard qwen3moe GGUF:
python3 tools/emit_moe_gguf.py --base Qwen/Qwen3-4B --ckpt CKPT.pt \
  --layers 32,33,34,35 --experts 8 --top-k 1 --dtype f16 --out model.gguf
```

`make help` lists everything. `docs/status.md` is the one-page verdict;
`docs/experiment_log.md` E22–E28 cover the MoCM era.

## Scope and limitations

Stated up front: proxy experiments run at 0.8–4B (what fits free VRAM);
27B claims rest on range-fetched real weights, the tier model (~2x), and
resident benchmarks — not on training 27B. Task probes sit near floor level
(dense itself scores 1/16, 4/50); matching dense there means "equally weak".
The converted GGUFs are padded-tile exact, not byte-saving. Full-model
conversion fails on every construction tried; partial conversion is the only
regime that works. All of this is measured in `results/`, not asserted here.

## Licence and provenance

Study code: do what you like with it. Model artefacts remain under their own
licences (Qwen is Apache-2.0). No model weights are redistributed here;
download on demand from the original repositories.
