# Lightning AI track: the one 27B experiment per month

Free tier (as reported): ~5 h/month A100-80GB + ~10 h/month A100-40GB.
That buys exactly ONE well-designed run. This file designs it so the hours
are not wasted.

Tier plan (all three files):
- `A100-40-10h.md` — real-activation oracle table (frozen, all 64 layers) +
  gate dry-runs. Run FIRST, on free hours. Replaces synthetic numbers.
- `README.md` (this file) — the month-one 80GB gate: E8/k1 late-4, trained.
- `L4-31h.md` — the 9B program (frozen oracle, depth map, distill, MoCM).
## The run: E15's gate, on the real 27B (not the 0.8B proxy)

At the ratio a conversion needs, does *training* beat the frozen ceiling —
measured on Qwen3.8-27B itself, late-4 layers (60-63), E=8 k=1:

```bash
pip install -q transformers accelerate bitsandbytes sentencepiece datasets huggingface_hub
pip install -q causal-conv1d  # Qwen GDN path; needs ~10 min build if no wheel
huggingface-cli login        # Qwen license
python3 experiments/upcycle_poc.py --model Qwen/Qwen3.8-27B --load-bit 8bit \
  --experts 8 --topks 1 --layers 60 61 62 63 \
  --steps 300 --seq-len 512 --eval-blocks 4 --corpus-chars 4000000 \
  --out results/upcycle_27b_E8.json
```

What it decides: `beats_frozen_oracle` on the report. True (by >0.01) =
training creates structure at 27B scale and the proxy verdict was scale-bound;
False = E15's "no" holds on the real model and the route is closed properly.

## VRAM math (A100-40GB)

- Backbone 8-bit: ~28GB (frozen, `device_map=auto` keeps it on GPU)
- 4 converted layers fp32/bf16: 4 x 267M x (2B + Adam 8B) ~ 10GB
- Activations batch-1 x4 accum, seq 512: ~1-2GB
- Total ~40GB: tight. Fallbacks in order: `--load-bit 4bit` (~14GB base),
  fewer layers (`--layers 62 63`), `--seq-len 256`.

80GB card alternative: same command, `--load-bit` still 8bit (headroom for
all-8-late-layers or E=32 cell if E8 wins and hours remain).

## Cost control

- Dry-run `--self-test` first (offline, seconds) — it caught a device bug
  that would have burned a GPU session once already.
- `--max-seconds 12600` caps training per config (3.5 h) so a slow device
  returns evidence, not a killed session.
- Checkpoints resume (`--resume`); results write after every config.
- Do NOT run E=32/64 in month one. One cell, one verdict.

## If E8 wins

Month two: E=32 k=1 (the 3.1% ratio) same protocol, then the BTX domain
stage (`experiments/branch_poc.py --btx` needs `--load-bit` porting — not
done yet, see `src/moeme/branch.py::from_hf_mlp` which already dequantizes).
