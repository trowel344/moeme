#!/usr/bin/env bash
# Re-quantize the Top-12 BF16 staging artifact to Q5_K_M, guided by the
# MoE-native importance matrix. Q5_K_M is the highest precision that still fits
# the 16 GiB host RAM + 8 GB VRAM budget (~19.9 GB vs ~23.6 GB combined), and it
# is a much larger precision jump than the failed mixed-precision Q4_K_M recipe:
# the routed expert gate/up/down projections all leave Q4_K/Q8_0 for Q5_K/Q8_0.
set -euo pipefail
cd /home/cleanerbox/Documents/moeme
export PYTHONPATH=src
exec /usr/bin/python3 scripts/quantize_gguf.py \
  --binary /home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-quantize \
  --input artifacts/gguf/moeme-27b-top12-bf16.gguf \
  --output artifacts/gguf/moeme-27b-top12-imatrix-q5_k_m.gguf \
  --type Q5_K_M \
  --imatrix .moeme/moeme-top12-imatrix.gguf \
  --threads 8 \
  --log .moeme/requantize-q5.log
