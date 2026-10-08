#!/usr/bin/env bash
# Re-quantize the Top-12 BF16 staging artifact with mixed precision: the routed
# expert gate/up projections at Q5_K and the token embedding at Q6_K, guided by
# the MoE-native importance matrix. Everything else stays at the Q4_K_M base
# type, which already assigns a high fallback precision to the tensors that
# cannot use k-quants.
#
# NOTE: do NOT widen the expert pattern to `ffn_.*_exps`. The Q4_K_M mix fails
# to k-quantize ffn_down_exps (ncols 1088 is not divisible by 256) and falls
# back to Q8_0; forcing it to Q5_K instead lands on the much worse Q5_1
# fallback and measurably degrades perplexity and top-token agreement.
set -euo pipefail
cd /home/cleanerbox/Documents/moeme
export PYTHONPATH=src
exec /usr/bin/python3 scripts/quantize_gguf.py \
  --binary /home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-quantize \
  --input artifacts/gguf/moeme-27b-top12-bf16.gguf \
  --output artifacts/gguf/moeme-27b-top12-mixed-q4_k_m.gguf \
  --type Q4_K_M \
  --imatrix .moeme/moeme-top12-imatrix.gguf \
  --tensor-type "ffn_gate_exps=Q5_K" \
  --tensor-type "ffn_up_exps=Q5_K" \
  --output-tensor-type Q6_K \
  --token-embedding-type Q6_K \
  --threads 8 \
  --log .moeme/requantize-mixed.log
