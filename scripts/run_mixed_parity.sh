#!/usr/bin/env bash
# Strict three-corpus source-logit parity for the mixed-precision candidate
# (routed experts Q5_K, output head + token embeddings Q6_K, attention Q4_K_M).
set -euo pipefail
cd /home/cleanerbox/Documents/moeme
export PYTHONPATH=src
exec /usr/bin/python3 scripts/logit_parity_gate.py \
  --pipeline-manifest .moeme/top12-parity-manifest.json \
  --reference checkpoints/qwen3.8-27b-gguf-reference/Qwen3.8-27B-Q4_K_M.gguf \
  --candidate artifacts/gguf/moeme-27b-top12-mixed-q4_k_m.gguf \
  --corpus data/calibration/router-v1.txt \
  --corpus scripts/train_sparse_layer.py \
  --corpus README.md \
  --binary /home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-perplexity \
  --logits .moeme/mixed-ref-512.kld \
  --report .moeme/logit-parity-top12-mixed-q4.json \
  --reference-log .moeme/logit-parity-mixed-reference.log \
  --candidate-log .moeme/logit-parity-mixed-candidate.log \
  --context 512 --chunks 1 \
  --max-ppl-ratio 1.01 --max-mean-kld 0.02 --min-same-top-percent 90
