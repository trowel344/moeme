#!/usr/bin/env bash
set -euo pipefail
cd /home/cleanerbox/Documents/moeme
export PYTHONPATH=src
exec /usr/bin/python3 scripts/logit_parity_gate.py \
  --pipeline-manifest .moeme/top12-parity-manifest.json \
  --reference checkpoints/qwen3.8-27b-gguf-reference/Qwen3.8-27B-Q4_K_M.gguf \
  --candidate artifacts/gguf/moeme-27b-top12-imatrix-q4_k_m.gguf \
  --corpus data/calibration/router-v1.txt \
  --corpus scripts/train_sparse_layer.py \
  --corpus README.md \
  --binary /home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-perplexity \
  --logits .moeme/imatrix-ref-512.kld \
  --report .moeme/logit-parity-top12-imatrix-q4.json \
  --reference-log .moeme/logit-parity-imatrix-reference.log \
  --candidate-log .moeme/logit-parity-imatrix-candidate.log \
  --context 512 --chunks 1 \
  --max-ppl-ratio 1.01 --max-mean-kld 0.02 --min-same-top-percent 90
