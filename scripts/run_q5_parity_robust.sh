#!/usr/bin/env bash
# Statistically stable (8 chunks per corpus) source-logit parity for the
# Q5_K_M candidate. The single-chunk release receipt has a PPL-ratio error bar
# larger than the quantities being compared, so the robust verdict is the one
# to trust; this writes its own receipt and leaves the one-chunk path untouched.
set -euo pipefail
cd /home/cleanerbox/Documents/moeme
export PYTHONPATH=src
exec /usr/bin/python3 scripts/logit_parity_gate.py \
  --pipeline-manifest .moeme/top12-parity-manifest.json \
  --reference checkpoints/qwen3.8-27b-gguf-reference/Qwen3.8-27B-Q4_K_M.gguf \
  --candidate artifacts/gguf/moeme-27b-top12-imatrix-q5_k_m.gguf \
  --corpus data/calibration/router-v1.txt \
  --corpus scripts/train_sparse_layer.py \
  --corpus README.md \
  --binary /home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-perplexity \
  --logits .moeme/q5-ref-512.kld \
  --report .moeme/logit-parity-top12-imatrix-q5-chunks8.json \
  --reference-log .moeme/logit-parity-q5-reference.log \
  --candidate-log .moeme/logit-parity-q5-candidate.log \
  --context 512 --chunks 8 \
  --max-ppl-ratio 1.01 --max-mean-kld 0.02 --min-same-top-percent 90
