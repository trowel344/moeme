#!/usr/bin/env bash
# Release parity for the recommended Top-12 imatrix Q4 artifact.
#
# 8 chunks per corpus (not 1): the one-chunk PPL-ratio error bar is larger than
# the differences being compared, so the robust measurement is the release
# verdict. The same-top threshold is 90, not 95: the ceiling study recorded in
# docs/moeme-serving-status.md shows even the near-lossless BF16 model agrees
# with the Q4 reference on only ~92.5% of top-1 tokens on the wiki-prose corpus,
# so 95 is unreachable by any artifact.
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
  --logits .moeme/imatrix8-ref-512.kld \
  --report .moeme/logit-parity-top12-imatrix-q4-chunks8.json \
  --reference-log .moeme/logit-parity-imatrix8-reference.log \
  --candidate-log .moeme/logit-parity-imatrix8-candidate.log \
  --context 512 --chunks 8 \
  --max-ppl-ratio 1.01 --max-mean-kld 0.02 --min-same-top-percent 90
