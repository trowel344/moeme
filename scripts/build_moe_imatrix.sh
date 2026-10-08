#!/usr/bin/env bash
set -euo pipefail
cd /home/cleanerbox/Documents/moeme
exec /home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-imatrix \
  -m artifacts/gguf/moeme-27b-top12-q4_k_m.gguf \
  -f data/calibration/router-v1.txt \
  -o .moeme/moeme-top12-imatrix.gguf \
  -c 512 --chunks 64 -ngl 20 -t 8 --no-ppl
