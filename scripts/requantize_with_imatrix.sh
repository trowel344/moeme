#!/usr/bin/env bash
set -euo pipefail
cd /home/cleanerbox/Documents/moeme
rm -f artifacts/gguf/.moeme-27b-top12-imatrix-q4.partial
exec /home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-quantize \
  --imatrix .moeme/moeme-top12-imatrix.gguf \
  artifacts/gguf/moeme-27b-top12-bf16.gguf \
  artifacts/gguf/.moeme-27b-top12-imatrix-q4.partial \
  Q4_K_M 8
