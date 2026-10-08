#!/usr/bin/env bash
# Measure the same-top / KLD ceiling of the release gate on the failing corpus.
#
# The MoEMe conversion reconstructs the dense FFN to ~5e-8 relative L2, so the
# BF16 staging artifact is numerically the dense model. Comparing it against the
# Q4-quantized dense reference therefore isolates the reference's own
# quantization error: no candidate can agree with the Q4 reference more than the
# true BF16 model does.
#
# This runs directly (not through logit_parity_gate.py) because the host needs
# the mandatory `--fit off` and the BF16 file is too large for the fitter.
set -euo pipefail
cd /home/cleanerbox/Documents/moeme
PERPLEXITY=/home/cleanerbox/.cache/moeme/llama-nocache/build/bin/llama-perplexity
REFERENCE=checkpoints/qwen3.8-27b-gguf-reference/Qwen3.8-27B-Q4_K_M.gguf
CANDIDATE=artifacts/gguf/moeme-27b-top12-bf16.gguf
CORPUS=data/calibration/router-v1.txt
COMMON=(-f "$CORPUS" -c 512 -b 512 -ub 64 -t 8 -tb 8 --chunks 1 --no-warmup --fit off)

# The 18 GiB dense reference fits partly on the 8 GB GPU; the 50 GiB BF16
# candidate does not, so it runs on CPU (identical metric, slow but safe).
echo "=== reference (dense Q4) ==="
"$PERPLEXITY" -m "$REFERENCE" "${COMMON[@]}" -ngl 20 \
  --kl-divergence-base .moeme/ceiling-ref-1.kld \
  > .moeme/logit-parity-ceiling-reference.log 2>&1
echo "=== candidate (MoEMe BF16) ==="
"$PERPLEXITY" -m "$CANDIDATE" "${COMMON[@]}" -ngl 0 \
  --kl-divergence-base .moeme/ceiling-ref-1.kld --kl-divergence \
  > .moeme/logit-parity-ceiling-candidate.log 2>&1
echo "=== done ==="
grep -E "Mean PPL\(Q\)/PPL\(base\)|Mean    KLD|Same top" .moeme/logit-parity-ceiling-candidate.log
