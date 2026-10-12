#!/usr/bin/env bash
# The E14 configuration, re-run to record corpus provenance (E14's report
# predates that field). Writes its own PID so the guard cannot be aimed at a
# wrapper, saves no checkpoints (does not touch .cache/moe/poc_ckpt), and writes
# to a NEW out file so the original results/upcycle_poc.json is preserved.
cd /home/cleanerbox/Documents/moeme || exit 1
echo $$ > logs/upcycle_rerun.pid
exec python3 experiments/upcycle_poc.py \
  --model Qwen/Qwen3.5-0.8B \
  --experts 8 --topks 1 2 4 --combine sum --plan random \
  --layers 0 3 8 23 --steps 300 --lr 1e-4 --seq-len 512 \
  --grad-accum 4 --corpus-chars 4000000 --save-every 0 \
  --max-seconds 1500 --out results/upcycle_poc_rerun.json

# NOTE: --eval-blocks was NOT passed, so this run used the current default of 4.
# The report it wrote (results/upcycle_poc_rerun.json) predates the eval_blocks
# field, so its window is evidenced by this command and by logs/blocks8.log +
# logs/hdr.log, which bracket it (23.4212 at 8 blocks, and a field-carrying
# repeat at 4 blocks that reproduced 19.57067870397227 exactly).
# Guard: tools/mem_watchdog.py --pidfile logs/upcycle_rerun.pid, floor 2.0 GiB.
