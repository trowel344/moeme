#!/usr/bin/env python3
"""
minimal.py -- the smallest useful demonstration of the moeme API.

Run it:
    python3 examples/minimal.py

It shows the three things the library does, in order:

  1. Partition a swiGLU FFN's intermediate axis into E experts, under one of
     several plans.
  2. Verify the all-active partitioned forward is exactly the dense forward.
  3. Measure what a top-k router could actually recover (the oracle ceiling),
     which is the number that decides whether sparsification is worth pursuing.

Uses synthetic weights at 27B-like geometry so it runs in under a second with
no downloads.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from moeme.ffn import (
    DenseFFN,
    PartitionedFFN,
    contiguous_plan,
    contribution_concentration,
    oracle_topk_residual,
)

# ---------------------------------------------------------------------------
# 1. Build a dense FFN at Qwen3.8-27B's FFN geometry.
#    hidden = 5120, intermediate = 17408  (the real numbers)
#    Full-size weights would be 535 MB; use a narrow slice of the intermediate
#    axis so the example is instant while keeping the plan maths identical.
# ---------------------------------------------------------------------------
HIDDEN, INTERMEDIATE = 5120, 2048
EPOCHS = 1  # (not training anything -- just naming the scale)

g = torch.Generator().manual_seed(0)
ffn = DenseFFN(
    gate=torch.randn(INTERMEDIATE, HIDDEN, generator=g) * 0.02,
    up=torch.randn(INTERMEDIATE, HIDDEN, generator=g) * 0.02,
    down=torch.randn(HIDDEN, INTERMEDIATE, generator=g) * 0.02,
)
print(f"dense FFN: hidden={ffn.hidden} intermediate={ffn.intermediate} "
      f"params={ffn.params():,}")

# ---------------------------------------------------------------------------
# 2. Partition it into experts. `contiguous_plan` is one of
#    contiguous / interleaved / random / clustered.
# ---------------------------------------------------------------------------
N_EXPERTS = 16
plan = contiguous_plan(ffn.intermediate, N_EXPERTS)
part = PartitionedFFN(ffn, plan)
print(f"plan: {N_EXPERTS} experts, sizes={plan.sizes()[:4]}... "
      f"covers_all={plan.covers_all()}")

x = torch.randn(8, HIDDEN)   # 8 "tokens"

# The exactness contract: every expert active == the dense layer.
err = part.dense_equivalent_error(x)
print(f"\nall-active partitioned vs dense: relative L2 error = {err:.3e}")
assert err < 1e-5, "partitioning is not dense-equivalent"
print("  -> exact (this is what tests/test_ffn_partition.py enforces)")

# ---------------------------------------------------------------------------
# 3. What could a top-k router recover? This is the decisive measurement.
# ---------------------------------------------------------------------------
outs = part.expert_outputs(x)          # (E, tokens, hidden) per-expert output
conc = contribution_concentration(outs)
print(f"\ncontribution concentration: participation_ratio="
      f"{conc['participation_ratio']:.1f} of {N_EXPERTS} experts, "
      f"top-1 energy share={conc['top1_energy_share']:.3f}")
print("  (participation_ratio close to E means energy is spread uniformly, so")
print("   there is no structure for a top-k router to exploit)\n")

print(f"{'k':>3} {'active':>8} {'oracle rel. residual':>21}")
for k in (1, 2, 4, 8, 16):
    r = oracle_topk_residual(outs, k)
    print(f"{k:>3} {100*k//N_EXPERTS:>7}% {r['mean_rel_residual']:>21.4f}")

print("\nReading: the oracle is handed the exact per-expert output norms, so it is")
print("the best any router could do. Residuals only reach 0 at k = E, i.e. when")
print("nothing is sparse. Real measurements on Qwen3.8-27B weights show the same")
print("shape -- see results/real_ffn.json and docs/feasibility_report.md.")
