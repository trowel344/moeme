"""Triton single-launch fused MoE layer for batch-1 decode (SM89-safe).

One program does: router GEMV + softmax/top-k + per-expert gate/up GEMV +
SiLU-mul + down GEMV + weighted accumulate. No TMA/WGMMA (SM90+), classic
pointer arithmetic only. Exact up to bf16ancement order vs the loop path.

Constraints (decode specialization): T=1 row input, E<=64, k<=4, H,U
arbitrary (masked tails), combine=softmax weights.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_decode_kernel(
    x_ptr,            # (H,) bf16
    rout_ptr,         # (E, H) fp32
    rout_b_ptr,       # (E,) fp32
    gate_ptr,         # (E, U, H) bf16
    up_ptr,           # (E, U, H) bf16
    down_ptr,         # (E, H, U) bf16
    y_ptr,            # (H,) bf16 out
    H, U, E, K: tl.constexpr,
    BH: tl.constexpr = 128,
    BU: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    if pid != 0:
        return
    h_off = tl.arange(0, BH)
    # ---- router GEMV: s[e] = x . rout[e] ----
    # fixed-size local buffer (E<=64), masked
    scr = tl.zeros((64,), dtype=tl.float32)
    for e in range(E):
        acc = tl.zeros((1,), dtype=tl.float32)
        for hb in range((H + BH - 1) // BH):
            cols = hb * BH + h_off
            m = cols < H
            xv = tl.load(x_ptr + cols, mask=m).to(tl.float32)
            wv = tl.load(rout_ptr + e * H + cols, mask=m).to(tl.float32)
            acc += tl.sum(xv * wv)
        rb = tl.load(rout_b_ptr + e).to(tl.float32)
        scr = tl.where(tl.arange(0, 64) == e, acc + rb, scr)
    s = scr
    # ---- softmax + top-k (scalar, E<=64) ----
    em = tl.arange(0, 64) < E
    se = tl.where(em, s, float("-inf"))
    se = se - tl.max(se)
    pe = tl.exp(se) * em.to(tl.float32)
    pe = pe / tl.sum(pe)
    # iterative argmax k times
    # NOTE: H<=1024 assumed for the accumulator below; guard by masking
    hid = tl.arange(0, 1024)
    hm = hid < H
    # zero the output once (disjoint rows per hb-block below make this safe)
    tl.store(y_ptr + hid, tl.zeros((1024,), dtype=tl.bfloat16), mask=hm)
    pk = tl.zeros((64,), dtype=tl.float32)
    ek = tl.zeros((64,), dtype=tl.int32)
    tmp = pe
    for j in range(4):
        active = j < tl.minimum(K, E)
        bj = tl.argmax(tmp, axis=0)
        bv = tl.max(tmp)
        ek = tl.where((tl.arange(0, 64) == j) & active, bj.to(tl.int32), ek)
        pk = tl.where((tl.arange(0, 64) == j) & active, bv, pk)
        tmp = tl.where(tl.arange(0, 64) == bj, float("-inf"), tmp)
    # renormalize selected weights
    psum = tl.sum(pk) + 1e-12
    # ---- per selected expert GEMVs ----
    u_off = tl.arange(0, BU)
    for j in range(4):
        on = j < tl.minimum(K, E)
        e = tl.sum(ek * (tl.arange(0, 64) == j).to(tl.int32))
        w = tl.sum(pk * (tl.arange(0, 64) == j).to(tl.float32)) / psum
        # gate/up rows for expert e
        for ub in range((U + BU - 1) // BU):
            cols = ub * BU + u_off
            um = cols < U
            gacc = tl.zeros((BU,), dtype=tl.float32)
            uacc = tl.zeros((BU,), dtype=tl.float32)
            for hb in range((H + BH - 1) // BH):
                rows = hb * BH + h_off
                hm2 = rows < H
                xv = tl.load(x_ptr + rows, mask=hm2).to(tl.float32)
                gv = tl.load(gate_ptr + e * U * H + cols[:, None] * H + rows[None, :],
                             mask=um[:, None] & hm2[None, :]).to(tl.float32)
                uv = tl.load(up_ptr + e * U * H + cols[:, None] * H + rows[None, :],
                             mask=um[:, None] & hm2[None, :]).to(tl.float32)
                gacc += tl.sum(gv * xv[None, :], axis=1)
                uacc += tl.sum(uv * xv[None, :], axis=1)
            sil = gacc / (1.0 + tl.exp(-gacc))
            hact = (sil * uacc) * um.to(tl.float32)
            # down: y += w * (hact . down[e])
            for hb in range((H + BH - 1) // BH):
                rows = hb * BH + h_off
                hm3 = rows < H
                dv = tl.load(down_ptr + e * H * U + rows[:, None] * U + cols[None, :],
                             mask=hm3[:, None] & um[None, :]).to(tl.float32)
                contrib = tl.sum(dv * hact[None, :], axis=1)
                old = tl.load(y_ptr + rows, mask=hm3).to(tl.float32)
                new = old + tl.where(on, w, 0.0) * contrib
                tl.store(y_ptr + rows, new.to(tl.bfloat16), mask=hm3)


def fused_moe_decode(x: torch.Tensor, router: torch.Tensor,
                     router_bias: torch.Tensor,
                     gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                     top_k: int) -> torch.Tensor:
    """x: (H,) bf16 CUDA; router: (E,H) fp32; router_bias: (E,) fp32;
    gate/up: (E,U,H) bf16; down: (E,H,U) bf16. Returns (H,) bf16. T=1 only."""
    assert x.dim() == 1 and x.is_cuda
    H = x.shape[0]
    E, U, Hh = gate.shape
    assert Hh == H and up.shape == (E, U, H) and down.shape == (E, H, U)
    assert E <= 64 and top_k <= 4 and H <= 1024
    y = torch.empty(H, device=x.device, dtype=torch.bfloat16)
    _fused_moe_decode_kernel[(1,)](x, router, router_bias, gate, up, down, y,
                                   H, U, E, top_k)
    return y
