"""Single-launch Triton MoE decode kernel: exactness vs the loop path.

The kernel computes in fp32 and rounds once at the store; agreement with the
bf16 loop path is therefore at the bf16 floor (~4e-3 relative), not bitwise.
"""
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from moeme.branch import BranchConfig, BranchMoEFFN
from moeme.triton_moe import fused_moe_decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("k", [1, 2])
def test_triton_matches_loop_to_bf16_floor(k):
    torch.manual_seed(0)
    H, I, E = 256, 512, 8
    g, u, d = (torch.randn(I, H), torch.randn(I, H), torch.randn(H, I))
    cfg = BranchConfig(n_experts=E, top_k=k, init="tile",
                       expert_units=I // E, combine="softmax", seed=0)
    m = BranchMoEFFN(g, u, d, cfg).cuda().bfloat16()
    m.router.float()
    m.eval()
    x = torch.randn(1, H, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref = m.fast_forward(x, stats=False)
        got = fused_moe_decode(x.reshape(-1), m.router.weight.detach(),
                               m.router.bias.detach(), m.gate, m.up, m.down, k)
    err = float((ref.reshape(-1).float() - got.float()).norm()
                / ref.float().norm().clamp_min(1e-12))
    assert err < 1e-2, err
