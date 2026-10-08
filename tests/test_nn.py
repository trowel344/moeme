import pytest
import torch

from moeme.layout import ExpertLayout
from moeme.nn import MoEMeMLP
from moeme.partition import dense_swiglu


def dense_weights(dtype=torch.float64):
    generator = torch.Generator().manual_seed(81)
    gate = torch.randn(32, 12, generator=generator, dtype=dtype)
    up = torch.randn(32, 12, generator=generator, dtype=dtype)
    down = torch.randn(12, 32, generator=generator, dtype=dtype)
    return gate, up, down


def test_zero_router_top_all_exactly_reconstructs_dense() -> None:
    gate, up, down = dense_weights()
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=6)
    model = MoEMeMLP.from_dense(gate, up, down, layout)
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    actual, stats = model(x, return_router_stats=True)
    expected = dense_swiglu(x, gate, up, down)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert stats.counts.tolist() == [10] * 6
    assert model.shared_expert_gate is not None
    assert torch.count_nonzero(model.shared_expert_gate.weight) == 0
    torch.testing.assert_close(stats.entropy, torch.full((10,), torch.log(torch.tensor(6.0))))


def test_module_supports_no_shared_ablation_and_gradients() -> None:
    gate, up, down = dense_weights(dtype=torch.float32)
    layout = ExpertLayout(32, groups=8, shared_groups=0, routed_experts=8, top_k=4)
    model = MoEMeMLP.from_dense(gate, up, down, layout)
    x = torch.randn(7, 12, requires_grad=True)
    loss = model(x).square().mean()
    loss.backward()
    assert model.shared_expert is None
    assert model.shared_expert_gate is None
    assert x.grad is not None
    assert model.router.weight.grad is not None


def test_top_k_can_be_annealed_with_bounds() -> None:
    gate, up, down = dense_weights()
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=6)
    model = MoEMeMLP.from_dense(gate, up, down, layout)
    for top_k in (6, 4, 2):
        model.set_top_k(top_k)
        assert model.top_k == top_k
    with pytest.raises(ValueError, match="must be in"):
        model.set_top_k(0)
