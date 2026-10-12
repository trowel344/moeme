"""
Trainable top-k MoE upcycling tests.

The contract under test, in the same spirit as `tests/test_ffn_partition.py`:

* the dense model must be a **strict special case** of `MoEFFN` (top_k == E with
  `combine="sum"` reproduces the dense layer exactly), so upcycling cannot make
  things worse by construction at the dense limit;
* the router, the experts and the auxiliary loss must all be *trainable* -- a
  module whose gradients silently vanish would make the PoC report a training
  failure as if it were a model failure.

Whether training actually recovers quality is a *measurement*, not a test:
see `experiments/upcycle_poc.py`.
"""

import math

import torch
import torch.nn as nn
import pytest

from moeme.ffn import DenseFFN
from moeme.upcycle import (
    MoEConfig,
    MoEFFN,
    build_plan,
    collect_total_aux_loss,
    iter_moe_modules,
    reconstruction_residual,
    set_top_k,
    upcycle_model,
)

HIDDEN, INTER = 32, 64


def make_dense(hidden=HIDDEN, inter=INTER, seed=0, dtype=torch.float32) -> DenseFFN:
    g = torch.Generator().manual_seed(seed)
    s = 0.2
    return DenseFFN(
        gate=(torch.randn(inter, hidden, generator=g) * s).to(dtype),
        up=(torch.randn(inter, hidden, generator=g) * s).to(dtype),
        down=(torch.randn(hidden, inter, generator=g) * s).to(dtype),
    )


def make_moe(n_experts=4, top_k=2, combine="sum", plan="contiguous", seed=0,
             **kw) -> MoEFFN:
    cfg = MoEConfig(n_experts=n_experts, top_k=top_k, combine=combine,
                    plan=plan, seed=seed, **kw)
    return MoEFFN.from_dense(make_dense(seed=seed), cfg)


# ---------------------------------------------------------------------------
# fake HF model for the integration path
# ---------------------------------------------------------------------------

class _FakeMLP(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x))
                              * self.up_proj(x))


class _FakeBlock(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.mlp = _FakeMLP(h, i)

    def forward(self, x):
        return self.mlp(x)


class _FakeModel(nn.Module):
    def __init__(self, h=HIDDEN, i=INTER, n=3):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeBlock(h, i) for _ in range(n)])

    def forward(self, x):
        for block in self.model.layers:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------

def test_config_rejects_out_of_range_top_k():
    with pytest.raises(ValueError):
        MoEConfig(n_experts=4, top_k=0)
    with pytest.raises(ValueError):
        MoEConfig(n_experts=4, top_k=5)


def test_config_rejects_unknown_modes():
    with pytest.raises(ValueError):
        MoEConfig(combine="nonsense")
    with pytest.raises(ValueError):
        MoEConfig(plan="nonsense")


def test_config_rejects_negative_noise():
    with pytest.raises(ValueError):
        MoEConfig(expert_noise=-1.0)


def test_more_experts_than_intermediate_is_rejected():
    with pytest.raises(ValueError):
        MoEFFN.from_dense(make_dense(inter=4), MoEConfig(n_experts=8))


def test_shape_mismatch_is_rejected():
    d = make_dense()
    with pytest.raises(ValueError):
        MoEFFN(d.gate, d.up[:, : HIDDEN - 1], d.down, MoEConfig(n_experts=2))


def test_build_plan_rejects_unknown_name():
    d = make_dense()
    with pytest.raises(ValueError):
        build_plan("nope", d.gate, d.up, 2)


# ---------------------------------------------------------------------------
# the dense limit -- the central contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plan", ["contiguous", "interleaved", "random", "clustered"])
@pytest.mark.parametrize("n_experts", [1, 2, 3, 4, 8, 16])
def test_top_k_equal_experts_with_sum_is_exactly_dense(plan, n_experts):
    dense = make_dense(inter=96)
    cfg = MoEConfig(n_experts=n_experts, top_k=n_experts, combine="sum",
                    plan=plan, seed=3)
    moe = MoEFFN.from_dense(dense, cfg)
    x = torch.randn(6, dense.hidden)
    torch.testing.assert_close(moe(x), dense.forward(x), rtol=1e-5, atol=1e-5)


def test_all_active_forward_matches_dense_forward():
    dense = make_dense(inter=96)
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=7, top_k=2, plan="random"))
    x = torch.randn(5, dense.hidden)
    torch.testing.assert_close(moe.all_active_forward(x), dense.forward(x),
                               rtol=1e-5, atol=1e-5)


def test_dense_equivalent_error_is_negligible():
    moe = make_moe(n_experts=12, top_k=3, plan="interleaved")
    assert moe.dense_equivalent_error(torch.randn(9, HIDDEN)) < 1e-5


def test_single_expert_module_is_the_dense_layer():
    dense = make_dense()
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=1, top_k=1))
    x = torch.randn(4, HIDDEN)
    torch.testing.assert_close(moe(x), dense.forward(x), rtol=1e-6, atol=1e-6)


def test_mean_combine_at_full_k_equals_dense_over_k():
    dense = make_dense()
    E = 4
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=E, top_k=E,
                                             combine="mean"))
    x = torch.randn(3, HIDDEN)
    torch.testing.assert_close(moe(x), dense.forward(x) / E,
                               rtol=1e-5, atol=1e-5)


def test_softmax_combine_is_not_the_dense_limit():
    """Documents a deliberate design choice: softmax weights sum to 1, so a
    full-k softmax *combination* is a weighted mean, not the dense layer."""
    dense = make_dense()
    E = 4
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=E, top_k=E,
                                             combine="softmax"))
    x = torch.randn(3, HIDDEN)
    assert not torch.allclose(moe(x), dense.forward(x), rtol=1e-3, atol=1e-3)


def test_softmax_combine_with_top1_equals_sum_with_top1():
    dense = make_dense()
    # routers are initialised with the global RNG, so seed before each build or
    # the two modules would route to different experts for unrelated reasons
    torch.manual_seed(0)
    a = MoEFFN.from_dense(dense, MoEConfig(n_experts=8, top_k=1, combine="sum",
                                           plan="interleaved"))
    torch.manual_seed(0)
    b = MoEFFN.from_dense(dense, MoEConfig(n_experts=8, top_k=1,
                                           combine="softmax", plan="interleaved"))
    x = torch.randn(5, HIDDEN)
    torch.testing.assert_close(a(x), b(x), rtol=1e-6, atol=1e-6)


def test_rescaled_combine_at_full_k_is_still_the_dense_limit():
    """E/k is exactly 1.0 at k == E, so unlike mean/softmax the output rescaling
    does *not* break the partition's dense limit. That is the whole reason it is
    safe to offer as a combine mode."""
    dense = make_dense()
    E = 4
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=E, top_k=E,
                                             combine="sum_rescaled"))
    x = torch.randn(3, HIDDEN)
    torch.testing.assert_close(moe(x), dense.forward(x), rtol=1e-5, atol=1e-5)


def test_rescaled_combine_scales_a_sparse_sum_by_n_over_k():
    dense = make_dense()
    E, k = 8, 2

    def build(combine):
        torch.manual_seed(0)
        return MoEFFN.from_dense(dense, MoEConfig(n_experts=E, top_k=k,
                                                  combine=combine,
                                                  plan="interleaved"))

    x = torch.randn(5, HIDDEN)
    torch.testing.assert_close(build("sum_rescaled")(x), build("sum")(x) * (E / k),
                               rtol=1e-5, atol=1e-5)


def test_rescaled_combine_is_a_global_scale_so_the_all_active_limit_holds():
    """`all_active_forward` hardcodes `sum`, so the dense-equivalence check is
    unaffected by the configured combine mode. Without that, a `sum_rescaled`
    config would look like a broken partition at k < E."""
    dense = make_dense(inter=96)
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=7, top_k=2,
                                             combine="sum_rescaled",
                                             plan="random"))
    x = torch.randn(5, dense.hidden)
    torch.testing.assert_close(moe.all_active_forward(x), dense.forward(x),
                               rtol=1e-5, atol=1e-5)
    assert moe.dense_equivalent_error(x) < 1e-5


# ---------------------------------------------------------------------------
# shapes and plumbing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(HIDDEN,), (7, HIDDEN), (2, 5, HIDDEN)])
def test_forward_preserves_leading_dimensions(shape):
    moe = make_moe(n_experts=4, top_k=2)
    y = moe(torch.randn(*shape))
    assert tuple(y.shape) == shape


def test_param_count_is_dense_plus_router():
    dense = make_dense()
    E = 8
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=E, top_k=2))
    router_params = sum(p.numel() for p in moe.router.parameters())
    assert moe.params == dense.params() + router_params
    # partition, not replication: total expert params equal the dense FFN
    assert moe.gate.numel() + moe.up.numel() + moe.down.numel() == dense.params()


def test_expert_sizes_partition_the_intermediate_axis():
    moe = MoEFFN.from_dense(make_dense(inter=100), MoEConfig(n_experts=7, top_k=2))
    sizes = moe.expert_sizes()
    assert sum(sizes) == 100
    assert max(sizes) - min(sizes) <= 1
    d = moe.describe()
    assert d["n_experts"] == 7 and d["intermediate"] == 100


def test_from_hf_mlp_preserves_the_mlp_function_at_full_k():
    mlp = _FakeMLP(HIDDEN, INTER)
    x = torch.randn(4, HIDDEN)
    ref = mlp(x)
    moe = MoEFFN.from_hf_mlp(mlp, MoEConfig(n_experts=4, top_k=4, combine="sum"))
    torch.testing.assert_close(moe(x), ref, rtol=1e-5, atol=1e-5)


def test_from_hf_mlp_rejects_a_non_swiglu_module():
    with pytest.raises(ValueError):
        MoEFFN.from_hf_mlp(nn.Linear(4, 4), MoEConfig(n_experts=2))


# ---------------------------------------------------------------------------
# model-level integration
# ---------------------------------------------------------------------------

def test_upcycle_model_replaces_only_requested_layers():
    model = _FakeModel(n=4)
    moes = upcycle_model(model, MoEConfig(n_experts=4, top_k=2), layers=[1, 3])
    assert [i for i, _ in moes] == [1, 3]
    assert isinstance(model.model.layers[1].mlp, MoEFFN)
    assert isinstance(model.model.layers[3].mlp, MoEFFN)
    assert isinstance(model.model.layers[0].mlp, _FakeMLP)
    assert len(list(iter_moe_modules(model))) == 2


def test_upcycle_model_all_layers_by_default():
    model = _FakeModel(n=3)
    upcycle_model(model, MoEConfig(n_experts=2, top_k=1))
    assert len(list(iter_moe_modules(model))) == 3


def test_upcycle_model_rejects_bad_layer_index():
    model = _FakeModel(n=3)
    with pytest.raises(ValueError):
        upcycle_model(model, MoEConfig(n_experts=2, top_k=1), layers=[9])


def test_forward_through_upcycled_model_matches_dense_at_full_k():
    model = _FakeModel(n=3)
    x = torch.randn(5, HIDDEN)
    with torch.no_grad():
        ref = model(x)
    upcycle_model(model, MoEConfig(n_experts=4, top_k=4, combine="sum"))
    with torch.no_grad():
        got = model(x)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_collect_total_aux_loss_sums_over_layers():
    model = _FakeModel(n=3)
    upcycle_model(model, MoEConfig(n_experts=4, top_k=2, aux_loss_coef=1.0))
    x = torch.randn(6, HIDDEN)
    with torch.no_grad():
        model(x)
    total = collect_total_aux_loss(model)
    per_layer = [m.last_aux_loss for m in iter_moe_modules(model)]
    assert torch.allclose(total, torch.stack(per_layer).sum())


def test_set_top_k_mutates_existing_modules():
    moe = make_moe(n_experts=8, top_k=1)
    set_top_k([(0, moe)], 8)
    assert moe.top_k == 8


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def test_aux_loss_is_one_for_a_uniform_router_at_full_k():
    """Switch aux = E * sum_i f_i P_i. With uniform dispatch and uniform
    probabilities it is exactly 1.0 -- the balanced optimum."""
    moe = make_moe(n_experts=4, top_k=4)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.zero_()
        moe(torch.randn(64, HIDDEN))
    assert moe.last_aux_loss.item() == pytest.approx(1.0, abs=1e-4)


def test_total_aux_loss_respects_coefficients():
    moe = make_moe(n_experts=4, top_k=2, aux_loss_coef=0.0, z_loss_coef=0.0)
    with torch.no_grad():
        moe(torch.randn(8, HIDDEN))
    assert moe.total_aux_loss().item() == 0.0


def test_z_loss_is_positive_and_grows_with_logit_scale():
    moe = make_moe(n_experts=4, top_k=2)
    with torch.no_grad():
        moe(torch.randn(8, HIDDEN))
        small = moe.last_z_loss.item()
        moe.router.weight.mul_(10.0)
        moe(torch.randn(8, HIDDEN))
        big = moe.last_z_loss.item()
    assert small > 0 and big > small


# ---------------------------------------------------------------------------
# routing metrics
# ---------------------------------------------------------------------------

def test_metrics_are_balanced_for_a_zeroed_router_at_full_k():
    moe = make_moe(n_experts=6, top_k=6)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.zero_()
        m = moe.routing_metrics(torch.randn(48, HIDDEN))
    assert m.effective_experts == pytest.approx(6.0, abs=1e-3)
    assert m.load_max_over_mean == pytest.approx(1.0, abs=1e-3)
    assert m.usage_entropy_normalised == pytest.approx(1.0, abs=1e-3)


def test_metrics_detect_a_collapsed_router():
    moe = make_moe(n_experts=6, top_k=1)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.zero_()
        moe.router.bias[0] = 10.0
        m = moe.routing_metrics(torch.randn(48, HIDDEN))
    assert m.effective_experts == pytest.approx(1.0, abs=1e-3)
    assert m.load_max_over_mean == pytest.approx(6.0, abs=1e-2)
    assert m.usage_fraction[0] == pytest.approx(1.0)


def test_usage_fraction_sums_to_one():
    moe = make_moe(n_experts=5, top_k=2)
    m = moe.routing_metrics(torch.randn(40, HIDDEN))
    assert abs(sum(m.usage_fraction) - 1.0) < 1e-5
    assert m.n_tokens == 40


def test_stats_are_stored_on_forward():
    moe = make_moe(n_experts=4, top_k=2)
    moe(torch.randn(16, HIDDEN))
    assert set(moe.last_stats) >= {"usage_fraction", "effective_experts"}
    assert moe.last_stats["n_tokens"] == 16


# ---------------------------------------------------------------------------
# gradients -- the point of the module
# ---------------------------------------------------------------------------

def test_gradients_reach_every_expert_and_the_router_at_full_k():
    moe = make_moe(n_experts=4, top_k=4)
    x = torch.randn(12, HIDDEN)
    loss = moe(x).pow(2).mean() + moe.total_aux_loss()
    loss.backward()
    for name, p in (("gate", moe.gate), ("up", moe.up), ("down", moe.down),
                    ("router", moe.router.weight), ("router_bias", moe.router.bias)):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_router_gradient_is_nonzero_at_sparse_top_k():
    moe = make_moe(n_experts=8, top_k=2)
    x = torch.randn(32, HIDDEN)
    loss = moe(x).pow(2).mean() + moe.total_aux_loss()
    loss.backward()
    assert moe.router.weight.grad.abs().sum() > 0


def test_training_step_decreases_a_linear_probe_loss():
    """A single upcycled layer must be able to fit a held-out target."""
    torch.manual_seed(0)
    dense = make_dense()
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=4, top_k=4, combine="sum"))
    x = torch.randn(128, HIDDEN)
    # a target *independent* of the initial layer, so the first loss is not
    # already zero (which would make the comparison vacuous)
    target = torch.randn(128, HIDDEN) * 0.3
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = (moe(x) - target).pow(2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert first > 1e-3
    assert loss.item() < first


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_reconstruction_residual_bounds():
    a = torch.randn(10, HIDDEN)
    assert reconstruction_residual(a, a.clone()) == pytest.approx(0.0, abs=1e-9)
    assert reconstruction_residual(a, torch.zeros_like(a)) == pytest.approx(1.0)


def test_expert_noise_changes_initial_weights_but_keeps_shape():
    d = make_dense()
    clean = MoEFFN.from_dense(d, MoEConfig(n_experts=4, top_k=2, seed=0))
    noisy = MoEFFN.from_dense(d, MoEConfig(n_experts=4, top_k=2, seed=0,
                                           expert_noise=0.05))
    assert noisy.gate.shape == clean.gate.shape
    assert not torch.allclose(noisy.gate, clean.gate)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="needs a CUDA device")
def test_module_follows_weights_onto_the_gpu():
    """The partition plan is built on the CPU; the module's weights may not be.

    A CPU index buffer next to CUDA weights raises inside `index_select`, which
    is exactly the failure the Kaggle self-test was written to catch.
    """
    d = make_dense()
    dense = DenseFFN(d.gate.cuda(), d.up.cuda(), d.down.cuda())
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=4, top_k=2))
    assert moe.assignment.device.type == "cuda"
    y = moe(torch.randn(4, HIDDEN, device="cuda"))
    assert y.device.type == "cuda"
    assert moe.dense_equivalent_error(torch.randn(4, HIDDEN, device="cuda")) < 1e-4


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("combine", ["sum", "sum_rescaled", "mean", "softmax"])
def test_forward_works_in_half_precision(dtype, combine):
    """The router runs in float32, so its combination weights must be cast to
    the activation dtype before `index_add_`.

    This was a real bug: every unit test ran in float32 and passed, while the
    first bf16 model raised `self (BFloat16) and source (Float) must have the
    same scalar type`. Testing only in float32 hid it completely.
    """
    d = make_dense(dtype=dtype)
    E = 8
    moe = MoEFFN.from_dense(d, MoEConfig(n_experts=E, top_k=2, combine=combine))
    x = torch.randn(5, HIDDEN, dtype=dtype)
    y = moe(x)
    assert y.dtype == dtype
    assert torch.isfinite(y.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA autocast")
@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_forward_under_cuda_autocast(amp_dtype):
    """Autocast promotes `.sum()` to float32 while `y` stays half precision.

    A plain half-precision test does NOT catch this, because it never enters an
    autocast region. The first real training run failed here with
    `index_add_(): self (BFloat16) and source (Float)`. Keeping the reduction
    inside an autocast block is the whole point of the test.
    """
    d = make_dense(dtype=amp_dtype)
    dense = DenseFFN(d.gate.cuda(), d.up.cuda(), d.down.cuda())
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=8, top_k=2))
    x = torch.randn(5, HIDDEN, device="cuda", dtype=amp_dtype)
    with torch.autocast("cuda", dtype=amp_dtype):
        y = moe(x)
    assert y.dtype == amp_dtype
    assert torch.isfinite(y.float()).all()


def test_dense_limit_holds_in_bfloat16():
    d = make_dense(inter=96, dtype=torch.bfloat16)
    E = 8
    moe = MoEFFN.from_dense(d, MoEConfig(n_experts=E, top_k=E, combine="sum"))
    x = torch.randn(6, d.hidden, dtype=torch.bfloat16)
    torch.testing.assert_close(moe(x).float(), d.forward(x).float(),
                               rtol=2e-2, atol=2e-2)


def test_construction_is_deterministic_for_a_fixed_seed():
    d = make_dense()
    cfg = MoEConfig(n_experts=6, top_k=2, plan="random", seed=5)
    a = MoEFFN.from_dense(d, cfg)
    b = MoEFFN.from_dense(d, cfg)
    assert torch.equal(a.assignment, b.assignment)
