"""
Dense-equivalent FFN partitioning tests.

The contract under test is *exactness*: with every partition active, the
partitioned FFN must reproduce the dense FFN to floating-point tolerance, for
any expert count, any plan, including ragged (non-dividing) splits.

Nothing here asserts that the *sparse* (top-k) path preserves quality. That is
deliberately not a test but a measurement -- see experiments/pilot_sparsity.py.
"""

import torch
import pytest

from moeme.ffn import (
    DenseFFN,
    PartitionPlan,
    PartitionedFFN,
    clustered_plan,
    contiguous_plan,
    contribution_concentration,
    interleaved_plan,
    oracle_topk_residual,
    random_plan,
    router_topk_residual,
    train_linear_router,
)

HIDDEN, INTER = 64, 96


def make_ffn(hidden=HIDDEN, inter=INTER, seed=0, dtype=torch.float32) -> DenseFFN:
    g = torch.Generator().manual_seed(seed)
    scale = 0.1
    return DenseFFN(
        gate=(torch.randn(inter, hidden, generator=g) * scale).to(dtype),
        up=(torch.randn(inter, hidden, generator=g) * scale).to(dtype),
        down=(torch.randn(hidden, inter, generator=g) * scale).to(dtype),
    )


ALL_PLANS = ["contiguous", "interleaved", "random"]


def plan_by_name(name, ffn, n_experts, seed=0):
    if name == "contiguous":
        return contiguous_plan(ffn.intermediate, n_experts)
    if name == "interleaved":
        return interleaved_plan(ffn.intermediate, n_experts)
    if name == "random":
        return random_plan(ffn.intermediate, n_experts, seed=seed)
    if name == "clustered":
        return clustered_plan(ffn.gate, ffn.up, n_experts, seed=seed)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# plan structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PLANS)
@pytest.mark.parametrize("n_experts", [1, 2, 3, 4, 8, 16])
def test_plan_covers_every_intermediate_unit_exactly_once(name, n_experts):
    ffn = make_ffn()
    plan = plan_by_name(name, ffn, n_experts)
    assert plan.covers_all()
    idx = torch.cat(plan.expert_indices())
    assert int(idx.numel()) == ffn.intermediate
    # no duplicates
    assert len(set(idx.tolist())) == ffn.intermediate


@pytest.mark.parametrize("name", ALL_PLANS)
def test_plans_are_balanced_when_they_claim_to_be(name):
    ffn = make_ffn(inter=100)
    plan = plan_by_name(name, ffn, 7)  # 100 / 7 does not divide
    sizes = plan.sizes()
    assert sum(sizes) == 100
    assert max(sizes) - min(sizes) <= 1


def test_ragged_split_is_supported():
    ffn = make_ffn(inter=17)
    plan = contiguous_plan(17, 5)
    assert plan.sizes() == [4, 4, 3, 3, 3]


def test_more_experts_than_units_is_rejected():
    with pytest.raises(ValueError):
        contiguous_plan(4, 8)
    ffn = make_ffn(inter=4)
    with pytest.raises(ValueError):
        clustered_plan(ffn.gate, ffn.up, 8)


def test_plan_rejects_non_contiguous_expert_ids():
    with pytest.raises(ValueError):
        PartitionPlan(torch.tensor([0, 2, 2]), n_experts=2)


def test_plan_rejects_unused_expert():
    with pytest.raises(ValueError):
        PartitionPlan(torch.tensor([0, 0, 0]), n_experts=2)


def test_plan_rejects_wrong_length():
    ffn = make_ffn()
    with pytest.raises(ValueError):
        PartitionedFFN(ffn, contiguous_plan(ffn.intermediate - 1, 2))


# ---------------------------------------------------------------------------
# exactness of the all-active partitioned forward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PLANS)
@pytest.mark.parametrize("n_experts", [1, 2, 3, 5, 8, 16, 32])
def test_all_active_partition_equals_dense(name, n_experts):
    ffn = make_ffn()
    p = PartitionedFFN(ffn, plan_by_name(name, ffn, n_experts))
    x = torch.randn(7, ffn.hidden)
    torch.testing.assert_close(p.all_active_forward(x), ffn.forward(x),
                               rtol=1e-5, atol=1e-5)


def test_single_expert_is_bitwise_dense():
    ffn = make_ffn()
    p = PartitionedFFN(ffn, contiguous_plan(ffn.intermediate, 1))
    x = torch.randn(4, ffn.hidden)
    assert torch.equal(p.all_active_forward(x), ffn.forward(x))


def test_all_active_exactness_holds_in_bfloat16():
    ffn = make_ffn(dtype=torch.bfloat16)
    p = PartitionedFFN(ffn, random_plan(ffn.intermediate, 8, seed=3))
    x = torch.randn(5, ffn.hidden, dtype=torch.bfloat16)
    ref, got = ffn.forward(x).float(), p.all_active_forward(x).float()
    assert float((ref - got).norm() / ref.norm()) < 2e-2


def test_dense_equivalent_error_is_near_zero():
    ffn = make_ffn()
    p = PartitionedFFN(ffn, contiguous_plan(ffn.intermediate, 12))
    assert p.dense_equivalent_error(torch.randn(13, ffn.hidden)) < 1e-5


def test_reassembly_restores_original_weights():
    """Concatenating expert slices in assignment order rebuilds the weights."""
    ffn = make_ffn()
    plan = random_plan(ffn.intermediate, 6, seed=11)
    rebuild = torch.empty_like(ffn.gate)
    for e, idx in enumerate(plan.expert_indices()):
        rebuild[idx] = ffn.gate[idx]
    assert torch.equal(rebuild, ffn.gate)


def test_expert_params_accounting():
    ffn = make_ffn(inter=30, hidden=10)
    p = PartitionedFFN(ffn, contiguous_plan(30, 3))
    assert p.expert_params(0) == 3 * 10 * 10
    assert sum(p.expert_params(e) for e in range(3)) == ffn.params()


# ---------------------------------------------------------------------------
# top-k path
# ---------------------------------------------------------------------------

def test_topk_with_all_experts_selected_matches_dense():
    ffn = make_ffn()
    E = 4
    p = PartitionedFFN(ffn, contiguous_plan(ffn.intermediate, E))
    x = torch.randn(6, ffn.hidden)
    topk = torch.arange(E).expand(6, E).contiguous()
    torch.testing.assert_close(p.topk_forward(x, topk), ffn.forward(x),
                               rtol=1e-5, atol=1e-5)


def test_topk_forward_matches_manual_subset_sum():
    ffn = make_ffn()
    E = 4
    p = PartitionedFFN(ffn, contiguous_plan(ffn.intermediate, E))
    x = torch.randn(3, ffn.hidden)
    sel = torch.tensor([[0, 2], [1, 3], [0, 1]])
    got = p.topk_forward(x, sel)
    outs = p.expert_outputs(x)                      # (E, T, hidden)
    want = torch.stack([outs[sel[t, 0], t] + outs[sel[t, 1], t]
                        for t in range(3)])
    assert got.shape == want.shape == (3, ffn.hidden)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_topk_weights_scale_contributions():
    ffn = make_ffn()
    p = PartitionedFFN(ffn, contiguous_plan(ffn.intermediate, 4))
    x = torch.randn(2, ffn.hidden)
    sel = torch.tensor([[0, 1], [2, 3]])
    plain = p.topk_forward(x, sel)
    halved = p.topk_forward(x, sel, weights=torch.full((2, 2), 0.5))
    torch.testing.assert_close(halved * 2, plain, rtol=1e-5, atol=1e-5)


def test_topk_rejects_out_of_range_ids():
    ffn = make_ffn()
    p = PartitionedFFN(ffn, contiguous_plan(ffn.intermediate, 2))
    with pytest.raises(ValueError):
        p.topk_forward(torch.randn(2, ffn.hidden), torch.tensor([[0, 5]]))


# ---------------------------------------------------------------------------
# oracle diagnostics
# ---------------------------------------------------------------------------

def _oracle_setup(n_experts=8, n_tokens=64, seed=0):
    ffn = make_ffn(seed=seed)
    p = PartitionedFFN(ffn, contiguous_plan(ffn.intermediate, n_experts))
    x = torch.randn(n_tokens, ffn.hidden, generator=torch.Generator().manual_seed(seed + 1))
    return p, p.expert_outputs(x)


def test_oracle_with_all_experts_has_zero_residual():
    p, outs = _oracle_setup(8)
    r = oracle_topk_residual(outs, k=8)
    assert r["mean_rel_residual"] < 1e-6


def test_oracle_residual_is_monotone_non_increasing_in_k():
    p, outs = _oracle_setup(16)
    means = [oracle_topk_residual(outs, k)["mean_rel_residual"] for k in (1, 2, 4, 8, 16)]
    for a, b in zip(means, means[1:]):
        assert b <= a + 1e-9, means


def test_oracle_single_expert_of_many_is_poor():
    """A post-hoc split of a dense FFN has no natural sparsity to exploit.

    With 16 uniformly-sized partitions and a random-init FFN, keeping 1/16 of
    the intermediate units retains far less than 1/16 of the output -- the
    oracle residual stays high. This is the negative result that must be
    established before training compute is spent.
    """
    p, outs = _oracle_setup(16, n_tokens=256)
    r = oracle_topk_residual(outs, k=1)
    assert r["mean_rel_residual"] > 0.3, r


def test_contribution_concentration_bounds():
    p, outs = _oracle_setup(8, n_tokens=32)
    c = contribution_concentration(outs)
    assert 1.0 <= c["participation_ratio"] <= 8.0 + 1e-6
    assert c["top1_energy_share"] > 0
    assert 0.0 <= c["entropy_normalised"] <= 1.0 + 1e-6


def test_concentration_is_maximal_for_one_sided_contributions():
    E, T, H = 6, 4, 5
    outs = torch.zeros(E, T, H)
    outs[0] = 1.0
    c = contribution_concentration(outs)
    assert c["participation_ratio"] == pytest.approx(1.0)
    assert c["top1_energy_share"] == pytest.approx(1.0)
    assert c["entropy_normalised"] == pytest.approx(0.0)


def test_concentration_is_uniform_for_equal_contributions():
    E, T, H = 8, 4, 5
    outs = torch.ones(E, T, H)
    c = contribution_concentration(outs)
    assert c["participation_ratio"] == pytest.approx(E, abs=1e-4)
    assert c["entropy_normalised"] == pytest.approx(1.0, abs=1e-4)


def test_router_residual_is_worse_than_oracle():
    p, outs = _oracle_setup(8, n_tokens=128)
    oracle = oracle_topk_residual(outs, k=2)["mean_rel_residual"]
    scores = torch.rand(128, 8, generator=torch.Generator().manual_seed(0))
    random_router = router_topk_residual(outs, scores, k=2)["mean_rel_residual"]
    assert random_router >= oracle - 1e-9


def test_linear_router_beats_random_and_is_worse_than_oracle():
    p, outs = _oracle_setup(8, n_tokens=256)
    x = torch.randn(256, p.dense.hidden, generator=torch.Generator().manual_seed(7))
    norms = outs.norm(dim=2)                                   # (E, T)
    targets = norms.argmax(dim=0)                              # oracle top-1
    wb = train_linear_router(x, targets, 8, steps=300)
    scores = torch.cat([x, torch.ones(256, 1)], dim=1) @ wb
    learned = router_topk_residual(outs, scores, k=2)["mean_rel_residual"]
    oracle = oracle_topk_residual(outs, k=2)["mean_rel_residual"]
    assert oracle <= learned + 1e-6


# ---------------------------------------------------------------------------
# clustering plan
# ---------------------------------------------------------------------------

def test_clustered_plan_recovers_planted_structure():
    """If the intermediate axis really has groups, k-means must find them.

    Without this property the clustered plan is just a slower random plan, so it
    is worth pinning down rather than assuming.
    """
    n_groups, per_group, hidden = 4, 8, 16
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(n_groups, 2 * hidden, generator=g) * 5.0
    X = torch.cat([
        centres[i].repeat(per_group, 1) + torch.randn(per_group, 2 * hidden,
                                                      generator=g) * 0.01
        for i in range(n_groups)
    ])
    gate, up = X[:, :hidden], X[:, hidden:]
    plan = clustered_plan(gate, up, n_groups, seed=1)
    assign = plan.assignment
    for i in range(n_groups):
        block = assign[i * per_group:(i + 1) * per_group]
        assert len(set(block.tolist())) == 1, "planted group was split"
    assert len(set(assign.tolist())) == n_groups


def test_clustered_plan_is_deterministic():
    ffn = make_ffn(inter=48)
    a = clustered_plan(ffn.gate, ffn.up, 4, seed=5).assignment
    b = clustered_plan(ffn.gate, ffn.up, 4, seed=5).assignment
    assert torch.equal(a, b)


def test_clustered_plan_output_is_balanced_after_repair():
    ffn = make_ffn(inter=60)
    plan = clustered_plan(ffn.gate, ffn.up, 7, seed=2)
    assert plan.covers_all()
    assert sum(plan.sizes()) == 60


@pytest.mark.parametrize("name", ALL_PLANS + ["clustered"])
def test_every_plan_supports_exact_all_active_forward(name):
    ffn = make_ffn(inter=60, hidden=32)
    p = PartitionedFFN(ffn, plan_by_name(name, ffn, 4))
    x = torch.randn(9, 32)
    torch.testing.assert_close(p.all_active_forward(x), ffn.forward(x),
                               rtol=1e-5, atol=1e-5)
