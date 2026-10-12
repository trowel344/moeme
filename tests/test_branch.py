"""
Branched (replicated-expert) MoE tests.

The contract under test, and why each part of it matters:

* a **full-width replica** expert set with `top_k=1` reproduces the dense layer
  exactly. This is the whole reason the module exists -- it is the claim that
  "experts are complete modules, not slices" makes losslessness possible, and it
  has to be exact rather than approximate or the comparison against the
  partitioned case is meaningless;
* **replicating a narrow slice does not help**. All the replicas are identical,
  so there is nothing for a router to choose between and the residual stays at
  the one-slice level. If this test ever fails, the earlier conclusion that
  "it is active *width*, not the tiling, that decides losslessness" is wrong;
* **tiled experts sum to dense when all are active** -- the dense limit is a
  strict special case, as in `moeme.upcycle`;
* a **shared expert is unconditionally on and additive**, and `unit_order` is
  honoured when slicing, because that is the mechanism by which the
  always-active units are meant to be collected into the shared block.

Whether any of this *trains* into a usable model is a measurement, not a test:
see `experiments/branch_poc.py`.
"""

import torch
import torch.nn as nn
import pytest

from moeme.branch import BranchConfig, BranchMoEFFN

HIDDEN, INTER = 32, 64


def make_weights(inter=INTER, hidden=HIDDEN, seed=0, scale=0.3):
    """A small random swiGLU FFN in HF layout: gate/up (I, H), down (H, I)."""
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(inter, hidden, generator=g) * scale,
        torch.randn(inter, hidden, generator=g) * scale,
        torch.randn(hidden, inter, generator=g) * scale,
    )


def make_branch(config, inter=INTER, hidden=HIDDEN, seed=0):
    gate, up, down = make_weights(inter=inter, hidden=hidden, seed=seed)
    return BranchMoEFFN(gate, up, down, config)


def make_input(n=8, hidden=HIDDEN, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, hidden, generator=g)


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# the central claim: complete experts are lossless, slices are not
# ---------------------------------------------------------------------------

def test_full_width_replica_reproduces_dense_exactly():
    """top_k=1 over full-width replicas IS the dense layer.

    This is the module's reason to exist. If it were only approximately true,
    the branch-vs-partition comparison would be measuring floating point noise
    rather than architecture.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica"))
    x = make_input()
    assert m.U == INTER                      # expert_units=None => full width
    assert rel_l2(m.dense_forward(x), m(x)) < 1e-6


def test_full_width_replica_residual_report_is_zero():
    """Both the oracle and the router land on the dense output at width 1.0.

    Unlike the partitioned case, there is no partial-sum deficit for either to
    fight, so the *sparse* report should be ~0 rather than merely "better".
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica"))
    rep = m.residual_report(make_input())
    assert rep["residual_oracle_topk"] < 1e-6
    assert rep["residual_router"] < 1e-6
    assert rep["residual_all_experts_sum"] > 1.0   # E copies sum to E x dense


def test_tiled_experts_sum_to_dense_when_all_active():
    """A disjoint tile partition with every expert active is exactly dense.

    The dense model must remain a strict special case, matching the guarantee
    `tests/test_ffn_partition.py` pins for the partitioned implementation.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=4, init="tile",
                                 expert_units=INTER // 4))
    x = make_input()
    assert rel_l2(m.dense_forward(x), m(x)) < 1e-6
    assert m.residual_report(x)["residual_all_experts_sum"] < 1e-6


def test_single_tile_does_not_reproduce_dense():
    """One tile of four loses most of the layer output.

    This is the contrast that motivates the module. With independent random
    contributions the expected relative residual is ~sqrt(1 - 1/4) = 0.87, so a
    threshold of 0.5 is comfortably below the real value and above noise.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="tile",
                                 expert_units=INTER // 4))
    x = make_input()
    assert rel_l2(m.dense_forward(x), m(x)) > 0.5


def test_tiled_experts_are_distinct_but_still_bad():
    """Tiling at least buys *diversity*; it does not buy accuracy.

    Worth separating: the router in the tiled case has genuinely different
    options, and the residual is still unusable. Diversity alone is not the fix.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="tile",
                                 expert_units=INTER // 4))
    outs = m.expert_outputs(make_input())
    with torch.no_grad():
        assert not torch.allclose(outs[0], outs[1])


# ---------------------------------------------------------------------------
# replication alone must not help
# ---------------------------------------------------------------------------

def test_replicated_narrow_experts_are_identical():
    """Replicas of one narrow slice carry no options for the router.

    This is the direct test of "replication alone does not help". Asserting the
    outputs are *identical* is exact and is the mechanism; asserting a residual
    number would depend on the random draw.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    outs = m.expert_outputs(make_input())
    for j in range(1, m.n_experts):
        assert torch.allclose(outs[0], outs[j], atol=1e-6)


def test_replicated_narrow_experts_keep_the_one_slice_residual():
    """Replication leaves the one-slice residual untouched, i.e. still unusable.

    Narrow replicas are the obvious "cut it up and stitch it back" move, and
    this pins that it buys nothing: the active width is still 1/E.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    rep = m.residual_report(make_input())
    assert rep["residual_best_single_expert"] > 0.5
    assert rep["active_width_fraction"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# geometry reported by the module
# ---------------------------------------------------------------------------

def test_width_fractions_track_active_and_stored_width():
    """The two width numbers are the axes the whole comparison is plotted on.

    `active` is what a token costs to read; `stored` is what the layer costs in
    memory. A full-width replica is lossless *and* stores E x the FFN, so these
    two must disagree exactly where the trade-off lives.
    """
    narrow = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                      expert_units=INTER // 4))
    assert narrow.active_width_fraction == pytest.approx(0.25)
    assert narrow.total_width_fraction == pytest.approx(1.0)

    full = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica"))
    assert full.active_width_fraction == pytest.approx(1.0)
    assert full.total_width_fraction == pytest.approx(4.0)


def test_active_and_stored_params_account_for_shared_experts():
    """Shared experts are read by every token, so they belong in both counts."""
    shared = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                      expert_units=INTER // 4, n_shared=1))
    plain = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                     expert_units=INTER // 4))
    one = 3 * (INTER // 4) * HIDDEN
    assert shared.active_params == plain.active_params + one
    assert shared.stored_params == plain.stored_params + one


# ---------------------------------------------------------------------------
# shared experts
# ---------------------------------------------------------------------------

def test_shared_expert_is_always_on_and_additive():
    """With one full-width shared expert the output is exactly two dense layers.

    The shared path must be unconditional -- no router decision may gate it --
    so the strongest statement is an exact 2x.

    `residual_report` splits the two paths: `residual_router` measures the
    routed path alone and `residual_shared_only` the shared block. Both are 0
    here (each is a full-width exact copy of the dense layer) while their sum is
    exactly 2x dense -- which is exactly why the report must keep them apart.
    If `residual_router` folded the shared expert in, it would read 1.0 here and
    a tile-vs-replica table would move with a term that is not routing at all.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 n_shared=1))
    x = make_input()
    torch.testing.assert_close(m(x), 2.0 * m.dense_forward(x), rtol=1e-5,
                               atol=1e-5)
    rep = m.residual_report(x)
    assert rep["residual_router"] == pytest.approx(0.0, abs=1e-5)
    assert rep["residual_shared_only"] == pytest.approx(0.0, abs=1e-5)


def test_shared_expert_contributes_for_every_token():
    """Changing only the shared weights must move every token's output.

    Guards the failure mode where the shared block is computed but silently not
    added, which would make the "always-on" idea look like it did nothing.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4, n_shared=1))
    x = make_input()
    before = m(x).detach().clone()
    with torch.no_grad():
        m.s_gate.mul_(1.5)
    after = m(x).detach()
    assert not torch.allclose(before, after)
    assert bool((after - before).abs().sum(dim=-1).gt(0).all())


# ---------------------------------------------------------------------------
# unit_order: how the always-active units get collected
# ---------------------------------------------------------------------------

def test_unit_order_is_applied_to_expert_slices():
    """`unit_order` must actually permute which units each expert owns.

    This is the mechanism for gathering always-active units into the shared
    expert (the CMoE idea), so it has to be exact. Asserted against the manually
    permuted dense weights rather than against a residual, so the test cannot
    pass for an unrelated reason.
    """
    inter = 8
    order = tuple(reversed(range(inter)))
    gate, up, down = make_weights(inter=inter, hidden=HIDDEN, seed=3)
    m = BranchMoEFFN(gate, up, down,
                     BranchConfig(n_experts=4, top_k=1, init="tile",
                                  expert_units=2, unit_order=order))
    idx0 = torch.as_tensor(order[:2])
    idx3 = torch.as_tensor(order[6:8])
    assert torch.equal(m.gate[0], gate[idx0])
    assert torch.equal(m.up[0], up[idx0])
    assert torch.equal(m.down[0], down[:, idx0])
    assert torch.equal(m.gate[3], gate[idx3])
    assert torch.equal(m.down[3], down[:, idx3])


def test_unit_order_rejects_a_non_permutation():
    """A non-permutation would silently drop or duplicate intermediate units."""
    gate, up, down = make_weights(inter=8)
    with pytest.raises(ValueError):
        BranchMoEFFN(gate, up, down,
                     BranchConfig(n_experts=2, top_k=1, init="tile",
                                  expert_units=4,
                                  unit_order=(0, 0, 1, 2, 3, 4, 5, 6)))


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------

def test_config_rejects_out_of_range_top_k():
    with pytest.raises(ValueError):
        BranchConfig(n_experts=4, top_k=0)
    with pytest.raises(ValueError):
        BranchConfig(n_experts=4, top_k=5)


def test_config_rejects_unknown_modes():
    with pytest.raises(ValueError):
        BranchConfig(combine="nonsense")
    with pytest.raises(ValueError):
        BranchConfig(init="nonsense")


def test_config_rejects_negative_shared_count_and_noise():
    with pytest.raises(ValueError):
        BranchConfig(n_shared=-1)
    with pytest.raises(ValueError):
        BranchConfig(init_noise=-1.0)


def test_tile_init_requires_the_experts_to_cover_the_axis():
    """Tiles that do not cover the axis would leave units unreachable."""
    gate, up, down = make_weights()
    with pytest.raises(ValueError):
        BranchMoEFFN(gate, up, down,
                     BranchConfig(n_experts=3, top_k=1, init="tile",
                                  expert_units=INTER // 4))


def test_expert_units_must_fit_within_the_intermediate_axis():
    gate, up, down = make_weights()
    with pytest.raises(ValueError):
        BranchMoEFFN(gate, up, down,
                     BranchConfig(n_experts=2, top_k=1, init="replica",
                                  expert_units=INTER + 1))


def test_weight_shapes_are_validated():
    gate, up, down = make_weights()
    with pytest.raises(ValueError):
        BranchMoEFFN(gate, up, down.T.contiguous(),
                     BranchConfig(n_experts=2, top_k=1))


# ---------------------------------------------------------------------------
# interface the experiment harness duck-types on
# ---------------------------------------------------------------------------

def test_expert_outputs_and_router_scores_have_the_expected_shapes():
    """`experiments/branch_poc.py` and the upcycle harness read these directly."""
    m = make_branch(BranchConfig(n_experts=4, top_k=2, init="replica"))
    x = make_input(n=6)
    assert m.expert_outputs(x).shape == (4, 6, HIDDEN)
    assert m.router_scores(x).shape == (6, 4)
    assert m.n_experts == 4
    assert m.top_k == 2


def test_residual_report_is_complete_and_finite():
    """Every key the report promises is present, finite and non-negative.

    A NaN here would silently poison a comparison table rather than fail.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="tile",
                                 expert_units=INTER // 4))
    rep = m.residual_report(make_input())
    for key in ("residual_best_single_expert", "residual_oracle_topk",
                "residual_router", "residual_all_experts_sum",
                "active_width_fraction", "stored_width_fraction"):
        assert key in rep, key
        assert torch.isfinite(torch.tensor(rep[key])), key
        assert rep[key] >= 0.0, key


def test_aux_loss_is_a_zero_scalar_before_any_forward():
    """The aux loss must be a real tensor before training touches the module.

    `collect_total_aux_loss` sums it across layers, so a missing or None value
    would break the first training step rather than report a routing problem.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1))
    loss = m.total_aux_loss()
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert float(loss) == 0.0


def test_forward_populates_routing_stats():
    """`last_stats` is what the routing-balance reporting reads."""
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    m(make_input())
    stats = m.last_stats
    assert stats["n_tokens"] == 8
    assert len(stats["usage_fraction"]) == 4
    assert stats["effective_experts"] > 0.0


def test_all_experts_are_trainable_when_every_one_is_active():
    """With top_k == E every expert must receive gradient.

    Guards against a module whose experts are built but never wired into the
    graph, which would show up as a training-time no-op rather than an error.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=4, init="replica",
                                 expert_units=INTER // 4))
    m(make_input()).sum().backward()
    touched = sum(1 for j in range(m.n_experts)
                  if float(m.gate.grad[j].abs().sum()) > 0.0)
    assert touched == m.n_experts


def test_sum_combine_gives_the_router_no_gradient_from_the_output():
    """`combine="sum"` makes the router undifferentiable through the output.

    With `sum` the combination weights are a constant 1.0, so the router's only
    influence on the output is the *discrete* choice of experts -- which carries
    no gradient. The router is therefore reached only by the load-balancing aux
    loss. This is a property of the design, shared with `upcycle.MoEFFN` (whose
    default it is), not a defect introduced here -- but it means a `sum`-mode
    router is never optimised for *quality*, only for balance. Pinned because it
    invalidates any reading of a trained router as "a router that learned to
    route well".
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=4, init="replica",
                                 expert_units=INTER // 4, combine="sum"))
    m(make_input()).sum().backward()
    assert m.router.weight.grad is None


def test_softmax_combine_gives_the_router_a_real_gradient():
    """`combine="softmax"` is the mode that actually trains the router.

    The weighted combination depends on the router logits, so the language-model
    objective reaches the router directly. Measured at roughly five orders of
    magnitude more signal than `sum` (+ aux loss). This is why the branch
    experiment must not read router quality off a `sum` run.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=4, init="replica",
                                 expert_units=INTER // 4, combine="softmax"))
    m(make_input()).sum().backward()
    assert m.router.weight.grad is not None
    assert float(m.router.weight.grad.abs().sum()) > 1e-6


def test_only_selected_experts_receive_gradient():
    """Experts the router never picked must get exactly zero gradient.

    This is the invariant that keeps the model honestly sparse during training.
    If an unselected expert accumulated gradient, the sparse forward would be a
    mask over a dense update and any "sparsity is preserved" reading of the PoC
    would be wrong.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    m(make_input(n=32)).sum().backward()
    usage = torch.tensor(m.last_stats["usage_fraction"])
    selected = int((usage > 0.0).sum())
    assert selected >= 1
    touched = sum(1 for j in range(m.n_experts)
                  if float(m.gate.grad[j].abs().sum()) > 0.0)
    assert touched == selected


class _FakeMLP(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)


def test_from_hf_mlp_matches_direct_construction():
    """The HF entry point must produce the same module as the raw entry point.

    The PoC builds from a real HF `*MLP`, so a shape confusion here would only
    show up after a model download.
    """
    mlp = _FakeMLP(HIDDEN, INTER)
    cfg = BranchConfig(n_experts=4, top_k=1, init="replica",
                       expert_units=INTER // 4)
    # The router is drawn from the global RNG, so reseed between constructions
    # or the two modules would route differently and the comparison would be
    # measuring the RNG rather than the entry points.
    torch.manual_seed(0)
    a = BranchMoEFFN.from_hf_mlp(mlp, cfg)
    torch.manual_seed(0)
    b = BranchMoEFFN(mlp.gate_proj.weight, mlp.up_proj.weight,
                     mlp.down_proj.weight, cfg)
    x = make_input()
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


def test_from_hf_mlp_rejects_a_non_swiglu_mlp():
    """A missing projection must fail loudly rather than silently mis-slice."""
    class _NoGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(HIDDEN, INTER, bias=False)

    with pytest.raises(ValueError):
        BranchMoEFFN.from_hf_mlp(_NoGate(), BranchConfig(n_experts=2, top_k=1))


# ---------------------------------------------------------------------------
# force_expert: training one expert on its own data (branch-train-miX stage 2)
# ---------------------------------------------------------------------------
#
# Stage 2 of branch-train-miX trains every expert *alone* on its own data, before
# any router exists to choose between them. `set_force_expert(e)` is the
# mechanism: it bypasses the router and sends every token to expert `e` at
# weight 1.0. The properties pinned below are what make that stage sound -- an
# expert that is trained through the router, or that is charged a balancing
# penalty while it is supposed to be diverging, would make the stage mean
# something different from what the experiment reports.

def test_force_expert_reproduces_the_dense_layer_at_full_width():
    """Forcing a full-width replica *is* the dense layer, for every expert.

    This is what lets stage 2 train an expert on its own data without the router
    in the way: at full width the forced path has no partial-sum deficit, so
    whatever the expert learns is learned against the true dense function.
    Asserted for every `e`, because the mechanism is per-expert and a bug that
    happened to work for expert 0 would be invisible in a single check.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica"))
    x = make_input()
    dense = m.dense_forward(x)
    for e in range(m.n_experts):
        m.set_force_expert(e)
        assert rel_l2(dense, m(x)) < 1e-6, e


def test_forced_narrow_replicas_are_indistinguishable_from_routing():
    """With narrow replicas, forcing changes nothing -- the router has no choice.

    Every replica is a copy of the same slice, so all forced outputs coincide
    *and* coincide with what the router already produces. This separates "the
    router picks badly" from "the expert is narrow": in the replicated case
    there is nothing to pick badly, and the loss is the width's, not the
    router's. Without this, a routing result at low width could be misread as a
    routing failure.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    x = make_input(n=16)
    with torch.no_grad():
        routed = m(x).clone()
        first = None
        for e in range(m.n_experts):
            m.set_force_expert(e)
            forced = m(x)
            if first is None:
                first = forced
            torch.testing.assert_close(forced, first, rtol=1e-5, atol=1e-6)
        # `combine` defaults to "sum", whose weights are a constant 1.0, so the
        # routed output is exactly the selected expert's output -- comparable
        # with the forced path rather than merely similar to it.
        torch.testing.assert_close(first, routed, rtol=1e-5, atol=1e-6)


def test_force_expert_is_recorded_in_last_stats():
    """Forcing must be visible in the diagnostics, not silent.

    The PoC writes `last_stats` into its report, so if forcing did not show up
    there an all-forced run would look like a routing-concentrated one. The
    usage fraction is a hard 1.0/0.0 split by construction.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    m.set_force_expert(2)
    m(make_input(n=8))
    stats = m.last_stats
    assert stats["forced_expert"] == 2
    usage = stats["usage_fraction"]
    assert usage[2] == pytest.approx(1.0)
    assert all(u == pytest.approx(0.0) for j, u in enumerate(usage) if j != 2)


def test_force_expert_suppresses_the_balancing_loss():
    """No load-balancing penalty is charged while an expert trains alone.

    Stage 2 exists to let experts *diverge*; a balancing loss pushes them the
    other way, and it would also be charged against a routing distribution the
    router did not actually choose. Both the raw and the total loss must be a
    clean zero, or the stage would quietly carry a term the report does not
    mention.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4, z_loss_coef=0.5))
    m.set_force_expert(1)
    m(make_input(n=8))
    assert float(m.last_aux_loss) == 0.0
    assert float(m.total_aux_loss()) == 0.0


def test_forced_path_is_differentiable_but_the_router_is_not():
    """Gradient flows to the forced expert's weights and never to the router.

    Stage 2 has to actually train the expert -- so the expert's own weights must
    be differentiable -- while the router's parameters must stay untouched, or
    the router would be shaped by a stage whose whole premise is that no routing
    decision has been made yet. The router still runs, but under `no_grad`, so
    its gradient is absent rather than merely small.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    m.set_force_expert(2)
    m(make_input(n=8)).sum().backward()
    assert float(m.gate.grad[2].abs().sum()) > 0.0
    assert float(m.up.grad[2].abs().sum()) > 0.0
    assert float(m.down.grad[2].abs().sum()) > 0.0
    assert m.router.weight.grad is None


def test_forced_gradient_reaches_only_the_forced_expert():
    """Only the forced expert's slice of the stacked weights moves.

    If a forced step leaked gradient into the other experts, stage 2 would be a
    joint update wearing an independent-training label, and the divergence the
    stage is meant to create would be partly an artefact. Checked on all three
    stacked tensors because `down` is sliced along the other axis and is the one
    most easily mis-indexed.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    m.set_force_expert(3)
    m(make_input(n=8)).sum().backward()
    for j in range(m.n_experts):
        if j == 3:
            continue
        assert float(m.gate.grad[j].abs().sum()) == 0.0, j
        assert float(m.up.grad[j].abs().sum()) == 0.0, j
        assert float(m.down.grad[j].abs().sum()) == 0.0, j


def test_forced_path_still_trains_the_shared_expert():
    """An always-on shared expert is in the forced path, so it gets gradient.

    The shared block is added unconditionally, so it must keep training while
    the routed experts train independently -- otherwise the one component that
    every token uses would be the only one left behind in stage 2.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4, n_shared=1))
    m.set_force_expert(0)
    m(make_input(n=8)).sum().backward()
    assert float(m.s_gate.grad.abs().sum()) > 0.0
    assert float(m.s_up.grad.abs().sum()) > 0.0
    assert float(m.s_down.grad.abs().sum()) > 0.0


def test_clearing_force_expert_restores_routing():
    """`set_force_expert(None)` must return the module to normal operation.

    The PoC forces each expert in turn and then hands the same modules to the
    cooperation finetune; a flag that did not clear would leave that stage
    silently forcing one expert instead of routing, which would look like a
    router that never learned anything.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    x = make_input(n=16)
    with torch.no_grad():
        routed = m(x).clone()
    m.set_force_expert(1)
    with torch.no_grad():
        m(x)
    assert "forced_expert" in m.last_stats
    m.set_force_expert(None)
    assert m.force_expert is None
    with torch.no_grad():
        after = m(x).clone()
    torch.testing.assert_close(after, routed, rtol=0, atol=0)
    assert "forced_expert" not in m.last_stats
    # The balancing loss is live again: `f >= 0` and the softmax is strictly
    # positive, so the aux loss is strictly positive, not merely non-zero.
    assert float(m.last_aux_loss) > 0.0


def test_set_force_expert_rejects_an_out_of_range_index():
    """An out-of-range index would silently address the wrong expert's weights.

    `-1` is the dangerous one: it is a valid Python index into the stacked
    expert tensors and would quietly force the *last* expert.
    """
    m = make_branch(BranchConfig(n_experts=4, top_k=1, init="replica",
                                 expert_units=INTER // 4))
    with pytest.raises(ValueError):
        m.set_force_expert(-1)
    with pytest.raises(ValueError):
        m.set_force_expert(m.n_experts)
    assert m.force_expert is None
