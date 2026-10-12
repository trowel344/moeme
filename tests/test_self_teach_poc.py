"""The corpus-free self-teaching driver's own logic.

`experiments/self_teach_poc.py` asks whether a model can teach its own sparse
layers using its own dense FFN as the teacher, with no labels and no corpus.
Two things in it are pure enough to pin on the CPU with no download, and both
carry the experiment's conclusions:

  * `magnitude_analysis` -- the least-squares optimal output scale `s*` and the
    residual at that scale. This is what decides whether the field's E/k output
    rescaling applies to a *partition* (measured: it does not, `s*` ~ 1.1-1.6
    against E/k of 8/4/2), so the property that `s*` really is the best scalar
    -- and not merely a plausible one -- has to be tested rather than trusted.
  * the calibration sentence set, which is the *only* training data in the
    experiment. If it moved between runs, the run would not be comparable to
    anything, and `sentence count` is quoted as a headline number.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

import self_teach_poc as st                      # noqa: E402
from moeme.ffn import DenseFFN                   # noqa: E402
from moeme.upcycle import MoEConfig, MoEFFN      # noqa: E402

HIDDEN, INTER = 32, 96


def make_moe(n_experts=8, top_k=1, plan="random", seed=0, inter=INTER):
    g = torch.Generator().manual_seed(seed)
    dense = DenseFFN(
        gate=torch.randn(inter, HIDDEN, generator=g) * 0.2,
        up=torch.randn(inter, HIDDEN, generator=g) * 0.2,
        down=torch.randn(HIDDEN, inter, generator=g) * 0.2,
    )
    torch.manual_seed(seed)
    moe = MoEFFN.from_dense(dense, MoEConfig(n_experts=n_experts, top_k=top_k,
                                             plan=plan, seed=seed))
    return dense, moe


# ---------------------------------------------------------------------------
# the calibration set: the experiment's only data
# ---------------------------------------------------------------------------

def test_calibration_sentences_are_fixed_and_unique():
    assert len(st.SENTENCES) >= 20
    assert all(isinstance(s, str) and s.strip() for s in st.SENTENCES)
    assert len(set(st.SENTENCES)) == len(st.SENTENCES), "duplicate sentences"
    # the run's headline provenance field; a set that drifts makes runs
    # incomparable and makes the quoted token count wrong
    assert st.SENTENCES[0] == "Paris is the capital of France."
    assert st.SENTENCES[1] == "The world is round."


def test_calibration_set_includes_code_as_well_as_prose():
    """All-prose activations would probe one grammatical shape only. The set is
    documented as mixed, so it is checked as mixed."""
    assert any("def " in s or "for " in s for s in st.SENTENCES)


# ---------------------------------------------------------------------------
# the magnitude analysis: s* is the best scalar, not just a plausible one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selection", ["oracle", "router"])
def test_best_scale_beats_every_other_scale(selection):
    """The whole conclusion rests on `s*` being the least-squares optimum, so it
    is checked against the alternatives the report prints -- not asserted."""
    dense, moe = make_moe()
    x = torch.randn(24, HIDDEN)
    ref = dense.forward(x)
    m = st.magnitude_analysis(moe, x, ref, selection)
    best = m["residual_at_best_scale"]
    assert best <= m["residual_at_scale_1"] + 1e-6
    assert best <= m["residual_at_E_over_k"] + 1e-6


def test_best_scale_is_one_when_the_kept_slices_are_the_whole_output():
    """A degenerate case with a known answer: one expert owning everything means
    one selected expert *is* the dense layer, so the residual is 0 at scale 1 and
    the optimal scale is 1."""
    dense, moe = make_moe(n_experts=1, top_k=1)
    x = torch.randn(8, HIDDEN)
    ref = dense.forward(x)
    m = st.magnitude_analysis(moe, x, ref, "oracle")
    assert m["best_scale_mean"] == pytest.approx(1.0, abs=1e-4)
    assert m["residual_at_best_scale"] < 1e-5


def test_all_experts_active_has_no_reconstruction_error():
    """The partition identity, measured through the same function the experiment
    reports: with E=k the selected set is every expert, so the residual against
    the dense layer must vanish. If this ever fails, the published residual
    numbers are measuring a plumbing bug."""
    dense, moe = make_moe(n_experts=4, top_k=4)
    x = torch.randn(8, HIDDEN)
    m = st.magnitude_analysis(moe, x, dense.forward(x), "oracle")
    assert m["residual_at_scale_1"] < 1e-5


def test_analysis_records_the_scale_the_literature_uses():
    """E/k is the rescaling LLaMA-MoE and NVIDIA call essential. The experiment
    exists to compare it against a measured s*, so the field must be present and
    correct or the comparison silently becomes a tautology."""
    _, moe = make_moe(n_experts=8, top_k=1)
    m = st.magnitude_analysis(moe, torch.randn(4, HIDDEN), torch.randn(4, HIDDEN),
                              "oracle")
    assert m["E_over_k"] == 8.0
    _, moe2 = make_moe(n_experts=8, top_k=4)
    m2 = st.magnitude_analysis(moe2, torch.randn(4, HIDDEN),
                               torch.randn(4, HIDDEN), "oracle")
    assert m2["E_over_k"] == 2.0


def test_analysis_reports_the_selection_it_used():
    _, moe = make_moe()
    x = torch.randn(4, HIDDEN)
    ref = moe.dense_forward(x)
    assert st.magnitude_analysis(moe, x, ref, "oracle")["selection"] == "oracle"
    assert st.magnitude_analysis(moe, x, ref, "router")["selection"] == "router"


def test_oracle_and_router_residuals_stay_in_range():
    _, moe = make_moe(inter=128)
    x = torch.randn(32, HIDDEN)
    ref = moe.dense_forward(x)
    for sel in ("oracle", "router"):
        m = st.magnitude_analysis(moe, x, ref, sel)
        assert 0.0 <= m["residual_at_scale_1"] <= 2.0
        assert 0.0 <= m["residual_at_best_scale"] <= 2.0


# ---------------------------------------------------------------------------
# the shared plumbing the driver relies on
# ---------------------------------------------------------------------------

def test_calibration_token_count_reads_the_activation_shapes():
    caps = {0: {"x": torch.zeros(7, HIDDEN)}, 1: {"x": torch.zeros(7, HIDDEN)}}
    assert st.n_calibration_tokens(caps) == 7


def test_reconstruction_loss_is_scale_sensitive_by_construction():
    """The loss the driver optimises is a plain relative L2, so scaling the MoE
    output away from the teacher *increases* it. That is why the E/k arm starts
    from a worse loss (2.45 vs 0.92) rather than the same one -- and why the
    rescaling cannot help a partition whose optimal scale is ~1."""
    dense, moe = make_moe()
    x = torch.randn(16, HIDDEN)
    caps = {0: {"x": x, "y": dense.forward(x)}}
    base = float(st.reconstruction_loss([(0, moe)], caps, torch.float32))
    with torch.no_grad():
        moe.down.mul_(8.0)
    worse = float(st.reconstruction_loss([(0, moe)], caps, torch.float32))
    assert worse > base
