"""
Tests for the eval-window recovery tool (experiments/recover_eval_window.py).

Why this file exists: the tool's whole job is to turn "this perplexity came from
some window, nobody wrote down which" into an identification. That means it can
fail in two opposite ways, and both are worth pinning offline:

1. Claiming a window that no measurement supports -- curve-fitting a number that
   came from something else. A recovered window must be reproducible to within
   tolerance, and a target that no window can produce must say so.
2. Refusing a window the evidence does support, because the target was rounded by
   an equivalent code path. The driver's `dense_limit_val_ppl` is exactly that
   case: the same evaluation re-run through the sum-of-all-experts MoE, which
   reproduces the dense FFN only to ~2e-4. Read strictly, it looks like a
   refutation of the correct answer.

Both cases are on real numbers here, from E14's lost `--eval-blocks = 6`.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import recover_eval_window as rec  # noqa: E402
from recover_eval_window import (  # noqa: E402
    match_window,
    nearest_window,
    ppl_curve,
    verdict,
)

# Equal-weight losses whose running perplexity is analytic: exp(mean of logs).
LOGS = [math.log(4.0), math.log(8.0), math.log(16.0), math.log(32.0)]
EQUAL = ppl_curve(LOGS, [1, 1, 1, 1])


# ---------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------

def test_curve_is_the_running_weighted_mean_the_driver_computes():
    """`eval_perplexity` accumulates loss * tokens and divides by tokens."""
    assert [round(v, 9) for v in EQUAL] == [
        round(v, 9) for v in
        [4.0, math.sqrt(4 * 8), (4 * 8 * 16) ** (1 / 3), (4 * 8 * 16 * 32) ** 0.25]]


def test_curve_weights_by_tokens_so_a_short_block_is_not_a_full_one():
    """v1 of this curve divided by the block count. That agrees with the driver
    only while every block is full, and silently drifts when the last one is not."""
    weighted = ppl_curve([math.log(100.0), math.log(2.0)], [100, 1])
    exact = math.exp((100 * math.log(100.0) + math.log(2.0)) / 101)
    naive = math.exp((math.log(100.0) + math.log(2.0)) / 2)
    assert abs(weighted[-1] - exact) < 1e-9
    assert abs(weighted[-1] - naive) > 1.0


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------

def test_a_perplexity_from_the_curve_recovers_its_own_window():
    for k, value in enumerate(EQUAL, start=1):
        assert match_window(EQUAL, value) == [k], (k, EQUAL)
        assert verdict(EQUAL, value)[0] == "recovered", k


def test_an_aliased_curve_reports_every_window_that_produces_it():
    """Losses (9, 1, 9, 1) give running perplexities (9, 3, 81**(1/3), 3): two
    windows produce 3.0, and the honest answer is both rather than the first."""
    aliased = ppl_curve([math.log(9), math.log(1), math.log(9), math.log(1)], [1] * 4)
    assert match_window(aliased, 3.0) == [2, 4]


# ---------------------------------------------------------------------------
# the two failure modes
# ---------------------------------------------------------------------------

def test_a_number_no_window_produces_is_refused_not_fitted():
    """The guard against curve-fitting: 500.0 is nowhere near this curve, so the
    tool must say `none` and refuse to name a window -- even though `nearest_window`
    will always return *some* k, which is exactly why it must not be trusted alone."""
    assert match_window(EQUAL, 500.0) == []
    kind, hits, k, rel = verdict(EQUAL, 500.0)
    assert kind == "none" and hits == []
    assert k is not None and rel > 0.1, (k, rel)


def test_the_dense_limit_path_reads_as_near_and_not_as_refuted():
    """E14's `dense_limit_val_ppl` is the same evaluation through the sum-of-all-
    experts path, which reproduces the dense number to ~2e-4. It must read as
    `near` -- pointing at the window -- because `none` would refute the right
    answer on evidence that supports it."""
    rounded = EQUAL[-1] * (1 + 2e-4)
    kind, hits, k, rel = verdict(EQUAL, rounded)
    assert kind == "near" and hits == []
    assert k == len(EQUAL) and rel <= 1e-3


def test_the_near_band_does_not_swallow_real_differences():
    """`near` must stay a narrow band: a genuine 1% difference is not a rounding
    artefact and must not be reported as a window."""
    for off in (0.01, 0.05, 0.2):
        assert verdict(EQUAL, EQUAL[-1] * (1 + off))[0] == "none", off


def test_nearest_window_is_relative_not_absolute():
    """A residual is only meaningful against the size of the target, since these
    perplexities span 15-24 on the same corpus. Stated as the property -- the
    smallest *relative* residual, at the matching index -- because the largest
    value here (11.31) is the nearest to 400 while 4.0 looks nearer in absolute
    terms if you only eyeball the first entry."""
    target = 400.0
    k, rel = nearest_window(EQUAL, target)
    relative = [abs(v - target) / target for v in EQUAL]
    assert abs(rel - min(relative)) < 1e-12
    assert k == relative.index(min(relative)) + 1
    assert k == len(EQUAL), "the closest value to 400 in a rising curve is the last"


# ---------------------------------------------------------------------------
# the delivered self-test
# ---------------------------------------------------------------------------

def test_self_test_passes_offline():
    assert rec.self_test() == 0
