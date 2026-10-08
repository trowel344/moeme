from scripts.analyze_sparse_curve import analyze_curve, linear_slope


def curve(errors: list[float]) -> dict:
    return {
        "points": [
            {"step": (index + 1) * 2500, "metrics": {"relative_l2": error}}
            for index, error in enumerate(errors)
        ]
    }


def test_curve_analysis_reports_promising_gap_closure() -> None:
    result = analyze_curve(
        curve([0.09, 0.075, 0.06, 0.045]), baseline_error=0.1064, target_error=0.01
    )
    assert result["status"] == "promising"
    assert result["headroom_closed_fraction"] > 0.6
    assert result["recent_slope_per_step"] < 0
    assert result["projected_target_step"] > 10000


def test_curve_analysis_does_not_call_flat_high_curve_success() -> None:
    result = analyze_curve(
        curve([0.08, 0.079, 0.081, 0.082]), baseline_error=0.1064, target_error=0.01
    )
    assert result["status"] == "flat_or_regressing_high"
    assert result["projected_target_step"] is None


def test_curve_analysis_requires_evidence_and_recognizes_target() -> None:
    empty = analyze_curve({"points": []}, baseline_error=0.1064, target_error=0.01)
    assert empty["status"] == "insufficient_data"
    reached = analyze_curve(curve([0.02, 0.009]), baseline_error=0.1064, target_error=0.01)
    assert reached["status"] == "target_reached"
    assert linear_slope([(1, 1.0)]) is None


def test_curve_analysis_uses_measured_initial_point_by_default() -> None:
    measured = {
        "points": [
            {"step": 0, "metrics": {"relative_l2": 0.12}},
            {"step": 2500, "metrics": {"relative_l2": 0.08}},
            {"step": 5000, "metrics": {"relative_l2": 0.06}},
            {"step": 7500, "metrics": {"relative_l2": 0.04}},
        ]
    }
    result = analyze_curve(measured, baseline_error=None, target_error=0.01)
    assert result["baseline_error"] == 0.12
    assert result["first"] == {"step": 0, "relative_l2": 0.12}


def test_historical_best_cannot_hide_a_regressed_exported_checkpoint() -> None:
    result = analyze_curve(
        curve([0.07, 0.06, 0.09, 0.085]),
        baseline_error=0.1064,
        target_error=0.01,
    )
    assert result["best_headroom_closed_fraction"] > 0.25
    assert result["headroom_closed_fraction"] < 0.25
    assert result["status"] != "promising"
