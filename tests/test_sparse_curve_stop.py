from scripts.train_sparse_layer import flat_curve_stop


def point(step: int, error: float, stage: int = 0) -> dict:
    return {"stage_index": stage, "step": step, "metrics": {"relative_l2": error}}


def test_flat_high_curve_stops_only_after_minimum_step() -> None:
    curve = [
        point(7500, 0.08),
        point(10000, 0.081),
        point(12500, 0.082),
        point(15000, 0.083),
    ]
    evidence = flat_curve_stop(
        curve,
        stage_index=0,
        minimum_step=15000,
        recent_points=4,
        target_error=0.01,
    )
    assert evidence is not None
    assert evidence["reason"] == "flat_or_regressing_high"
    assert evidence["step"] == 15000


def test_improving_or_near_target_curve_keeps_training() -> None:
    improving = [
        point(step, error)
        for step, error in [(7500, 0.08), (10000, 0.07), (12500, 0.06), (15000, 0.05)]
    ]
    near_target = [
        point(step, error)
        for step, error in [(7500, 0.029), (10000, 0.028), (12500, 0.029), (15000, 0.03)]
    ]
    options = {
        "stage_index": 0,
        "minimum_step": 15000,
        "recent_points": 4,
        "target_error": 0.01,
    }
    assert flat_curve_stop(improving, **options) is None
    assert flat_curve_stop(near_target, **options) is None
