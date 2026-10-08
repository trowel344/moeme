#!/usr/bin/env python3
"""Summarize sparse-training progress against the calibrated per-layer target."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def linear_slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def analyze_curve(
    document: dict,
    *,
    baseline_error: float | None,
    target_error: float,
    recent_points: int = 4,
) -> dict:
    raw_points = document.get("points", [])
    points = []
    for item in raw_points:
        try:
            step = int(item["step"])
            error = float(item["metrics"]["relative_l2"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(error):
            points.append((step, error))
    points.sort()
    if not points:
        return {
            "format": "moeme-sparse-curve-analysis-v1",
            "status": "insufficient_data",
            "point_count": 0,
            "baseline_error": baseline_error,
            "target_error": target_error,
        }

    first_step, first_error = points[0]
    baseline_error = first_error if baseline_error is None else baseline_error
    best_step, best_error = min(points, key=lambda point: point[1])
    last_step, last_error = points[-1]
    denominator = baseline_error - target_error
    best_headroom_closed = (
        (baseline_error - best_error) / denominator if denominator > 0 else float("nan")
    )
    headroom_closed = (
        (baseline_error - last_error) / denominator if denominator > 0 else float("nan")
    )
    recent = points[-recent_points:]
    slope = linear_slope(recent)
    projected_target_step = None
    if slope is not None and slope < 0 and last_error > target_error:
        projected_target_step = round(last_step + (target_error - last_error) / slope)

    if best_error <= target_error:
        status = "target_reached"
    elif len(points) < recent_points:
        status = "insufficient_data"
    elif headroom_closed >= 0.25 and slope is not None and slope < 0:
        status = "promising"
    elif slope is not None and slope >= 0 and best_error > target_error * 3:
        status = "flat_or_regressing_high"
    else:
        status = "weak_or_uncertain"

    return {
        "format": "moeme-sparse-curve-analysis-v1",
        "status": status,
        "point_count": len(points),
        "baseline_error": baseline_error,
        "target_error": target_error,
        "first": {"step": first_step, "relative_l2": first_error},
        "best": {"step": best_step, "relative_l2": best_error},
        "last": {"step": last_step, "relative_l2": last_error},
        "headroom_closed_fraction": headroom_closed,
        "best_headroom_closed_fraction": best_headroom_closed,
        "recent_point_count": len(recent),
        "recent_slope_per_step": slope,
        "projected_target_step": projected_target_step,
        "caveat": (
            "This is a per-layer de-risk signal, not full-model acceptance. A candidate still "
            "requires injection plus three-corpus source-logit and capability gates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline-error",
        type=float,
        help="Known baseline error; defaults to the first measured curve point.",
    )
    parser.add_argument("--target-error", type=float, default=0.01)
    parser.add_argument("--recent-points", type=int, default=4)
    args = parser.parse_args()
    if args.recent_points < 2:
        parser.error("--recent-points must be at least 2")
    result = analyze_curve(
        json.loads(args.curve.read_text()),
        baseline_error=args.baseline_error,
        target_error=args.target_error,
        recent_points=args.recent_points,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
