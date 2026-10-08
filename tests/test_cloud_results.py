import json
from pathlib import Path

import pytest

from moeme.cloud_results import (
    candidate_checkpoint_name,
    configured_top_k,
    scientific_candidate_eligibility,
    verify_result_manifest,
    write_result_manifest,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def complete_result(root: Path) -> None:
    write_json(root / "cloud-preflight.json", {"format": "preflight"})
    write_json(root / "cloud-run.json", {"status": "passed"})
    write_json(root / "progress.json", {"phase": "complete"})
    write_json(root / "curve.json", {"points": []})
    write_json(
        root / "curve-analysis.json",
        {
            "status": "promising",
            "baseline_error": 0.1064,
            "target_error": 0.01,
            "last": {"relative_l2": 0.07},
        },
    )
    write_json(
        root / "report.json",
        {
            "layer": 63,
            "passed": True,
            "stages": [
                {
                    "checkpoint": "/cloud/layer-63-top4.safetensors",
                    "validation": {"relative_l2": 0.075},
                }
            ],
        },
    )
    (root / "layer-63-top4.safetensors").write_bytes(b"checkpoint")


def test_result_manifest_binds_complete_candidate(tmp_path: Path) -> None:
    complete_result(tmp_path)
    label_chunk = tmp_path / "oracle-labels/training/chunk-00000.safetensors"
    label_chunk.parent.mkdir(parents=True)
    label_chunk.write_bytes(b"labels")
    manifest = write_result_manifest(tmp_path, 63)
    assert manifest["candidate_ready"] is True
    verified = verify_result_manifest(tmp_path)
    assert verified["candidate_ready"] is True
    assert "layer-63-top4.safetensors" in verified["files"]
    assert "oracle-labels/training/chunk-00000.safetensors" in verified["files"]


def test_result_verification_rejects_transfer_corruption(tmp_path: Path) -> None:
    complete_result(tmp_path)
    write_result_manifest(tmp_path, 63)
    (tmp_path / "layer-63-top4.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_result_manifest(tmp_path)


def test_candidate_checkpoint_follows_the_trained_top_k(tmp_path: Path) -> None:
    """A de-risk routed at Top-8 must not be disqualified by a Top-4 filename."""
    complete_result(tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    report["configuration"] = {"top_k_schedule": [8]}
    write_json(tmp_path / "report.json", report)
    (tmp_path / "layer-63-top4.safetensors").unlink()
    (tmp_path / "layer-63-top8.safetensors").write_bytes(b"checkpoint")
    manifest = write_result_manifest(tmp_path, 63)
    assert manifest["candidate_top_k"] == 8
    assert manifest["candidate_ready"] is True
    verified = verify_result_manifest(tmp_path)
    assert verified["candidate_ready"] is True
    assert "layer-63-top8.safetensors" in verified["files"]
    assert "layer-63-top4.safetensors" not in verified["files"]


def test_configured_top_k_prefers_configuration_then_command() -> None:
    assert configured_top_k({}, {}) == 4
    assert configured_top_k({}, {"configuration": {"top_k_schedule": [12, 8]}}) == 8
    assert configured_top_k({"command": ["python", "t.py", "--top-k-schedule", "8"]}, {}) == 8
    assert (
        configured_top_k(
            {"command": ["python", "t.py", "--top-k-schedule", "10", "8", "6", "--x"]}, {}
        )
        == 6
    )
    assert candidate_checkpoint_name(63, 8) == "layer-63-top8.safetensors"


def test_candidate_eligibility_binds_last_curve_and_final_checkpoint() -> None:
    curve = {
        "status": "promising",
        "baseline_error": 0.1064,
        "target_error": 0.01,
        "last": {"relative_l2": 0.07},
    }
    eligible = scientific_candidate_eligibility(
        {"passed": True, "stages": [{"validation": {"relative_l2": 0.075}}]}, curve
    )
    assert eligible["passed"] is True
    regressed = scientific_candidate_eligibility(
        {"passed": True, "stages": [{"validation": {"relative_l2": 0.11}}]}, curve
    )
    assert regressed["passed"] is False
    assert regressed["checks"]["final_validation_within_limit"] is False
