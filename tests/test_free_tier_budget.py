import hashlib
import json
from pathlib import Path

from scripts.analyze_free_tier_budget import budget_report


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    stage = tmp_path / "stage"
    smoke = tmp_path / "smoke"
    stage.mkdir()
    smoke.mkdir()
    (stage / "upload-manifest.json").write_text(
        json.dumps({"format": "moeme-free-cloud-upload-v1", "logical_bytes": 4_000})
    )
    (smoke / "cloud-preflight.json").write_text(json.dumps({"activation": {"tokens": 100}}))
    (smoke / "training-state.pt").write_bytes(b"s" * 1_000)
    (smoke / "layer-63-top4.safetensors").write_bytes(b"c" * 500)
    labels = smoke / "oracle-labels/train"
    labels.mkdir(parents=True)
    (labels / "chunk.safetensors").write_bytes(b"l" * 100)
    (smoke / "report.json").write_bytes(b"r" * 50)
    return stage, smoke


def test_budget_uses_atomic_double_state_and_scales_labels(tmp_path: Path) -> None:
    stage, smoke = make_fixture(tmp_path)
    report = budget_report(
        stage,
        smoke,
        production_tokens=1_000,
        lightning_unbilled_bytes=20_000,
        kaggle_working_bytes=20_000,
        reserve_bytes=200,
    )
    components = report["components"]
    assert (
        report["measured_from"]["stage_manifest_sha256"]
        == hashlib.sha256((stage / "upload-manifest.json").read_bytes()).hexdigest()
    )
    assert components["scaled_oracle_label_bytes"] == 1_000
    assert components["fixed_smoke_result_bytes"] == 50 + len(
        (smoke / "cloud-preflight.json").read_bytes()
    )
    assert components["estimated_output_atomic_peak_bytes"] == (
        2_000 + 500 + 1_000 + components["fixed_smoke_result_bytes"] + 200
    )
    assert report["recommended_provider"] == "lightning"
    assert report["passed"] is True
    assert report["limit_assumptions"]["requires_account_usage_check"] is True


def test_budget_falls_back_to_kaggle_then_fails_closed(tmp_path: Path) -> None:
    stage, smoke = make_fixture(tmp_path)
    fallback = budget_report(
        stage,
        smoke,
        production_tokens=1_000,
        lightning_unbilled_bytes=1,
        kaggle_working_bytes=20_000,
        reserve_bytes=0,
    )
    assert fallback["recommended_provider"] == "kaggle"
    assert fallback["providers"]["lightning"]["fits_unbilled_storage"] is False

    rejected = budget_report(
        stage,
        smoke,
        production_tokens=1_000,
        lightning_unbilled_bytes=1,
        kaggle_working_bytes=1,
        reserve_bytes=0,
    )
    assert rejected["recommended_provider"] is None
    assert rejected["passed"] is False
