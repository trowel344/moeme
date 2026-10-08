import hashlib
import json
from pathlib import Path

import pytest

from scripts.upload_cloud_stage import lightning_command, main, verify_free_tier_budget


def staged_fixture(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    stage.mkdir()
    payload = stage / "payload.bin"
    payload.write_bytes(b"verified payload")
    manifest = {
        "format": "moeme-free-cloud-upload-v1",
        "files": {
            "payload.bin": {
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }
    (stage / "upload-manifest.json").write_text(json.dumps(manifest))
    return stage


def budget_fixture(tmp_path: Path, stage: Path, provider: str = "lightning") -> Path:
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {
                "format": "moeme-free-tier-budget-v1",
                "passed": True,
                "measured_from": {
                    "stage_manifest_sha256": hashlib.sha256(
                        (stage / "upload-manifest.json").read_bytes()
                    ).hexdigest()
                },
                "providers": {
                    provider: {
                        (
                            "fits_unbilled_storage"
                            if provider == "lightning"
                            else "fits_working_storage"
                        ): True
                    }
                },
            }
        )
    )
    return path


def test_lightning_command_rejects_non_lightning_and_multiline_destinations(tmp_path: Path) -> None:
    assert lightning_command(tmp_path, "lit:///studios/moeme/job") == [
        "lightning",
        "cp",
        "-r",
        str(tmp_path.resolve()),
        "lit:///studios/moeme/job",
    ]
    with pytest.raises(ValueError, match="lit://"):
        lightning_command(tmp_path, "/tmp/not-remote")
    with pytest.raises(ValueError, match="single-line"):
        lightning_command(tmp_path, "lit:///studio/job\n--danger")


def test_budget_must_fit_provider_and_bind_exact_stage(tmp_path: Path) -> None:
    stage = staged_fixture(tmp_path)
    budget = budget_fixture(tmp_path, stage)
    assert verify_free_tier_budget(stage, budget, "lightning")["verified"] is True
    value = json.loads(budget.read_text())
    value["providers"]["lightning"]["fits_unbilled_storage"] = False
    budget.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not fit"):
        verify_free_tier_budget(stage, budget, "lightning")
    value["providers"]["lightning"]["fits_unbilled_storage"] = True
    value["measured_from"]["stage_manifest_sha256"] = "0" * 64
    budget.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="stale"):
        verify_free_tier_budget(stage, budget, "lightning")


def test_execute_requires_signed_in_account_usage_check(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "upload_cloud_stage.py",
            "--destination",
            "lit:///studios/moeme/layer63-200k",
            "--execute",
        ],
    )
    with pytest.raises(SystemExit):
        main()


def test_dry_run_verifies_stage_without_external_process(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    stage = staged_fixture(tmp_path)
    budget = budget_fixture(tmp_path, stage)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "upload_cloud_stage.py",
            "--stage",
            str(stage),
            "--destination",
            "lit:///studios/moeme/layer63-200k",
            "--budget",
            str(budget),
            "--receipt",
            str(receipt),
        ],
    )
    monkeypatch.setattr(
        "scripts.upload_cloud_stage.subprocess.run",
        lambda *args, **kwargs: pytest.fail("dry run attempted an external process"),
    )
    assert main() == 0
    value = json.loads(receipt.read_text())
    assert value["status"] == "verified_dry_run"
    assert value["executed"] is False
    assert value["stage_verification"]["verified"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "verified_dry_run"


def test_execute_uses_resolved_cli_and_receipts_result(tmp_path: Path, monkeypatch) -> None:
    stage = staged_fixture(tmp_path)
    budget = budget_fixture(tmp_path, stage)
    receipt = tmp_path / "receipt.json"
    observed = []

    class Completed:
        returncode = 0

    monkeypatch.setattr("scripts.upload_cloud_stage.shutil.which", lambda _: "/bin/lightning")
    monkeypatch.setattr(
        "scripts.upload_cloud_stage.subprocess.run",
        lambda command, check: observed.append((command, check)) or Completed(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "upload_cloud_stage.py",
            "--stage",
            str(stage),
            "--destination",
            "lit:///studios/moeme/layer63-200k",
            "--budget",
            str(budget),
            "--receipt",
            str(receipt),
            "--execute",
            "--account-usage-checked",
        ],
    )
    assert main() == 0
    assert observed == [
        (
            [
                "/bin/lightning",
                "cp",
                "-r",
                str(stage.resolve()),
                "lit:///studios/moeme/layer63-200k",
            ],
            False,
        )
    ]
    value = json.loads(receipt.read_text())
    assert value["status"] == "passed"
    assert value["returncode"] == 0
