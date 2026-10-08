import hashlib
import json
from pathlib import Path

import pytest

from scripts.upload_kaggle_stage import (
    build_upload_view,
    dataset_metadata,
    kaggle_command,
    main,
)


def staged_fixture(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    payload = stage / "nested/payload.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"verified payload")
    manifest = {
        "format": "moeme-free-cloud-upload-v1",
        "files": {
            "nested/payload.bin": {
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }
    (stage / "upload-manifest.json").write_text(json.dumps(manifest))
    return stage


def budget_fixture(tmp_path: Path, stage: Path) -> Path:
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
                "providers": {"kaggle": {"fits_working_storage": True}},
            }
        )
    )
    return path


def test_private_metadata_and_command_reject_unsafe_identifiers(tmp_path: Path) -> None:
    metadata = dataset_metadata("owner/moeme-private", "MoEMe Private Stage")
    assert metadata["id"] == "owner/moeme-private"
    assert metadata["licenses"] == [{"name": "other"}]
    assert kaggle_command(tmp_path) == [
        "kaggle",
        "datasets",
        "create",
        "--path",
        str(tmp_path.resolve()),
        "--dir-mode",
        "tar",
        "--keep-tabular",
    ]
    assert "--public" not in kaggle_command(tmp_path)
    with pytest.raises(ValueError, match="owner/slug"):
        dataset_metadata("owner/moeme\n--public", "MoEMe Private Stage")


def test_version_notes_select_the_update_command_for_an_existing_dataset(
    tmp_path: Path,
) -> None:
    """`create` cannot refresh a published dataset; `version` with notes can."""
    assert kaggle_command(tmp_path, version_notes="K=8 recipe") == [
        "kaggle",
        "datasets",
        "version",
        "--path",
        str(tmp_path.resolve()),
        "--dir-mode",
        "tar",
        "--keep-tabular",
        "-m",
        "K=8 recipe",
    ]
    assert "create" in kaggle_command(tmp_path)
    assert "--public" not in kaggle_command(tmp_path, version_notes="K=8 recipe")


def test_execute_requires_signed_in_account_usage_check(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "upload_kaggle_stage.py",
            "--dataset",
            "owner/moeme-private",
            "--execute",
        ],
    )
    with pytest.raises(SystemExit):
        main()


def test_upload_view_is_exact_zero_copy_mirror(tmp_path: Path) -> None:
    stage = staged_fixture(tmp_path)
    view = tmp_path / "view"
    metadata = dataset_metadata("owner/moeme-private", "MoEMe Private Stage")
    result = build_upload_view(stage, view, metadata)
    assert result["verified"] is True
    assert (view / "nested/payload.bin").samefile(stage / "nested/payload.bin")
    assert json.loads((view / "dataset-metadata.json").read_text()) == metadata
    assert build_upload_view(stage, view, metadata) == result


def test_dry_run_never_calls_external_process(tmp_path: Path, monkeypatch) -> None:
    stage = staged_fixture(tmp_path)
    budget = budget_fixture(tmp_path, stage)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "upload_kaggle_stage.py",
            "--stage",
            str(stage),
            "--dataset",
            "owner/moeme-private",
            "--budget",
            str(budget),
            "--view",
            str(tmp_path / "view"),
            "--receipt",
            str(receipt),
        ],
    )
    monkeypatch.setattr(
        "scripts.upload_kaggle_stage.subprocess.run",
        lambda *args, **kwargs: pytest.fail("dry run attempted an external process"),
    )
    assert main() == 0
    value = json.loads(receipt.read_text())
    assert value["provider"] == "kaggle"
    assert value["privacy"] == "private"
    assert value["status"] == "verified_dry_run"
    assert value["executed"] is False


def test_execute_uses_resolved_cli_and_receipts_result(tmp_path: Path, monkeypatch) -> None:
    stage = staged_fixture(tmp_path)
    budget = budget_fixture(tmp_path, stage)
    view = tmp_path / "view"
    receipt = tmp_path / "receipt.json"
    observed = []

    class Completed:
        returncode = 0

    monkeypatch.setattr("scripts.upload_kaggle_stage.shutil.which", lambda _: "/bin/kaggle")
    monkeypatch.setattr(
        "scripts.upload_kaggle_stage.subprocess.run",
        lambda command, check: observed.append((command, check)) or Completed(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "upload_kaggle_stage.py",
            "--stage",
            str(stage),
            "--dataset",
            "owner/moeme-private",
            "--budget",
            str(budget),
            "--view",
            str(view),
            "--receipt",
            str(receipt),
            "--execute",
            "--account-usage-checked",
        ],
    )
    assert main() == 0
    assert observed == [(kaggle_command(view, "/bin/kaggle"), False)]
    value = json.loads(receipt.read_text())
    assert value["status"] == "passed"
    assert value["returncode"] == 0
