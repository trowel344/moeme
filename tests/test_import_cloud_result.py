import json
import zipfile
from pathlib import Path

import pytest

from moeme.cloud_results import build_result_manifest
from scripts.import_cloud_result import import_result, validate_zip


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_result(path: Path, *, eligible: bool) -> None:
    write_json(path / "cloud-preflight.json", {"format": "preflight"})
    write_json(path / "progress.json", {"phase": "complete" if eligible else "training"})
    write_json(path / "cloud-run.json", {"status": "passed" if eligible else "interrupted"})
    if eligible:
        checkpoint = path / "layer-63-top4.safetensors"
        checkpoint.write_bytes(b"trained")
        write_json(
            path / "report.json",
            {
                "passed": True,
                "stages": [
                    {
                        "checkpoint": str(checkpoint),
                        "validation": {"relative_l2": 0.075},
                    }
                ],
            },
        )
        write_json(path / "curve.json", {"points": []})
        write_json(
            path / "curve-analysis.json",
            {
                "status": "promising",
                "baseline_error": 0.1064,
                "target_error": 0.01,
                "last": {"relative_l2": 0.07},
            },
        )
    write_json(path / "cloud-result-manifest.json", build_result_manifest(path, 63))


def test_directory_result_is_verified_then_atomically_published(tmp_path: Path) -> None:
    source = tmp_path / "download/result"
    target = tmp_path / "cloud-results/layer63-200k"
    receipt = tmp_path / "transfer.json"
    make_result(source, eligible=True)
    result = import_result(source, target, receipt)
    assert result["candidate_ready"] is True
    assert result["atomic_publish"] is True
    assert result["source_preserved"] is True
    assert source.is_dir()
    assert (target / "layer-63-top4.safetensors").read_bytes() == b"trained"
    assert json.loads(receipt.read_text())["candidate_ready"] is True


def test_directory_result_rejects_file_added_after_manifest(tmp_path: Path) -> None:
    source = tmp_path / "download/result"
    make_result(source, eligible=True)
    (source / "undeclared.bin").write_bytes(b"not in the cloud manifest")
    with pytest.raises(ValueError, match="unexpected=.*undeclared.bin"):
        import_result(source, tmp_path / "target", tmp_path / "receipt.json")


def test_nested_zip_result_is_located_and_imported(tmp_path: Path) -> None:
    source = tmp_path / "build/nested/result"
    make_result(source, eligible=False)
    archive = tmp_path / "download.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for item in source.rglob("*"):
            if item.is_file():
                handle.write(item, Path("provider-output/result") / item.relative_to(source))
    target = tmp_path / "imported"
    result = import_result(archive, target, tmp_path / "receipt.json")
    assert result["candidate_ready"] is False
    assert result["run_status"] == "interrupted"
    assert archive.is_file()
    assert (target / "cloud-result-manifest.json").is_file()


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as writer:
        writer.writestr("../escape", "bad")
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="unsafe path"):
        validate_zip(archive)


def test_import_refuses_to_overlay_existing_result(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    make_result(source, eligible=False)
    target.mkdir()
    (target / "unrelated").write_text("preserve")
    with pytest.raises(FileExistsError, match="refusing to overlay"):
        import_result(source, target, tmp_path / "receipt.json")
    assert (target / "unrelated").read_text() == "preserve"
