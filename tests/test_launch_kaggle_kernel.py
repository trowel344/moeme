import hashlib
import json
from pathlib import Path

import pytest

from scripts.launch_kaggle_kernel import (
    build_kernel_view,
    kaggle_kernel_command,
    kernel_metadata,
    kernel_source,
    refuse_duplicate_launch,
    title_slug,
    verify_upload_receipt,
)


def test_private_t4_metadata_attaches_only_requested_dataset() -> None:
    value = kernel_metadata(
        "owner/moeme-layer-63-private-training",
        "owner/moeme-layer63-private",
        "MoEMe Layer 63 Private Training",
    )
    assert value["is_private"] is True
    assert value["enable_gpu"] is True
    assert value["machine_shape"] == "NvidiaTeslaT4"
    assert value["dataset_sources"] == ["owner/moeme-layer63-private"]
    with pytest.raises(ValueError, match="owner/slug"):
        kernel_metadata("owner/bad\nslug", "owner/dataset", "Valid Private Title")


def test_title_slug_matches_kaggles_title_derived_slug() -> None:
    assert title_slug("MoEMe Layer 63 Private Training") == "moeme-layer-63-private-training"
    assert title_slug("MoEMe Layer63 Training") == "moeme-layer63-training"


def test_kernel_id_must_match_the_slug_kaggle_derives_from_the_title() -> None:
    # Kaggle ignores a non-resolving id and creates the kernel at the title slug, so a
    # mismatch must fail before the push instead of binding a kernel that cannot be polled.
    with pytest.raises(ValueError, match="moeme-layer-63-private-training"):
        kernel_metadata(
            "owner/moeme-layer63-training",
            "owner/moeme-layer63-private",
            "MoEMe Layer 63 Private Training",
        )


def test_kernel_source_locates_exactly_one_hash_bound_stage(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"manifest").hexdigest()
    source = kernel_source(digest)
    assert digest in source
    assert 'Path("/kaggle/input").rglob("upload-manifest.json")' in source
    assert "len(candidates) != 1" in source
    view = tmp_path / "kernel"
    metadata = kernel_metadata("owner/private-moeme-run", "owner/dataset", "Private MoEMe Run")
    result = build_kernel_view(view, metadata, source)
    assert result["verified"] is True
    assert result["file_count"] == 2
    assert build_kernel_view(view, metadata, source) == result


def test_kernel_push_command_requests_free_t4_pair(tmp_path: Path) -> None:
    assert kaggle_kernel_command(tmp_path) == [
        "kaggle",
        "kernels",
        "push",
        "--path",
        str(tmp_path.resolve()),
        "--accelerator",
        "NvidiaTeslaT4",
    ]


def test_execute_requires_exact_passing_private_upload_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "upload.json"
    digest = "a" * 64
    value = {
        "format": "moeme-cloud-upload-v1",
        "provider": "kaggle",
        "privacy": "private",
        "destination": "owner/dataset",
        "stage_manifest_sha256": digest,
        "status": "passed",
        "account_usage_checked": True,
    }
    receipt.write_text(json.dumps(value))
    assert verify_upload_receipt(receipt, "owner/dataset", digest)["verified"] is True
    value["privacy"] = "public"
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="receipt mismatch"):
        verify_upload_receipt(receipt, "owner/dataset", digest)


@pytest.mark.parametrize("status", ["running", "submitted", "passed"])
def test_duplicate_kernel_submission_is_refused(tmp_path: Path, status: str) -> None:
    receipt = tmp_path / "kernel.json"
    receipt.write_text(json.dumps({"kernel": "owner/kernel", "status": status}))
    with pytest.raises(RuntimeError, match="refusing to submit another version"):
        refuse_duplicate_launch(receipt, "owner/kernel")


def test_failed_or_other_kernel_receipt_does_not_block_explicit_launch(tmp_path: Path) -> None:
    receipt = tmp_path / "kernel.json"
    receipt.write_text(json.dumps({"kernel": "owner/kernel", "status": "failed"}))
    refuse_duplicate_launch(receipt, "owner/kernel")
    receipt.write_text(json.dumps({"kernel": "owner/other", "status": "submitted"}))
    refuse_duplicate_launch(receipt, "owner/kernel")
