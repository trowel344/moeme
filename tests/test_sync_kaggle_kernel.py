import json
from pathlib import Path

import pytest

from moeme.cloud_results import build_result_manifest
from scripts.sync_kaggle_kernel import (
    load_launch_receipt,
    output_command,
    parse_kernel_status,
    promote_download,
    should_collect,
    status_command,
)


def make_result(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "cloud-preflight.json").write_text('{"format":"preflight"}')
    (path / "cloud-run.json").write_text('{"status":"interrupted"}')
    (path / "progress.json").write_text('{"phase":"training"}')
    (path / "cloud-result-manifest.json").write_text(json.dumps(build_result_manifest(path, 63)))


def test_status_parser_is_strict() -> None:
    assert parse_kernel_status('owner/kernel has status "KernelWorkerStatus.RUNNING"') == "RUNNING"
    with pytest.raises(ValueError, match="found 0"):
        parse_kernel_status("unstructured success")
    with pytest.raises(ValueError, match="found 2"):
        parse_kernel_status("KernelWorkerStatus.RUNNING KernelWorkerStatus.COMPLETE")


def test_collection_is_opt_in_per_terminal_class() -> None:
    # a COMPLETE run is only downloaded when --sync is requested
    assert should_collect("COMPLETE", sync=True, collect_failed=False) is True
    assert should_collect("COMPLETE", sync=False, collect_failed=True) is False
    # a terminated run may still hold an early-stopped result tree
    assert should_collect("ERROR", sync=True, collect_failed=True) is True
    assert should_collect("ERROR", sync=True, collect_failed=False) is False
    assert should_collect("CANCEL_ACKNOWLEDGED", sync=False, collect_failed=True) is True
    # non-terminal statuses never download
    assert should_collect("RUNNING", sync=True, collect_failed=True) is False
    assert should_collect("QUEUED", sync=True, collect_failed=True) is False


def test_commands_are_scoped_to_one_receipted_kernel(tmp_path: Path) -> None:
    assert status_command("owner/kernel") == ["kaggle", "kernels", "status", "owner/kernel"]
    assert output_command("owner/kernel", tmp_path) == [
        "kaggle",
        "kernels",
        "output",
        "owner/kernel",
        "--path",
        str(tmp_path.resolve()),
        "--quiet",
    ]


def test_launch_receipt_must_be_private_kaggle(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "format": "moeme-kaggle-kernel-v1",
                "provider": "kaggle",
                "privacy": "private",
                "kernel": "owner/kernel",
            }
        )
    )
    assert load_launch_receipt(path)["kernel"] == "owner/kernel"
    value = json.loads(path.read_text())
    value["privacy"] = "public"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="not private"):
        load_launch_receipt(path)


def test_collect_failed_imports_the_result_left_by_an_errored_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exit-3 early stop surfaces as Kaggle ERROR but still publishes a result."""
    import sys as _sys

    import scripts.sync_kaggle_kernel as sync

    src = Path(__file__).resolve().parents[1] / "src"
    fake = tmp_path / "kaggle"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        "from moeme.cloud_results import build_result_manifest\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['kernels', 'status']:\n"
        "    print('owner/kernel has status \"KernelWorkerStatus.ERROR\"')\n"
        "    raise SystemExit(0)\n"
        "if args[:2] == ['kernels', 'output']:\n"
        "    root = Path(args[args.index('--path') + 1]) / 'moeme/cloud-results/layer63-200k'\n"
        "    root.mkdir(parents=True)\n"
        "    (root / 'cloud-preflight.json').write_text('{\"format\":\"preflight\"}')\n"
        "    (root / 'cloud-run.json').write_text('{\"status\":\"early_stopped\"}')\n"
        "    (root / 'progress.json').write_text('{\"phase\":\"early_stopped\"}')\n"
        "    manifest = build_result_manifest(root, 63)\n"
        "    (root / 'cloud-result-manifest.json').write_text(json.dumps(manifest))\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(sync.shutil, "which", lambda name: str(fake))

    receipt = tmp_path / "kernel.json"
    receipt.write_text(
        json.dumps(
            {
                "format": "moeme-kaggle-kernel-v1",
                "provider": "kaggle",
                "privacy": "private",
                "kernel": "owner/kernel",
                "status": "failed",
            }
        )
    )
    target = tmp_path / "cloud-results/layer63-200k"
    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "sync_kaggle_kernel.py",
            "--receipt",
            str(receipt),
            "--inbox",
            str(tmp_path / "cloud-inbox/layer63-200k"),
            "--target",
            str(target),
            "--transfer-receipt",
            str(tmp_path / "transfer.json"),
            "--collect-failed",
        ],
    )
    # a terminated run still reports failure, but its result is now secured
    assert sync.main() == 2
    assert target.is_dir()
    saved = json.loads(receipt.read_text())
    assert saved["remote_status"] == "ERROR"
    assert saved["status"] == "imported_after_failure"
    assert saved["transfer"]["run_status"] == "early_stopped"


def test_watch_holds_through_running_then_collects_on_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--watch keeps polling an in-progress kernel and imports once it completes."""
    import sys as _sys

    import scripts.sync_kaggle_kernel as sync

    src = Path(__file__).resolve().parents[1] / "src"
    counter = tmp_path / "polls.txt"
    fake = tmp_path / "kaggle"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        "from moeme.cloud_results import build_result_manifest\n"
        "args = sys.argv[1:]\n"
        f"counter = Path({str(counter)!r})\n"
        "seen = int(counter.read_text()) if counter.exists() else 0\n"
        "if args[:2] == ['kernels', 'status']:\n"
        "    counter.write_text(str(seen + 1))\n"
        "    status = ['RUNNING', 'RUNNING', 'COMPLETE'][min(seen, 2)]\n"
        "    print(f'owner/kernel has status \"KernelWorkerStatus.{status}\"')\n"
        "    raise SystemExit(0)\n"
        "if args[:2] == ['kernels', 'output']:\n"
        "    root = Path(args[args.index('--path') + 1]) / 'moeme/cloud-results/layer63-200k'\n"
        "    root.mkdir(parents=True)\n"
        "    (root / 'cloud-preflight.json').write_text('{\"format\":\"preflight\"}')\n"
        "    (root / 'cloud-run.json').write_text('{\"status\":\"passed\"}')\n"
        "    (root / 'progress.json').write_text('{\"phase\":\"complete\"}')\n"
        "    (root / 'curve.json').write_text('{\"format\":\"moeme-sparse-learning-curve-v1\"}')\n"
        "    (root / 'curve-analysis.json').write_text('{\"status\":\"target_reached\"}')\n"
        "    (root / 'report.json').write_text('{\"passed\":true}')\n"
        "    (root / 'layer-63-top4.safetensors').write_bytes(b'checkpoint')\n"
        "    manifest = build_result_manifest(root, 63)\n"
        "    (root / 'cloud-result-manifest.json').write_text(json.dumps(manifest))\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(sync.shutil, "which", lambda name: str(fake))

    receipt = tmp_path / "kernel.json"
    receipt.write_text(
        json.dumps(
            {
                "format": "moeme-kaggle-kernel-v1",
                "provider": "kaggle",
                "privacy": "private",
                "kernel": "owner/kernel",
                "status": "running",
            }
        )
    )
    target = tmp_path / "cloud-results/layer63-200k"
    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "sync_kaggle_kernel.py",
            "--receipt",
            str(receipt),
            "--inbox",
            str(tmp_path / "cloud-inbox/layer63-200k"),
            "--target",
            str(target),
            "--transfer-receipt",
            str(tmp_path / "transfer.json"),
            "--watch",
            "--interval",
            "0",
            "--sync",
        ],
    )
    assert sync.main() == 0
    # it polled past the two RUNNING states before collecting
    assert counter.read_text() == "3"
    assert target.is_dir()
    saved = json.loads(receipt.read_text())
    assert saved["status"] == "imported"
    assert saved["transfer"]["run_status"] == "passed"


def test_download_is_verified_before_atomic_inbox_publish(tmp_path: Path) -> None:
    download = tmp_path / "download/moeme/cloud-results/layer63-200k"
    make_result(download)
    inbox = tmp_path / "cloud-inbox/layer63-200k"
    result = promote_download(tmp_path / "download", inbox)
    assert result["run_status"] == "interrupted"
    assert result["already_present"] is False
    assert inbox.is_dir()
    assert not download.exists()
    again = promote_download(inbox, inbox)
    assert again["already_present"] is True
