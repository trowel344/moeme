import json
from pathlib import Path

from scripts.moeme_status import (
    active_progress,
    atomic_text,
    capture_progress_estimate,
    compact_summary,
    descendants,
    duration_estimate,
    parse_compute_apps,
    parse_service_show,
    summarize_command,
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_active_progress_ignores_completed_layers(tmp_path: Path) -> None:
    write(tmp_path / "layer-0/progress.json", {"layer": 0, "phase": "complete"})
    write(
        tmp_path / "layer-1/progress.json",
        {"layer": 1, "phase": "training", "step": 700, "steps": 2000},
    )
    assert active_progress(tmp_path) == {
        "layer": 1,
        "phase": "training",
        "step": 700,
        "steps": 2000,
        "loss": None,
        "reconstruction_loss": None,
        "routing_loss": None,
        "updated_at": None,
        "receipt": str(tmp_path / "layer-1/progress.json"),
    }


def test_duration_estimate_uses_completed_layer_cadence() -> None:
    campaign = {
        "started_at": "2026-01-01T00:00:00+00:00",
        "layers": {
            "0": {"finished_at": "2026-01-01T00:05:00+00:00"},
            "1": {"finished_at": "2026-01-01T00:11:00+00:00"},
        },
    }
    assert duration_estimate(campaign) == {
        "median_layer_seconds": 330.0,
        "training_seconds_remaining": 20460,
    }


def test_service_show_requires_active_pid() -> None:
    assert (
        parse_service_show("ActiveState=active\nSubState=running\nMainPID=123\nResult=success\n")[
            "active"
        ]
        is True
    )
    assert (
        parse_service_show("ActiveState=active\nSubState=exited\nMainPID=0\nResult=success\n")[
            "active"
        ]
        is False
    )
    activating = parse_service_show(
        "ActiveState=activating\nSubState=start\nMainPID=456\nResult=success\n"
    )
    assert activating["active"] is False
    assert activating["running"] is True


def test_capture_progress_estimate_uses_service_monotonic_start() -> None:
    assert capture_progress_estimate(
        [{"bytes": 25, "completion_fraction": 0.25}],
        {"running": True, "started_monotonic_usec": 100_000_000},
        200.0,
    ) == {
        "completion_fraction": 0.25,
        "elapsed_seconds": 100,
        "estimated_total_seconds": 400,
        "estimated_remaining_seconds": 300,
        "method": "observed-bytes-linear",
    }


def test_atomic_text_replaces_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "STATUS.json"
    snapshot.write_text("old")
    atomic_text(snapshot, "new\n")
    assert snapshot.read_text() == "new\n"
    assert list(tmp_path.iterdir()) == [snapshot]


def test_descendants_follows_entire_pipeline_tree() -> None:
    processes = {
        10: {"pid": 10, "ppid": 1},
        11: {"pid": 11, "ppid": 10},
        12: {"pid": 12, "ppid": 11},
        20: {"pid": 20, "ppid": 1},
    }
    assert [value["pid"] for value in descendants(processes, 10)] == [11, 12]


def test_compute_app_parser_skips_transient_bad_rows() -> None:
    assert parse_compute_apps("123, 5776\nN/A, N/A\n") == [{"pid": 123, "used_memory_mib": 5776}]


def test_command_summary_keeps_only_worker_identity() -> None:
    command = (
        "/usr/bin/python3 scripts/train_sparse_layer.py --activations huge/path "
        "--layer 4 --output-dir artifacts/layer-4 --steps 2000 --device cuda"
    )
    assert summarize_command(command) == (
        "/usr/bin/python3 scripts/train_sparse_layer.py --layer 4 "
        "--output-dir artifacts/layer-4 --device cuda"
    )


def test_compact_summary_excludes_process_inventory() -> None:
    full = {
        "runtime": {"processes": [{"pid": value} for value in range(1000)]},
        "free_cloud": {
            "activation_capture": {
                "service": {"running": True},
                "ready": False,
                "progress": {"completion_fraction": 0.5},
                "partial_files": [{"bytes": 10}],
                "manifest": None,
                "imatrix_receipt": None,
            },
            "postcapture_supervisor": {"status": "waiting"},
            "upload_stage": {"ready": False},
            "upload_transfer": {"status": "verified_dry_run"},
            "kaggle_kernel": {"status": "queued"},
            "free_tier_budget": {"passed": True, "recommended_provider": "lightning"},
            "local_smoke": {},
            "training": {"curve_points": 2},
            "acceptance": {"current_phase": "quantization", "phases": {}},
        },
        "storage": {"free_bytes": 100},
        "artifacts": {"final_q4": {"exists": True}},
    }
    compact = compact_summary(full)
    assert compact["capture"]["progress"]["completion_fraction"] == 0.5
    assert compact["acceptance"]["current_phase"] == "quantization"
    assert compact["training"]["curve_points"] == 2
    assert compact["free_tier_budget"]["recommended_provider"] == "lightning"
    assert compact["upload_transfer"]["status"] == "verified_dry_run"
    assert compact["kaggle_kernel"]["status"] == "queued"
    assert "runtime" not in compact
