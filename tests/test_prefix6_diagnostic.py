import json
import os
from datetime import UTC, datetime

from scripts.run_prefix6_diagnostic import (
    checkpoint_ready,
    main_pipeline_command,
    recoverable_evaluation_report,
)


def test_checkpoint_boundary_requires_all_atomic_outputs(tmp_path) -> None:
    layer = tmp_path / "layer-5"
    layer.mkdir()
    (layer / "progress.json").write_text(json.dumps({"phase": "complete"}))
    assert checkpoint_ready(tmp_path, 5) is False
    (layer / "report.json").write_text("{}")
    (layer / "layer-5-top4.safetensors").write_bytes(b"checkpoint")
    assert checkpoint_ready(tmp_path, 5) is True


def test_resume_command_is_bound_to_v3_and_8k(tmp_path) -> None:
    command = main_pipeline_command(tmp_path)
    assert command[command.index("--manifest") + 1] == ".moeme/all64-pipeline-v3.json"
    assert command[command.index("--training-dir") + 1].endswith("all64-v2")
    assert command[command.index("--server-context") + 1] == "8192"


def test_valid_report_can_recover_post_report_ledger_failure(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "passed": True,
                "quality_passed": True,
                "stability_passed": True,
                "performance_passed": True,
            }
        )
    )
    timestamp = report.stat().st_mtime - 1
    state = {
        "phases": {
            "evaluation": {
                "status": "failed",
                "started_at": datetime.fromtimestamp(timestamp, UTC).isoformat(),
            }
        }
    }
    assert recoverable_evaluation_report(state, report)["passed"] is True
    os.utime(report, (timestamp - 2, timestamp - 2))
    assert recoverable_evaluation_report(state, report) is None
