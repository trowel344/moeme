import json
from pathlib import Path

import pytest

from moeme.ledger import ExperimentLedger


def test_ledger_records_attempts_and_localizes_first_bad_gate(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "experiments.sqlite3")
    first = ledger.start("partition", "a", {"top_k": 12})
    ledger.metric(first, "text", "max_error", 1e-12, "absolute", gate="<=1e-10", passed=True)
    ledger.finish(first, "passed", {"valid": True})

    second = ledger.start("top-k-anneal", "b", {"top_k": 10}, parent_run_id=first)
    ledger.metric(
        second,
        "ocr",
        "accuracy_delta",
        -0.08,
        "fraction",
        gate=">=-0.01",
        passed=False,
    )
    ledger.finish(second, "failed", {"reason": "OCR regression"})

    diagnosis = ledger.diagnosis()
    assert diagnosis["first_failed_gate"]["stage"] == "top-k-anneal"
    assert diagnosis["first_failed_gate"]["capability"] == "ocr"
    assert diagnosis["last_passing_predecessor"]["stage"] == "partition"
    assert diagnosis["first_failed_run"]["error"] is None
    assert [record.attempt for record in ledger.history()] == [1, 1]


def test_diagnosis_ignores_failures_resolved_by_newer_stage_attempt(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "experiments.sqlite3")
    failed = ledger.start("convert", "old", {})
    ledger.metric(failed, "format", "valid", 0, "bool", gate="==1", passed=False)
    ledger.finish(failed, "failed", {}, error="truncated")

    fixed = ledger.start("convert", "new", {})
    ledger.metric(fixed, "format", "valid", 1, "bool", gate="==1", passed=True)
    ledger.finish(fixed, "passed", {"valid": True})

    current = ledger.diagnosis()
    assert current["all_clear"] is True
    assert current["first_failed_gate"] is None
    assert current["first_failed_run"] is None
    assert current["historical_first_failed_gate"]["id"] == failed
    assert current["historical_first_failed_run"]["id"] == failed


def test_diagnosis_keeps_latest_unresolved_failure(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "experiments.sqlite3")
    passed = ledger.start("convert", "old", {})
    ledger.finish(passed, "passed", {})
    failed = ledger.start("convert", "new", {})
    ledger.finish(failed, "failed", {}, error="regressed")

    current = ledger.diagnosis()
    assert current["all_clear"] is False
    assert current["first_failed_run"]["id"] == failed
    assert current["last_passing_predecessor"]["id"] == passed


def test_context_manager_records_exception(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "experiments.sqlite3")
    with pytest.raises(RuntimeError, match="boom"), ledger.run("broken", "digest", {}):
        raise RuntimeError("boom")
    assert ledger.history()[0].status == "failed"
    assert ledger.history()[0].error == "RuntimeError: boom"


def test_terminal_run_cannot_be_finished_twice(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "experiments.sqlite3")
    run_id = ledger.start("stage", "digest", {})
    ledger.finish(run_id, "passed", {})
    with pytest.raises(ValueError, match="already finished"):
        ledger.finish(run_id, "passed", {})


def test_phase_import_is_idempotent(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "experiments.sqlite3")
    first = ledger.import_phase("architecture", "abc", "passed", {"layers": 64})
    second = ledger.import_phase("architecture", "abc", "passed", {"layers": 64})
    assert first == second
    assert len(ledger.history()) == 1


def test_stage_transitions_refresh_compact_handoff(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "experiments.sqlite3")
    run_id = ledger.start("quantize", "digest", {"type": "Q4_K_M"})
    summary_path = tmp_path / "ledger-summary.json"
    running = json.loads(summary_path.read_text())
    assert running["latest_by_stage"]["quantize"]["status"] == "running"

    ledger.finish(run_id, "passed", {"bytes": 123, "nested": {"omitted": True}})
    passed = json.loads(summary_path.read_text())
    compact = passed["latest_by_stage"]["quantize"]
    assert compact["status"] == "passed"
    assert compact["summary"] == {"bytes": 123}


def test_concurrent_starters_get_distinct_attempts(tmp_path: Path) -> None:
    """Concurrent campaigns must not collide on UNIQUE(stage, attempt)."""
    import threading

    path = tmp_path / "ledger.sqlite3"
    ExperimentLedger(path)  # initialize schema once, single-threaded
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        ledger = ExperimentLedger(path)
        for _ in range(5):
            try:
                ledger.start("sparse-layer-distillation", "digest", {})
            except BaseException as error:  # noqa: BLE001 - reported to the test
                with lock:
                    errors.append(error)
                return

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    records = ExperimentLedger(path).history(limit=100)
    attempts = sorted(
        record.attempt for record in records if record.stage == "sparse-layer-distillation"
    )
    assert attempts == list(range(1, len(attempts) + 1))
