from scripts.start_sparse_candidate_service import start_decision


def test_candidate_starter_waits_for_verified_transfer() -> None:
    assert start_decision(None, None, None, {"ActiveState": "inactive"}) == {
        "status": "not_ready",
        "start": False,
        "reason": "no transfer receipt",
    }


def test_candidate_starter_starts_exactly_once_when_eligible() -> None:
    decision = start_decision(
        {"candidate_ready": True},
        None,
        None,
        {"ActiveState": "inactive", "MainPID": "0"},
    )
    assert decision == {"status": "ready", "start": True}
    running = start_decision(
        {"candidate_ready": True},
        None,
        None,
        {"ActiveState": "active", "MainPID": "123"},
    )
    assert running == {"status": "running", "start": False}


def test_candidate_starter_refuses_failed_or_ineligible_attempt() -> None:
    refused = start_decision(
        {"candidate_ready": True},
        {"status": "failed"},
        None,
        {"ActiveState": "failed", "MainPID": "0"},
    )
    assert refused["status"] == "refused"
    assert "inspect before resume" in refused["reason"]

    ineligible = start_decision(
        {"candidate_ready": False},
        None,
        None,
        {"ActiveState": "inactive", "MainPID": "0"},
    )
    assert ineligible["status"] == "refused"


def test_candidate_starter_resumes_content_bound_interruption() -> None:
    assert start_decision(
        {"candidate_ready": True},
        {"status": "interrupted"},
        {"passed": False, "phases": {"quantization": {"status": "failed"}}},
        {"ActiveState": "inactive", "MainPID": "0"},
    ) == {"status": "ready", "start": True}


def test_candidate_starter_refuses_orphaned_failed_phase() -> None:
    decision = start_decision(
        {"candidate_ready": True},
        {"status": "running"},
        {"passed": False, "phases": {"evaluation": {"status": "failed"}}},
        {"ActiveState": "inactive", "MainPID": "0"},
    )
    assert decision["status"] == "refused"
    assert "evaluation" in decision["reason"]


def test_candidate_starter_recognizes_completed_acceptance() -> None:
    assert start_decision(
        {"candidate_ready": True},
        {"status": "passed"},
        {"passed": True},
        {"ActiveState": "inactive", "MainPID": "0"},
    ) == {"status": "complete", "start": False}
