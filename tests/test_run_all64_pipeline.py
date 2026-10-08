import json

from scripts.run_all64_pipeline import begin_phase, fail_phase, pass_phase


def test_phase_lifecycle_is_persisted_atomically(tmp_path) -> None:
    manifest = tmp_path / "pipeline.json"
    state = {"phases": {}}

    begin_phase(manifest, state, "training", {"layers": 64})
    running = json.loads(manifest.read_text())
    assert running["current_phase"] == "training"
    assert running["phases"]["training"]["status"] == "running"
    assert running["phases"]["training"]["attempt"] == 1
    assert running["phases"]["training"]["layers"] == 64

    pass_phase(manifest, state, "training", {"campaign": "campaign.json"})
    passed = json.loads(manifest.read_text())
    assert "current_phase" not in passed
    assert passed["phases"]["training"]["status"] == "passed"
    assert passed["phases"]["training"]["campaign"] == "campaign.json"
    assert "finished_at" in passed["phases"]["training"]


def test_resumed_phase_records_attempt_and_failure(tmp_path) -> None:
    manifest = tmp_path / "pipeline.json"
    state = {"phases": {}}

    begin_phase(manifest, state, "quantization")
    first_started_at = state["phases"]["quantization"]["started_at"]
    begin_phase(manifest, state, "quantization")
    assert state["phases"]["quantization"]["attempt"] == 2
    assert state["phases"]["quantization"]["started_at"] == first_started_at
    assert "resumed_at" in state["phases"]["quantization"]

    fail_phase(manifest, state, RuntimeError("quantizer exited 2"))
    failed = json.loads(manifest.read_text())
    assert failed["current_phase"] == "quantization"
    assert failed["phases"]["quantization"]["status"] == "failed"
    assert failed["phases"]["quantization"]["error"] == "RuntimeError: quantizer exited 2"
    assert failed["last_error"] == "RuntimeError: quantizer exited 2"

    pass_phase(manifest, state, "quantization", {"sha256": "recovered"})
    recovered = json.loads(manifest.read_text())
    assert recovered["phases"]["quantization"]["status"] == "passed"
    assert recovered["phases"]["quantization"]["sha256"] == "recovered"
    assert "error" not in recovered["phases"]["quantization"]
    assert "failed_at" not in recovered["phases"]["quantization"]
