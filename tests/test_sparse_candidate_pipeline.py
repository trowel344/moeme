import hashlib
import json
from pathlib import Path

from scripts.run_sparse_candidate_pipeline import (
    completed_phase_valid,
    configuration_digest,
    training_eligibility,
)


def test_training_eligibility_requires_cloud_curve_report_and_layer() -> None:
    eligible = training_eligibility(
        {"status": "passed"},
        {
            "status": "promising",
            "baseline_error": 0.1064,
            "target_error": 0.01,
            "last": {"relative_l2": 0.07},
            "best": {"relative_l2": 0.06},
        },
        {
            "passed": True,
            "layer": 63,
            "stages": [{"validation": {"relative_l2": 0.075}}],
        },
        63,
    )
    assert eligible["passed"] is True
    rejected = training_eligibility(
        {"status": "passed"},
        {"status": "weak_or_uncertain"},
        {"passed": True, "layer": 63, "stages": []},
        63,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["curve_supports_escalation"] is False


def test_configuration_digest_is_order_independent() -> None:
    assert configuration_digest({"layer": 63, "type": "Q5"}) == configuration_digest(
        {"type": "Q5", "layer": 63}
    )


def test_completed_candidate_phases_require_matching_artifacts(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.gguf"
    server = tmp_path / "server.json"
    parity = tmp_path / "parity.json"
    candidate.write_bytes(b"candidate")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    server.write_text(json.dumps({"passed": True}))
    parity.write_text(json.dumps({"passed": True, "candidate_sha256": digest}))
    state = {
        "phases": {
            "quantization": {
                "status": "passed",
                "bytes": candidate.stat().st_size,
                "sha256": digest,
            },
            "evaluation": {"status": "passed", "candidate_sha256": digest},
            "quantized_parity": {"status": "passed"},
        }
    }
    paths = {"candidate": candidate, "server_report": server, "parity_report": parity}
    assert completed_phase_valid(state, "quantization", **paths) == (True, None)
    assert completed_phase_valid(state, "evaluation", **paths) == (True, None)
    assert completed_phase_valid(state, "quantized_parity", **paths) == (True, None)

    candidate.write_bytes(b"tampered")
    valid, reason = completed_phase_valid(state, "quantization", **paths)
    assert valid is False
    assert "size changed" in reason
    candidate.write_bytes(b"candidate")
    server.unlink()
    assert completed_phase_valid(state, "evaluation", **paths)[0] is False
    parity.write_text(json.dumps({"passed": True, "candidate_sha256": "wrong"}))
    valid, reason = completed_phase_valid(state, "quantized_parity", **paths)
    assert valid is False
    assert "different candidate" in reason
