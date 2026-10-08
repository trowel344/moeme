import hashlib
import json
from pathlib import Path

import pytest

from scripts.assemble_release_manifest import assemble


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def receipts(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"qualified model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    parity = tmp_path / "parity.json"
    server = tmp_path / "server.json"
    write(parity, {"passed": True, "candidate_sha256": digest, "results": [{}, {}, {}]})
    write(
        server,
        {
            "passed": True,
            "quality_passed": True,
            "stability_passed": True,
            "performance_passed": True,
            "cases_passed": 16,
            "cases_total": 16,
        },
    )
    return model, parity, server


def test_assemble_records_bound_quantization_and_evaluation(tmp_path: Path) -> None:
    model, parity, server = receipts(tmp_path)
    state = assemble(model, parity, server)
    quantization = state["phases"]["quantization"]
    assert quantization["status"] == "passed"
    assert quantization["sha256"] == hashlib.sha256(b"qualified model").hexdigest()
    evaluation = state["phases"]["evaluation"]
    assert evaluation["status"] == "passed"
    assert evaluation["capability_cases"] == 16
    assert evaluation["parity_corpora"] == 3


def test_assemble_rejects_failed_capability_receipt(tmp_path: Path) -> None:
    model, parity, server = receipts(tmp_path)
    write(server, {"passed": True, "quality_passed": True, "stability_passed": False})
    with pytest.raises(ValueError, match="capability report"):
        assemble(model, parity, server)


def test_assemble_rejects_stale_parity_receipt(tmp_path: Path) -> None:
    model, parity, server = receipts(tmp_path)
    write(parity, {"passed": True, "candidate_sha256": "old", "results": []})
    with pytest.raises(ValueError, match="not bound"):
        assemble(model, parity, server)


def test_assemble_rejects_failed_parity_receipt(tmp_path: Path) -> None:
    model, parity, server = receipts(tmp_path)
    write(parity, {"passed": False})
    with pytest.raises(ValueError, match="parity report"):
        assemble(model, parity, server)
