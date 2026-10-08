import hashlib
import json
import signal
import struct
from pathlib import Path

import numpy as np
import pytest

from moeme.activations import MAGIC, VERSION
from scripts.run_postcapture_supervisor import (
    PostcaptureInterrupted,
    capture_is_running,
    imatrix_receipt_command,
    recover_completed_capture,
    run_step,
    validate_capture_ready,
)


def test_capture_running_requires_live_state_and_pid() -> None:
    assert capture_is_running({"ActiveState": "active", "MainPID": "123"}) is True
    assert capture_is_running({"ActiveState": "failed", "MainPID": "123"}) is False
    assert capture_is_running({"ActiveState": "active", "MainPID": "0"}) is False


def test_capture_readiness_fails_closed_on_wrong_contract(tmp_path: Path) -> None:
    directory = tmp_path / "cloud-inputs/layer63-200k-activations"
    directory.mkdir(parents=True)
    payload = directory / "layer-63.f32"
    payload.write_bytes(b"tiny")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "tokens_per_layer": 1,
                "layers": {"63": {"tokens": 1, "width": 5120, "bytes": 4, "sha256": digest}},
            }
        )
    )
    with pytest.raises(ValueError, match="shape/size contract mismatch"):
        validate_capture_ready(tmp_path)


def test_completed_temporary_capture_is_recovered_only_when_unambiguous(tmp_path: Path) -> None:
    output = tmp_path / "captures/final"
    partial = output.parent / ".final.partial"
    partial.mkdir(parents=True)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("capture corpus")
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    with (partial / "layer-63.f32").open("wb") as handle:
        handle.write(struct.pack("<III", MAGIC, VERSION, 4))
        handle.write(struct.pack("<I", 3))
        handle.write(values.tobytes())
    recovered = recover_completed_capture(
        tmp_path,
        output=output,
        corpus=corpus,
        expected_tokens=3,
        expected_width=4,
    )
    assert recovered is not None
    assert recovered["manifest"]["recovered_from_completed_temporary"] is True
    assert (output / "manifest.json").is_file()


def test_ambiguous_completed_temporary_captures_are_never_guessed(tmp_path: Path) -> None:
    output = tmp_path / "captures/final"
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("capture corpus")
    for suffix in ("one", "two"):
        partial = output.parent / f".final.{suffix}"
        partial.mkdir(parents=True)
        with (partial / "layer-63.f32").open("wb") as handle:
            handle.write(struct.pack("<III", MAGIC, VERSION, 1))
            handle.write(struct.pack("<I", 1))
            handle.write(struct.pack("<f", 1.0))
    with pytest.raises(RuntimeError, match="multiple complete"):
        recover_completed_capture(
            tmp_path,
            output=output,
            corpus=corpus,
            expected_tokens=1,
            expected_width=1,
        )


def test_imatrix_receipt_is_created_only_when_missing_then_verified(tmp_path: Path) -> None:
    command = imatrix_receipt_command(tmp_path, "/usr/bin/python3")
    assert "--verify" not in command
    receipt = tmp_path / ".moeme/layer63-200k-imatrix.receipt.json"
    receipt.parent.mkdir()
    receipt.write_text("{}")
    assert imatrix_receipt_command(tmp_path, "/usr/bin/python3")[-1] == "--verify"


@pytest.mark.parametrize("received", [signal.SIGINT, signal.SIGTERM])
def test_postcapture_forwards_interrupt_and_receipts_stage(
    tmp_path: Path, monkeypatch, received: signal.Signals
) -> None:
    handlers = {}
    forwarded = []

    def fake_signal(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    class FakeProcess:
        def poll(self):
            return None

        def send_signal(self, signum):
            forwarded.append(signum)

        def wait(self):
            handlers[received](received, None)
            return 128 + received

    monkeypatch.setattr("scripts.run_postcapture_supervisor.signal.signal", fake_signal)
    monkeypatch.setattr(
        "scripts.run_postcapture_supervisor.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    state = {"stages": {}}
    receipt = tmp_path / "receipt.json"
    with pytest.raises(PostcaptureInterrupted) as caught:
        run_step(tmp_path, state, receipt, "smoke", ["trainer"])
    assert caught.value.signum == received
    assert forwarded == [received]
    assert json.loads(receipt.read_text())["stages"]["smoke"]["status"] == "interrupted"


def test_passed_stage_is_reused_only_while_required_outputs_exist(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class FakeProcess:
        def poll(self):
            return 0

        def wait(self):
            calls.append(True)
            return 0

    monkeypatch.setattr(
        "scripts.run_postcapture_supervisor.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    state = {"stages": {"verify": {"status": "passed"}}}
    receipt = tmp_path / "receipt.json"
    output = Path("result.json")
    run_step(
        tmp_path,
        state,
        receipt,
        "verify",
        ["verifier"],
        required_outputs=(output,),
    )
    assert calls == [True]
    run_step(
        tmp_path,
        state,
        receipt,
        "verify",
        ["verifier"],
        required_outputs=(output,),
        revalidate_passed=True,
    )
    assert calls == [True, True]
    (tmp_path / output).write_text("{}")
    run_step(
        tmp_path,
        state,
        receipt,
        "verify",
        ["verifier"],
        required_outputs=(output,),
    )
    assert calls == [True, True]
