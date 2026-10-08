import signal
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.capture_activations import (
    ChildSignalForwarder,
    terminate_process,
    wait_for_capture_start,
)


def test_capture_start_guard_observes_first_complete_row(tmp_path: Path) -> None:
    capture = tmp_path / "layer-63.f32"
    child_code = (
        "import pathlib,time; time.sleep(.05); "
        f"pathlib.Path({str(capture)!r}).write_bytes(b'x' * 16)"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_code,
        ]
    )
    wait_for_capture_start(process, [capture], timeout_seconds=2, poll_seconds=0.01)
    assert process.wait(timeout=2) == 0


def test_capture_start_guard_times_out_and_process_can_be_terminated(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with pytest.raises(TimeoutError, match="wrote no activation rows"):
        wait_for_capture_start(
            process,
            [tmp_path / "layer-63.f32"],
            timeout_seconds=0.05,
            poll_seconds=0.01,
        )
    terminate_process(process)
    assert process.poll() is not None


@pytest.mark.parametrize("received", [signal.SIGINT, signal.SIGTERM])
def test_capture_signal_forwarder_restores_handlers(monkeypatch, received: signal.Signals) -> None:
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

    monkeypatch.setattr("scripts.capture_activations.signal.signal", fake_signal)
    process = FakeProcess()
    with ChildSignalForwarder(process) as forwarder:
        handlers[received](received, None)
        assert forwarder.signum == received
    assert forwarded == [received]
    assert handlers[signal.SIGINT] == signal.SIG_DFL
    assert handlers[signal.SIGTERM] == signal.SIG_DFL
