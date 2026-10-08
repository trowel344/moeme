import json
import os
import signal
from pathlib import Path

import pytest

from scripts.launch_free_cloud_job import (
    restore_resume,
    run_forwarding_signals,
    runtime_configuration,
)


def test_kaggle_runtime_config_keeps_inputs_read_only_and_redirects_output(tmp_path: Path) -> None:
    stage = tmp_path / "input/job"
    config_path = stage / "configs/free-cloud-layer63-200k.json"
    activation = stage / "cloud-inputs/activations"
    checkpoint = stage / "cloud-inputs/checkpoint"
    seed = stage / "cloud-inputs/seed/model.safetensors"
    for path in (activation, checkpoint, seed.parent):
        path.mkdir(parents=True, exist_ok=True)
    seed.write_bytes(b"seed")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "activations": "../cloud-inputs/activations",
                "checkpoint": "../cloud-inputs/checkpoint",
                "seed_checkpoint": "../cloud-inputs/seed/model.safetensors",
                "layer": 63,
                "output_dir": "../cloud-results/layer63-200k",
            }
        )
    )
    workdir = tmp_path / "working/moeme"
    config, runtime_path = runtime_configuration(stage, workdir, config_path)
    assert config["activations"] == str(activation.resolve())
    assert config["checkpoint"] == str(checkpoint.resolve())
    assert config["seed_checkpoint"] == str(seed.resolve())
    assert config["output_dir"] == str(workdir.resolve() / "cloud-results/layer63-200k")
    assert runtime_path == workdir.resolve() / ".moeme-runtime/free-cloud-layer63-200k.json"


def test_resume_restore_refuses_to_overlay_existing_state(tmp_path: Path) -> None:
    source = tmp_path / "prior"
    output = tmp_path / "working/results"
    source.mkdir()
    (source / "training-state.pt").write_bytes(b"state")
    restore_resume(source, output)
    assert (output / "training-state.pt").read_bytes() == b"state"
    with pytest.raises(FileExistsError, match="refusing to overlay"):
        restore_resume(source, output)


@pytest.mark.parametrize("received", [signal.SIGINT, signal.SIGTERM])
def test_launcher_forwards_provider_and_user_signals(
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

        def wait(self, timeout=None):
            del timeout
            handlers[received](received, None)
            return 128 + received

    monkeypatch.setattr("scripts.launch_free_cloud_job.signal.signal", fake_signal)
    monkeypatch.setattr(
        "scripts.launch_free_cloud_job.subprocess.Popen", lambda *args, **kwargs: FakeProcess()
    )
    returncode = run_forwarding_signals(["trainer"], cwd=tmp_path, environment=os.environ.copy())
    assert forwarded == [received]
    assert returncode == 128 + received
