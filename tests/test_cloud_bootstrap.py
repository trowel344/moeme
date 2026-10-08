import importlib.metadata
import os
import signal
import sys
from pathlib import Path

import pytest

from scripts.cloud_bootstrap import (
    configured_output,
    launcher_command,
    missing_requirements,
    numeric_version,
    run_step,
    staged_configuration,
)


def test_numeric_version_ignores_non_numeric_suffixes() -> None:
    assert numeric_version("2.1.0+cu130")[:3] == (2, 1, 0)
    assert numeric_version("0.6.2") == (0, 6, 2)


def test_missing_requirements_installs_only_old_or_absent_packages(monkeypatch) -> None:
    versions = {"numpy": "2.1.0"}

    def fake_version(name: str) -> str:
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr("scripts.cloud_bootstrap.importlib.metadata.version", fake_version)
    assert missing_requirements() == ["safetensors>=0.6"]


def test_launcher_command_is_one_training_call_with_optional_resume(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    config = stage / "configs/job.json"
    workdir = tmp_path / "work"
    resume = tmp_path / "prior"
    command = launcher_command(stage, config, "kaggle", workdir, resume)
    assert command == [
        sys.executable,
        str(stage / "scripts/launch_free_cloud_job.py"),
        "--stage",
        str(stage),
        "--config",
        str(config),
        "--provider",
        "kaggle",
        "--workdir",
        str(workdir),
        "--resume-from",
        str(resume.resolve()),
    ]
    assert "--preflight-only" not in command
    assert "--resume-terminal" not in command
    assert launcher_command(stage, config, "kaggle", workdir, None, True)[-1] == (
        "--resume-terminal"
    )


def test_manifest_selects_generic_staged_configuration_and_output(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    config = stage / "configs/free-cloud-layer12-1m.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"output_dir":"../cloud-results/layer12-1m","layer":12}')
    (stage / "upload-manifest.json").write_text(
        '{"configuration":"configs/free-cloud-layer12-1m.json"}'
    )
    selected = staged_configuration(stage, None)
    assert selected == config.resolve()
    assert configured_output(stage, tmp_path / "work", selected) == (
        tmp_path / "work/cloud-results/layer12-1m"
    )


@pytest.mark.parametrize("received", [signal.SIGINT, signal.SIGTERM])
def test_run_step_forwards_termination_signal_and_records_it(
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
            return -received

    monkeypatch.setattr("scripts.cloud_bootstrap.signal.signal", fake_signal)
    monkeypatch.setattr(
        "scripts.cloud_bootstrap.subprocess.Popen", lambda *args, **kwargs: FakeProcess()
    )
    result = run_step(
        "training",
        ["trainer"],
        cwd=tmp_path,
        environment=os.environ.copy(),
        log=tmp_path / "bootstrap.log",
    )
    assert forwarded == [received]
    assert result["signal"] == received
    assert result["returncode"] == -received
