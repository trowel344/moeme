import json
import os
import signal
import sys
import threading
from pathlib import Path

from scripts.run_free_cloud_training import (
    acquire_run_lock,
    existing_ancestor,
    existing_run_action,
    load_config,
    next_attempt,
    preflight_receipt_path,
    provider_name,
    run_and_tee,
    selected_compute_dtype,
    terminal_status,
    trainer_command,
    verify_receipt,
)


def test_load_config_requires_core_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"activations": "a"}))
    try:
        load_config(path)
    except ValueError as error:
        assert "checkpoint" in str(error)
        assert "output_dir" in str(error)
    else:
        raise AssertionError("invalid config was accepted")


def test_preflight_dtype_selection_honors_explicit_request() -> None:
    assert selected_compute_dtype("auto", True) == "bfloat16"
    assert selected_compute_dtype("auto", False) == "float16"
    assert selected_compute_dtype("float16", True) == "float16"
    try:
        selected_compute_dtype("bfloat16", False)
    except RuntimeError as error:
        assert "does not support" in str(error)
    else:
        raise AssertionError("unsupported explicit bfloat16 was accepted")


def test_trainer_command_auto_resumes_atomic_state(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    state = output / "training-state.pt"
    state.write_bytes(b"state")
    config = {
        "early_stop_flat": True,
        "layer": 63,
        "top_k_schedule": [4],
        "train_shared": True,
    }
    paths = {
        "activations": tmp_path / "activations",
        "checkpoint": tmp_path / "checkpoint",
        "output_dir": output,
    }
    command = trainer_command(config, paths, tmp_path)
    assert command[-2:] == ["--resume-state", str(state)]
    assert "--train-shared" in command
    assert "--systems-smoke" not in command
    assert command[command.index("--checkpoint-every") + 1] == "0"
    assert command[command.index("--state-every") + 1] == "500"
    assert command[command.index("--ledger") + 1] == str(output / "experiments.sqlite3")
    assert command[command.index("--early-stop-min-step") + 1] == "15000"
    assert command[command.index("--validation-fraction") + 1] == "0.25"
    assert provider_name() in {"kaggle", "lightning", "local-or-unknown"}
    assert existing_ancestor(tmp_path / "not-created" / "output") == tmp_path


def test_trainer_command_marks_explicit_systems_smoke(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    config = {
        "layer": 63,
        "systems_smoke": True,
        "top_k_schedule": [4],
    }
    paths = {
        "activations": tmp_path / "activations",
        "checkpoint": tmp_path / "checkpoint",
        "output_dir": output,
    }
    assert "--systems-smoke" in trainer_command(config, paths, tmp_path)


def test_trainer_command_passes_clean_cloud_session_budget(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    config = {
        "layer": 63,
        "max_runtime_seconds": 36000,
        "top_k_schedule": [4],
    }
    paths = {
        "activations": tmp_path / "activations",
        "checkpoint": tmp_path / "checkpoint",
        "output_dir": output,
    }
    command = trainer_command(config, paths, tmp_path)
    assert command[command.index("--max-runtime-seconds") + 1] == "36000"


def test_trainer_command_uses_seed_only_before_exact_state_exists(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    seed = tmp_path / "seed/model.safetensors"
    config = {"layer": 63, "top_k_schedule": [4], "train_shared": True}
    paths = {
        "activations": tmp_path / "activations",
        "checkpoint": tmp_path / "checkpoint",
        "seed_checkpoint": seed,
        "output_dir": output,
    }
    initial = trainer_command(config, paths, tmp_path)
    assert initial[-5:] == [
        "--partition-indices-from",
        str(seed),
        "--resume",
        str(seed),
        "--allow-partial-resume",
    ]
    (output / "training-state.pt").write_bytes(b"state")
    resumed = trainer_command(config, paths, tmp_path)
    assert "--allow-partial-resume" not in resumed
    assert resumed[-2:] == ["--resume-state", str(output / "training-state.pt")]


def test_verify_receipt_checks_size_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"moeme")
    receipt = {
        "bytes": 5,
        "sha256": "3cf0bbd36373c3f538eaa2aa46a1e283b6b9f2367806dc7d0291c9ae5106bca9",
    }
    verified = verify_receipt(path, receipt, "payload", True)
    assert verified["hash_verified"] is True
    assert verified["sha256"] == receipt["sha256"]


def test_next_attempt_survives_missing_or_truncated_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "cloud-run.json"
    assert next_attempt(receipt) == 1
    receipt.write_text("{")
    assert next_attempt(receipt) == 1
    receipt.write_text(json.dumps({"attempt": 3, "status": "interrupted"}))
    assert next_attempt(receipt) == 4


def test_existing_terminal_run_requires_explicit_resume() -> None:
    assert existing_run_action(None, False) == "run"
    assert existing_run_action({"status": "interrupted"}, False) == "run"
    assert existing_run_action({"status": "running"}, False) == "run"
    assert existing_run_action({"status": "passed"}, False) == "complete"
    for status in ("failed", "early_stopped"):
        try:
            existing_run_action({"status": status}, False)
        except RuntimeError as error:
            assert status in str(error)
        else:
            raise AssertionError(f"terminal {status} run was accepted")
        assert existing_run_action({"status": status}, True) == "run"


def test_run_lock_refuses_concurrent_trainer(tmp_path: Path) -> None:
    first = acquire_run_lock(tmp_path / "output")
    try:
        try:
            acquire_run_lock(tmp_path / "output")
        except RuntimeError as error:
            assert "owns the run lock" in str(error)
        else:
            raise AssertionError("second trainer acquired the same output lock")
    finally:
        first.close()
    second = acquire_run_lock(tmp_path / "output")
    second.close()


def test_local_preflight_can_write_outside_future_cloud_result(tmp_path: Path) -> None:
    output = tmp_path / "cloud-results/layer63-200k"
    assert preflight_receipt_path(output, None, tmp_path) == output / "cloud-preflight.json"
    assert (
        preflight_receipt_path(output, Path(".moeme/preflight/layer63-200k.json"), tmp_path)
        == (tmp_path / ".moeme/preflight/layer63-200k.json").resolve()
    )


def test_terminal_status_never_calls_failed_training_report_passed() -> None:
    assert terminal_status(0, 0, True) == ("passed", "complete")
    assert terminal_status(0, 0, False) == ("failed", "training-report")
    assert terminal_status(0, 0, None) == ("failed", "training-report")
    assert terminal_status(0, 2, True) == ("failed", "curve-analysis")
    assert terminal_status(3, 0, True) == ("early_stopped", "curve-policy")
    assert terminal_status(130, None, None) == ("interrupted", "training-interrupted")
    assert terminal_status(143, None, None) == ("interrupted", "training-interrupted")
    assert terminal_status(75, None, None) == ("interrupted", "session-budget")
    assert terminal_status(0, 0, False, require_report_passed=False) == (
        "passed",
        "systems-smoke-complete",
    )
    assert terminal_status(0, 0, None, require_report_passed=False) == (
        "passed",
        "systems-smoke-complete",
    )


def test_run_and_tee_preserves_visible_child_output(tmp_path: Path, capsys) -> None:
    log = tmp_path / "cloud-training.log"
    returncode, child_pid = run_and_tee(
        [sys.executable, "-c", "print('measured progress')"],
        cwd=tmp_path,
        environment={},
        log_path=log,
    )
    assert returncode == 0
    assert child_pid > 0
    assert capsys.readouterr().out == "measured progress\n"
    assert log.read_text() == "measured progress\n"


def test_run_and_tee_forwards_interrupts_without_killing_parent(tmp_path: Path) -> None:
    for received in (signal.SIGINT, signal.SIGTERM):
        expected = 128 + received
        timer = threading.Timer(0.2, lambda value=received: os.kill(os.getpid(), value))
        timer.start()
        try:
            returncode, _ = run_and_tee(
                [
                    sys.executable,
                    "-c",
                    (
                        "import signal,sys,time; "
                        f"signal.signal(signal.Signals({received}), lambda *_: sys.exit({expected})); "
                        "print('ready', flush=True); time.sleep(30)"
                    ),
                ],
                cwd=tmp_path,
                environment={},
                log_path=tmp_path / "signal.log",
            )
        finally:
            timer.cancel()
        assert returncode == expected
