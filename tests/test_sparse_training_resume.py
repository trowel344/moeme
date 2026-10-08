import json
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from moeme.activations import MAGIC, VERSION
from scripts import train_sparse_layer
from scripts.oracle_router_analysis import load_layer_weights


def build_tiny_checkpoint(root: Path) -> None:
    root.mkdir()
    hidden = 4
    width = 2
    experts = 4
    base = "model.language_model.layers.0.mlp."
    generator = torch.Generator().manual_seed(9)
    tensors = {}
    for expert in range(experts):
        tensors[f"{base}experts.{expert}.gate_proj.weight"] = torch.randn(
            width, hidden, generator=generator
        )
        tensors[f"{base}experts.{expert}.up_proj.weight"] = torch.randn(
            width, hidden, generator=generator
        )
        tensors[f"{base}experts.{expert}.down_proj.weight"] = torch.randn(
            hidden, width, generator=generator
        )
    tensors[f"{base}shared_expert.gate_proj.weight"] = torch.randn(
        width, hidden, generator=generator
    )
    tensors[f"{base}shared_expert.up_proj.weight"] = torch.randn(width, hidden, generator=generator)
    tensors[f"{base}shared_expert.down_proj.weight"] = torch.randn(
        hidden, width, generator=generator
    )
    save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
    )
    (root / "moeme-manifest.json").write_text(
        json.dumps(
            {
                "layout": {
                    "groups": 5,
                    "shared_groups": 1,
                    "routed_experts": experts,
                    "intermediate_size": width * 5,
                }
            }
        )
    )


def build_tiny_capture(root: Path) -> None:
    root.mkdir()
    values = np.random.default_rng(7).standard_normal((16, 4)).astype(np.float32)
    with (root / "layer-0.f32").open("wb") as handle:
        handle.write(struct.pack("<III", MAGIC, VERSION, values.shape[1]))
        handle.write(struct.pack("<I", len(values)))
        handle.write(values.tobytes())


def test_resume_input_identity_is_content_bound_not_location_bound(tmp_path: Path) -> None:
    first = tmp_path / "first-checkpoint"
    second = tmp_path / "second-checkpoint"
    build_tiny_checkpoint(first)
    shutil.copytree(first, second)
    assert train_sparse_layer.checkpoint_identity(
        first, 0
    ) == train_sparse_layer.checkpoint_identity(second, 0)
    first_seed = tmp_path / "first-seed.safetensors"
    second_seed = tmp_path / "second-seed.safetensors"
    first_seed.write_bytes(b"portable partition seed")
    shutil.copy2(first_seed, second_seed)
    assert train_sparse_layer.partition_seed_identity(
        first_seed
    ) == train_sparse_layer.partition_seed_identity(second_seed)


def test_oracle_label_build_checks_interrupt_before_each_batch() -> None:
    calls = []

    def interrupt() -> None:
        calls.append(True)
        raise train_sparse_layer.TrainingInterrupted(signal.SIGTERM, "oracle")

    with pytest.raises(train_sparse_layer.TrainingInterrupted):
        train_sparse_layer.build_oracle_labels(
            torch.zeros((4, 4)),
            {"expert_gate": torch.zeros((48, 1, 1))},
            (4,),
            2,
            "cpu",
            torch.float32,
            interrupt,
        )
    assert calls == [True]


def test_completed_training_state_can_resume_without_retraining(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    activations = tmp_path / "activations"
    output = tmp_path / "output"
    build_tiny_checkpoint(checkpoint)
    build_tiny_capture(activations)
    repository = Path(__file__).resolve().parents[1]
    base = [
        sys.executable,
        str(repository / "scripts/train_sparse_layer.py"),
        "--activations",
        str(activations),
        "--checkpoint",
        str(checkpoint),
        "--layer",
        "0",
        "--output-dir",
        str(output),
        "--partition-mode",
        "checkpoint",
        "--top-k-schedule",
        "4",
        "--steps-per-stage",
        "2",
        "--router-warmup-steps",
        "2",
        "--batch-size",
        "2",
        "--checkpoint-every",
        "1",
        "--progress-every",
        "1",
        "--evaluate-every",
        "1",
        "--evaluation-tokens",
        "2",
        "--device",
        "cpu",
        "--compute-dtype",
        "float32",
        "--max-validation-relative-l2",
        "10",
        "--ledger",
        str(tmp_path / "ledger.sqlite3"),
    ]
    environment = {"PYTHONPATH": str(repository / "src")}
    first = subprocess.run(
        base,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    state_path = output / "training-state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    assert state["stage_index"] == 1
    assert state["stage_step"] == 0
    assert state["model"]["expert_down"].dtype == torch.float32
    assert state["grad_scaler"] == {}
    assert len(state["curve"]) == 3
    report = json.loads((output / "report.json").read_text())
    assert report["training_tokens"] > 2
    assert report["training_evaluation_tokens"] == 2
    assert report["validation_tokens"] > 2
    curve = json.loads((output / "curve.json").read_text())
    assert [point["step"] for point in curve["points"]] == [0, 1, 2]
    assert curve["points"][0]["initial"] is True
    label_chunks = sorted((output / "oracle-labels").glob("*/*.safetensors"))
    assert len(label_chunks) == 2
    label_mtimes = {path: path.stat().st_mtime_ns for path in label_chunks}

    second = subprocess.run(
        [*base, "--resume-state", str(state_path)],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert {path: path.stat().st_mtime_ns for path in label_chunks} == label_mtimes
    progress = json.loads((output / "progress.json").read_text())
    assert progress["phase"] == "complete"


def test_training_consumes_completed_activation_shards_without_concatenation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    shards = tmp_path / "shards"
    output = tmp_path / "output"
    build_tiny_checkpoint(checkpoint)
    shards.mkdir()
    for index in range(2):
        shard = shards / f"shard-{index:04d}"
        build_tiny_capture(shard)
        (shard / "manifest.json").write_text(
            json.dumps(
                {
                    "tokens_per_layer": 16,
                    "layers": {"0": {"tokens": 16, "width": 4}},
                }
            )
        )
    (shards / "campaign.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "shards": {
                    "0": {"status": "passed"},
                    "1": {"status": "passed"},
                },
            }
        )
    )
    repository = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repository / "scripts/train_sparse_layer.py"),
        "--activations",
        str(shards),
        "--checkpoint",
        str(checkpoint),
        "--layer",
        "0",
        "--output-dir",
        str(output),
        "--partition-mode",
        "checkpoint",
        "--top-k-schedule",
        "4",
        "--steps-per-stage",
        "2",
        "--router-warmup-steps",
        "2",
        "--batch-size",
        "2",
        "--checkpoint-every",
        "0",
        "--progress-every",
        "1",
        "--device",
        "cpu",
        "--compute-dtype",
        "float32",
        "--max-validation-relative-l2",
        "10",
        "--ledger",
        str(tmp_path / "ledger.sqlite3"),
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        env={"PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["training_tokens"] == 16
    assert report["training_evaluation_tokens"] == 16
    assert report["validation_tokens"] == 16


def test_partial_weight_seed_freezes_partition_and_initializes_missing_features(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    activations = tmp_path / "activations"
    output = tmp_path / "output"
    seed = tmp_path / "seed.safetensors"
    build_tiny_checkpoint(checkpoint)
    build_tiny_capture(activations)
    calibration = torch.randn(4, 4, generator=torch.Generator().manual_seed(4))
    weights = load_layer_weights(
        checkpoint,
        0,
        "cpu",
        "checkpoint",
        calibration,
        shared_groups=1,
    )
    save_file(
        {
            "expert_down": weights["expert_down"],
            "router": torch.ones((4, 4)),
            "shared_indices": weights["shared_indices"],
            "expert_indices": weights["expert_indices"],
            "shared_gate": weights["shared_gate"],
            "shared_up": weights["shared_up"],
            "shared_down": weights["shared_down"],
        },
        seed,
    )
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/train_sparse_layer.py"),
            "--activations",
            str(activations),
            "--checkpoint",
            str(checkpoint),
            "--layer",
            "0",
            "--output-dir",
            str(output),
            "--partition-mode",
            "checkpoint",
            "--partition-indices-from",
            str(seed),
            "--resume",
            str(seed),
            "--allow-partial-resume",
            "--top-k-schedule",
            "4",
            "--steps-per-stage",
            "2",
            "--router-warmup-steps",
            "20",
            "--batch-size",
            "2",
            "--checkpoint-every",
            "0",
            "--progress-every",
            "1",
            "--train-projections",
            "all",
            "--train-shared",
            "--device",
            "cpu",
            "--compute-dtype",
            "float32",
            "--max-validation-relative-l2",
            "10",
            "--ledger",
            str(tmp_path / "ledger.sqlite3"),
        ],
        cwd=repository,
        env={"PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    final = torch.load(output / "training-state.pt", map_location="cpu", weights_only=True)
    assert "expert_gate" in final["model"]
    assert "expert_up" in final["model"]
    assert torch.equal(final["model"]["shared_indices"], weights["shared_indices"])


@pytest.mark.parametrize("received", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_saves_exact_completed_step_before_exit(
    tmp_path: Path, received: signal.Signals
) -> None:
    checkpoint = tmp_path / "checkpoint"
    activations = tmp_path / "activations"
    output = tmp_path / "output"
    build_tiny_checkpoint(checkpoint)
    build_tiny_capture(activations)
    repository = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [
            sys.executable,
            str(repository / "scripts/train_sparse_layer.py"),
            "--activations",
            str(activations),
            "--checkpoint",
            str(checkpoint),
            "--layer",
            "0",
            "--output-dir",
            str(output),
            "--partition-mode",
            "checkpoint",
            "--top-k-schedule",
            "4",
            "--steps-per-stage",
            "10000",
            "--router-warmup-steps",
            "2",
            "--batch-size",
            "2",
            "--checkpoint-every",
            "0",
            "--state-every",
            "0",
            "--progress-every",
            "1",
            "--device",
            "cpu",
            "--compute-dtype",
            "float32",
            "--ledger",
            str(tmp_path / "ledger.sqlite3"),
        ],
        cwd=repository,
        env={"PYTHONPATH": str(repository / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 30
    observed_step = 0
    while time.monotonic() < deadline:
        progress_path = output / "progress.json"
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text())
            except json.JSONDecodeError:
                progress = {}
            if progress.get("phase") == "training" and int(progress.get("step", 0)) > 0:
                observed_step = int(progress["step"])
                break
        time.sleep(0.01)
    assert observed_step > 0
    process.send_signal(received)
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 128 + received, (stdout, stderr)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["phase"] == "interrupted"
    assert progress["interrupted_stage"] == "training"
    state = torch.load(output / "training-state.pt", map_location="cpu", weights_only=True)
    assert state["stage_step"] >= observed_step


def test_atomic_safetensors_failure_preserves_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"previous-complete-checkpoint")

    def fail_save(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated interrupted write")

    monkeypatch.setattr(train_sparse_layer, "save_file", fail_save)
    with pytest.raises(RuntimeError, match="interrupted write"):
        train_sparse_layer.atomic_safetensors_save(
            checkpoint,
            {"weight": torch.ones(1)},
            {"format": "test"},
        )
    assert checkpoint.read_bytes() == b"previous-complete-checkpoint"
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_wall_clock_budget_exits_cleanly_with_resumable_evidence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    activations = tmp_path / "activations"
    output = tmp_path / "output"
    build_tiny_checkpoint(checkpoint)
    build_tiny_capture(activations)
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/train_sparse_layer.py"),
            "--activations",
            str(activations),
            "--checkpoint",
            str(checkpoint),
            "--layer",
            "0",
            "--output-dir",
            str(output),
            "--partition-mode",
            "checkpoint",
            "--top-k-schedule",
            "4",
            "--steps-per-stage",
            "10000",
            "--router-warmup-steps",
            "2",
            "--batch-size",
            "2",
            "--progress-every",
            "1",
            "--device",
            "cpu",
            "--compute-dtype",
            "float32",
            "--max-runtime-seconds",
            "0.000001",
            "--ledger",
            str(tmp_path / "ledger.sqlite3"),
        ],
        cwd=repository,
        env={"PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 75, completed.stderr
    progress = json.loads((output / "progress.json").read_text())
    assert progress["phase"] == "session_budget_reached"
    assert progress["max_runtime_seconds"] == pytest.approx(0.000001)
