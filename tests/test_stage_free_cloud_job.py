import hashlib
import json
import struct
from pathlib import Path

import pytest

from moeme.activations import MAGIC, VERSION
from scripts.stage_free_cloud_job import SOURCE_FILES, build_stage, verify_stage


def receipt(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_stage_is_atomic_zero_copy_and_self_verifying(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    for name in SOURCE_FILES:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{name}\n")
    module = repository / "src/moeme/example.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n")
    activation = repository / "cloud-inputs/activations"
    checkpoint = repository / "cloud-inputs/checkpoint"
    seed = repository / "cloud-inputs/seed/model.safetensors"
    activation.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    seed.parent.mkdir(parents=True)
    activation_file = activation / "layer-63.f32"
    activation_file.write_bytes(
        struct.pack("<IIII", MAGIC, VERSION, 2, 1) + struct.pack("<ff", 1.0, 2.0)
    )
    (activation / "manifest.json").write_text(
        json.dumps(
            {
                "tokens_per_layer": 1,
                "layers": {"63": {**receipt(activation_file), "tokens": 1, "width": 2}},
            }
        )
    )
    checkpoint_file = checkpoint / "model.safetensors"
    checkpoint_file.write_bytes(b"checkpoint")
    (checkpoint / "receipt.json").write_text(
        json.dumps({**receipt(checkpoint_file), "tensor_file": checkpoint_file.name})
    )
    (checkpoint / "moeme-manifest.json").write_text(json.dumps({"layer": 63}))
    (checkpoint / "model.safetensors.index.json").write_text("{}")
    seed.write_bytes(b"seed")
    (seed.parent / "receipt.json").write_text(json.dumps({**receipt(seed), "layer": 63}))
    config = {
        "activations": "../cloud-inputs/activations",
        "checkpoint": "../cloud-inputs/checkpoint",
        "seed_checkpoint": "../cloud-inputs/seed/model.safetensors",
        "layer": 63,
        "output_dir": "../cloud-results/layer63-200k",
    }
    config_path = repository / "configs/free-cloud-layer63-200k.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    output = repository / "cloud-jobs/layer63-200k"
    result = build_stage(repository, config_path, output)
    assert result["verified"] is True
    staged_activation = output / "cloud-inputs/activations/layer-63.f32"
    assert staged_activation.stat().st_ino == (activation / "layer-63.f32").stat().st_ino
    manifest = json.loads((output / "upload-manifest.json").read_text())
    assert manifest["large_inputs_are_hardlinked"] is True
    assert manifest["input_validation"]["activation_tokens"] == 1
    assert manifest["input_validation"]["critical_hashes_verified"] == 3
    runbook = (output / "RUN.md").read_text()
    assert "Kaggle (selected by the measured free-tier budget)" in runbook
    assert "cloud_bootstrap.py --provider lightning" in runbook
    assert 'cloud_bootstrap.py"' in runbook
    assert "--preflight-only" not in runbook
    assert "pip install -e" not in runbook
    assert "pip install torch" not in runbook.lower()
    assert verify_stage(output)["file_count"] == len(manifest["files"])
    source_trainer = repository / "scripts/train_sparse_layer.py"
    source_trainer.write_text("updated trainer\n")
    rebuilt = build_stage(repository, config_path, output, rebuild=True)
    assert rebuilt["verified"] is True
    assert (output / "scripts/train_sparse_layer.py").read_text() == "updated trainer\n"
    assert not (output.parent / ".layer63-200k.rebuild-backup").exists()
    extra = output / "undeclared.txt"
    extra.write_text("not hash-bound")
    with pytest.raises(ValueError, match="unexpected=.*undeclared.txt"):
        verify_stage(output)
    extra.unlink()
    link = output / "external-link"
    link.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        verify_stage(output)
    link.unlink()
    (output / "scripts/train_sparse_layer.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="mismatch"):
        verify_stage(output)
