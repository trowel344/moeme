import json
from pathlib import Path

import pytest

from scripts.extract_layer_checkpoint import extract_layer_checkpoint
from scripts.oracle_router_analysis import load_layer_weights
from tests.test_oracle_router_analysis import STORED_EXPERTS, build_checkpoint


def test_extract_layer_checkpoint_is_loadable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    build_checkpoint(source)
    output = tmp_path / "portable"

    receipt = extract_layer_checkpoint(source, 0, output)

    assert receipt["tensor_count"] == STORED_EXPERTS * 3 + 3
    assert receipt["bytes"] == (output / "model.safetensors").stat().st_size
    assert len(receipt["sha256"]) == 64
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {"model.safetensors"}
    weights = load_layer_weights(output, 0, "cpu", "checkpoint", None, shared_groups=1)
    assert weights["expert_gate"].shape[0] == STORED_EXPERTS


def test_extract_refuses_overwrite_and_missing_layer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    build_checkpoint(source)
    output = tmp_path / "portable"
    output.mkdir()
    with pytest.raises(FileExistsError):
        extract_layer_checkpoint(source, 0, output)
    with pytest.raises(ValueError, match="no MLP tensors"):
        extract_layer_checkpoint(source, 63, tmp_path / "missing")
