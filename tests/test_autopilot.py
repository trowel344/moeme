import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from moeme.autopilot import CheckpointAutopilot
from moeme.layout import ExpertLayout


def test_autopilot_advances_once_then_waits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    state = tmp_path / "state"
    source.mkdir()
    shard = "model-00001-of-00002.safetensors"
    missing = "model-00002-of-00002.safetensors"
    name = "model.language_model.layers.0.input_layernorm.weight"
    save_file({name: torch.ones(12)}, source / shard)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard, "future": missing}})
    )
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=2)
    autopilot = CheckpointAutopilot(source, output, state, layout)

    first = autopilot.advance_once()
    second = autopilot.advance_once()
    assert first.status == "advanced"
    assert first.bytes_validated == 48
    assert second.status == "waiting"
    assert second.converted_shards == 1
    assert len(autopilot.ledger.history()) == 2
