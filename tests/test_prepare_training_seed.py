from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.prepare_training_seed import prepare_seed


def test_prepare_seed_reflinks_and_receipts_partial_projection_state(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    save_file(
        {
            "expert_down": torch.ones((2, 2)),
            "router": torch.ones((2, 2)),
            "shared_indices": torch.tensor([0, 1]),
            "expert_indices": torch.tensor([[2, 3]]),
        },
        source,
    )
    receipt = prepare_seed(source, tmp_path / "seed", 63)
    assert receipt["layer"] == 63
    assert receipt["partial_projection_seed"] is True
    assert (tmp_path / "seed/model.safetensors").read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_seed(source, tmp_path / "seed", 63)
