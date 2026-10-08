import torch
from test_oracle_router_analysis import HIDDEN, build_checkpoint

from scripts.oracle_router_analysis import load_layer_weights
from scripts.patch_gguf_layer import frozen_partition_weights, resolve_frozen_projections


def test_frozen_projections_rebuild_partition_rows(tmp_path) -> None:
    """Down-only receipts omit gate/up; they must be rebuildable bit-for-bit.

    The rebuild concatenates a partition checkpoint's stored dense rows and
    selects them with the receipt's saved channel indices, so it has to agree
    exactly with the tensors the training run actually used.
    """
    checkpoint = tmp_path / "checkpoint"
    build_checkpoint(checkpoint, dtype=torch.bfloat16)
    inputs = torch.randn(6, HIDDEN, generator=torch.Generator().manual_seed(17))
    weights = load_layer_weights(
        checkpoint, 0, "cpu", "importance_contiguous", inputs, shared_groups=1, groups=2
    )
    receipt = {
        "shared_indices": weights["shared_indices"],
        "expert_indices": weights["expert_indices"],
    }
    resolved = resolve_frozen_projections(receipt, frozen_partition_weights(checkpoint, 0))

    for key in ("shared_gate", "shared_up", "expert_gate", "expert_up"):
        assert torch.equal(resolved[key], weights[key]), key
    # Trained tensors already present must be passed through untouched.
    assert "expert_down" not in resolved
