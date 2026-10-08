import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.oracle_router_analysis import load_layer_weights, project_experts, project_shared
from scripts.train_sparse_layer import checkpoint_shared_groups

HIDDEN = 4
INTERMEDIATE = 8
STORED_GROUPS = 4
STORED_SHARED_GROUPS = 1
STORED_EXPERTS = 3
STORED_WIDTH = INTERMEDIATE // STORED_GROUPS


def build_checkpoint(
    root: Path, seed: int = 7, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    root.mkdir(parents=True, exist_ok=True)
    base = "model.language_model.layers.0.mlp."
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(STORED_EXPERTS):
        for projection, shape in (
            ("gate_proj", (STORED_WIDTH, HIDDEN)),
            ("up_proj", (STORED_WIDTH, HIDDEN)),
            ("down_proj", (HIDDEN, STORED_WIDTH)),
        ):
            tensors[f"{base}experts.{expert}.{projection}.weight"] = torch.randn(
                shape, generator=generator, dtype=dtype
            )
    for projection, shape in (
        ("gate_proj", (STORED_SHARED_GROUPS * STORED_WIDTH, HIDDEN)),
        ("up_proj", (STORED_SHARED_GROUPS * STORED_WIDTH, HIDDEN)),
        ("down_proj", (HIDDEN, STORED_SHARED_GROUPS * STORED_WIDTH)),
    ):
        tensors[f"{base}shared_expert.{projection}.weight"] = torch.randn(
            shape, generator=generator, dtype=dtype
        )
    save_file(tensors, root / "model-00001-of-00001.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}})
    )
    (root / "moeme-manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "layout": {
                    "intermediate_size": INTERMEDIATE,
                    "groups": STORED_GROUPS,
                    "shared_groups": STORED_SHARED_GROUPS,
                    "routed_experts": STORED_EXPERTS,
                },
            }
        )
    )
    return tensors


def test_checkpoint_mode_reconstructs_dense_mlp(tmp_path: Path) -> None:
    """The stored shared down projection is 1x dense; the runtime slot is 2x.

    `project_shared` applies the llama.cpp sigmoid(shared_expert_gate=0) == 0.5
    compensation, so `load_layer_weights` must return the shared down projection
    at 2x for shared + every routed expert to reproduce the dense FFN.
    """
    tensors = build_checkpoint(tmp_path / "checkpoint")
    inputs = torch.randn(6, HIDDEN, generator=torch.Generator().manual_seed(11))
    weights = load_layer_weights(
        tmp_path / "checkpoint",
        0,
        "cpu",
        "checkpoint",
        inputs,
        shared_groups=STORED_SHARED_GROUPS,
    )
    base = "model.language_model.layers.0.mlp."
    assert torch.allclose(
        weights["shared_down"], 2 * tensors[f"{base}shared_expert.down_proj.weight"]
    )

    dense_gate = torch.cat(
        [tensors[f"{base}shared_expert.gate_proj.weight"]]
        + [tensors[f"{base}experts.{expert}.gate_proj.weight"] for expert in range(STORED_EXPERTS)]
    )
    dense_up = torch.cat(
        [tensors[f"{base}shared_expert.up_proj.weight"]]
        + [tensors[f"{base}experts.{expert}.up_proj.weight"] for expert in range(STORED_EXPERTS)]
    )
    dense_down = torch.cat(
        [tensors[f"{base}shared_expert.down_proj.weight"]]
        + [tensors[f"{base}experts.{expert}.down_proj.weight"] for expert in range(STORED_EXPERTS)],
        dim=1,
    )
    hidden = torch.nn.functional.silu(inputs @ dense_gate.T) * (inputs @ dense_up.T)
    expected = hidden @ dense_down.T

    rebuilt = project_shared(inputs, weights) + project_experts(inputs, weights).sum(dim=1)
    relative = ((rebuilt - expected).square().sum() / expected.square().sum()) ** 0.5
    assert relative.item() < 1e-5


def test_groups_override_repartitions_stored_dense_weights(tmp_path: Path) -> None:
    """A group-count override re-partitions the same dense weights at a new width."""
    # Stored partition tensors are BF16 in production; match that here so the
    # override exercises the same dtype path.
    build_checkpoint(tmp_path / "checkpoint", dtype=torch.bfloat16)
    inputs = torch.randn(6, HIDDEN, generator=torch.Generator().manual_seed(13))
    weights = load_layer_weights(
        tmp_path / "checkpoint",
        0,
        "cpu",
        "importance_contiguous",
        inputs,
        shared_groups=1,
        groups=2,
    )
    assert weights["shared_gate"].shape == (INTERMEDIATE // 2, HIDDEN)
    assert weights["expert_gate"].shape == (1, INTERMEDIATE // 2, HIDDEN)
    assert weights["expert_indices"].shape == (1, INTERMEDIATE // 2)
    assert weights["shared_down"].shape == (HIDDEN, INTERMEDIATE // 2)
    # The override describes a partition of the same dense channels: every
    # intermediate channel lands in exactly one shared or routed slot.
    covered = torch.cat((weights["shared_indices"], weights["expert_indices"].flatten()))
    assert sorted(covered.tolist()) == list(range(INTERMEDIATE))
    assert weights["shared_down"].dtype == torch.bfloat16


def test_fixed_partition_indices_bypass_activation_dependent_repartition(tmp_path: Path) -> None:
    build_checkpoint(tmp_path / "checkpoint")
    inputs = torch.randn(6, HIDDEN, generator=torch.Generator().manual_seed(17))
    fixed = {
        "shared_indices": torch.tensor([7, 1, 4, 0]),
        "expert_indices": torch.tensor([[6, 2, 5, 3]]),
    }
    weights = load_layer_weights(
        tmp_path / "checkpoint",
        0,
        "cpu",
        "importance_contiguous",
        inputs,
        shared_groups=1,
        groups=2,
        partition_indices=fixed,
    )
    assert torch.equal(weights["shared_indices"], fixed["shared_indices"])
    assert torch.equal(weights["expert_indices"], fixed["expert_indices"])
    invalid = {**fixed, "expert_indices": torch.tensor([[6, 2, 5, 7]])}
    with pytest.raises(ValueError, match="not an exact channel partition"):
        load_layer_weights(
            tmp_path / "checkpoint",
            0,
            "cpu",
            "importance_contiguous",
            inputs,
            shared_groups=1,
            groups=2,
            partition_indices=invalid,
        )


def test_checkpoint_shared_groups_reads_the_manifest(tmp_path: Path) -> None:
    build_checkpoint(tmp_path / "checkpoint")
    assert checkpoint_shared_groups(tmp_path / "checkpoint") == STORED_SHARED_GROUPS
