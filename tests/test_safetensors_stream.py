import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from moeme.layout import ExpertLayout
from moeme.safetensors_stream import (
    convert_checkpoint_streaming,
    plan_moeme_shard,
    read_safetensors_header,
    validate_converted_shards,
    write_safetensors_streaming,
)


def test_streaming_split_preserves_all_values_and_adds_zero_router(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(4)
    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(
            32, 12, generator=generator
        ),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(
            32, 12, generator=generator
        ),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(
            12, 32, generator=generator
        ),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(
            12, generator=generator
        ),
    }
    source_path = tmp_path / "source.safetensors"
    output_path = tmp_path / "output.safetensors"
    save_file(tensors, source_path, metadata={"format": "pt"})
    source = read_safetensors_header(source_path)
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=2)
    plans = plan_moeme_shard(source, layout)
    write_safetensors_streaming(output_path, plans, metadata=source.metadata)
    converted = load_file(output_path)

    for projection in ("gate_proj", "up_proj"):
        pieces = [converted[f"model.language_model.layers.0.mlp.shared_expert.{projection}.weight"]]
        pieces += [
            converted[f"model.language_model.layers.0.mlp.experts.{i}.{projection}.weight"]
            for i in range(6)
        ]
        torch.testing.assert_close(
            torch.cat(pieces, dim=0),
            tensors[f"model.language_model.layers.0.mlp.{projection}.weight"],
        )
    down_pieces = [converted["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"]]
    down_pieces += [
        converted[f"model.language_model.layers.0.mlp.experts.{i}.down_proj.weight"]
        for i in range(6)
    ]
    torch.testing.assert_close(
        torch.cat(down_pieces, dim=1),
        tensors["model.language_model.layers.0.mlp.down_proj.weight"],
    )
    torch.testing.assert_close(
        converted["model.language_model.layers.0.input_layernorm.weight"],
        tensors["model.language_model.layers.0.input_layernorm.weight"],
    )
    assert torch.count_nonzero(converted["model.language_model.layers.0.mlp.router.weight"]) == 0
    assert converted["model.language_model.layers.0.mlp.router.weight"].shape == (6, 12)
    assert (
        torch.count_nonzero(
            converted["model.language_model.layers.0.mlp.shared_expert_gate.weight"]
        )
        == 0
    )
    assert converted["model.language_model.layers.0.mlp.shared_expert_gate.weight"].shape == (
        1,
        12,
    )


def test_invalid_header_offsets_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    header = {
        "x": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]},
    }
    encoded = json.dumps(header).encode()
    padding = (-len(encoded)) % 8
    path.write_bytes(
        len(encoded + b" " * padding).to_bytes(8, "little") + encoded + b" " * padding + b"12345"
    )
    try:
        read_safetensors_header(path)
    except ValueError as error:
        assert "invalid/non-contiguous offsets" in str(error)
    else:
        raise AssertionError("malformed offsets were accepted")


def test_checkpoint_conversion_resumes_completed_shards(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.arange(32 * 12).reshape(32, 12),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(32 * 12).reshape(32, 12),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.arange(12 * 32).reshape(12, 32),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, source_dir / shard)
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard for name in tensors},
            }
        )
    )
    (source_dir / "config.json").write_text("{}")
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=2)

    first = convert_checkpoint_streaming(source_dir, output_dir, layout)
    mtime = (output_dir / shard).stat().st_mtime_ns
    second = convert_checkpoint_streaming(source_dir, output_dir, layout)
    assert first == second
    assert (output_dir / shard).stat().st_mtime_ns == mtime
    progress = json.loads((output_dir / "conversion-progress.json").read_text())
    assert progress["status"] == "complete"
    validation = validate_converted_shards(source_dir, output_dir, layout)
    assert validation["valid"] is True
    assert validation["complete"] is True
    assert validation["tensors_checked"] == len(load_file(output_dir / shard))


def test_incomplete_checkpoint_converts_available_shards(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    available = "model-00001-of-00002.safetensors"
    missing = "model-00002-of-00002.safetensors"
    tensor_name = "model.language_model.layers.0.input_layernorm.weight"
    save_file({tensor_name: torch.ones(12)}, source_dir / available)
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {tensor_name: available, "future.weight": missing},
            }
        )
    )
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=2)
    manifest = convert_checkpoint_streaming(source_dir, output_dir, layout, allow_incomplete=True)
    assert manifest["complete"] is False
    assert manifest["converted_shards"] == 1
    assert manifest["missing_shards"] == [missing]
    assert (output_dir / available).is_file()
    assert json.loads((output_dir / "conversion-progress.json").read_text())["status"] == (
        "waiting-for-source-shards"
    )
    validation = validate_converted_shards(source_dir, output_dir, layout, allow_incomplete=True)
    assert validation["complete"] is False
    assert validation["validated_shards"] == 1
