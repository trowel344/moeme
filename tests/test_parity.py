import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from moeme.layout import ExpertLayout
from moeme.parity import validate_layer_forward_parity
from moeme.safetensors_stream import convert_checkpoint_streaming


def test_converted_layer_executes_with_dense_parity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    converted = tmp_path / "converted"
    source.mkdir()
    generator = torch.Generator().manual_seed(3)
    prefix = "model.language_model.layers.0.mlp"
    tensors = {
        f"{prefix}.gate_proj.weight": torch.randn(32, 12, generator=generator),
        f"{prefix}.up_proj.weight": torch.randn(32, 12, generator=generator),
        f"{prefix}.down_proj.weight": torch.randn(12, 32, generator=generator),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, source / shard)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard for name in tensors}})
    )
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=2)
    convert_checkpoint_streaming(source, converted, layout)
    result = validate_layer_forward_parity(
        source, converted, layout, layer=0, tokens=5, device="cpu"
    )
    assert result["relative_l2_error"] < 1e-6
    assert result["cosine_similarity"] > 0.999999
    assert result["finite"] is True
