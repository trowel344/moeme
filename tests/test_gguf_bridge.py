import json
from pathlib import Path

import pytest

from moeme.gguf_bridge import prepare_gguf_staging


def checkpoint(tmp_path: Path, *, complete: bool = True) -> Path:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {"model_type": "qwen3_5_text"},
            }
        )
    )
    (root / "model.safetensors.index.json").write_text("{}")
    (root / "moeme-manifest.json").write_text(
        json.dumps(
            {
                "complete": complete,
                "layout": {
                    "intermediate_size": 17408,
                    "groups": 16,
                    "shared_groups": 4,
                    "routed_experts": 12,
                },
            }
        )
    )
    return root


def test_staging_is_no_copy_and_has_exactness_metadata(tmp_path: Path) -> None:
    source = checkpoint(tmp_path)
    staging = tmp_path / "staging"
    result = prepare_gguf_staging(source, staging, top_k=12)
    config = json.loads((staging / "config.json").read_text())
    text = config["text_config"]
    assert config["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]
    assert text["num_experts"] == 12
    assert text["num_experts_per_tok"] == 12
    assert text["moe_intermediate_size"] == 1088
    assert text["shared_expert_intermediate_size"] == 4352
    assert text["routed_scaling_factor"] == 12.0
    assert (staging / "model.safetensors.index.json").is_symlink()
    assert "doubles shared down projection" in result["shared_gate"]
    assert result["runtime_contract"]["expert_weights_scale_required"] is True
    assert result["runtime_contract"]["expert_weights_scale_key"] == (
        "qwen35moe.expert_weights_scale"
    )


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        prepare_gguf_staging(checkpoint(tmp_path, complete=False), tmp_path / "staging", top_k=4)
