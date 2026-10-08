import json

import pytest

from moeme.architecture import inspect_config


def config() -> dict:
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "max_position_embeddings": 262144,
            "vocab_size": 248320,
        },
        "vision_config": {"depth": 27, "out_hidden_size": 5120},
    }


def test_inspection_counts_layer_kinds() -> None:
    facts = inspect_config(config())
    assert facts.hidden_size == 5120
    assert facts.intermediate_size == 17408
    assert facts.linear_attention_layers == 3
    assert facts.full_attention_layers == 1
    assert json.loads(json.dumps(facts.to_dict()))["vision_depth"] == 27


def test_mismatched_layer_count_is_rejected() -> None:
    value = config()
    value["text_config"]["num_hidden_layers"] = 64
    with pytest.raises(ValueError, match="does not match"):
        inspect_config(value)
