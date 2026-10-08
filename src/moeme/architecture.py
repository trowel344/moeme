from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ArchitectureFacts:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    linear_attention_layers: int
    full_attention_layers: int
    max_position_embeddings: int
    vocab_size: int
    vision_depth: int | None
    vision_output_size: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_config(config: dict[str, Any]) -> ArchitectureFacts:
    try:
        text = config["text_config"]
        layer_types = text["layer_types"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"not a supported Qwen multimodal config: missing {error}") from error
    if len(layer_types) != text["num_hidden_layers"]:
        raise ValueError("layer_types length does not match num_hidden_layers")
    unknown = set(layer_types) - {"linear_attention", "full_attention"}
    if unknown:
        raise ValueError(f"unsupported layer types: {sorted(unknown)}")
    vision = config.get("vision_config") or {}
    return ArchitectureFacts(
        model_type=config["model_type"],
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        num_hidden_layers=text["num_hidden_layers"],
        linear_attention_layers=layer_types.count("linear_attention"),
        full_attention_layers=layer_types.count("full_attention"),
        max_position_embeddings=text["max_position_embeddings"],
        vocab_size=text["vocab_size"],
        vision_depth=vision.get("depth"),
        vision_output_size=vision.get("out_hidden_size"),
    )


def inspect_config_file(path: str | Path) -> ArchitectureFacts:
    with Path(path).open(encoding="utf-8") as handle:
        return inspect_config(json.load(handle))
