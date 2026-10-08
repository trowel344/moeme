from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def prepare_gguf_staging(
    checkpoint_dir: str | Path,
    staging_dir: str | Path,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Create a no-copy HF view consumable as native Qwen35MoE by llama.cpp."""

    checkpoint_dir = Path(checkpoint_dir)
    staging_dir = Path(staging_dir)
    with (checkpoint_dir / "moeme-manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("complete"):
        raise ValueError("converted checkpoint is incomplete")
    layout = manifest["layout"]
    if not 1 <= top_k <= layout["routed_experts"]:
        raise ValueError("top_k is outside the converted expert count")

    with (checkpoint_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    config["model_type"] = "qwen3_5_moe"
    text = config["text_config"]
    text.update(
        {
            "model_type": "qwen3_5_moe_text",
            "num_experts": layout["routed_experts"],
            "num_experts_per_tok": top_k,
            "moe_intermediate_size": layout["intermediate_size"] // layout["groups"],
            "shared_expert_intermediate_size": (
                layout["intermediate_size"] * layout["shared_groups"] // layout["groups"]
            ),
            "norm_topk_prob": True,
            "score_function": "softmax",
            "routed_scaling_factor": float(top_k),
        }
    )

    staging_dir.mkdir(parents=True, exist_ok=True)
    _write_json(staging_dir / "config.json", config)
    linked = []
    for source in checkpoint_dir.iterdir():
        if source.name in {"config.json", "conversion-progress.json", "moeme-manifest.json"}:
            continue
        if not (source.is_file() or source.is_symlink()):
            continue
        destination = staging_dir / source.name
        target = os.path.relpath(source.resolve(), staging_dir.resolve())
        if destination.is_symlink() and os.readlink(destination) == target:
            linked.append(source.name)
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"staging entry already exists with a different target: {destination}"
            )
        destination.symlink_to(target)
        linked.append(source.name)
    bridge_manifest = {
        "format": "moeme-gguf-bridge-v1",
        "source_checkpoint": str(checkpoint_dir.resolve()),
        "top_k": top_k,
        "routed_scale": float(top_k),
        "shared_gate": "sigmoid(0)=0.5; GGUF conversion doubles shared down projection",
        "runtime_contract": {
            "expert_weights_scale_key": "qwen35moe.expert_weights_scale",
            "expert_weights_scale_required": True,
            "reason": "Top-K normalization must be scaled back to the dense routed sum",
        },
        "linked_files": sorted(linked),
    }
    _write_json(staging_dir / "moeme-gguf-bridge.json", bridge_manifest)
    return bridge_manifest


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
