from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torch import nn
from transformers import AutoConfig, AutoModelForMultimodalLM

from .layout import ExpertLayout
from .nn import MoEMeMLP


def install_moeme_mlps(
    model: nn.Module,
    layout: ExpertLayout,
    *,
    copy_dense_weights: bool,
) -> nn.Module:
    """Replace only language FFNs, leaving every other module untouched."""

    language_model = model.model.language_model
    if len(language_model.layers) == 0:
        raise ValueError("model has no language layers")
    for index, layer in enumerate(language_model.layers):
        dense = layer.mlp
        required = ("gate_proj", "up_proj", "down_proj")
        if not all(hasattr(dense, name) for name in required):
            raise TypeError(f"layer {index} does not contain a supported dense SwiGLU MLP")
        gate = dense.gate_proj.weight
        if gate.shape != (layout.intermediate_size, language_model.config.hidden_size):
            raise ValueError(f"layer {index} MLP shape {tuple(gate.shape)} does not match layout")
        if copy_dense_weights:
            if gate.device.type == "meta":
                raise ValueError("cannot copy dense weights from a meta-initialized model")
            replacement = MoEMeMLP.from_dense(
                gate,
                dense.up_proj.weight,
                dense.down_proj.weight,
                layout.with_top_k(layout.routed_experts),
            )
            replacement.set_top_k(layout.top_k)
        else:
            replacement = MoEMeMLP(
                language_model.config.hidden_size,
                layout,
                dtype=gate.dtype,
                device=gate.device,
            )
        layer.mlp = replacement
    return model


def load_moeme_checkpoint(
    checkpoint: str | Path,
    *,
    top_k: int = 12,
    device_map: str | dict[str, Any] = "auto",
    max_memory: dict[Any, Any] | None = None,
    offload_folder: str | Path | None = None,
) -> nn.Module:
    """Load a converted checkpoint with RAM/disk dispatch support.

    Top-12 is the parity/reference default. Sparse Top-K must only be selected
    after the corresponding router has been trained.
    """

    checkpoint = Path(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    layout = ExpertLayout(config.text_config.intermediate_size, top_k=top_k)
    with init_empty_weights():
        model = AutoModelForMultimodalLM.from_config(config)
        install_moeme_mlps(model, layout, copy_dense_weights=False)
    return load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint,
        device_map=device_map,
        max_memory=max_memory,
        no_split_module_classes=["Qwen3_5DecoderLayer", "MoEMeMLP"],
        offload_folder=str(offload_folder) if offload_folder is not None else None,
        offload_state_dict=True,
        dtype=torch.bfloat16,
        strict=True,
    )
