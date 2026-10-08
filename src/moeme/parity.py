from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .layout import ExpertLayout
from .nn import MoEMeMLP
from .partition import dense_swiglu


class TensorStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        with (self.root / "model.safetensors.index.json").open(encoding="utf-8") as handle:
            self.index = json.load(handle)["weight_map"]

    def load(self, name: str, device: torch.device) -> torch.Tensor:
        shard = self.index.get(name)
        if shard is None:
            raise KeyError(f"tensor not found in checkpoint index: {name}")
        with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
            if device.type == "cpu":
                return tensor.clone()
            return tensor.to(device=device, non_blocking=False)


def validate_layer_forward_parity(
    source_dir: str | Path,
    converted_dir: str | Path,
    layout: ExpertLayout,
    *,
    layer: int,
    tokens: int = 4,
    device: str | torch.device = "cuda",
    seed: int = 17,
) -> dict[str, Any]:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity was requested but no CUDA device is available")
    source = TensorStore(source_dir)
    converted = TensorStore(converted_dir)
    prefix = f"model.language_model.layers.{layer}.mlp"
    gate = source.load(f"{prefix}.gate_proj.weight", device)
    up = source.load(f"{prefix}.up_proj.weight", device)
    down = source.load(f"{prefix}.down_proj.weight", device)
    hidden_size = gate.shape[1]
    if gate.shape[0] != layout.intermediate_size:
        raise ValueError("source layer intermediate size does not match layout")

    model = MoEMeMLP(hidden_size, layout.with_top_k(layout.routed_experts), dtype=gate.dtype).to(
        device
    )
    with torch.no_grad():
        for local_name, parameter in model.named_parameters():
            value = converted.load(f"{prefix}.{local_name}", device)
            if parameter.shape != value.shape:
                raise ValueError(
                    f"converted shape mismatch for {prefix}.{local_name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value)

        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(tokens, hidden_size, generator=generator, device=device, dtype=gate.dtype)
        dense = dense_swiglu(x, gate, up, down)
        sparse = model(x)
        difference = (dense.float() - sparse.float()).abs()
        dense_float = dense.float()
        relative_l2 = difference.norm() / dense_float.norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            dense_float.reshape(1, -1), sparse.float().reshape(1, -1)
        )
    return {
        "layer": layer,
        "tokens": tokens,
        "device": str(device),
        "dtype": str(gate.dtype).removeprefix("torch."),
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "relative_l2_error": relative_l2.item(),
        "cosine_similarity": cosine.item(),
        "finite": bool(torch.isfinite(sparse).all().item()),
    }
