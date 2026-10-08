from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


def validate_mlp_tensors(
    tensors: dict[str, TensorSpec],
    *,
    layers: int,
    hidden_size: int,
    intermediate_size: int,
    expected_dtype: str | None = None,
) -> dict[str, Any]:
    """Validate every dense MLP triplet without loading tensor payloads."""

    missing: list[str] = []
    mismatched: list[dict[str, Any]] = []
    expected = {
        "gate_proj": (intermediate_size, hidden_size),
        "up_proj": (intermediate_size, hidden_size),
        "down_proj": (hidden_size, intermediate_size),
    }
    for layer in range(layers):
        prefix = f"model.language_model.layers.{layer}.mlp"
        for projection, shape in expected.items():
            name = f"{prefix}.{projection}.weight"
            spec = tensors.get(name)
            if spec is None:
                missing.append(name)
                continue
            reasons: dict[str, Any] = {}
            if spec.shape != shape:
                reasons["shape"] = {"expected": shape, "actual": spec.shape}
            if expected_dtype is not None and spec.dtype != expected_dtype:
                reasons["dtype"] = {"expected": expected_dtype, "actual": spec.dtype}
            if reasons:
                mismatched.append({"tensor": name, **reasons})
    return {
        "layers_checked": layers,
        "mlp_tensors_checked": layers * 3,
        "missing": missing,
        "mismatched": mismatched,
        "valid": not missing and not mismatched,
    }
