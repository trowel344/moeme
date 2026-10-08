from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .layout import ExpertLayout


def dense_swiglu(x: Tensor, gate: Tensor, up: Tensor, down: Tensor) -> Tensor:
    """Reference bias-free SwiGLU using Transformers weight orientation."""

    _validate_dense_shapes(x, gate, up, down)
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


@dataclass(frozen=True, slots=True)
class SwiGLUWeights:
    gate: Tensor
    up: Tensor
    down: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return dense_swiglu(x, self.gate, self.up, self.down)


class PartitionedSwiGLU:
    """Exact channel partition of a dense SwiGLU.

    The shared and routed branches own disjoint intermediate channels. Summing
    all branch outputs is algebraically identical to the dense FFN; no fitting
    or calibration is involved.
    """

    def __init__(self, shared: SwiGLUWeights, experts: tuple[SwiGLUWeights, ...]) -> None:
        if not experts:
            raise ValueError("at least one routed expert is required")
        self.shared = shared
        self.experts = experts

    @classmethod
    def from_dense(
        cls,
        gate: Tensor,
        up: Tensor,
        down: Tensor,
        layout: ExpertLayout,
        *,
        permutation: Tensor | None = None,
    ) -> PartitionedSwiGLU:
        _validate_weight_shapes(gate, up, down)
        if gate.shape[0] != layout.intermediate_size:
            raise ValueError(
                f"weight intermediate size {gate.shape[0]} != layout {layout.intermediate_size}"
            )
        if permutation is None:
            permutation = torch.arange(layout.intermediate_size, device=gate.device)
        _validate_permutation(permutation, layout.intermediate_size)

        def take(index: Tensor) -> SwiGLUWeights:
            return SwiGLUWeights(
                gate=gate.index_select(0, index),
                up=up.index_select(0, index),
                down=down.index_select(1, index),
            )

        width = layout.group_width
        groups = tuple(permutation[i * width : (i + 1) * width] for i in range(layout.groups))
        shared_index = torch.cat(groups[: layout.shared_groups])
        shared = take(shared_index)
        experts = tuple(take(index) for index in groups[layout.shared_groups :])
        return cls(shared, experts)

    def forward_all(self, x: Tensor) -> Tensor:
        """Exact conversion-stage forward: shared + every routed partition."""

        output = self.shared.forward(x)
        for expert in self.experts:
            output = output + expert.forward(x)
        return output

    def forward_selected(
        self,
        x: Tensor,
        expert_indices: Iterable[int],
        weights: Tensor | None = None,
        *,
        preserve_sum_scale: bool = True,
    ) -> Tensor:
        """Run selected experts.

        If normalized router weights are supplied, ``preserve_sum_scale``
        multiplies them by K. This is required for a uniform Top-all router to
        reproduce the additive dense partition. Setting it false implements a
        conventional normalized MoE mixture, which is intentionally *not* an
        exact initialization.
        """

        indices = tuple(expert_indices)
        if not indices:
            raise ValueError("at least one expert must be selected")
        if len(set(indices)) != len(indices):
            raise ValueError("expert indices must be unique")
        if any(index < 0 or index >= len(self.experts) for index in indices):
            raise IndexError("expert index out of range")

        if weights is None:
            scales = torch.ones(len(indices), device=x.device, dtype=x.dtype)
        else:
            if weights.ndim != 1 or weights.numel() != len(indices):
                raise ValueError("weights must be a 1D tensor with one value per selected expert")
            scales = weights.to(device=x.device, dtype=x.dtype)
            if preserve_sum_scale:
                scales = scales * len(indices)

        output = self.shared.forward(x)
        for index, scale in zip(indices, scales, strict=True):
            output = output + scale * self.experts[index].forward(x)
        return output


def _validate_dense_shapes(x: Tensor, gate: Tensor, up: Tensor, down: Tensor) -> None:
    _validate_weight_shapes(gate, up, down)
    if x.shape[-1] != gate.shape[1]:
        raise ValueError(f"input hidden size {x.shape[-1]} != weight hidden size {gate.shape[1]}")


def _validate_weight_shapes(gate: Tensor, up: Tensor, down: Tensor) -> None:
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("gate, up, and down weights must all be matrices")
    if gate.shape != up.shape:
        raise ValueError(f"gate shape {tuple(gate.shape)} != up shape {tuple(up.shape)}")
    expected_down = (gate.shape[1], gate.shape[0])
    if tuple(down.shape) != expected_down:
        raise ValueError(f"down shape {tuple(down.shape)} != expected {expected_down}")


def _validate_permutation(permutation: Tensor, size: int) -> None:
    if permutation.ndim != 1 or permutation.numel() != size:
        raise ValueError(f"permutation must contain exactly {size} indices")
    if permutation.dtype not in (torch.int32, torch.int64):
        raise ValueError("permutation must use an integer dtype")
    sorted_values = permutation.detach().cpu().sort().values
    expected = torch.arange(size, dtype=sorted_values.dtype)
    if not torch.equal(sorted_values, expected):
        raise ValueError("permutation must contain every intermediate index exactly once")
