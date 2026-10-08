from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layout import ExpertLayout


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, dtype=None, device=None):
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, **factory)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, **factory)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, **factory)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass(frozen=True, slots=True)
class RouterStats:
    counts: Tensor
    probabilities: Tensor
    entropy: Tensor


class MoEMeMLP(nn.Module):
    """Trainable reference MoEMe FFN.

    This is a correctness/training implementation, not the final paged runtime.
    It uses explicit token dispatch so routing behavior can be audited.
    """

    def __init__(
        self,
        hidden_size: int,
        layout: ExpertLayout,
        *,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layout = layout
        self.top_k = layout.top_k
        self.shared_expert = (
            SwiGLUExpert(hidden_size, layout.shared_width, dtype=dtype, device=device)
            if layout.shared_width
            else None
        )
        self.experts = nn.ModuleList(
            SwiGLUExpert(hidden_size, layout.expert_width, dtype=dtype, device=device)
            for _ in range(layout.routed_experts)
        )
        self.router = nn.Linear(
            hidden_size,
            layout.routed_experts,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        nn.init.zeros_(self.router.weight)
        self.shared_expert_gate = (
            nn.Linear(hidden_size, 1, bias=False, dtype=dtype, device=device)
            if layout.shared_width
            else None
        )
        if self.shared_expert_gate is not None:
            nn.init.zeros_(self.shared_expert_gate.weight)

    @classmethod
    def from_dense(
        cls,
        gate: Tensor,
        up: Tensor,
        down: Tensor,
        layout: ExpertLayout,
        *,
        permutation: Tensor | None = None,
    ) -> MoEMeMLP:
        if gate.ndim != 2 or up.shape != gate.shape:
            raise ValueError("gate/up weights must be matching matrices")
        if down.shape != (gate.shape[1], gate.shape[0]):
            raise ValueError("down weight has an incompatible shape")
        if gate.shape[0] != layout.intermediate_size:
            raise ValueError("weights do not match layout intermediate size")
        if permutation is None:
            permutation = torch.arange(layout.intermediate_size, device=gate.device)
        sorted_indices = permutation.detach().cpu().sort().values
        if not torch.equal(
            sorted_indices, torch.arange(layout.intermediate_size, dtype=sorted_indices.dtype)
        ):
            raise ValueError("permutation must contain every channel exactly once")

        model = cls(gate.shape[1], layout, dtype=gate.dtype, device=gate.device)
        width = layout.group_width
        groups = [permutation[i * width : (i + 1) * width] for i in range(layout.groups)]

        def copy(target: SwiGLUExpert, indices: Tensor) -> None:
            with torch.no_grad():
                target.gate_proj.weight.copy_(gate.index_select(0, indices))
                target.up_proj.weight.copy_(up.index_select(0, indices))
                target.down_proj.weight.copy_(down.index_select(1, indices))

        if model.shared_expert is not None:
            copy(model.shared_expert, torch.cat(groups[: layout.shared_groups]))
        for expert, indices in zip(model.experts, groups[layout.shared_groups :], strict=True):
            copy(expert, indices)
        return model

    def set_top_k(self, top_k: int) -> None:
        if not 1 <= top_k <= len(self.experts):
            raise ValueError(f"top_k must be in [1, {len(self.experts)}]")
        self.top_k = top_k

    def route(self, x: Tensor) -> tuple[Tensor, Tensor, RouterStats]:
        router_input = x if x.dtype == torch.float64 else x.float()
        logits = F.linear(router_input, self.router.weight.to(dtype=router_input.dtype))
        values, indices = torch.topk(logits, self.top_k, dim=-1)
        probabilities = torch.softmax(values, dim=-1).to(dtype=x.dtype)
        # The K factor preserves additive partition scale. At Top-all with zero
        # logits it gives every original dense channel coefficient one.
        weights = probabilities * self.top_k
        counts = torch.bincount(indices.reshape(-1), minlength=len(self.experts))
        entropy = -(probabilities.float() * probabilities.float().clamp_min(1e-12).log()).sum(
            dim=-1
        )
        return indices, weights, RouterStats(counts, probabilities, entropy)

    def forward(self, x: Tensor, *, return_router_stats: bool = False):
        original_shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        indices, weights, stats = self.route(flat)
        accumulator_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        output = (
            self.shared_expert(flat).to(dtype=accumulator_dtype)
            if self.shared_expert is not None
            else torch.zeros_like(flat, dtype=accumulator_dtype)
        )

        for expert_index, expert in enumerate(self.experts):
            selected = (indices == expert_index).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            token_indices = selected[:, 0]
            slots = selected[:, 1]
            expert_output = expert(flat.index_select(0, token_indices)).to(dtype=accumulator_dtype)
            scales = weights[token_indices, slots].to(dtype=accumulator_dtype).unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output * scales)

        output = output.to(dtype=x.dtype).reshape(original_shape)
        return (output, stats) if return_router_stats else output
