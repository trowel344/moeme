"""
moeme.mocm -- Mixture of Cut Modules (MoCM) v0.

A MoCM is a dense swiGLU FFN cut into disjoint slices ("Cut Modules")
with a top-k router. This is intentionally the *sliced* construction,
not the complete-replica construction:

* sliced (this file): 1x storage, k/E active bytes per token.
  Best for efficiency / local runtime. Lossy at k<E.
* replica (moeme.branch): Ex storage, full-width active.
  Lossless at full width, zero byte saving. Best for quality.

That tradeoff is measured, not assumed: E16 shows a full-width replica
reproduces dense (ppl 15.49 vs 15.48) at 4.00x stored FFN and 100%
active width, while a 12.5%-active slice loses 42-81% of layer output
even with a perfect oracle router. Losslessness and byte savings are
the same axis.

Cut semantics (exactness):
  A cut is a partition of the intermediate axis (moeme.ffn). All modules
  active with combine="sum" IS the dense layer (worst rel err 5.1e-07
  on real 27B weights). Any k<E is an approximation, and on SwiGLU
  there is no latent structure to find (participation ratio ~31.6/32).

This module is a thin, named wrapper over moeme.upcycle.MoEFFN with
plan="random" forced (v0), plus the v1 stubborn path for 25%/12.5% active:
shared always-on expert + activation-ordered cut + softmax routing on
moeme.branch.BranchMoEFFN. v1 exists because random cuts scatter the
always-active neurons across experts, forcing the router to fire
everything (E15: eff 33.2/64, load 9.25); the shared expert holds those
units outside the router's remit (CMoE fix).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .branch import BranchConfig, BranchMoEFFN
from .upcycle import MoEConfig, MoEFFN, upcycle_model


@dataclass(frozen=True)
class MoCMConfig:
    """Configuration for one MoCM layer."""

    n_modules: int = 8
    top_k: int = 1
    router_hidden: int | None = None
    aux_loss_coef: float = 1e-2
    z_loss_coef: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_modules < 1:
            raise ValueError("n_modules must be >= 1")
        if not 1 <= self.top_k <= self.n_modules:
            raise ValueError(
                f"top_k={self.top_k} must be in [1, n_modules={self.n_modules}]"
            )

    def to_moe_config(self) -> MoEConfig:
        return MoEConfig(
            n_experts=self.n_modules,
            top_k=self.top_k,
            combine="sum",  # only mode preserving the exact dense limit at k=E
            plan="random",  # MoCM v0: semi-random cut per user spec
            aux_loss_coef=self.aux_loss_coef,
            z_loss_coef=self.z_loss_coef,
            router_hidden=self.router_hidden,
            seed=self.seed,
        )


class MoCMLayer(MoEFFN):
    """One FFN cut into random modules. Identical math to MoEFFN(plan=random)."""

    def __init__(self, gate: Tensor, up: Tensor, down: Tensor, config: MoCMConfig):
        super().__init__(gate, up, down, config.to_moe_config())
        self.mocm_config = config

    @classmethod
    def from_hf_mlp(cls, mlp: nn.Module, config: MoCMConfig) -> "MoCMLayer":
        missing = [n for n in ("gate_proj", "up_proj", "down_proj")
                   if not hasattr(mlp, n)]
        if missing:
            raise ValueError(f"mlp is missing {missing}; not a swiGLU MLP")
        return cls(
            mlp.gate_proj.weight.detach(),
            mlp.up_proj.weight.detach(),
            mlp.down_proj.weight.detach(),
            config,
        )

    @property
    def n_modules(self) -> int:
        return self.mocm_config.n_modules

    def active_params(self) -> int:
        """FFN params read per token (router excluded)."""
        per_module = [s * self.hidden_dim * 3 for s in self.expert_sizes()]
        # top-k picks vary by token; report mean over uniform sizes
        return sum(sorted(per_module, reverse=True)[: self.top_k])

    def stored_params(self) -> int:
        return int(self.gate.numel() + self.up.numel() + self.down.numel())

    def describe(self) -> dict:
        d = super().describe()
        d["architecture"] = "MoCM-random-cut"
        d["n_modules"] = self.n_modules
        return d


def mocm_cut_model(
    model: nn.Module, config: MoCMConfig, layers: list[int] | None = None
) -> list[tuple[int, MoEFFN]]:
    """Replace model.model.layers[i].mlp with MoCM layers. Returns [(idx, mod)]."""
    return upcycle_model(model, config.to_moe_config(), layers)


# ---------------------------------------------------------------------------
# v1: stubborn path for 25% / 12.5% active width
# ---------------------------------------------------------------------------

def activation_order(
    gate: Tensor, up: Tensor, sample_inputs: Tensor, threshold: float = 0.0
) -> tuple[int, ...]:
    """Order intermediate units by activation frequency (descending).

    Runs the dense gate/up on `sample_inputs` (T, H), scores each unit by
    how often |h| exceeds `threshold` (default: mean-activation ranking),
    and returns high-frequency units first. The caller puts the head of
    this order in the shared always-on expert.
    """
    with torch.no_grad():
        x = sample_inputs.reshape(-1, sample_inputs.shape[-1]).float()
        h = F.silu(F.linear(x, gate.float())) * F.linear(x, up.float())
        if threshold > 0:
            score = (h.abs() > threshold).float().mean(dim=0)
        else:
            score = h.abs().mean(dim=0)
        return tuple(int(i) for i in torch.argsort(score, descending=True))


@dataclass(frozen=True)
class MoCMv1Config:
    """One MoCM v1 layer: shared always-on + routed tiles, softmax gating.

    Active width fraction = (shared_units + top_k * (I / n_routed)) / I.
    Two sanctioned cells:
      25%:   n_routed=8, top_k=1, shared=I//8  -> 12.5% + 12.5%
      12.5%: n_routed=16, top_k=1, shared=I//16 -> 6.25% + 6.25%
    """

    n_routed: int = 8
    top_k: int = 1
    shared_units: int | None = None  # None => I // 8
    router_hidden: int | None = None
    aux_loss_coef: float = 1e-2
    z_loss_coef: float = 0.0
    seed: int = 0
    init_noise: float = 0.0  # relative std of symmetry-breaking init noise
    unit_order: tuple[int, ...] | None = None  # None => tiled in weight order

    def resolved_shared(self, intermediate: int) -> int:
        return int(intermediate // 8) if self.shared_units is None else int(self.shared_units)

    def to_branch_config(self, intermediate: int) -> BranchConfig:
        if intermediate % self.n_routed != 0:
            raise ValueError(
                f"tile init needs n_routed * expert_units == I, "
                f"got {self.n_routed} x ? != {intermediate}"
            )
        shared = self.resolved_shared(intermediate)
        return BranchConfig(
            n_experts=self.n_routed,
            top_k=self.top_k,
            expert_units=intermediate // self.n_routed,
            init="tile",
            n_shared=0 if shared <= 0 else 1,
            shared_units=max(shared, 1),
            unit_order=self.unit_order,
            combine="softmax",  # router must receive LM gradient; sum gives none
            aux_loss_coef=self.aux_loss_coef,
            z_loss_coef=self.z_loss_coef,
            router_hidden=self.router_hidden,
            init_noise=self.init_noise,
            seed=self.seed,
        )

    def active_width_fraction(self, intermediate: int) -> float:
        return (self.resolved_shared(intermediate)
                + self.top_k * (intermediate // self.n_routed)) / intermediate


def build_mocm_v1_layer(
    gate: Tensor, up: Tensor, down: Tensor, config: MoCMv1Config
) -> BranchMoEFFN:
    """Build one v1 layer (shared + routed tiles, softmax)."""
    return BranchMoEFFN(gate, up, down, config.to_branch_config(int(gate.shape[0])))


def align_moe_dtype(m: BranchMoEFFN, dtype) -> BranchMoEFFN:
    """Cast expert weights to the activation dtype, router stays float32.

    Needed when experts are built from dequantized (fp32) copies while the
    model runs bf16: without this the expert GEMMs raise a dtype mismatch.
    No-op when everything already matches.
    """
    import torch as _t
    for name in ("gate", "up", "down", "s_gate", "s_up", "s_down"):
        p = getattr(m, name, None)
        if isinstance(p, _t.nn.Parameter) and p.dtype != dtype:
            p.data = p.data.to(dtype)
    m.router.to(dtype=_t.float32)
    return m


def mocm_v1_presets(intermediate: int) -> dict:
    """The two sanctioned cells with their width arithmetic."""
    out = {}
    for name, cfg in (
        ("mocm25", MoCMv1Config(n_routed=8, top_k=1, shared_units=intermediate // 8)),
        ("mocm12", MoCMv1Config(n_routed=16, top_k=1, shared_units=intermediate // 16)),
    ):
        out[name] = {
            "active_width_fraction": cfg.active_width_fraction(intermediate),
            "stored_width_fraction": 1.0 + cfg.resolved_shared(intermediate) / intermediate,
        }
    return out
