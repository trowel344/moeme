"""
moeme.upcycle -- turn a dense swiGLU FFN into a *trainable* top-k MoE.

Why this module exists
----------------------
`experiments/pilot_sparsity.py` measured the **no-retrain ceiling**: slicing a
*trained* dense FFN into E experts and keeping k of them loses most of the
layer output (Qwen3.5-0.8B layer 0, E=8, k=1: mean relative residual 0.746).
Nothing in that experiment touches any weight, so it is a ceiling for a
*frozen* partition -- and it says nothing about what gradient descent could do.

This module implements the thing the pilot deliberately did not: a partition
whose expert weights *and* router are trained. The two experiments together
answer the question that matters:

    can training beat the frozen ceiling, i.e. does a sparse partition become
    usable once it is allowed to specialise?

Design decisions, stated plainly (they are choices, not accidents)
------------------------------------------------------------------
* Experts are a **partition** of the intermediate axis (via `moeme.ffn`), not
  replicas. Total parameters therefore stay ~constant, which is what a
  "27B total / 9B active" target requires. It also means `top_k == n_experts`
  is an **exact** decomposition of the dense layer, so the dense model is a
  strict special case of this module and cannot be beaten at `top_k=E`.
* `combine="sum"` is the default. It keeps that exact limit *and* matches the
  semantics of `ffn.oracle_topk_residual` (a partial sum of expert outputs),
  so "trained vs frozen" is an apples-to-apples comparison. The price is that a
  sparse sum is scaled by ~k/E relative to the dense layer; training has to
  learn that scale. `combine="mean"` and `combine="softmax"` are provided and
  documented, but they do *not* preserve the exact limit at top_k=E.
* `combine="sum_rescaled"` multiplies the sparse sum by **E/k** -- the output
  rescaling LLaMA-MoE and NVIDIA both call essential for upcycling -- so a
  k-of-E partial sum starts at the dense layer's *magnitude* instead of ~k/E of
  it. It is exactly 1.0 when k == E, so unlike `mean`/`softmax` it **does**
  preserve the dense limit, and `all_active_forward` (which hardcodes `sum`)
  remains the exact reference either way.

  Note what this can and cannot fix, because it decides which metric to read. A
  global scale `s` multiplies `total` and `kept` together, so any residual
  written as `||total - kept|| / ||total||` is *invariant* to it -- which is
  what `ffn.oracle_topk_residual` and `router_topk_residual` compute. The
  rescaling therefore cannot move those two numbers at all; it can only move a
  residual measured against the **dense layer** (`dense_forward`, i.e. the
  unscaled sum), such as `reconstruction_residual`. A frozen 'oracle' table is
  structurally blind to this fix, so a negative result read off it is not
  evidence about the rescaling.
* For the PoC the full weight matrices stay resident as `Parameter`s and the
  partition is a fixed buffer. This is deliberately **not** a memory-efficient
  MoE -- it exists to answer a quality question, not a memory question.
  Sharding/dispatch is a deployment concern, covered in
  `docs/upcycling_design.md`.

Layout matches Hugging Face: gate/up are (intermediate, hidden), down is
(hidden, intermediate), so `F.linear(x, gate)` is the HF forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ffn import (
    DenseFFN,
    PartitionPlan,
    clustered_plan,
    contiguous_plan,
    interleaved_plan,
    random_plan,
)

COMBINE_MODES = ("sum", "sum_rescaled", "mean", "softmax")
PLAN_MODES = ("contiguous", "interleaved", "random", "clustered")


def _dequant(w: Tensor) -> Tensor:
    """A plain detached copy, dequantizing bitsandbytes storage if needed."""
    if type(w).__module__.startswith("bitsandbytes"):
        try:
            import bitsandbytes.functional as bnb_f
            qs = getattr(w, "quant_state", None)
            if qs is not None and getattr(w, "data", None) is not None \
                    and w.data.dtype == torch.uint8:
                out = bnb_f.dequantize_4bit(w.data, qs).to(torch.float32)
                return out
            if hasattr(w, "dequantize"):
                return w.dequantize().to(torch.float32)
        except Exception:
            pass
        return w.float().detach().clone()
    return w.detach().clone()


def build_plan(name: str, gate: Tensor, up: Tensor, n_experts: int,
               seed: int = 0) -> PartitionPlan:
    """Dispatch to the partition plans in `moeme.ffn`."""
    inter = int(gate.shape[0])
    if name == "contiguous":
        return contiguous_plan(inter, n_experts)
    if name == "interleaved":
        return interleaved_plan(inter, n_experts)
    if name == "random":
        return random_plan(inter, n_experts, seed=seed)
    if name == "clustered":
        return clustered_plan(gate, up, n_experts, seed=seed)
    raise ValueError(f"unknown plan {name!r}; expected one of {PLAN_MODES}")


@dataclass(frozen=True)
class MoEConfig:
    """Configuration for one upcycled MoE layer."""

    n_experts: int = 8
    top_k: int = 2
    combine: str = "sum"
    aux_loss_coef: float = 1e-2
    z_loss_coef: float = 0.0
    router_hidden: int | None = None   # None => a single linear router layer
    plan: str = "random"
    seed: int = 0
    expert_noise: float = 0.0          # relative std added to expert init

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(
                f"top_k={self.top_k} must be in [1, n_experts={self.n_experts}]"
            )
        if self.combine not in COMBINE_MODES:
            raise ValueError(f"combine={self.combine!r} not in {COMBINE_MODES}")
        if self.plan not in PLAN_MODES:
            raise ValueError(f"plan={self.plan!r} not in {PLAN_MODES}")
        if self.expert_noise < 0:
            raise ValueError("expert_noise must be >= 0")

    def with_(self, **kw) -> "MoEConfig":
        from dataclasses import replace
        return replace(self, **kw)


@dataclass
class RoutingMetrics:
    """Load balancing / specialisation diagnostics for one forward pass."""

    usage_fraction: list[float]        # dispatch share per expert, sums to 1
    effective_experts: float           # participation ratio of `usage_fraction`
    load_max_over_mean: float          # 1.0 == perfectly balanced
    usage_entropy_normalised: float    # 1.0 == uniform dispatch
    router_entropy_mean: float         # 1.0 == router is maximally unsure
    n_tokens: int

    def as_dict(self) -> dict:
        return {
            "usage_fraction": self.usage_fraction,
            "effective_experts": self.effective_experts,
            "load_max_over_mean": self.load_max_over_mean,
            "usage_entropy_normalised": self.usage_entropy_normalised,
            "router_entropy_mean": self.router_entropy_mean,
            "n_tokens": self.n_tokens,
        }


# ---------------------------------------------------------------------------
# the module
# ---------------------------------------------------------------------------

class MoEFFN(nn.Module):
    """A dense swiGLU FFN viewed as `n_experts` partition experts with a router.

    Drop-in replacement for a Hugging Face `*MLP` module: `forward(x)` takes the
    hidden states and returns the layer output. The auxiliary losses the
    training loop needs are stashed on the module (`last_aux_loss`,
    `last_z_loss`, `last_stats`) because HF decoder layers call `self.mlp(x)`
    with no channel to return them.
    """

    def __init__(self, gate: Tensor, up: Tensor, down: Tensor,
                 config: MoEConfig) -> None:
        super().__init__()
        if gate.shape != up.shape:
            raise ValueError("gate and up must share shape (intermediate, hidden)")
        hidden, inter = int(gate.shape[1]), int(gate.shape[0])
        if tuple(down.shape) != (hidden, inter):
            raise ValueError(
                f"down must be (hidden, intermediate)=({hidden}, {inter}), "
                f"got {tuple(down.shape)}"
            )
        if config.n_experts > inter:
            raise ValueError(
                f"n_experts={config.n_experts} exceeds intermediate={inter}"
            )

        self.config = config
        self.hidden_dim = hidden
        self.intermediate = inter

        plan = build_plan(config.plan, gate, up, config.n_experts, config.seed)
        # the plan is built on the CPU; the module's weights may already be on a
        # GPU, so the index buffer MUST follow the weights or index_select fails
        self.register_buffer("assignment", plan.assignment.to(gate.device).clone())

        if config.expert_noise > 0:
            g = torch.Generator(device="cpu").manual_seed(config.seed + 7919)
            def noisy(t: Tensor) -> Tensor:
                sd = t.float().std().item()
                n = torch.randn(t.shape, generator=g,
                                dtype=torch.float32).to(t.device)
                return (t.float() + n * (sd * config.expert_noise)).to(t.dtype)
            gate, up, down = noisy(gate), noisy(up), noisy(down)

        # Parameters keep the dense layout; experts are views onto row/column
        # subsets of them under `assignment`.
        self.gate = nn.Parameter(gate.detach().clone().contiguous())
        self.up = nn.Parameter(up.detach().clone().contiguous())
        self.down = nn.Parameter(down.detach().clone().contiguous())

        if config.router_hidden:
            self.router: nn.Module = nn.Sequential(
                nn.Linear(hidden, config.router_hidden),
                nn.SiLU(),
                nn.Linear(config.router_hidden, config.n_experts),
            )
        else:
            self.router = nn.Linear(hidden, config.n_experts)
        self.router = self.router.to(device=gate.device, dtype=torch.float32)

        self.last_aux_loss: Tensor = torch.zeros((), dtype=torch.float32)
        self.last_z_loss: Tensor = torch.zeros((), dtype=torch.float32)
        self.last_stats: dict = {}
        self._units_cache: list[Tensor] | None = None
        self._units_device: torch.device | None = None

    # ---- construction helpers -------------------------------------------

    @classmethod
    def from_dense(cls, dense: DenseFFN, config: MoEConfig) -> "MoEFFN":
        return cls(dense.gate, dense.up, dense.down, config)

    @classmethod
    def from_hf_mlp(cls, mlp: nn.Module, config: MoEConfig) -> "MoEFFN":
        """Upcycle an HF `*MLP` module (gate_proj/up_proj/down_proj).

        Quantized backbones (bitsandbytes 8-bit/4-bit) are dequantized into
        full-precision expert copies: the converted layers always train in
        fp32/bf16 even when the frozen backbone stays quantized. This is what
        makes 27B-class conversion fittable on a 40GB card.
        """
        missing = [n for n in ("gate_proj", "up_proj", "down_proj")
                   if not hasattr(mlp, n)]
        if missing:
            raise ValueError(f"mlp is missing {missing}; not a swiGLU MLP")
        return cls(
            _dequant(mlp.gate_proj.weight),
            _dequant(mlp.up_proj.weight),
            _dequant(mlp.down_proj.weight),
            config,
        )

    # ---- geometry --------------------------------------------------------

    @property
    def n_experts(self) -> int:
        return self.config.n_experts

    @property
    def top_k(self) -> int:
        return min(self.config.top_k, self.config.n_experts)

    @property
    def params(self) -> int:
        return (self.gate.numel() + self.up.numel() + self.down.numel()
                + sum(p.numel() for p in self.router.parameters()))

    def expert_units(self, e: int) -> Tensor:
        """Intermediate indices owned by expert `e`, on the weight device."""
        if self._units_cache is None or self._units_device != self.assignment.device:
            a = self.assignment
            self._units_cache = [
                torch.nonzero(a == i, as_tuple=False).flatten()
                for i in range(self.n_experts)
            ]
            self._units_device = self.assignment.device
        return self._units_cache[e]

    def expert_sizes(self) -> list[int]:
        return [int(self.expert_units(e).numel()) for e in range(self.n_experts)]

    # ---- forward ---------------------------------------------------------

    def dense_forward(self, x: Tensor) -> Tensor:
        """The dense swiGLU layer these experts partition (exact reference)."""
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        h = F.silu(F.linear(x2, self.gate)) * F.linear(x2, self.up)
        return F.linear(h, self.down).reshape(shape)

    def _router_logits(self, x2: Tensor) -> Tensor:
        # router is kept in float32 for stable softmax/top-k
        return self.router(x2.float())

    def _route(self, x2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self._router_logits(x2)                     # (T, E) float32
        k = self.top_k
        top_val, top_idx = torch.topk(logits, k, dim=1)      # (T, k)
        return logits, top_val, top_idx

    def _combine_weights(self, top_val: Tensor, combine: str,
                         x_dtype: torch.dtype) -> Tensor:
        """Combination weights, always cast to the *activation* dtype.

        The router deliberately runs in float32 for stable top-k, so its outputs
        must be cast before they meet bf16/fp16 activations -- otherwise
        `index_add_` raises on a dtype mismatch (found the first time this ran on
        a bf16 model).
        """
        k = top_val.shape[1]
        if combine == "sum":
            return torch.ones_like(top_val, dtype=x_dtype)
        if combine == "sum_rescaled":
            # E/k: the sparse partial sum carries the dense layer's magnitude.
            # Exactly 1.0 at k == E, so the dense limit is preserved exactly.
            return torch.full_like(top_val, self.n_experts / k, dtype=x_dtype)
        if combine == "mean":
            return torch.full_like(top_val, 1.0 / k, dtype=x_dtype)
        if combine == "softmax":
            return torch.softmax(top_val, dim=1).to(x_dtype)
        raise ValueError(f"combine={combine!r}")

    def _forward_impl(self, x: Tensor, k: int | None = None,
                      combine: str | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (y, logits, top_idx). `k`/`combine` override the config."""
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        logits = self._router_logits(x2)                     # (T, E)
        E = self.n_experts
        k = self.top_k if k is None else min(k, E)
        combine = combine or self.config.combine

        top_val, top_idx = torch.topk(logits, k, dim=1)      # (T, k)
        sel_w = self._combine_weights(top_val, combine, x2.dtype)

        y = torch.zeros_like(x2)
        for e in range(E):
            mask = top_idx == e                              # (T, k)
            tok = mask.any(dim=1)                            # (T,)
            if not bool(tok.any()):
                continue
            units = self.expert_units(e)
            xe = x2[tok]                                     # (Te, H)
            ge = self.gate.index_select(0, units)            # (Ie, H)
            ue = self.up.index_select(0, units)
            he = F.silu(F.linear(xe, ge)) * F.linear(xe, ue)  # (Te, Ie)
            de = self.down.index_select(1, units)            # (H, Ie)
            ye = F.linear(he, de)                            # (Te, H)
            # `.sum()` is on autocast's float32-promote list, so under autocast
            # this reduction comes back float32 even from bf16 inputs. Cast it
            # (and the product) back to the activation dtype or index_add_ raises
            # a dtype mismatch -- which is what happened on the first real run.
            w = (sel_w * mask.to(sel_w.dtype)).sum(dim=1).to(y.dtype)   # (T,)
            rows = torch.nonzero(tok, as_tuple=False).flatten()
            y.index_add_(0, rows, (ye * w[tok].unsqueeze(1)).to(y.dtype))

        return y.reshape(shape), logits, top_idx

    def forward(self, x: Tensor) -> Tensor:
        y, logits, top_idx = self._forward_impl(x)
        self.last_aux_loss, self.last_z_loss = self._losses(logits, top_idx)
        # diagnostics are pure logging: keep them off the autograd graph
        with torch.no_grad():
            self.last_stats = self.routing_metrics_from(
                logits.detach(), top_idx
            ).as_dict()
        return y

    # ---- losses ----------------------------------------------------------

    def _losses(self, logits: Tensor, top_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Switch-style load-balancing aux loss, plus the router z-loss."""
        T = logits.shape[0]
        E = self.n_experts
        probs = torch.softmax(logits, dim=1)                  # (T, E)
        # dispatch fraction f_i: share of (token, slot) assignments on expert i
        onehot = F.one_hot(top_idx, num_classes=E).to(probs.dtype)   # (T, k, E)
        f = onehot.sum(dim=(0, 1)) / float(max(T * top_idx.shape[1], 1))  # (E,)
        P = probs.mean(dim=0)                                 # (E,)
        aux = E * (f * P).sum()
        z = torch.logsumexp(logits, dim=1)
        z_loss = (z ** 2).mean()
        return aux, z_loss

    def aux_loss(self) -> Tensor:
        return self.last_aux_loss

    def total_aux_loss(self) -> Tensor:
        return (self.config.aux_loss_coef * self.last_aux_loss
                + self.config.z_loss_coef * self.last_z_loss)

    # ---- diagnostics -----------------------------------------------------

    @staticmethod
    def routing_metrics_from(logits: Tensor, top_idx: Tensor) -> RoutingMetrics:
        E = int(logits.shape[1])
        T = int(logits.shape[0])
        onehot = F.one_hot(top_idx, num_classes=E).float()
        counts = onehot.sum(dim=(0, 1))                       # (E,)
        total = counts.sum().clamp_min(1.0)
        f = counts / total
        pr = 1.0 / f.pow(2).sum().clamp_min(1e-30)
        ent = -(f * f.clamp_min(1e-30).log()).sum() / torch.log(
            torch.tensor(float(E))
        ) if E > 1 else torch.zeros(())
        probs = torch.softmax(logits, dim=1)
        if E > 1:
            r_ent = (-(probs * probs.clamp_min(1e-30).log()).sum(dim=1)
                     / torch.log(torch.tensor(float(E)))).mean()
        else:
            r_ent = torch.zeros(())
        return RoutingMetrics(
            usage_fraction=[float(v) for v in f],
            effective_experts=float(pr),
            load_max_over_mean=float(f.max() / f.mean().clamp_min(1e-30)),
            usage_entropy_normalised=float(ent),
            router_entropy_mean=float(r_ent),
            n_tokens=T,
        )

    def routing_metrics(self, x: Tensor) -> RoutingMetrics:
        x2 = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            logits, _, top_idx = self._route(x2)
            return self.routing_metrics_from(logits, top_idx)

    def all_active_forward(self, x: Tensor) -> Tensor:
        """Partial sum over *every* expert -- must equal `dense_forward`.

        `combine="sum"` is hardcoded rather than taken from the config: the
        identity this checks is about the *unscaled* sum of every expert, which
        is the dense layer. With `sum_rescaled` the sparse path is scaled by
        E/k, so reading the limit through the config would make the check fail
        at k<E for a reason that has nothing to do with partitioning.
        """
        y, _, _ = self._forward_impl(x, k=self.n_experts, combine="sum")
        return y

    def expert_outputs(self, x: Tensor) -> Tensor:
        """Per-expert contributions, shape (n_experts, T, hidden).

        Materialising every expert is what the pilot called the *oracle* setting;
        exposing it here lets the PoC reuse `ffn.oracle_topk_residual` on the
        trained weights, so "trained" and "frozen" are measured the same way.
        """
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        outs = []
        for e in range(self.n_experts):
            units = self.expert_units(e)
            he = F.silu(F.linear(x2, self.gate.index_select(0, units))) \
                * F.linear(x2, self.up.index_select(0, units))
            outs.append(F.linear(he, self.down.index_select(1, units)))
        return torch.stack(outs, dim=0)

    def router_scores(self, x: Tensor) -> Tensor:
        """Router logits as (T, n_experts), float32, no grad graph."""
        x2 = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            return self._router_logits(x2).detach()

    def dense_equivalent_error(self, x: Tensor) -> float:
        with torch.no_grad():
            ref = self.dense_forward(x)
            got = self.all_active_forward(x)
            return float((ref.detach() - got.detach()).norm()
                         / ref.detach().norm().clamp_min(1e-12))

    def describe(self) -> dict:
        return {
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "combine": self.config.combine,
            "plan": self.config.plan,
            "hidden": self.hidden_dim,
            "intermediate": self.intermediate,
            "expert_sizes": self.expert_sizes(),
            "router_params": int(sum(p.numel() for p in self.router.parameters())),
            "expert_params": int(self.gate.numel() + self.up.numel()
                                 + self.down.numel()),
        }


# ---------------------------------------------------------------------------
# model-level helpers
# ---------------------------------------------------------------------------

def upcycle_model(model: nn.Module, config: MoEConfig,
                  layers: list[int] | None = None,
                  mlp_attr: str = "mlp") -> list[tuple[int, MoEFFN]]:
    """Replace `model.model.layers[i].mlp` with a `MoEFFN` for each target layer.

    Returns [(layer_index, module), ...]. `layers=None` means every layer.
    """
    blocks = model.model.layers
    if layers is None:
        layers = list(range(len(blocks)))
    out: list[tuple[int, MoEFFN]] = []
    for i in layers:
        if not 0 <= i < len(blocks):
            raise ValueError(f"layer {i} out of range (0..{len(blocks) - 1})")
        block = blocks[i]
        moe = MoEFFN.from_hf_mlp(getattr(block, mlp_attr), config)
        setattr(block, mlp_attr, moe)
        out.append((i, moe))
    return out


def iter_moe_modules(model: nn.Module):
    for m in model.modules():
        if isinstance(m, MoEFFN):
            yield m


def collect_total_aux_loss(model: nn.Module) -> Tensor:
    """Sum of every MoE layer's weighted aux + z loss for the last forward."""
    total = None
    for m in iter_moe_modules(model):
        t = m.total_aux_loss()
        total = t if total is None else total + t
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total


def router_parameters(moes) -> list[Parameter]:  # noqa: F821
    params = []
    for _, m in moes:
        params += list(m.router.parameters())
    return params


def expert_parameters(moes) -> list[Parameter]:  # noqa: F821
    params = []
    for _, m in moes:
        params += [m.gate, m.up, m.down]
    return params


def set_top_k(moes, k: int) -> None:
    """Mutate the top-k of already-built modules (for the dense-limit check)."""
    for _, m in moes:
        m.config = m.config.with_(top_k=k)


def reconstruction_residual(reference: Tensor, got: Tensor) -> float:
    """Relative L2 error of `got` against `reference`.

    Same metric as `ffn.oracle_topk_residual`, so a trained MoE's residual can
    be compared directly against the frozen ceiling the pilot measured.
    """
    ref = reference.float()
    return float((ref - got.float()).norm() / ref.norm().clamp_min(1e-12))
