"""
moeme.branch -- the *branching* alternative to partitioning a dense FFN.

Why this module exists
----------------------
`moeme.upcycle` (and the pilot before it) answers "can a **disjoint slice** of a
trained dense FFN stand in for the whole thing?" and finds that it cannot: a
top-k partial sum leaves ~(1 - k/E) of the layer output, and the residual only
approaches zero as k -> E. That is a statement about slicing a sum, not about
MoE.

The obvious objection -- and it is the right objection -- is that real MoE
models (Mixtral, Switch, BTX) do not slice anything: their experts are
*complete* feed-forward networks. Activating one gives a whole layer's maths,
not a fraction of it, so there is no partial-sum deficit to begin with.

This module implements that alternative so the two can be measured against each
other on the same model, activations and metric:

    expert = a complete swiGLU FFN, initialised from the dense one, routed

The parameter that matters is **how much intermediate width the active set
covers**, `top_k * expert_units / intermediate`. The module makes it possible to
vary that while holding the *shape* of the construction (tiled vs replicated)
fixed, which is what distinguishes "slicing hurts" from "reading less hurts".

Two constructions, one width knob
---------------------------------
* `init="tile"`      expert j is initialised from the j-th disjoint block of
                     intermediate units. At `expert_units = I / E` this is
                     exactly the pilot's partition, and `sum` over all experts is
                     the dense layer. Total parameters stay at 1x the dense FFN.
* `init="replica"`   every expert is initialised from the *same* block. At
                     `expert_units = I` every expert is a complete copy of the
                     dense FFN, so `top_k = 1` reproduces the dense layer
                     exactly and the residual is 0 -- at the price of storing E
                     complete FFNs (E x the FFN parameters).

That contrast is the whole point: the lossless case is lossless *because it
reads full width*, and it is not sparse. Nothing here claims otherwise; the
module exists to let the claim be measured rather than argued.

Shared experts
--------------
`n_shared` always-on experts are added to every token's output. This is the
DeepSeek-MoE / CMoE idea: the units that are active for *most* inputs (and so
cannot be routed away) belong in an always-on block, leaving the router to
choose only among genuinely input-dependent experts. `unit_order` lets the
caller put high-activation units first, which is how that block is chosen in
practice.

A warning about `combine="sum"`, which is also `upcycle.MoEFFN`'s default
-----------------------------------------------------------------------
With `sum` the combination weights are a constant 1.0, so the router's only
influence on the output is the *discrete* expert choice -- and a discrete choice
carries no gradient. The router is then reached **only** by the load-balancing
auxiliary loss, never by the language-model objective: from that loss alone the
router receives **no gradient at all** (`router.weight.grad is None`), against
~2.6e1 under `softmax`; with the auxiliary loss included -- the form a training
loop actually backprops -- `sum` reaches ~1e-2 against `softmax`'s ~2.6e1 at
E=8, k=2, a factor of ~1.6e3. (An earlier note here quoted 6e-9 against 2.8e-3,
~5 orders; neither figure reproduces, so it is replaced by the measured one. At
`k=1` the two modes coincide exactly, since a single selected logit has softmax
weight 1.0, so the difference exists only for `k>1`.) Pinned by
`tests/test_branch.py::test_softmax_carries_far_more_router_gradient_than_sum`.

So a `sum`-mode router is optimised for *balance*, never for *quality*, and a
router-underperforms-oracle result from a `sum` run says nothing about what a
properly differentiable router could do. `softmax` is the mode to train a router
in; `sum` is the mode to demonstrate exact dense equivalence in (it is the only
mode that keeps `top_k == E` an exact decomposition).

Layout matches Hugging Face: gate/up are (intermediate, hidden), down is
(hidden, intermediate), so `F.linear(x, W)` is the HF forward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

COMBINE_MODES = ("sum", "mean", "softmax")
INIT_MODES = ("tile", "replica")


@dataclass(frozen=True)
class BranchConfig:
    """Configuration for one branched MoE layer."""

    n_experts: int = 4
    top_k: int = 1
    #: intermediate units each routed expert owns. None => the full width.
    expert_units: int | None = None
    init: str = "replica"
    n_shared: int = 0
    #: intermediate units each shared expert owns. None => `expert_units`.
    shared_units: int | None = None
    #: permutation of the intermediate axis; high-priority units first. Used to
    #: place the always-active units in the shared expert.
    unit_order: tuple[int, ...] | None = None
    combine: str = "sum"
    aux_loss_coef: float = 1e-2
    z_loss_coef: float = 0.0
    router_hidden: int | None = None
    init_noise: float = 0.0     # relative std added to the expert initialisation
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(
                f"top_k={self.top_k} must be in [1, n_experts={self.n_experts}]"
            )
        if self.combine not in COMBINE_MODES:
            raise ValueError(f"combine={self.combine!r} not in {COMBINE_MODES}")
        if self.init not in INIT_MODES:
            raise ValueError(f"init={self.init!r} not in {INIT_MODES}")
        if self.n_shared < 0:
            raise ValueError("n_shared must be >= 0")
        if self.init_noise < 0:
            raise ValueError("init_noise must be >= 0")

    def with_(self, **kw) -> "BranchConfig":
        return replace(self, **kw)


def _residual(reference: Tensor, got: Tensor) -> float:
    """Relative L2 error, the same metric the pilot and `upcycle` use."""
    ref = reference.detach().float()
    return float((ref - got.detach().float()).norm() / ref.norm().clamp_min(1e-12))


class BranchMoEFFN(nn.Module):
    """`n_experts` complete swiGLU FFNs with a top-k router, plus shared experts.

    Duck-types the parts of `MoEFFN`'s interface the experiment harness uses:
    `expert_outputs`, `router_scores`, `n_experts`, `last_stats`,
    `total_aux_loss()` and a `router` module.
    """

    def __init__(self, gate: Tensor, up: Tensor, down: Tensor,
                 config: BranchConfig):
        super().__init__()
        I, H = int(gate.shape[0]), int(gate.shape[1])
        if tuple(up.shape) != (I, H) or tuple(down.shape) != (H, I):
            raise ValueError(
                f"expected gate/up (intermediate, hidden) and down "
                f"(hidden, intermediate); got {tuple(gate.shape)}, "
                f"{tuple(up.shape)}, {tuple(down.shape)}"
            )
        self.config = config
        self.intermediate = I
        self.hidden = H
        self.E = config.n_experts

        U = I if config.expert_units is None else int(config.expert_units)
        if not 1 <= U <= I:
            raise ValueError(f"expert_units={U} must be in [1, intermediate={I}]")
        if config.init == "tile" and config.n_experts * U != I:
            raise ValueError(
                f"init='tile' needs n_experts*expert_units == intermediate, "
                f"got {config.n_experts}*{U} != {I}"
            )
        self.U = U

        if config.unit_order is not None:
            order = torch.as_tensor(config.unit_order, dtype=torch.long)
            if order.numel() != I or \
                    not torch.equal(torch.sort(order).values,
                                    torch.arange(I)):
                raise ValueError("unit_order must be a permutation of all "
                                 "intermediate units")
        else:
            order = torch.arange(I)
        self.order = order

        S = U if config.shared_units is None else int(config.shared_units)
        self.S = S
        if config.n_shared and not 1 <= S <= I:
            raise ValueError(f"shared_units={S} must be in [1, intermediate={I}]")

        # The dense layer's own weights, kept as the exact reference.
        self.register_buffer("dense_gate", gate.detach().clone(), persistent=False)
        self.register_buffer("dense_up", up.detach().clone(), persistent=False)
        self.register_buffer("dense_down", down.detach().clone(), persistent=False)

        def make_slices(n: int, width: int, tiled: bool) -> list[Tensor]:
            out = []
            for j in range(n):
                a = j * width if tiled else 0
                out.append(order[a:a + width])
            return out

        self.expert_slices = make_slices(self.E, U, config.init == "tile")
        self.shared_slices = make_slices(config.n_shared, S, tiled=False)

        # Stacked expert weights: (n_experts, units, hidden) / (n_experts, hidden, units)
        self.gate = nn.Parameter(self._stack(gate, self.expert_slices))
        self.up = nn.Parameter(self._stack(up, self.expert_slices))
        self.down = nn.Parameter(self._stack_down(down, self.expert_slices))
        if config.n_shared:
            self.s_gate = nn.Parameter(self._stack(gate, self.shared_slices))
            self.s_up = nn.Parameter(self._stack(up, self.shared_slices))
            self.s_down = nn.Parameter(self._stack_down(down, self.shared_slices))

        if config.init_noise:
            g = torch.Generator().manual_seed(config.seed)
            for p in (self.gate, self.up, self.down):
                p.data.add_(torch.randn(p.shape, generator=g) * config.init_noise
                            * p.data.std())
            if config.n_shared:
                for p in (self.s_gate, self.s_up, self.s_down):
                    p.data.add_(torch.randn(p.shape, generator=g)
                                * config.init_noise * p.data.std())

        if config.router_hidden:
            self.router: nn.Module = nn.Sequential(
                nn.Linear(H, config.router_hidden),
                nn.SiLU(),
                nn.Linear(config.router_hidden, self.E),
            )
        else:
            self.router = nn.Linear(H, self.E)
        self.router = self.router.to(device=gate.device, dtype=torch.float32)

        # Set to an expert index to bypass the router and send every token to
        # that expert at weight 1.0. This is the mechanism for **independent**
        # expert training (branch-train-miX stage 2): an expert can be trained
        # on its own data alone, before a router exists to choose between
        # experts. `None` means normal routed behaviour.
        self.force_expert: int | None = None

        self.last_aux_loss: Tensor = torch.zeros((), dtype=torch.float32)
        self.last_z_loss: Tensor = torch.zeros((), dtype=torch.float32)
        self.last_stats: dict = {}

    def set_force_expert(self, e: int | None) -> None:
        """Route every token to expert `e` (or restore routing with `None`)."""
        if e is not None and not 0 <= int(e) < self.E:
            raise ValueError(f"force_expert={e} out of range for E={self.E}")
        self.force_expert = None if e is None else int(e)

    # ---- construction ----------------------------------------------------

    @staticmethod
    def _stack(w: Tensor, slices: list[Tensor]) -> Tensor:
        return torch.stack([w[s].detach().clone() for s in slices])

    @staticmethod
    def _stack_down(w: Tensor, slices: list[Tensor]) -> Tensor:
        return torch.stack([w[:, s].detach().clone() for s in slices])

    @classmethod
    def from_hf_mlp(cls, mlp: nn.Module, config: BranchConfig) -> "BranchMoEFFN":
        missing = [n for n in ("gate_proj", "up_proj", "down_proj")
                   if not hasattr(mlp, n)]
        if missing:
            raise ValueError(f"mlp is missing {missing}; not a swiGLU MLP")
        return cls(mlp.gate_proj.weight.detach(),
                   mlp.up_proj.weight.detach(),
                   mlp.down_proj.weight.detach(), config)

    # ---- geometry --------------------------------------------------------

    @property
    def n_experts(self) -> int:
        return self.E

    @property
    def top_k(self) -> int:
        return min(self.config.top_k, self.E)

    @property
    def active_width_fraction(self) -> float:
        """Share of the intermediate axis the active routed set covers."""
        return self.top_k * self.U / self.intermediate

    @property
    def total_width_fraction(self) -> float:
        """Share of the dense FFN this layer *stores* (routed experts only)."""
        return self.E * self.U / self.intermediate

    @property
    def active_params(self) -> int:
        """Params read per token: shared experts + the top-k routed experts."""
        per_expert = 3 * self.U * self.hidden
        return (self.config.n_shared * 3 * self.S * self.hidden
                + self.top_k * per_expert)

    @property
    def stored_params(self) -> int:
        per_expert = 3 * self.U * self.hidden
        return (self.config.n_shared * 3 * self.S * self.hidden
                + self.E * per_expert
                + sum(p.numel() for p in self.router.parameters()))

    # ---- forward ---------------------------------------------------------

    def _expert_forward(self, x2: Tensor, gate: Tensor, up: Tensor,
                        down: Tensor, j: int) -> Tensor:
        h = F.silu(F.linear(x2, gate[j])) * F.linear(x2, up[j])
        return F.linear(h, down[j])

    def dense_forward(self, x: Tensor) -> Tensor:
        """The dense swiGLU layer this module was built from (exact reference)."""
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        h = F.silu(F.linear(x2, self.dense_gate)) * F.linear(x2, self.dense_up)
        return F.linear(h, self.dense_down).reshape(shape)

    def router_scores(self, x: Tensor) -> Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        return self.router(x2.float())

    def _shared_output(self, x2: Tensor) -> Tensor:
        y = torch.zeros_like(x2)
        for j in range(self.config.n_shared):
            y = y + self._expert_forward(x2, self.s_gate, self.s_up, self.s_down, j)
        return y

    def expert_outputs(self, x: Tensor) -> Tensor:
        """Every routed expert's output: (E, T, hidden). Routed experts only."""
        x2 = x.reshape(-1, x.shape[-1])
        return torch.stack([self._expert_forward(x2, self.gate, self.up,
                                                 self.down, j)
                            for j in range(self.E)])

    def _combine_weights(self, top_val: Tensor, combine: str,
                         dtype: torch.dtype) -> Tensor:
        k = top_val.shape[1]
        if combine == "sum":
            return torch.ones_like(top_val, dtype=dtype)
        if combine == "mean":
            return torch.full_like(top_val, 1.0 / k, dtype=dtype)
        if combine == "softmax":
            return torch.softmax(top_val, dim=1).to(dtype)
        raise ValueError(f"combine={combine!r}")

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.force_expert is not None:
            return self._forced_forward(x2, shape)
        logits = self.router(x2.float())
        k = self.top_k
        top_val, top_idx = torch.topk(logits, k, dim=1)

        y = self._shared_output(x2) if self.config.n_shared \
            else torch.zeros_like(x2)

        w_sel = self._combine_weights(top_val, self.config.combine, x2.dtype)
        for e in range(self.E):
            mask = top_idx == e
            tok = mask.any(dim=1)
            if not bool(tok.any()):
                continue
            ye = self._expert_forward(x2[tok], self.gate, self.up, self.down, e)
            w = (w_sel * mask.to(w_sel.dtype)).sum(dim=1).to(y.dtype)
            rows = torch.nonzero(tok, as_tuple=False).flatten()
            y.index_add_(0, rows, (ye * w[tok].unsqueeze(1)).to(y.dtype))

        self.last_aux_loss, self.last_z_loss = self._losses(logits, top_idx)
        with torch.no_grad():
            self.last_stats = self.routing_metrics(logits.detach(),
                                                   top_idx).as_dict()
        return y.reshape(shape)

    def _forced_forward(self, x2: Tensor, shape) -> Tensor:
        """Every token to one expert, weight 1.0, no router in the path.

        The router still runs (under `no_grad`) purely so the diagnostics stay
        populated, but it does not touch the output and no balancing loss is
        charged -- stage 2 exists to let experts *diverge*, and a load-balancing
        penalty would be pushing the wrong way.
        """
        e = self.force_expert
        y = self._shared_output(x2) if self.config.n_shared \
            else torch.zeros_like(x2)
        y = y + self._expert_forward(x2, self.gate, self.up, self.down,
                                     e).to(y.dtype)
        self.last_aux_loss = torch.zeros((), dtype=torch.float32)
        self.last_z_loss = torch.zeros((), dtype=torch.float32)
        with torch.no_grad():
            logits = self.router(x2.float())
            top = torch.full((x2.shape[0], 1), int(e), dtype=torch.long,
                             device=logits.device)
            stats = self.routing_metrics(logits, top).as_dict()
        stats["forced_expert"] = int(e)
        self.last_stats = stats
        return y.reshape(shape)

    # ---- losses / diagnostics -------------------------------------------

    def _losses(self, logits: Tensor, top_idx: Tensor) -> tuple[Tensor, Tensor]:
        T = logits.shape[0]
        E = self.E
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(top_idx, num_classes=E).to(probs.dtype)
        f = onehot.sum(dim=(0, 1)) / float(max(T * top_idx.shape[1], 1))
        P = probs.mean(dim=0)
        aux = E * (f * P).sum()
        z = torch.logsumexp(logits, dim=1)
        return aux, (z ** 2).mean()

    def total_aux_loss(self) -> Tensor:
        return (self.config.aux_loss_coef * self.last_aux_loss
                + self.config.z_loss_coef * self.last_z_loss)

    def aux_loss(self) -> Tensor:
        return self.last_aux_loss

    def routing_metrics(self, logits: Tensor, top_idx: Tensor):
        from .upcycle import RoutingMetrics
        T = logits.shape[0]
        E = self.E
        onehot = F.one_hot(top_idx, num_classes=E).float()
        usage = onehot.sum(dim=(0, 1)) / float(max(T * top_idx.shape[1], 1))
        u = usage.clamp_min(0)
        s = float(u.sum())
        p = (u / s if s > 0 else u).clamp_min(1e-12)
        eff = float(1.0 / (p ** 2).sum())
        mx = float(u.max() / u.mean().clamp_min(1e-12))
        ent = float(-(p * p.log()).sum() / torch.log(torch.tensor(float(E))))
        probs = torch.softmax(logits, dim=1)
        r_ent = float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()
                      / torch.log(torch.tensor(float(E))))
        return RoutingMetrics(usage.tolist(), eff, mx, ent, r_ent, int(T))

    # ---- measurement -----------------------------------------------------

    @torch.no_grad()
    def residual_report(self, x: Tensor, top_k: int | None = None) -> dict:
        """Faithfulness of the sparse layer to the dense function it replaced.

        All residuals are against the *original dense* output, which is the only
        reference that means the same thing for a tile and a replica. The
        harness's `oracle_topk_residual` compares against the sum of all
        experts, which equals the dense layer for a tile but is E x the dense
        layer for a replica, so it is not usable here.
        """
        # The experts may be bf16/fp16 while the captured activations are
        # float32; a matmul across the two dtypes raises. Match the weight
        # dtype for the forwards and promote to float32 only for the metric.
        x2 = x.reshape(-1, x.shape[-1]).to(self.gate.dtype)
        k = self.top_k if top_k is None else min(top_k, self.E)
        ref = self.dense_forward(x2).detach().float()
        outs = self.expert_outputs(x2).detach().float()       # (E, T, H)
        # Per-token error is used ONLY to rank which experts to keep; every
        # residual reported below is then the same global norm ratio, so the
        # numbers are directly comparable with each other and with the pilot's.
        # Mixing conventions here would make a tile-vs-replica table show
        # definitional differences as if they were architectural.
        #
        # Every `residual_*` below measures the **routed path only**, with the
        # always-on shared experts excluded. Shared experts contribute the same
        # additive term to all four, so including them would move the numbers
        # by an amount that has nothing to do with routing; their own
        # contribution is reported separately as `residual_shared_only`.
        err = (outs - ref.unsqueeze(0)).norm(dim=-1) / \
            ref.norm(dim=-1).clamp_min(1e-12).unsqueeze(0)
        order = err.argsort(dim=0)                           # (E, T)

        def take(n: int) -> Tensor:
            idx = order[:n].unsqueeze(-1).expand(-1, -1, outs.shape[-1])
            return outs.gather(0, idx).sum(0)

        report = {
            "residual_best_single_expert": _residual(ref, take(1)),
            "residual_oracle_topk": _residual(ref, take(k)),
            "residual_router": _residual(ref, self._routed_forward(x2)),
            "residual_all_experts_sum": _residual(ref, outs.sum(0)),
            "active_width_fraction": self.active_width_fraction,
            "stored_width_fraction": self.total_width_fraction,
        }
        if self.config.n_shared:
            report["residual_shared_only"] = _residual(
                ref, self._shared_output(x2).detach().float())
        return report

    def _routed_forward(self, x2: Tensor) -> Tensor:
        """`forward` without the always-on shared experts: the routed path."""
        logits = self.router(x2.float())
        top_val, top_idx = torch.topk(logits, self.top_k, dim=1)
        w_sel = self._combine_weights(top_val, self.config.combine, x2.dtype)
        y = torch.zeros_like(x2)
        for e in range(self.E):
            mask = top_idx == e
            tok = mask.any(dim=1)
            if not bool(tok.any()):
                continue
            ye = self._expert_forward(x2[tok], self.gate, self.up, self.down, e)
            w = (w_sel * mask.to(w_sel.dtype)).sum(dim=1).to(y.dtype)
            rows = torch.nonzero(tok, as_tuple=False).flatten()
            y.index_add_(0, rows, (ye * w[tok].unsqueeze(1)).to(y.dtype))
        return y
