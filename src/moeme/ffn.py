"""
moeme.ffn -- dense-equivalent FFN partitioning.

The core idea
-------------
A swiGLU FFN computes

    h = silu(x @ Wg^T) * (x @ Wu^T)        # h has shape (..., intermediate)
    y = h @ Wd^T                           # (..., hidden)

The intermediate axis is *separable*: if we split the intermediate index set
into disjoint groups g_0 .. g_{E-1}, then

    y = sum_e  ( silu(x @ Wg[g_e]^T) * (x @ Wu[g_e]^T) ) @ Wd[:, g_e]^T

holds **exactly**, because the down-projection is linear and sums over
intermediate units independently. That makes an "all experts active" partition
a mathematically exact decomposition of the dense layer -- the property this
module exists to exploit and to test.

An MoE drops the sum to a top-k subset, which is where approximation (and
quality loss) enters. Nothing here claims that dropping partitions preserves
quality: `oracle_topk_residual` measures the *best case* for a given k, which
is the number that must be checked before spending training compute.

Layout conventions match Hugging Face:
    Wg, Wu : (intermediate, hidden)
    Wd     : (hidden, intermediate)
so `nn.functional.linear(x, W)` gives the HF forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# partition plans
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartitionPlan:
    """Assignment of every intermediate unit to exactly one expert."""

    assignment: Tensor  # int64, shape (intermediate,) -> expert id in [0, n_experts)
    n_experts: int

    def __post_init__(self) -> None:
        if self.assignment.ndim != 1:
            raise ValueError("assignment must be 1-D over the intermediate axis")
        ids = torch.unique(self.assignment)
        if ids.numel() != self.n_experts:
            raise ValueError(
                f"plan declares {self.n_experts} experts but uses {ids.numel()}"
            )
        if int(ids.min()) != 0 or int(ids.max()) != self.n_experts - 1:
            raise ValueError("expert ids must be contiguous and 0-based")

    @property
    def intermediate(self) -> int:
        return int(self.assignment.numel())

    def expert_indices(self, device: torch.device | None = None) -> list[Tensor]:
        """Intermediate indices owned by each expert (ragged sizes allowed)."""
        a = self.assignment if device is None else self.assignment.to(device)
        return [
            torch.nonzero(a == e, as_tuple=False).flatten()
            for e in range(self.n_experts)
        ]

    def sizes(self) -> list[int]:
        return [int(i.numel()) for i in self.expert_indices()]

    def covers_all(self) -> bool:
        idx = torch.cat(self.expert_indices())
        return int(idx.numel()) == self.intermediate and \
            bool(torch.equal(torch.sort(idx).values,
                             torch.arange(self.intermediate, device=idx.device)))


def _balanced_sizes(intermediate: int, n_experts: int) -> list[int]:
    """Split `intermediate` into n_experts sizes differing by at most 1."""
    if n_experts > intermediate:
        raise ValueError("cannot have more experts than intermediate units")
    base, rem = divmod(intermediate, n_experts)
    return [base + (1 if e < rem else 0) for e in range(n_experts)]


def contiguous_plan(intermediate: int, n_experts: int) -> PartitionPlan:
    """Consecutive blocks. Cheap, but splits correlated neighbours apart only
    at block edges, which is usually the *worst* case for oracle recovery."""
    assign = torch.empty(intermediate, dtype=torch.long)
    off = 0
    for e, s in enumerate(_balanced_sizes(intermediate, n_experts)):
        assign[off:off + s] = e
        off += s
    return PartitionPlan(assign, n_experts)


def interleaved_plan(intermediate: int, n_experts: int) -> PartitionPlan:
    """Round-robin: expert e owns indices e, e+E, e+2E, ..."""
    assign = torch.arange(intermediate, dtype=torch.long) % n_experts
    return PartitionPlan(assign, n_experts)


def random_plan(intermediate: int, n_experts: int, seed: int = 0) -> PartitionPlan:
    """Deterministic random assignment subject to the balanced sizes."""
    g = torch.Generator().manual_seed(seed)
    assign = torch.empty(intermediate, dtype=torch.long)
    off = 0
    for e, s in enumerate(_balanced_sizes(intermediate, n_experts)):
        assign[off:off + s] = e
        off += s
    assign = assign[torch.randperm(intermediate, generator=g)]
    return PartitionPlan(assign, n_experts)


def clustered_plan(gate: Tensor, up: Tensor, n_experts: int,
                   iters: int = 25, seed: int = 0) -> PartitionPlan:
    """Cluster intermediate rows of [Wg; Wu] with k-means.

    This is the interesting non-trivial plan: if intermediate units that behave
    similarly are grouped together, a top-k selection has a chance of capturing
    correlated mass, so oracle recovery can beat the contiguous baseline. We
    implement it exactly to be able to measure whether that hope is real.
    """
    if gate.shape[0] != up.shape[0]:
        raise ValueError("gate/up must share the intermediate axis")
    X = torch.cat([gate, up], dim=1).float()
    n = X.shape[0]
    if n_experts > n:
        raise ValueError("cannot have more experts than intermediate units")

    # generator must match X's device, otherwise torch.multinomial rejects it
    g = torch.Generator(device=X.device).manual_seed(seed)
    # k-means++ style spread init
    first = int(torch.randint(n, (1,), generator=g, device=X.device).item())
    centres = [X[first]]
    d = torch.cdist(X, centres[0][None]).flatten()
    for _ in range(1, n_experts):
        p = d.clamp_min(0) ** 2
        p = p / p.sum()
        nxt = int(torch.multinomial(p, 1, generator=g).item())
        centres.append(X[nxt])
        d = torch.minimum(d, torch.cdist(X, centres[-1][None]).flatten())
    C = torch.stack(centres)

    assign = torch.zeros(n, dtype=torch.long, device=X.device)
    for _ in range(iters):
        dist = torch.cdist(X, C)
        new_assign = dist.argmin(dim=1)
        if torch.equal(new_assign, assign) and _ > 0:
            assign = new_assign
            break
        assign = new_assign
        for e in range(n_experts):
            m = assign == e
            if bool(m.any()):
                C[e] = X[m].mean(dim=0)
    # repair empty clusters so the plan is valid
    for e in range(n_experts):
        if not bool((assign == e).any()):
            counts = torch.bincount(assign, minlength=n_experts)
            donor = int(counts.argmax())
            assign[int(torch.nonzero(assign == donor)[0])] = e
    # compact ids
    uniq, remap = torch.unique(assign, return_inverse=True)
    if uniq.numel() != n_experts:
        raise RuntimeError("clustering produced fewer distinct clusters than requested")
    return PartitionPlan(remap.to(torch.long), n_experts)


# ---------------------------------------------------------------------------
# dense reference + partitioned forward
# ---------------------------------------------------------------------------

@dataclass
class DenseFFN:
    """A plain swiGLU FFN with Hugging Face weight layout."""

    gate: Tensor  # (intermediate, hidden)
    up: Tensor    # (intermediate, hidden)
    down: Tensor  # (hidden, intermediate)

    def __post_init__(self) -> None:
        if self.gate.shape != self.up.shape:
            raise ValueError("gate and up must have the same shape")
        if self.down.shape != (self.gate.shape[1], self.gate.shape[0]):
            raise ValueError(
                f"down must be (hidden, intermediate) = "
                f"{(self.gate.shape[1], self.gate.shape[0])}, got {tuple(self.down.shape)}"
            )

    @property
    def hidden(self) -> int:
        return int(self.gate.shape[1])

    @property
    def intermediate(self) -> int:
        return int(self.gate.shape[0])

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_units(x, slice(None))

    def _forward_units(self, x: Tensor, idx) -> Tensor:
        gg, uu = self.gate[idx], self.up[idx]
        h = F.silu(F.linear(x, gg)) * F.linear(x, uu)
        return F.linear(h, self.down[:, idx])

    def params(self) -> int:
        return int(self.gate.numel() + self.up.numel() + self.down.numel())


class PartitionedFFN:
    """A DenseFFN viewed as `n_experts` experts under a PartitionPlan."""

    def __init__(self, dense: DenseFFN, plan: PartitionPlan):
        if plan.intermediate != dense.intermediate:
            raise ValueError(
                f"plan covers {plan.intermediate} units but FFN has "
                f"{dense.intermediate}"
            )
        self.dense = dense
        self.plan = plan
        # Plans may be built on the CPU (contiguous/interleaved/random) while the
        # weights live on the GPU. Index tensors must follow the weights.
        device = dense.gate.device
        self._indices = [i.to(device) for i in plan.expert_indices()]

    @property
    def n_experts(self) -> int:
        return self.plan.n_experts

    def expert_size(self, e: int) -> int:
        return int(self._indices[e].numel())

    def expert_params(self, e: int) -> int:
        s = self.expert_size(e)
        return 3 * s * self.dense.hidden

    def all_active_forward(self, x: Tensor) -> Tensor:
        """Sum over *every* expert -- must equal DenseFFN.forward to fp tolerance."""
        idx = self._indices[0]
        out = self.dense._forward_units(x, idx)
        for e in range(1, self.n_experts):
            out = out + self.dense._forward_units(x, self._indices[e])
        return out

    def expert_outputs(self, x: Tensor) -> Tensor:
        """Per-expert contributions, shape (n_experts, *x.shape[:-1], hidden).

        Materialising all of them is the *oracle* setting: a real router cannot
        afford this, which is precisely why the oracle number is an upper bound.
        """
        ys = [self.dense._forward_units(x, self._indices[e])
              for e in range(self.n_experts)]
        return torch.stack(ys, dim=0)

    def topk_forward(self, x: Tensor, topk: Tensor, weights: Tensor | None = None) -> Tensor:
        """Sum only the selected experts.

        topk    : (n_tokens, k) expert ids
        weights : (n_tokens, k) optional combination weights, default 1.0
        """
        if topk.ndim != 2:
            raise ValueError("topk must be (n_tokens, k)")
        if int(topk.max()) >= self.n_experts:
            raise ValueError("topk contains an out-of-range expert id")
        outs = self.expert_outputs(x)               # (E, T, hidden)
        E, T, H = outs.shape
        k = topk.shape[1]
        rows = torch.arange(T, device=x.device)
        # outs[e, t, :] for each (t, j) pair -> (T, k, hidden)
        vals = torch.stack([outs[topk[:, j], rows, :] for j in range(k)], dim=1)
        if weights is not None:
            vals = vals * weights.reshape(T, k, 1).to(vals.dtype)
        return vals.sum(dim=1)

    def dense_equivalent_error(self, x: Tensor) -> float:
        """Relative L2 error of the all-active partition vs the dense layer."""
        ref = self.dense.forward(x)
        got = self.all_active_forward(x)
        return float((ref - got).norm() / ref.norm().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# oracle + router-free diagnostics
# ---------------------------------------------------------------------------

def oracle_topk_residual(expert_outputs: Tensor, k: int) -> dict:
    """Best possible top-k relative residual.

    expert_outputs : (E, T, hidden) per-expert contributions
    Returns per-token relative L2 residual ||y - y_hat_k|| / ||y|| and summary
    statistics. This is the ceiling any learned router is chasing: if it is bad,
    no router can be good, and the design must change (or be trained).
    """
    E, T, H = expert_outputs.shape
    if not 1 <= k <= E:
        raise ValueError(f"k={k} out of range for {E} experts")
    # diagnostics run in float32 regardless of the activation dtype: bf16/fp16
    # residuals would both lose precision and break `quantile` (which requires
    # float or double)
    eo = expert_outputs.float()
    total = eo.sum(dim=0)                                   # (T, hidden)
    norms = eo.norm(dim=2)                                  # (E, T)
    topk = torch.topk(norms, k, dim=0).indices              # (k, T)
    mask = torch.zeros_like(norms, dtype=torch.bool)
    mask.scatter_(0, topk, True)
    kept = (eo * mask.unsqueeze(-1)).sum(dim=0)
    resid = (total - kept).norm(dim=1) / total.norm(dim=1).clamp_min(1e-12)
    return {
        "k": k,
        "n_experts": E,
        "active_fraction": k / E,
        "mean_rel_residual": float(resid.mean()),
        "median_rel_residual": float(resid.median()),
        "p90_rel_residual": float(resid.quantile(0.9)),
        "max_rel_residual": float(resid.max()),
        "rms_rel_residual": float(resid.pow(2).mean().sqrt()),
    }


def contribution_concentration(expert_outputs: Tensor) -> dict:
    """How concentrated is the per-expert output energy?

    If a trained dense FFN has no natural expert structure, energy is spread
    almost uniformly across partitions and top-k cannot recover it. Measured by
    the participation ratio (1 / sum p_i^2), the effective expert count, and the
    share held by the single largest contributor.
    """
    energy = expert_outputs.float().pow(2).sum(dim=(1, 2))  # (E,)
    p = energy / energy.sum().clamp_min(1e-30)
    pr = 1.0 / p.pow(2).sum().clamp_min(1e-30)
    return {
        "n_experts": int(expert_outputs.shape[0]),
        "participation_ratio": float(pr),
        "top1_energy_share": float(p.max()),
        "entropy_normalised": float(
            -(p * p.clamp_min(1e-30).log()).sum() / torch.log(
                torch.tensor(float(expert_outputs.shape[0]))
            )
        ),
    }


def train_linear_router(features: Tensor, targets: Tensor, n_experts: int,
                        steps: int = 400, lr: float = 0.05,
                        seed: int = 0) -> Tensor:
    """Fit a linear E-way router by cross-entropy on oracle top-1 labels.

    This is a cheap proxy for "what could a learned router extract from the
    hidden state alone, with no expert computation". It deliberately has no
    access to the per-expert outputs, so its accuracy is a lower bound on a
    properly trained (and much larger) router.
    """
    torch.manual_seed(seed)
    W = torch.zeros(features.shape[1], n_experts,
                    device=features.device, dtype=torch.float32,
                    requires_grad=True)
    b = torch.zeros(n_experts, device=features.device, requires_grad=True)
    feats = features.float()
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(feats @ W + b, targets)
        loss.backward()
        opt.step()
    return torch.cat([W.detach(), b.detach()[None]], dim=0)


def router_topk_residual(expert_outputs: Tensor, router_scores: Tensor,
                         k: int) -> dict:
    """Realistic (non-oracle) top-k residual using router scores.

    router_scores : (T, E) scores computed without seeing expert outputs.
    """
    E, T, H = expert_outputs.shape
    eo = expert_outputs.float()                             # float32 diagnostics
    total = eo.sum(dim=0)
    if router_scores.shape != (T, E):
        raise ValueError(f"router_scores must be (T,{E}), got {tuple(router_scores.shape)}")
    topk = torch.topk(router_scores, k, dim=1).indices       # (T, k)
    mask = torch.zeros(T, E, dtype=torch.bool, device=eo.device)
    mask.scatter_(1, topk, True)
    kept = (eo * mask.t().unsqueeze(-1)).sum(dim=0)
    resid = (total - kept).norm(dim=1) / total.norm(dim=1).clamp_min(1e-12)
    return {
        "k": k,
        "n_experts": E,
        "mean_rel_residual": float(resid.mean()),
        "median_rel_residual": float(resid.median()),
        "p90_rel_residual": float(resid.quantile(0.9)),
    }
