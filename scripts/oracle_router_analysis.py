#!/usr/bin/env python3
"""Measure the best greedy Top-K reconstruction bound for partitioned FFNs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from moeme.activations import load_activation_capture
from moeme.ledger import ExperimentLedger


def balanced_activation_groups(
    profiles: torch.Tensor, experts: int, group_width: int, iterations: int = 8
) -> list[torch.Tensor]:
    """Capacity-balanced cosine k-means over channel activation profiles."""

    profiles = F.normalize(profiles.float(), dim=1)
    generator = torch.Generator().manual_seed(7)
    seeds = torch.randperm(len(profiles), generator=generator)[:experts].to(profiles.device)
    centroids = profiles.index_select(0, seeds)
    assignments = torch.empty(len(profiles), dtype=torch.long, device=profiles.device)
    for _ in range(iterations):
        scores = profiles @ centroids.T
        preferences = scores.argsort(dim=1, descending=True).cpu()
        order = scores.max(dim=1).values.argsort(descending=True).cpu().tolist()
        remaining = [group_width] * experts
        assigned = torch.empty(len(profiles), dtype=torch.long)
        for channel in order:
            for expert in preferences[channel].tolist():
                if remaining[expert]:
                    assigned[channel] = expert
                    remaining[expert] -= 1
                    break
        assignments = assigned.to(profiles.device)
        centroids = torch.stack(
            [profiles[assignments == expert].mean(dim=0) for expert in range(experts)]
        )
        centroids = F.normalize(centroids, dim=1)
    return [(assignments == expert).nonzero(as_tuple=False).flatten() for expert in range(experts)]


def load_layer_weights(
    checkpoint: Path,
    layer: int,
    device: str,
    partition_mode: str,
    calibration_inputs: torch.Tensor,
    shared_groups: int,
    groups: int | None = None,
    compute_dtype: torch.dtype | None = None,
    partition_indices: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    layout = json.loads((checkpoint / "moeme-manifest.json").read_text())["layout"]
    experts_total = int(layout["routed_experts"])
    # A checkpoint stores a faithful re-parameterization of the dense FFN, so the
    # stored group count only fixes how finely its tensors were sliced. Callers
    # may request a different group count to re-partition the same dense weights
    # (e.g. derive the 16-group spec layout from a 64-group checkpoint).
    groups_total = int(groups) if groups is not None else int(layout["groups"])
    stored_shared = int(layout["shared_groups"])
    intermediate = int(layout["intermediate_size"])
    width = intermediate // groups_total
    base = f"model.language_model.layers.{layer}.mlp."
    names = []
    for expert in range(experts_total):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"{base}experts.{expert}.{projection}.weight")
    for projection in ("gate_proj", "up_proj", "down_proj"):
        names.append(f"{base}shared_expert.{projection}.weight")

    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(index[name], []).append(name)
    tensors = {}
    for filename, file_names in by_file.items():
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for name in file_names:
                tensor = handle.get_tensor(name)
                if compute_dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype=compute_dtype)
                tensors[name] = tensor.to(device)
    result = {
        "expert_gate": torch.stack(
            [tensors[f"{base}experts.{expert}.gate_proj.weight"] for expert in range(experts_total)]
        ),
        "expert_up": torch.stack(
            [tensors[f"{base}experts.{expert}.up_proj.weight"] for expert in range(experts_total)]
        ),
        "expert_down": torch.stack(
            [tensors[f"{base}experts.{expert}.down_proj.weight"] for expert in range(experts_total)]
        ),
        "shared_gate": tensors[f"{base}shared_expert.gate_proj.weight"],
        "shared_up": tensors[f"{base}shared_expert.up_proj.weight"],
        "shared_down": tensors[f"{base}shared_expert.down_proj.weight"],
        "shared_indices": torch.arange(stored_shared * width, device=device),
        "expert_indices": torch.arange(shared_groups * width, intermediate, device=device).reshape(
            groups_total - shared_groups, width
        ),
    }
    if partition_mode == "checkpoint":
        if shared_groups != stored_shared:
            raise ValueError(f"checkpoint partition has exactly {stored_shared} shared groups")
        # The safetensors staging stores the shared down projection at 1x the
        # dense value; the runtime/GGUF slot and `project_shared`'s sigmoid(0)=0.5
        # convention expect 2x. Return 2x so shared + experts reconstruct dense.
        result["shared_down"] = 2 * result["shared_down"]
        return result
    if partition_mode not in (
        "strided",
        "importance_shared",
        "importance_contiguous",
        "activation_cluster",
    ):
        raise ValueError(f"unknown partition mode: {partition_mode}")

    dense_gate = torch.cat((result["shared_gate"], *result["expert_gate"]), dim=0)
    dense_up = torch.cat((result["shared_up"], *result["expert_up"]), dim=0)
    # Stored shared_down is 1x dense (exact reconstruction check: shared + all
    # experts == dense FFN only at 1x). Concatenating without the historical 0.5
    # factor recovers the true dense down projection.
    dense_down = torch.cat((result["shared_down"], *result["expert_down"]), dim=1)
    if partition_indices is not None:
        shared = partition_indices["shared_indices"].to(device=device, dtype=torch.long)
        expert_indices = partition_indices["expert_indices"].to(device=device, dtype=torch.long)
        expected_routed = groups_total - shared_groups
        if tuple(shared.shape) != (shared_groups * width,):
            raise ValueError("fixed shared partition has the wrong shape")
        if tuple(expert_indices.shape) != (expected_routed, width):
            raise ValueError("fixed routed partition has the wrong shape")
        combined = torch.cat((shared, expert_indices.flatten()))
        if not torch.equal(combined.sort().values, torch.arange(intermediate, device=device)):
            raise ValueError("fixed partition indices are not an exact channel partition")
        routed = list(expert_indices)
    elif partition_mode == "strided":
        groups = [
            torch.arange(group, dense_gate.shape[0], 16, device=device) for group in range(16)
        ]
        shared = torch.cat(groups[:shared_groups]) if shared_groups else groups[0][:0]
        routed = groups[shared_groups:]
    else:
        activation_energy = torch.zeros(dense_gate.shape[0], dtype=torch.float32, device=device)
        sample_count = min(len(calibration_inputs), 512)
        sample_indices = torch.linspace(0, len(calibration_inputs) - 1, sample_count).long()
        sampled_inputs = calibration_inputs.index_select(0, sample_indices)
        energy_inputs = (
            calibration_inputs
            if partition_mode in ("importance_shared", "importance_contiguous")
            else sampled_inputs
        )
        hidden_samples = []
        for start in range(0, len(energy_inputs), 64):
            x = energy_inputs[start : start + 64].to(device=device, dtype=dense_gate.dtype)
            hidden = F.silu(x @ dense_gate.T) * (x @ dense_up.T)
            activation_energy += hidden.float().square().sum(dim=0)
            if partition_mode == "activation_cluster":
                hidden_samples.append(hidden.float())
        down_energy = dense_down.float().square().sum(dim=0)
        order = (activation_energy * down_energy).argsort(descending=True)
        routed_experts = groups_total - shared_groups
        shared_width = shared_groups * width
        shared = order[:shared_width]
        remaining = order[shared_width:]
        if partition_mode == "importance_shared":
            routed = [remaining[group::routed_experts] for group in range(routed_experts)]
        elif partition_mode == "importance_contiguous":
            # Contiguous importance blocks: the top groups carry the highest
            # activation x down energy, so a Top-K selection is near-optimal.
            routed = [
                remaining[group * width : (group + 1) * width] for group in range(routed_experts)
            ]
        else:
            profiles = torch.cat(hidden_samples, dim=0).index_select(1, remaining).T
            local_groups = balanced_activation_groups(profiles, routed_experts, width)
            routed = [remaining.index_select(0, group) for group in local_groups]
    return {
        "expert_gate": torch.stack([dense_gate.index_select(0, indices) for indices in routed]),
        "expert_up": torch.stack([dense_up.index_select(0, indices) for indices in routed]),
        "expert_down": torch.stack([dense_down.index_select(1, indices) for indices in routed]),
        "shared_gate": dense_gate.index_select(0, shared),
        "shared_up": dense_up.index_select(0, shared),
        # Preserve the staging/runtime shared-gate compensation contract.
        "shared_down": 2 * dense_down.index_select(1, shared),
        "shared_indices": shared,
        "expert_indices": torch.stack(routed),
    }


def project_experts(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    expanded = x.unsqueeze(0).expand(len(weights["expert_gate"]), -1, -1)
    gate = torch.bmm(expanded, weights["expert_gate"].transpose(1, 2))
    up = torch.bmm(expanded, weights["expert_up"].transpose(1, 2))
    hidden = F.silu(gate) * up
    return torch.bmm(hidden, weights["expert_down"].transpose(1, 2)).transpose(0, 1)


def project_shared(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    gate = x @ weights["shared_gate"].T
    up = x @ weights["shared_up"].T
    # The staged checkpoint doubles shared down_proj because llama.cpp applies
    # sigmoid(shared_expert_gate=0) == 0.5. Reproduce that runtime gate here.
    return 0.5 * ((F.silu(gate) * up) @ weights["shared_down"].T)


def project_simplex(values: torch.Tensor, radius: float) -> torch.Tensor:
    """Project the final axis onto the nonnegative simplex with the given sum."""

    ordered = values.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(dim=-1) - radius
    indices = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    positive = ordered - cumulative / indices > 0
    rho = positive.sum(dim=-1, keepdim=True).sub(1)
    theta = cumulative.gather(-1, rho) / (rho + 1).to(values.dtype)
    return (values - theta).clamp_min(0)


def simplex_oracle(
    experts: torch.Tensor,
    routed: torch.Tensor,
    k: int,
    *,
    radius: float,
    iterations: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate the best legal per-token Top-K mixture by exhaustive subsets."""

    device = experts.device
    combinations = torch.tensor(
        list(itertools.combinations(range(experts.shape[1]), k)),
        dtype=torch.long,
        device=device,
    )
    gram = torch.bmm(experts, experts.transpose(1, 2))
    linear = torch.einsum("beh,bh->be", experts, routed)
    subset_gram = gram[:, combinations[:, :, None], combinations[:, None, :]]
    subset_linear = linear[:, combinations]
    mixture = torch.full(subset_linear.shape, radius / k, dtype=experts.dtype, device=device)
    # The maximum absolute row sum bounds the spectral norm and avoids the
    # multi-gigabyte cuSOLVER workspace used by thousands of tiny eigendecompositions.
    lipschitz = 2 * subset_gram.abs().sum(dim=-1).amax(dim=-1).clamp_min(1e-8)
    step = lipschitz.reciprocal()
    for _ in range(iterations):
        gradient = 2 * (torch.einsum("bcij,bcj->bci", subset_gram, mixture) - subset_linear)
        mixture = project_simplex(mixture - step.unsqueeze(-1) * gradient, radius)

    objective = torch.einsum("bci,bcij,bcj->bc", mixture, subset_gram, mixture)
    objective -= 2 * torch.einsum("bci,bci->bc", mixture, subset_linear)
    best = objective.argmin(dim=1)
    batch_indices = torch.arange(len(experts), device=device)
    best_combinations = combinations[best]
    best_mixture = mixture[batch_indices, best]
    chosen = experts[batch_indices[:, None], best_combinations]
    prediction = (chosen * best_mixture.unsqueeze(-1)).sum(dim=1)
    return prediction, best_combinations, best_mixture


def analyze_layer(
    activations: Path,
    checkpoint: Path,
    layer: int,
    batch_size: int,
    device: str,
    partition_mode: str,
    router_total_scale: str,
    shared_groups: int,
    top_ks: tuple[int, ...],
    groups: int | None = None,
) -> dict[str, Any]:
    inputs = torch.from_numpy(load_activation_capture(activations / f"layer-{layer}.f32"))
    weights = load_layer_weights(
        checkpoint, layer, device, partition_mode, inputs, shared_groups, groups
    )
    ks = top_ks
    expert_count = len(weights["expert_gate"])
    if any(k > expert_count for k in ks):
        raise ValueError(f"requested Top-K exceeds {expert_count} routed experts")
    totals = {
        mode: {k: {"error2": 0.0, "target2": 0.0, "cos_dot": 0.0, "pred2": 0.0} for k in ks}
        for mode in ("greedy_equal_weight", "simplex_oracle")
    }
    selection_counts = {
        mode: {k: torch.zeros(expert_count, dtype=torch.int64) for k in ks} for mode in totals
    }

    for start in range(0, len(inputs), batch_size):
        x = inputs[start : start + batch_size].to(device=device, dtype=torch.bfloat16)
        experts = project_experts(x, weights).float()
        shared = project_shared(x, weights).float()
        routed = experts.sum(dim=1)
        target = shared + routed
        snapshots: dict[str, dict[int, torch.Tensor]] = {mode: {} for mode in totals}
        for k in ks:
            total_scale = float(k) if router_total_scale == "k" else float(router_total_scale)
            coefficient = total_scale / k
            residual = routed.clone()
            selected_sum = torch.zeros_like(routed)
            selected = torch.zeros((len(x), expert_count), dtype=torch.bool, device=device)
            for _ in range(k):
                marginal = 2 * coefficient * torch.einsum("bh,beh->be", residual, experts)
                marginal -= coefficient * coefficient * experts.square().sum(dim=2)
                marginal.masked_fill_(selected, float("-inf"))
                choice = marginal.argmax(dim=1)
                selected.scatter_(1, choice[:, None], True)
                chosen = experts[torch.arange(len(x), device=device), choice]
                selected_sum += chosen
                residual -= coefficient * chosen
            snapshots["greedy_equal_weight"][k] = shared + coefficient * selected_sum
            selection_counts["greedy_equal_weight"][k] += selected.sum(dim=0).cpu()

        # Exhaustively consider every K-subset and optimize nonnegative weights
        # that sum to K. This matches softmax(selected logits) * K, while giving
        # every token independent weights and therefore an optimistic bound for
        # any learned linear router.
        for k in ks:
            total_scale = float(k) if router_total_scale == "k" else float(router_total_scale)
            prediction, chosen, _ = simplex_oracle(experts, routed, k, radius=total_scale)
            snapshots["simplex_oracle"][k] = shared + prediction
            selection_counts["simplex_oracle"][k] += torch.bincount(
                chosen.cpu().flatten(), minlength=expert_count
            )

        target2 = target.square().sum().item()
        for mode, mode_snapshots in snapshots.items():
            for k, prediction in mode_snapshots.items():
                error = prediction - target
                totals[mode][k]["error2"] += error.square().sum().item()
                totals[mode][k]["target2"] += target2
                totals[mode][k]["cos_dot"] += (prediction * target).sum().item()
                totals[mode][k]["pred2"] += prediction.square().sum().item()

    metrics = {
        mode: {
            str(k): {
                "relative_l2": (values["error2"] / values["target2"]) ** 0.5,
                "cosine_similarity": values["cos_dot"]
                / (values["pred2"] * values["target2"]) ** 0.5,
            }
            for k, values in mode_totals.items()
        }
        for mode, mode_totals in totals.items()
    }
    return {
        "layer": layer,
        "tokens": len(inputs),
        "partition_mode": partition_mode,
        "router_total_scale": router_total_scale,
        "shared_groups": shared_groups,
        "routed_experts": expert_count,
        "metrics": metrics,
        "selection_counts": {
            mode: {str(k): counts.tolist() for k, counts in mode_counts.items()}
            for mode, mode_counts in selection_counts.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--partition-mode",
        choices=(
            "checkpoint",
            "strided",
            "importance_shared",
            "importance_contiguous",
            "activation_cluster",
        ),
        default="checkpoint",
    )
    parser.add_argument(
        "--router-total-scale",
        choices=("k", "12"),
        default="k",
        help="Total selected-expert weight: current K or a fixed 12.",
    )
    parser.add_argument("--max-top4-relative-l2", type=float, default=0.20)
    parser.add_argument("--max-top12-relative-l2", type=float, default=1e-4)
    parser.add_argument("--shared-groups", type=int, default=4)
    parser.add_argument(
        "--groups",
        type=int,
        default=None,
        help="Override the stored group count when re-partitioning dense weights.",
    )
    parser.add_argument("--top-ks", type=int, nargs="+")
    parser.add_argument("--target-k", type=int, default=4)
    parser.add_argument("--report", type=Path, default=Path(".moeme/oracle-router-analysis.json"))
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()

    configuration = {
        "activations": str(args.activations.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "layers": args.layers,
        "batch_size": args.batch_size,
        "device": args.device,
        "partition_mode": args.partition_mode,
        "router_total_scale": args.router_total_scale,
        "max_top4_relative_l2": args.max_top4_relative_l2,
        "max_top12_relative_l2": args.max_top12_relative_l2,
        "shared_groups": args.shared_groups,
        "groups": args.groups,
        "top_ks": args.top_ks,
        "target_k": args.target_k,
    }
    ledger = ExperimentLedger(args.ledger)
    run_id = ledger.start(
        "oracle-router-bound",
        json.dumps(configuration, sort_keys=True),
        configuration,
    )
    try:
        routed_experts = 16 - args.shared_groups
        top_ks = tuple(args.top_ks or (4, 6, 8, 10, 12))
        if routed_experts not in top_ks:
            top_ks = (*top_ks, routed_experts)
        layers = []
        for layer in args.layers:
            layers.append(
                analyze_layer(
                    args.activations,
                    args.checkpoint,
                    layer,
                    args.batch_size,
                    args.device,
                    args.partition_mode,
                    args.router_total_scale,
                    args.shared_groups,
                    top_ks,
                    groups=args.groups,
                )
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    except BaseException as error:
        ledger.finish(run_id, "failed", {}, error=f"{type(error).__name__}: {error}")
        raise

    failed_gates = []
    for result in layers:
        oracle = result["metrics"]["simplex_oracle"]
        target_key = str(args.target_k)
        exact_key = str(result["routed_experts"])
        if oracle[target_key]["relative_l2"] > args.max_top4_relative_l2:
            failed_gates.append(
                {
                    "layer": result["layer"],
                    "gate": f"top{args.target_k}_relative_l2",
                    "value": oracle[target_key]["relative_l2"],
                    "maximum": args.max_top4_relative_l2,
                }
            )
        if oracle[exact_key]["relative_l2"] > args.max_top12_relative_l2:
            failed_gates.append(
                {
                    "layer": result["layer"],
                    "gate": f"top{result['routed_experts']}_reconstruction",
                    "value": oracle[exact_key]["relative_l2"],
                    "maximum": args.max_top12_relative_l2,
                }
            )
    report = {"layers": layers, "failed_gates": failed_gates}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ledger.artifact(run_id, "oracle-report", args.report, bytes_count=args.report.stat().st_size)
    for result in layers:
        for mode, mode_metrics in result["metrics"].items():
            for k, values in mode_metrics.items():
                ledger.metric(
                    run_id,
                    f"layer-{result['layer']}",
                    f"{mode}_top{k}_relative_l2",
                    values["relative_l2"],
                    "ratio",
                )
    status = "failed" if failed_gates else "passed"
    ledger.finish(
        run_id, status, report, error="oracle reconstruction gate failed" if failed_gates else None
    )
    print(json.dumps(report, indent=2))
    return 1 if failed_gates else 0


if __name__ == "__main__":
    raise SystemExit(main())
