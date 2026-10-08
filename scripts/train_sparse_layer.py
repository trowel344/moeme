#!/usr/bin/env python3
"""Distill one dense FFN into a sparse MoEMe layer on captured inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import tempfile
import time
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

try:
    from scripts.oracle_router_analysis import (
        load_layer_weights,
        project_experts,
        project_shared,
        simplex_oracle,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from oracle_router_analysis import (
        load_layer_weights,
        project_experts,
        project_shared,
        simplex_oracle,
    )
from safetensors.torch import load_file, save_file

from moeme.activations import (
    ActivationTensorDataset,
    IndexedActivationDataset,
    activation_block_split_indices,
    activation_capture_paths,
    inspect_activation_capture,
)
from moeme.ledger import ExperimentLedger

COMPUTE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class EarlyStop(RuntimeError):
    def __init__(self, evidence: dict[str, object]):
        super().__init__("held-out reconstruction curve is flat or regressing above target")
        self.evidence = evidence


class TrainingInterrupted(RuntimeError):
    def __init__(self, signum: int, stage: str):
        super().__init__(f"received signal {signum} during {stage}")
        self.signum = signum
        self.stage = stage


class TrainingSessionComplete(RuntimeError):
    def __init__(self, stage: str, elapsed_seconds: float):
        super().__init__(f"session runtime budget reached during {stage}")
        self.stage = stage
        self.elapsed_seconds = elapsed_seconds


def flat_curve_stop(
    curve: list[dict],
    *,
    stage_index: int,
    minimum_step: int,
    recent_points: int,
    target_error: float,
) -> dict[str, object] | None:
    """Return auditable stop evidence only for a clearly flat high-error curve."""

    points: list[tuple[int, float]] = []
    for item in curve:
        if item.get("stage_index") != stage_index:
            continue
        try:
            step = int(item["step"])
            error = float(item["metrics"]["relative_l2"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(error):
            points.append((step, error))
    points.sort()
    if len(points) < recent_points or points[-1][0] < minimum_step:
        return None
    recent = points[-recent_points:]
    mean_step = sum(step for step, _ in recent) / len(recent)
    mean_error = sum(error for _, error in recent) / len(recent)
    denominator = sum((step - mean_step) ** 2 for step, _ in recent)
    if denominator == 0:
        return None
    slope = sum((step - mean_step) * (error - mean_error) for step, error in recent) / denominator
    best_step, best_error = min(points, key=lambda point: point[1])
    if slope < 0 or best_error <= target_error * 3:
        return None
    return {
        "reason": "flat_or_regressing_high",
        "stage_index": stage_index,
        "step": points[-1][0],
        "point_count": len(points),
        "recent_point_count": recent_points,
        "recent_slope_per_step": slope,
        "best_step": best_step,
        "best_relative_l2": best_error,
        "target_error": target_error,
    }


def resolve_compute_dtype(name: str, device: str) -> torch.dtype:
    """Choose a cloud-portable compute dtype without silently emulating BF16."""
    device_type = torch.device(device).type
    if name == "auto":
        if device_type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dtype = COMPUTE_DTYPES[name]
    if dtype == torch.bfloat16 and device_type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "this GPU does not support bfloat16; use --compute-dtype float16 "
            "(required on Kaggle T4/P100) or --compute-dtype auto"
        )
    return dtype


def autocast_context(device: str, dtype: torch.dtype):
    device_type = torch.device(device).type
    if device_type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def checkpoint_shared_groups(checkpoint: Path) -> int:
    """Shared group count stored in a partition checkpoint's manifest."""
    manifest = json.loads((checkpoint / "moeme-manifest.json").read_text())
    return int(manifest["layout"]["shared_groups"])


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: object) -> None:
    """Publish a restart state only after torch has completely written it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_safetensors_save(
    path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def activation_identity(directory: Path, layer: int) -> dict[str, object]:
    files = []
    total_tokens = 0
    width = None
    for path in activation_capture_paths(directory, layer):
        info = inspect_activation_capture(path)
        manifest_path = path.parent / "manifest.json"
        item = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            item = manifest.get("layers", {}).get(str(layer), {})
        identity = {
            "path": str(path.relative_to(directory)),
            "bytes": info["bytes"],
            "tokens": info["tokens"],
        }
        if item.get("sha256"):
            identity["sha256"] = item["sha256"]
        else:
            identity["mtime_ns"] = path.stat().st_mtime_ns
        files.append(identity)
        total_tokens += int(info["tokens"])
        width = int(info["width"])
    return {"files": files, "tokens": total_tokens, "width": width}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(directory: Path, layer: int) -> dict[str, object]:
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    prefix = f"model.language_model.layers.{layer}.mlp."
    tensor_files = sorted(
        {
            str(value)
            for name, value in (index.get("weight_map") or {}).items()
            if str(name).startswith(prefix)
        }
    )
    if not tensor_files:
        raise ValueError(f"checkpoint index contains no MLP tensors for layer {layer}")
    files = []
    for name in tensor_files:
        path = directory / name
        files.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "layer": layer}


def partition_seed_identity(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def resume_contract(args: argparse.Namespace, compute_dtype: torch.dtype) -> dict[str, object]:
    """Fields that must remain identical for an exact optimizer-state resume."""
    return {
        "activation": activation_identity(args.activations, args.layer),
        "checkpoint": checkpoint_identity(args.checkpoint, args.layer),
        "layer": args.layer,
        "partition_mode": args.partition_mode,
        "partition_indices_from": (
            {
                **partition_seed_identity(args.partition_indices_from),
            }
            if args.partition_indices_from is not None
            else None
        ),
        "shared_groups": args.shared_groups,
        "groups": args.groups,
        "top_k_schedule": args.top_k_schedule,
        "steps_per_stage": args.steps_per_stage,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "feature_learning_rate": args.feature_learning_rate,
        "router_learning_rate": args.router_learning_rate,
        "router_selection_probability": args.router_selection_probability,
        "router_loss_weight": args.router_loss_weight,
        "train_projections": args.train_projections,
        "train_shared": args.train_shared,
        "routing_strategy": args.routing_strategy,
        "validation_fraction": args.validation_fraction,
        "split_strategy": "capture-chunk-v1",
        "evaluate_every": args.evaluate_every,
        "evaluation_tokens": args.evaluation_tokens,
        "seed": args.seed,
        "compute_dtype": str(compute_dtype),
    }


def training_exit_code(passed: bool, systems_smoke: bool) -> int:
    """Keep model-quality failure distinct from a completed systems smoke run."""
    return 0 if passed or systems_smoke else 1


def expert_hidden(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    expert_gate: torch.Tensor | None = None,
    expert_up: torch.Tensor | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> torch.Tensor:
    expert_gate = weights["expert_gate"] if expert_gate is None else expert_gate
    expert_up = weights["expert_up"] if expert_up is None else expert_up
    expanded = x.unsqueeze(0).expand(expert_gate.shape[0], -1, -1)
    with autocast_context(device, compute_dtype):
        gate = torch.bmm(expanded, expert_gate.transpose(1, 2))
        up = torch.bmm(expanded, expert_up.transpose(1, 2))
    return (F.silu(gate) * up).transpose(0, 1)


@torch.no_grad()
def greedy_oracle(
    experts: torch.Tensor, routed: torch.Tensor, k: int, *, radius: float | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalable Top-K selection for large expert counts.

    `simplex_oracle` enumerates every K-subset, which is exact at 12 experts
    (C(12,4)=495) but infeasible at 48 (C(48,16)~2e12). This greedy marginal-gain
    selector is O(E*K) and returns the same contract: (prediction, chosen, mixture).
    """

    radius = float(k) if radius is None else radius
    batch, expert_count, _ = experts.shape
    device = experts.device
    coefficient = radius / k
    norm2 = experts.square().sum(dim=2)
    selected = torch.zeros((batch, expert_count), dtype=torch.bool, device=device)
    residual = routed.clone()
    chosen = torch.empty((batch, k), dtype=torch.long, device=device)
    for slot in range(k):
        marginal = 2 * coefficient * torch.einsum("bh,beh->be", residual, experts)
        marginal -= coefficient * coefficient * norm2
        marginal.masked_fill_(selected, float("-inf"))
        pick = marginal.argmax(dim=1)
        contribution = experts[torch.arange(batch, device=device), pick]
        residual = residual - coefficient * contribution
        chosen[:, slot] = pick
        selected.scatter_(1, pick[:, None], True)
    prediction = coefficient * experts[torch.arange(batch, device=device).unsqueeze(1), chosen].sum(
        dim=1
    )
    mixture = torch.full((batch, k), coefficient, device=device, dtype=experts.dtype)
    return prediction, chosen, mixture


@torch.no_grad()
def build_oracle_labels(
    inputs: torch.Tensor,
    weights: dict[str, torch.Tensor],
    ks: tuple[int, ...],
    batch_size: int,
    device: str,
    compute_dtype: torch.dtype = torch.bfloat16,
    interrupt_check: Callable[[], None] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    indices = {k: [] for k in ks}
    mixtures = {k: [] for k in ks}
    expert_count = weights["expert_gate"].shape[0]
    # The exhaustive simplex oracle is only affordable for the 12-expert layout.
    exhaustive = expert_count <= 12
    for start in range(0, len(inputs), batch_size):
        if interrupt_check is not None:
            interrupt_check()
        x = inputs[start : start + batch_size].to(device=device, dtype=compute_dtype)
        experts = project_experts(x, weights).float()
        routed = experts.sum(dim=1)
        for k in ks:
            if exhaustive:
                _, chosen, mixture = simplex_oracle(
                    experts, routed, k, radius=float(k), iterations=32
                )
            else:
                _, chosen, mixture = greedy_oracle(experts, routed, k, radius=float(k))
            indices[k].append(chosen.cpu())
            mixtures[k].append(mixture.cpu())
    return {k: {"indices": torch.cat(indices[k]), "mixture": torch.cat(mixtures[k])} for k in ks}


def build_oracle_labels_resumable(
    inputs,
    weights: dict[str, torch.Tensor],
    ks: tuple[int, ...],
    batch_size: int,
    device: str,
    compute_dtype: torch.dtype,
    cache_dir: Path,
    contract_digest: str,
    chunk_tokens: int,
    progress_callback=None,
    interrupt_check: Callable[[], None] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    contract_path = cache_dir / "contract.json"
    contract = {
        "format": "moeme-oracle-label-cache-v1",
        "contract_digest": contract_digest,
        "tokens": len(inputs),
        "ks": list(ks),
        "chunk_tokens": chunk_tokens,
    }
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError(f"oracle label cache contract mismatch: {cache_dir}")
    else:
        atomic_json(contract_path, contract)

    collected = {k: {"indices": [], "mixture": []} for k in ks}
    for chunk_index, start in enumerate(range(0, len(inputs), chunk_tokens)):
        tokens = min(chunk_tokens, len(inputs) - start)
        chunk_path = cache_dir / f"chunk-{chunk_index:05d}.safetensors"
        expected_metadata = {
            "format": "moeme-oracle-label-chunk-v1",
            "contract_digest": contract_digest,
            "start": str(start),
            "tokens": str(tokens),
            "ks": ",".join(map(str, ks)),
        }
        labels = None
        if chunk_path.exists():
            with safe_open(chunk_path, framework="pt", device="cpu") as handle:
                if handle.metadata() != expected_metadata:
                    raise ValueError(f"oracle label chunk metadata mismatch: {chunk_path}")
                labels = {
                    k: {
                        "indices": handle.get_tensor(f"k{k}_indices"),
                        "mixture": handle.get_tensor(f"k{k}_mixture"),
                    }
                    for k in ks
                }
            if any(len(labels[k]["indices"]) != tokens for k in ks):
                raise ValueError(f"oracle label chunk has the wrong row count: {chunk_path}")
        else:
            labels = build_oracle_labels(
                inputs[start : start + tokens],
                weights,
                ks,
                batch_size,
                device,
                compute_dtype,
                interrupt_check,
            )
            tensors = {}
            for k in ks:
                tensors[f"k{k}_indices"] = labels[k]["indices"]
                tensors[f"k{k}_mixture"] = labels[k]["mixture"]
            atomic_safetensors_save(chunk_path, tensors, expected_metadata)
        for k in ks:
            collected[k]["indices"].append(labels[k]["indices"])
            collected[k]["mixture"].append(labels[k]["mixture"])
        if progress_callback is not None:
            progress_callback(min(len(inputs), start + tokens), len(inputs))
    return {
        k: {
            "indices": torch.cat(collected[k]["indices"]),
            "mixture": torch.cat(collected[k]["mixture"]),
        }
        for k in ks
    }


def routing_target(
    indices: torch.Tensor, mixture: torch.Tensor, top_k: int, experts: int
) -> torch.Tensor:
    target = torch.zeros((len(indices), experts), dtype=torch.float32, device=indices.device)
    return target.scatter_(1, indices, mixture.float() / top_k)


def routing_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * logits.log_softmax(dim=1)).sum(dim=1).mean()


def distillation_loss(
    reconstruction: torch.Tensor,
    logits: torch.Tensor,
    oracle_indices: torch.Tensor,
    oracle_mixture: torch.Tensor,
    top_k: int,
    weight: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = routing_target(oracle_indices, oracle_mixture, top_k, experts=logits.shape[1])
    router_objective = routing_loss(logits, target)
    return reconstruction + weight * router_objective, router_objective


def route_utilization(counts: torch.Tensor) -> dict[str, float]:
    total = counts.sum().item()
    if total == 0:
        return {
            "route_coverage": 0.0,
            "normalized_route_entropy": 0.0,
            "maximum_route_fraction": 0.0,
        }
    probabilities = counts.double() / total
    positive = probabilities[probabilities > 0]
    entropy = -(positive * positive.log()).sum().item()
    return {
        "route_coverage": float((counts > 0).sum().item() / len(counts)),
        "normalized_route_entropy": entropy / math.log(len(counts)),
        "maximum_route_fraction": probabilities.max().item(),
    }


def forward_shared(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    shared_gate: torch.Tensor | None = None,
    shared_up: torch.Tensor | None = None,
    shared_down: torch.Tensor | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> torch.Tensor:
    if shared_gate is None or shared_up is None or shared_down is None:
        return project_shared(x, weights).float()
    with autocast_context(device, compute_dtype):
        gate = x @ shared_gate.T
        up = x @ shared_up.T
        output = 0.5 * ((F.silu(gate) * up) @ shared_down.T)
    return output.float()


def forward_sparse(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    expert_down: torch.Tensor,
    router: torch.Tensor,
    top_k: int,
    expert_gate: torch.Tensor | None = None,
    expert_up: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
    shared_up: torch.Tensor | None = None,
    shared_down: torch.Tensor | None = None,
    chosen_override: torch.Tensor | None = None,
    scales_override: torch.Tensor | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = expert_hidden(x, weights, expert_gate, expert_up, compute_dtype, device)
    with autocast_context(device, compute_dtype):
        outputs = torch.einsum("bei,eoi->beo", hidden, expert_down)
    logits = x.float() @ router.T
    if chosen_override is None:
        values, chosen = logits.topk(top_k, dim=1)
        scales = values.softmax(dim=1) * top_k
    else:
        if scales_override is None:
            raise ValueError("scales_override is required with chosen_override")
        chosen = chosen_override
        scales = scales_override
    selected = outputs.gather(1, chosen.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]))
    routed = (selected.float() * scales.unsqueeze(-1)).sum(dim=1)
    return (
        forward_shared(
            x,
            weights,
            shared_gate,
            shared_up,
            shared_down,
            compute_dtype,
            device,
        )
        + routed,
        logits,
        chosen,
    )


@torch.no_grad()
def evaluate(
    inputs: torch.Tensor,
    labels: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    expert_down: torch.Tensor,
    router: torch.Tensor,
    top_k: int,
    batch_size: int,
    device: str,
    expert_gate: torch.Tensor | None = None,
    expert_up: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
    shared_up: torch.Tensor | None = None,
    shared_down: torch.Tensor | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    interrupt_check: Callable[[], None] | None = None,
) -> dict[str, float]:
    error2 = target2 = dot = pred2 = 0.0
    oracle_error2 = oracle_dot = oracle_pred2 = 0.0
    overlap = 0
    route_counts = torch.zeros(weights["expert_gate"].shape[0], dtype=torch.int64)
    for start in range(0, len(inputs), batch_size):
        if interrupt_check is not None:
            interrupt_check()
        x = inputs[start : start + batch_size].to(device=device, dtype=compute_dtype)
        original_experts = project_experts(x, weights).float()
        target = project_shared(x, weights).float() + original_experts.sum(dim=1)
        prediction, _, chosen = forward_sparse(
            x,
            weights,
            expert_down,
            router,
            top_k,
            expert_gate,
            expert_up,
            shared_gate,
            shared_up,
            shared_down,
            compute_dtype=compute_dtype,
            device=device,
        )
        expected = labels["indices"][start : start + len(x)].to(device)
        expected_scales = labels["mixture"][start : start + len(x)].to(device)
        route_counts += torch.bincount(
            chosen.detach().cpu().flatten(), minlength=route_counts.numel()
        )
        oracle_prediction, _, _ = forward_sparse(
            x,
            weights,
            expert_down,
            router,
            top_k,
            expert_gate,
            expert_up,
            shared_gate,
            shared_up,
            shared_down,
            expected,
            expected_scales,
            compute_dtype,
            device,
        )
        error2 += (prediction - target).square().sum().item()
        oracle_error2 += (oracle_prediction - target).square().sum().item()
        target2 += target.square().sum().item()
        dot += (prediction * target).sum().item()
        pred2 += prediction.square().sum().item()
        oracle_dot += (oracle_prediction * target).sum().item()
        oracle_pred2 += oracle_prediction.square().sum().item()
        overlap += (chosen.unsqueeze(2) == expected.unsqueeze(1)).any(dim=2).sum().item()
    return {
        "relative_l2": (error2 / target2) ** 0.5,
        "cosine_similarity": dot / (pred2 * target2) ** 0.5,
        "oracle_selection_recall": overlap / (len(inputs) * top_k),
        "teacher_forced_relative_l2": (oracle_error2 / target2) ** 0.5,
        "teacher_forced_cosine_similarity": oracle_dot / (oracle_pred2 * target2) ** 0.5,
        **route_utilization(route_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition-mode", default="importance_shared")
    parser.add_argument(
        "--shared-groups",
        type=int,
        default=None,
        help="Shared expert width in groups; defaults to the checkpoint layout.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=None,
        help="Override the stored group count when re-partitioning dense weights.",
    )
    parser.add_argument("--top-k-schedule", type=int, nargs="+", default=[10, 8, 6, 4])
    parser.add_argument("--steps-per-stage", type=int, default=50)
    parser.add_argument("--router-warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--feature-learning-rate", type=float, default=2e-6)
    parser.add_argument("--router-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--router-selection-probability",
        type=float,
        default=0.0,
        help=(
            "Fraction of oracle-routed steps that instead reconstruct the router's own "
            "Top-K selection (robustness to router mistakes)."
        ),
    )
    parser.add_argument(
        "--router-loss-weight",
        type=float,
        default=0.01,
        help="Weight of the router cross-entropy term against the oracle labels.",
    )
    parser.add_argument("--train-projections", choices=("down", "all"), default="down")
    parser.add_argument("--train-shared", action="store_true")
    parser.add_argument(
        "--routing-strategy",
        choices=("oracle", "random-generalist"),
        default="oracle",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-validation-relative-l2", type=float, default=0.20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--allow-partial-resume",
        action="store_true",
        help="Keep dense initial values for trainable tensors omitted by a weight-only seed.",
    )
    parser.add_argument(
        "--partition-indices-from",
        type=Path,
        help="Safetensors seed whose shared/expert indices freeze the channel partition.",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="Exact training-state resume, including optimizer, RNG and completed step.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        help="Atomic restart state path (default: OUTPUT_DIR/training-state.pt).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument(
        "--state-every",
        type=int,
        default=0,
        help="Atomically overwrite the exact restart state every N steps; 0 disables it.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=0,
        help="Run a bounded held-out evaluation every N training steps; 0 disables it.",
    )
    parser.add_argument(
        "--evaluation-tokens",
        type=int,
        default=2048,
        help="Maximum held-out tokens used for each periodic learning-curve point.",
    )
    parser.add_argument(
        "--early-stop-flat",
        action="store_true",
        help="Stop after a saved evaluation when the recent high-error curve is flat.",
    )
    parser.add_argument("--early-stop-min-step", type=int, default=15000)
    parser.add_argument("--early-stop-recent-points", type=int, default=4)
    parser.add_argument("--early-stop-target-error", type=float, default=0.01)
    parser.add_argument(
        "--label-cache-dir",
        type=Path,
        help="Chunked oracle-label cache (default: OUTPUT_DIR/oracle-labels).",
    )
    parser.add_argument("--label-chunk-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Training math dtype. Auto uses BF16 when supported, FP16 on T4/P100, FP32 on CPU.",
    )
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help=(
            "Cleanly checkpoint and exit 75 after this wall time; 0 disables the "
            "session budget. Use less than the provider hard limit."
        ),
    )
    parser.add_argument(
        "--systems-smoke",
        action="store_true",
        help=(
            "Return success after a finite completed run even when the scientific "
            "quality gate fails. The report still records passed=false."
        ),
    )
    args = parser.parse_args()
    if args.resume is not None and args.resume_state is not None:
        parser.error("--resume and --resume-state are mutually exclusive")
    if args.allow_partial_resume and args.resume is None:
        parser.error("--allow-partial-resume requires --resume")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.evaluate_every < 0:
        parser.error("--evaluate-every cannot be negative")
    if args.state_every < 0:
        parser.error("--state-every cannot be negative")
    if args.evaluation_tokens <= 0:
        parser.error("--evaluation-tokens must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between zero and one")
    if args.early_stop_min_step <= 0:
        parser.error("--early-stop-min-step must be positive")
    if args.early_stop_recent_points < 2:
        parser.error("--early-stop-recent-points must be at least 2")
    if args.early_stop_target_error <= 0:
        parser.error("--early-stop-target-error must be positive")
    if args.label_chunk_tokens <= 0:
        parser.error("--label-chunk-tokens must be positive")
    if args.max_runtime_seconds < 0:
        parser.error("--max-runtime-seconds cannot be negative")

    session_started = time.monotonic()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    compute_dtype = resolve_compute_dtype(args.compute_dtype, args.device)
    state_path = args.state_path or (args.output_dir / "training-state.pt")
    label_cache_dir = args.label_cache_dir or (args.output_dir / "oracle-labels")
    contract = resume_contract(args, compute_dtype)
    configuration = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "ledger"
    }
    configuration["top_k_schedule"] = args.top_k_schedule
    ledger = ExperimentLedger(args.ledger)
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    run_id = ledger.start("sparse-layer-distillation", digest, configuration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    termination = {"signum": None}
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def request_termination(signum, _frame) -> None:
        termination["signum"] = signum

    for signum in previous_handlers:
        signal.signal(signum, request_termination)

    def progress(phase: str, **details: object) -> None:
        atomic_json(
            progress_path,
            {
                "format": "moeme-sparse-layer-progress-v1",
                "layer": args.layer,
                "phase": phase,
                "updated_at": datetime.now(UTC).isoformat(),
                **details,
            },
        )

    def check_session_budget(stage: str) -> None:
        elapsed = time.monotonic() - session_started
        if args.max_runtime_seconds > 0 and elapsed >= args.max_runtime_seconds:
            raise TrainingSessionComplete(stage, elapsed)

    try:
        progress("loading_activations")
        inputs = ActivationTensorDataset.from_directory(args.activations, args.layer)
        validation_indices, training_indices = activation_block_split_indices(
            inputs, args.validation_fraction, args.seed
        )
        validation_inputs = IndexedActivationDataset(inputs, validation_indices)
        training_inputs = IndexedActivationDataset(inputs, training_indices)
        progress("loading_weights", training_tokens=len(training_inputs))
        shared_groups = (
            args.shared_groups
            if args.shared_groups is not None
            else checkpoint_shared_groups(args.checkpoint)
        )
        progress("loading_weights", shared_groups=shared_groups)
        partition_indices = None
        if args.partition_indices_from is not None:
            index_seed = load_file(args.partition_indices_from, device="cpu")
            required_indices = {"shared_indices", "expert_indices"}
            if not required_indices.issubset(index_seed):
                raise ValueError("partition index seed omits shared_indices or expert_indices")
            partition_indices = {name: index_seed[name] for name in required_indices}
            del index_seed
        weights = load_layer_weights(
            args.checkpoint,
            args.layer,
            args.device,
            args.partition_mode,
            training_inputs,
            shared_groups,
            args.groups,
            compute_dtype,
            partition_indices,
        )
        ks = tuple(dict.fromkeys((*args.top_k_schedule, 4)))
        label_contract = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        progress("building_training_labels")

        def label_progress(phase: str):
            def update(completed: int, total: int) -> None:
                progress(phase, completed_tokens=completed, total_tokens=total)
                if termination["signum"] is not None:
                    raise TrainingInterrupted(int(termination["signum"]), phase)
                check_session_budget(phase)

            return update

        def label_interrupt(phase: str) -> None:
            if termination["signum"] is not None:
                raise TrainingInterrupted(int(termination["signum"]), phase)
            check_session_budget(phase)

        training_labels = build_oracle_labels_resumable(
            training_inputs,
            weights,
            ks,
            args.batch_size,
            args.device,
            compute_dtype,
            label_cache_dir / "training",
            f"{label_contract}:training",
            args.label_chunk_tokens,
            label_progress("building_training_labels"),
            lambda: label_interrupt("building_training_labels"),
        )
        progress("building_validation_labels")
        validation_labels = build_oracle_labels_resumable(
            validation_inputs,
            weights,
            ks,
            args.batch_size,
            args.device,
            compute_dtype,
            label_cache_dir / "validation",
            f"{label_contract}:validation",
            args.label_chunk_tokens,
            label_progress("building_validation_labels"),
            lambda: label_interrupt("building_validation_labels"),
        )

        router = torch.nn.Parameter(
            torch.zeros(
                (weights["expert_gate"].shape[0], training_inputs.shape[1]),
                device=args.device,
            )
        )
        warmup = torch.optim.AdamW([router], lr=args.router_learning_rate, weight_decay=1e-4)
        final_k = args.top_k_schedule[-1]
        if args.routing_strategy == "oracle" and args.resume_state is None and args.resume is None:
            for _ in range(args.router_warmup_steps):
                check_session_budget("router_warmup")
                chosen = torch.randint(0, len(training_inputs), (args.batch_size,))
                x = training_inputs[chosen].to(args.device).float()
                target = routing_target(
                    training_labels[final_k]["indices"][chosen].to(args.device),
                    training_labels[final_k]["mixture"][chosen].to(args.device),
                    final_k,
                    weights["expert_gate"].shape[0],
                )
                loss = routing_loss(x @ router.T, target)
                warmup.zero_grad(set_to_none=True)
                loss.backward()
                warmup.step()

        expert_down = torch.nn.Parameter(weights["expert_down"].float().clone())
        expert_gate = (
            torch.nn.Parameter(weights["expert_gate"].float().clone())
            if args.train_projections == "all"
            else None
        )
        expert_up = (
            torch.nn.Parameter(weights["expert_up"].float().clone())
            if args.train_projections == "all"
            else None
        )
        shared_gate = (
            torch.nn.Parameter(weights["shared_gate"].float().clone())
            if args.train_shared
            else None
        )
        shared_up = (
            torch.nn.Parameter(weights["shared_up"].float().clone()) if args.train_shared else None
        )
        shared_down = (
            torch.nn.Parameter(weights["shared_down"].float().clone())
            if args.train_shared
            else None
        )
        if args.resume is not None:
            resumed = load_file(args.resume, device="cpu")
            if not torch.equal(resumed["shared_indices"], weights["shared_indices"].cpu()):
                raise ValueError("resume shared partition does not match this campaign")
            if not torch.equal(resumed["expert_indices"], weights["expert_indices"].cpu()):
                raise ValueError("resume routed partition does not match this campaign")
            with torch.no_grad():
                expert_down.copy_(resumed["expert_down"])
                router.copy_(resumed["router"])
                if expert_gate is not None and expert_up is not None:
                    missing_features = {"expert_gate", "expert_up"} - set(resumed)
                    if missing_features and not args.allow_partial_resume:
                        raise ValueError(
                            "full-projection training requires gate/up in resume state"
                        )
                    if not missing_features:
                        expert_gate.copy_(resumed["expert_gate"])
                        expert_up.copy_(resumed["expert_up"])
                if shared_gate is not None and shared_up is not None and shared_down is not None:
                    required = {"shared_gate", "shared_up", "shared_down"}
                    if not required.issubset(resumed):
                        raise ValueError("shared training requires shared tensors in resume state")
                    shared_gate.copy_(resumed["shared_gate"])
                    shared_up.copy_(resumed["shared_up"])
                    shared_down.copy_(resumed["shared_down"])
            del resumed
        trained_projections = [expert_down]
        optimizer_groups = [{"params": [expert_down], "lr": args.learning_rate}]
        if expert_gate is not None and expert_up is not None:
            trained_projections.extend((expert_gate, expert_up))
            optimizer_groups.append(
                {
                    "params": [expert_gate, expert_up],
                    "lr": args.feature_learning_rate,
                }
            )
        if shared_gate is not None and shared_up is not None and shared_down is not None:
            trained_projections.extend((shared_gate, shared_up, shared_down))
            optimizer_groups.extend(
                (
                    {"params": [shared_down], "lr": args.learning_rate},
                    {
                        "params": [shared_gate, shared_up],
                        "lr": args.feature_learning_rate,
                    },
                )
            )
        optimizer = torch.optim.AdamW(
            [
                *optimizer_groups,
                {"params": [router], "lr": args.router_learning_rate * 0.1},
            ],
            weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(torch.device(args.device).type == "cuda" and compute_dtype == torch.float16),
        )

        start_stage_index = 0
        resumed_stage_step = 0
        stages = []
        curve = []
        if args.resume_state is not None:
            # Loading a full FP32 model + Adam state directly onto an 8 GB T4
            # temporarily duplicates already-allocated parameters and can OOM
            # before restore begins. CPU loading preserves exact values while
            # copy_/load_state_dict move them into their owned GPU allocations.
            state = torch.load(args.resume_state, map_location="cpu", weights_only=True)
            if state.get("format") != "moeme-sparse-training-state-v1":
                raise ValueError("unsupported resume-state format")
            if state.get("contract") != contract:
                raise ValueError("resume-state contract does not match this training invocation")
            resumed = state["model"]
            if not torch.equal(resumed["shared_indices"], weights["shared_indices"].cpu()):
                raise ValueError("resume-state shared partition does not match this campaign")
            if not torch.equal(resumed["expert_indices"], weights["expert_indices"].cpu()):
                raise ValueError("resume-state routed partition does not match this campaign")
            with torch.no_grad():
                expert_down.copy_(resumed["expert_down"])
                router.copy_(resumed["router"])
                if expert_gate is not None and expert_up is not None:
                    expert_gate.copy_(resumed["expert_gate"])
                    expert_up.copy_(resumed["expert_up"])
                if shared_gate is not None and shared_up is not None and shared_down is not None:
                    shared_gate.copy_(resumed["shared_gate"])
                    shared_up.copy_(resumed["shared_up"])
                    shared_down.copy_(resumed["shared_down"])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("grad_scaler") is not None:
                scaler.load_state_dict(state["grad_scaler"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if torch.device(args.device).type == "cuda" and state.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            random.setstate(state["python_random_state"])
            start_stage_index = int(state["stage_index"])
            resumed_stage_step = int(state["stage_step"])
            stages = list(state.get("completed_stages", []))
            curve = list(state.get("curve", []))
            progress(
                "resumed",
                state_path=str(args.resume_state.resolve()),
                stage_index=start_stage_index,
                stage_step=resumed_stage_step,
            )
            del resumed, state

        def saved_tensors(*, preserve_precision: bool = False) -> dict[str, torch.Tensor]:
            def projection(tensor: torch.Tensor) -> torch.Tensor:
                tensor = tensor.detach().cpu()
                return tensor if preserve_precision else tensor.to(dtype=torch.bfloat16)

            tensors = {
                "expert_down": projection(expert_down),
                "router": router.detach().cpu(),
                "shared_indices": weights["shared_indices"].detach().cpu(),
                "expert_indices": weights["expert_indices"].detach().cpu(),
            }
            if expert_gate is not None and expert_up is not None:
                tensors["expert_gate"] = projection(expert_gate)
                tensors["expert_up"] = projection(expert_up)
            if shared_gate is not None and shared_up is not None and shared_down is not None:
                tensors["shared_gate"] = projection(shared_gate)
                tensors["shared_up"] = projection(shared_up)
                tensors["shared_down"] = projection(shared_down)
            return tensors

        def save_training_state(stage_index: int, stage_step: int) -> None:
            atomic_torch_save(
                state_path,
                {
                    "format": "moeme-sparse-training-state-v1",
                    "contract": contract,
                    "stage_index": stage_index,
                    "stage_step": stage_step,
                    "completed_stages": stages,
                    "curve": curve,
                    "model": saved_tensors(preserve_precision=True),
                    "optimizer": optimizer.state_dict(),
                    "grad_scaler": scaler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if torch.device(args.device).type == "cuda"
                        else None
                    ),
                    "python_random_state": random.getstate(),
                },
            )

        def interrupt_evaluation(stage: str, stage_index: int, stage_step: int) -> None:
            if termination["signum"] is not None:
                save_training_state(stage_index, stage_step)
                raise TrainingInterrupted(int(termination["signum"]), stage)
            try:
                check_session_budget(stage)
            except TrainingSessionComplete:
                save_training_state(stage_index, stage_step)
                raise

        if args.evaluate_every > 0 and not curve and start_stage_index < len(args.top_k_schedule):
            initial_top_k = args.top_k_schedule[start_stage_index]
            evaluation_count = min(args.evaluation_tokens, len(validation_inputs))
            initial_labels = {
                name: values[:evaluation_count]
                for name, values in validation_labels[initial_top_k].items()
            }
            initial_metrics = evaluate(
                validation_inputs[:evaluation_count],
                initial_labels,
                weights,
                expert_down,
                router,
                initial_top_k,
                args.batch_size,
                args.device,
                expert_gate,
                expert_up,
                shared_gate,
                shared_up,
                shared_down,
                compute_dtype,
                lambda: interrupt_evaluation(
                    "initial_evaluation", start_stage_index, resumed_stage_step
                ),
            )
            curve.append(
                {
                    "stage_index": start_stage_index,
                    "top_k": initial_top_k,
                    "step": resumed_stage_step,
                    "tokens": evaluation_count,
                    "metrics": initial_metrics,
                    "initial": True,
                }
            )
            atomic_json(
                args.output_dir / "curve.json",
                {"format": "moeme-sparse-learning-curve-v1", "points": curve},
            )
            progress(
                "initial_evaluation",
                top_k=initial_top_k,
                step=resumed_stage_step,
                periodic_validation=initial_metrics,
            )
            save_training_state(start_stage_index, resumed_stage_step)
            if termination["signum"] is not None:
                raise TrainingInterrupted(int(termination["signum"]), "initial_evaluation")

        for stage_index, top_k in enumerate(args.top_k_schedule):
            if stage_index < start_stage_index:
                continue
            first_step = resumed_stage_step + 1 if stage_index == start_stage_index else 1
            progress(
                "training",
                top_k=top_k,
                step=first_step - 1,
                steps=args.steps_per_stage,
            )
            for stage_step in range(first_step, args.steps_per_stage + 1):
                chosen = torch.randint(0, len(training_inputs), (args.batch_size,))
                x = training_inputs[chosen].to(device=args.device, dtype=compute_dtype)
                with torch.no_grad():
                    original = project_experts(x, weights).float()
                    target = project_shared(x, weights).float() + original.sum(dim=1)
                if args.routing_strategy == "oracle":
                    route_indices = training_labels[top_k]["indices"][chosen].to(args.device)
                    route_mixture = training_labels[top_k]["mixture"][chosen].to(args.device)
                    # An oracle-trained router is imperfect at inference, so an
                    # oracle-specialised projection is fragile: a single wrong
                    # expert costs a whole dropped block. Training a fraction of
                    # steps on the router's own selection makes the projections
                    # robust to those mistakes without changing inference.
                    if args.router_selection_probability > 0 and (
                        random.random() < args.router_selection_probability
                    ):
                        with torch.no_grad():
                            route_indices = (x.float() @ router.T).topk(top_k, dim=1).indices
                        route_mixture = torch.ones(
                            (len(x), top_k), device=args.device, dtype=torch.float32
                        )
                else:
                    route_indices = (
                        torch.rand((len(x), weights["expert_gate"].shape[0]), device=args.device)
                        .topk(top_k, dim=1)
                        .indices
                    )
                    route_mixture = torch.ones(
                        (len(x), top_k), device=args.device, dtype=torch.float32
                    )
                prediction, logits, _ = forward_sparse(
                    x,
                    weights,
                    expert_down,
                    router,
                    top_k,
                    expert_gate,
                    expert_up,
                    shared_gate,
                    shared_up,
                    shared_down,
                    route_indices,
                    route_mixture,
                    compute_dtype,
                    args.device,
                )
                reconstruction = (prediction - target).square().mean() / target.square().mean()
                oracle_indices = training_labels[top_k]["indices"][chosen].to(args.device)
                oracle_mixture = training_labels[top_k]["mixture"][chosen].to(args.device)
                # Random-generalist training deliberately exposes every expert to
                # arbitrary subsets, but the runtime router must still learn from
                # the oracle. Without this auxiliary term, chosen_override makes
                # reconstruction independent of logits and the router stays zero.
                loss, routing = distillation_loss(
                    reconstruction,
                    logits,
                    oracle_indices,
                    oracle_mixture,
                    top_k,
                    args.router_loss_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([*trained_projections, router], 1.0)
                scaler.step(optimizer)
                scaler.update()
                if stage_step == 1 or stage_step % args.progress_every == 0:
                    progress(
                        "training",
                        top_k=top_k,
                        step=stage_step,
                        steps=args.steps_per_stage,
                        loss=float(loss.detach()),
                        reconstruction_loss=float(reconstruction.detach()),
                        routing_loss=float(routing.detach()),
                        grad_scale=scaler.get_scale(),
                    )
                if args.evaluate_every > 0 and stage_step % args.evaluate_every == 0:
                    evaluation_count = min(args.evaluation_tokens, len(validation_inputs))
                    periodic_labels = {
                        name: values[:evaluation_count]
                        for name, values in validation_labels[top_k].items()
                    }
                    metrics = evaluate(
                        validation_inputs[:evaluation_count],
                        periodic_labels,
                        weights,
                        expert_down,
                        router,
                        top_k,
                        args.batch_size,
                        args.device,
                        expert_gate,
                        expert_up,
                        shared_gate,
                        shared_up,
                        shared_down,
                        compute_dtype,
                        lambda stage_index=stage_index, stage_step=stage_step: interrupt_evaluation(
                            "periodic_evaluation", stage_index, stage_step
                        ),
                    )
                    curve.append(
                        {
                            "stage_index": stage_index,
                            "top_k": top_k,
                            "step": stage_step,
                            "tokens": evaluation_count,
                            "metrics": metrics,
                        }
                    )
                    atomic_json(
                        args.output_dir / "curve.json",
                        {"format": "moeme-sparse-learning-curve-v1", "points": curve},
                    )
                    progress(
                        "training",
                        top_k=top_k,
                        step=stage_step,
                        steps=args.steps_per_stage,
                        periodic_validation=metrics,
                    )
                    save_training_state(stage_index, stage_step)
                    if args.early_stop_flat:
                        stop = flat_curve_stop(
                            curve,
                            stage_index=stage_index,
                            minimum_step=args.early_stop_min_step,
                            recent_points=args.early_stop_recent_points,
                            target_error=args.early_stop_target_error,
                        )
                        if stop is not None:
                            atomic_json(
                                args.output_dir / "early-stop.json",
                                {
                                    "format": "moeme-sparse-early-stop-v1",
                                    "stopped_at": datetime.now(UTC).isoformat(),
                                    **stop,
                                },
                            )
                            raise EarlyStop(stop)
                elif args.state_every > 0 and stage_step % args.state_every == 0:
                    save_training_state(stage_index, stage_step)
                if args.checkpoint_every > 0 and stage_step % args.checkpoint_every == 0:
                    step_checkpoint_path = (
                        args.output_dir
                        / f"layer-{args.layer}-top{top_k}-step{stage_step}.safetensors"
                    )
                    atomic_safetensors_save(
                        step_checkpoint_path,
                        saved_tensors(),
                        metadata={
                            "format": "moeme-sparse-layer-v1",
                            "top_k": str(top_k),
                            "stage_step": str(stage_step),
                        },
                    )
                    ledger.artifact(
                        run_id,
                        f"layer-{args.layer}-top{top_k}-step{stage_step}",
                        step_checkpoint_path,
                        bytes_count=step_checkpoint_path.stat().st_size,
                    )
                if termination["signum"] is not None:
                    save_training_state(stage_index, stage_step)
                    raise TrainingInterrupted(int(termination["signum"]), "training")
                try:
                    check_session_budget("training")
                except TrainingSessionComplete:
                    save_training_state(stage_index, stage_step)
                    raise

            final_training_count = min(args.evaluation_tokens, len(training_inputs))
            final_training_labels = {
                name: values[:final_training_count]
                for name, values in training_labels[top_k].items()
            }
            progress(
                "evaluating_training",
                top_k=top_k,
                evaluation_tokens=final_training_count,
            )
            training_metrics = evaluate(
                training_inputs[:final_training_count],
                final_training_labels,
                weights,
                expert_down,
                router,
                top_k,
                args.batch_size,
                args.device,
                expert_gate,
                expert_up,
                shared_gate,
                shared_up,
                shared_down,
                compute_dtype,
                lambda stage_index=stage_index, stage_step=stage_step: interrupt_evaluation(
                    "training_evaluation", stage_index, stage_step
                ),
            )
            progress("evaluating_validation", top_k=top_k)
            validation_metrics = evaluate(
                validation_inputs,
                validation_labels[top_k],
                weights,
                expert_down,
                router,
                top_k,
                args.batch_size,
                args.device,
                expert_gate,
                expert_up,
                shared_gate,
                shared_up,
                shared_down,
                compute_dtype,
                lambda stage_index=stage_index, stage_step=stage_step: interrupt_evaluation(
                    "validation_evaluation", stage_index, stage_step
                ),
            )
            progress("publishing_checkpoint", top_k=top_k)
            checkpoint_path = args.output_dir / f"layer-{args.layer}-top{top_k}.safetensors"
            atomic_safetensors_save(
                checkpoint_path,
                saved_tensors(),
                metadata={"format": "moeme-sparse-layer-v1", "top_k": str(top_k)},
            )
            ledger.artifact(
                run_id,
                f"layer-{args.layer}-top{top_k}",
                checkpoint_path,
                bytes_count=checkpoint_path.stat().st_size,
            )
            stages.append(
                {
                    "top_k": top_k,
                    "training": training_metrics,
                    "validation": validation_metrics,
                    "checkpoint": str(checkpoint_path.resolve()),
                }
            )
            save_training_state(stage_index + 1, 0)
            print(json.dumps(stages[-1]), flush=True)

        final_validation = stages[-1]["validation"]
        final_error = final_validation["relative_l2"]
        router_nonzero_fraction = router.detach().count_nonzero().item() / router.numel()
        router_abs_max = router.detach().abs().max().item()
        passed = (
            final_error <= args.max_validation_relative_l2
            and router_nonzero_fraction >= 0.99
            and final_validation["route_coverage"] == 1.0
        )
        report = {
            "layer": args.layer,
            "training_tokens": len(training_inputs),
            "training_evaluation_tokens": min(args.evaluation_tokens, len(training_inputs)),
            "validation_tokens": len(validation_inputs),
            "stages": stages,
            "curve": curve,
            "passed": passed,
            "systems_smoke": args.systems_smoke,
            "max_validation_relative_l2": args.max_validation_relative_l2,
            "router_nonzero_fraction": router_nonzero_fraction,
            "router_abs_max": router_abs_max,
        }
        report_path = args.output_dir / "report.json"
        atomic_json(report_path, report)
        ledger.artifact(run_id, "distillation-report", report_path)
        progress(
            "complete",
            passed=passed,
            validation_relative_l2=final_error,
            router_nonzero_fraction=router_nonzero_fraction,
            route_coverage=final_validation["route_coverage"],
            normalized_route_entropy=final_validation["normalized_route_entropy"],
            maximum_route_fraction=final_validation["maximum_route_fraction"],
        )
        execution_passed = passed or args.systems_smoke
        ledger.finish(
            run_id,
            "passed" if execution_passed else "failed",
            report,
            error=None if execution_passed else "validation reconstruction gate failed",
        )
        return training_exit_code(passed, args.systems_smoke)
    except EarlyStop as error:
        progress("early_stopped", **error.evidence)
        ledger.finish(
            run_id,
            "failed",
            {"early_stop": error.evidence},
            error=str(error),
        )
        return 3
    except TrainingInterrupted as error:
        progress("interrupted", signal=error.signum, interrupted_stage=error.stage)
        ledger.finish(
            run_id,
            "failed",
            {"signal": error.signum, "interrupted_stage": error.stage},
            error=str(error),
        )
        return 128 + error.signum
    except TrainingSessionComplete as error:
        evidence = {
            "planned_interruption": True,
            "session_stage": error.stage,
            "elapsed_seconds": error.elapsed_seconds,
            "max_runtime_seconds": args.max_runtime_seconds,
        }
        progress("session_budget_reached", **evidence)
        ledger.finish(run_id, "failed", evidence, error=None)
        return 75
    except BaseException as error:
        progress("failed", error=f"{type(error).__name__}: {error}")
        ledger.finish(run_id, "failed", {}, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
