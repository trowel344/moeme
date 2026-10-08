#!/usr/bin/env python3
"""Reflink a BF16 MoEMe GGUF and replace layers from training receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from safetensors.torch import load_file

from moeme.ledger import ExperimentLedger

TRAINED_KEYS = {
    "expert_down",
    "expert_gate",
    "expert_up",
    "router",
    "shared_down",
    "shared_gate",
    "shared_up",
}

# Keys that are frozen when only the down projection is trained. They are the
# partition's own rows, so they are rebuilt from a partition checkpoint instead
# of being stored in every layer receipt (which would double the campaign size).
FROZEN_KEYS = ("expert_gate", "expert_up")


def validate_trained_state(
    state: dict[str, torch.Tensor],
    layer: int,
    minimum_router_nonzero: float,
    required_keys: set[str] = TRAINED_KEYS,
) -> dict[str, float]:
    missing = sorted(required_keys - set(state))
    if missing:
        raise ValueError(f"checkpoint for layer {layer} is missing trained tensors: {missing}")
    for key in required_keys:
        if not torch.isfinite(state[key]).all():
            raise ValueError(f"checkpoint for layer {layer} has non-finite values in {key}")
    router = state["router"]
    nonzero_fraction = router.count_nonzero().item() / router.numel()
    if nonzero_fraction < minimum_router_nonzero:
        raise ValueError(
            f"checkpoint for layer {layer} router nonzero fraction "
            f"{nonzero_fraction:.6f} is below {minimum_router_nonzero:.6f}"
        )
    return {
        "router_nonzero_fraction": nonzero_fraction,
        "router_abs_max": router.abs().max().item(),
    }


def resolve_frozen_projections(
    state: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Rebuild frozen gate/up tensors from a partition checkpoint's dense rows.

    A checkpoint stores `shared` followed by every routed expert in the dense
    intermediate-channel order (verified exact against the dense source), and a
    receipt records `shared_indices`/`expert_indices` as positions in exactly
    that order. Selecting the concatenated rows with those indices therefore
    reproduces the frozen partition tensors bit-for-bit without storing them.
    """
    dense_gate = torch.cat((weights["shared_gate"], *weights["expert_gate"]), dim=0)
    dense_up = torch.cat((weights["shared_up"], *weights["expert_up"]), dim=0)
    shared_indices = state["shared_indices"].long()
    expert_indices = state["expert_indices"].long()
    resolved = dict(state)
    for key, dense, indices in (
        ("shared_gate", dense_gate, shared_indices),
        ("shared_up", dense_up, shared_indices),
        ("expert_gate", dense_gate, expert_indices.flatten()),
        ("expert_up", dense_up, expert_indices.flatten()),
    ):
        if key in resolved:
            continue
        selected = dense.index_select(0, indices)
        resolved[key] = (
            selected.reshape(*expert_indices.shape, dense.shape[1])
            if key.startswith("expert_")
            else selected
        )
    return resolved


def frozen_partition_weights(checkpoint: Path, layer: int) -> dict[str, torch.Tensor]:
    try:
        from scripts.oracle_router_analysis import load_layer_weights
    except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
        from oracle_router_analysis import load_layer_weights

    manifest = json.loads((checkpoint / "moeme-manifest.json").read_text())
    shared_groups = int(manifest["layout"]["shared_groups"])
    return load_layer_weights(checkpoint, layer, "cpu", "checkpoint", None, shared_groups)


def tensor_bytes(value: torch.Tensor, shape: tuple[int, ...]) -> np.ndarray:
    value = value.detach().cpu().contiguous()
    if value.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 tensor, got {value.dtype}")
    raw = value.view(torch.uint16).numpy().view(np.uint8)
    return raw.reshape(shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument(
        "--layer-checkpoint",
        action="append",
        default=[],
        metavar="LAYER=PATH",
        help="repeatable layer/checkpoint pair; may replace --layer/--checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-router-nonzero", type=float, default=0.99)
    parser.add_argument(
        "--frozen-checkpoint",
        type=Path,
        default=None,
        help=(
            "Partition checkpoint used to rebuild frozen gate/up tensors that a "
            "receipt omits (required when receipts come from down-only training)."
        ),
    )
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()

    layer_checkpoints: list[tuple[int, Path]] = []
    if args.layer is not None or args.checkpoint is not None:
        if args.layer is None or args.checkpoint is None:
            parser.error("--layer and --checkpoint must be provided together")
        layer_checkpoints.append((args.layer, args.checkpoint))
    for item in args.layer_checkpoint:
        try:
            layer_text, checkpoint_text = item.split("=", 1)
            layer_checkpoints.append((int(layer_text), Path(checkpoint_text)))
        except (TypeError, ValueError):
            parser.error(f"invalid --layer-checkpoint {item!r}; expected LAYER=PATH")
    if not layer_checkpoints:
        parser.error("provide --layer/--checkpoint or at least one --layer-checkpoint")
    layers = [layer for layer, _ in layer_checkpoints]
    if len(layers) != len(set(layers)):
        parser.error("each layer may be specified only once")

    configuration = {
        "source": str(args.source.resolve()),
        "frozen_checkpoint": (
            str(args.frozen_checkpoint.resolve()) if args.frozen_checkpoint else None
        ),
        "layer_checkpoints": [
            {"layer": layer, "checkpoint": str(checkpoint.resolve())}
            for layer, checkpoint in layer_checkpoints
        ],
        "output": str(args.output.resolve()),
        "copy_mode": "reflink-auto",
        "minimum_router_nonzero": args.minimum_router_nonzero,
    }
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    ledger = ExperimentLedger(args.ledger)
    run_id = ledger.start("gguf-layer-injection", digest, configuration)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    try:
        if args.output.exists() or temporary.exists():
            raise FileExistsError("output or partial output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cp", "--reflink=auto", "--sparse=auto", str(args.source), str(temporary)],
            check=True,
        )
        reader = GGUFReader(temporary, "r+")
        gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
        patched = {}
        checkpoint_health = {}
        for layer, checkpoint in layer_checkpoints:
            state = load_file(checkpoint, device="cpu")
            required = set(TRAINED_KEYS)
            if args.frozen_checkpoint is not None:
                required -= set(FROZEN_KEYS)
            checkpoint_health[str(layer)] = validate_trained_state(
                state, layer, args.minimum_router_nonzero, required
            )
            if set(FROZEN_KEYS) - set(state):
                if args.frozen_checkpoint is None:
                    raise ValueError(
                        f"layer {layer} omits {sorted(set(FROZEN_KEYS) - set(state))}; "
                        "pass --frozen-checkpoint to rebuild them"
                    )
                state = resolve_frozen_projections(
                    state, frozen_partition_weights(args.frozen_checkpoint, layer)
                )
            names = {
                "expert_down": f"blk.{layer}.ffn_down_exps.weight",
                "expert_gate": f"blk.{layer}.ffn_gate_exps.weight",
                "expert_up": f"blk.{layer}.ffn_up_exps.weight",
                "router": f"blk.{layer}.ffn_gate_inp.weight",
                "shared_down": f"blk.{layer}.ffn_down_shexp.weight",
                "shared_gate": f"blk.{layer}.ffn_gate_shexp.weight",
                "shared_up": f"blk.{layer}.ffn_up_shexp.weight",
            }
            for key, name in names.items():
                target = gguf_tensors[name]
                value = state[key]
                if key == "router":
                    if target.tensor_type != GGMLQuantizationType.F32:
                        raise TypeError(f"{name} is not F32")
                    source = value.float().numpy().reshape(target.data.shape)
                else:
                    if target.tensor_type != GGMLQuantizationType.BF16:
                        raise TypeError(f"{name} is not BF16")
                    source = tensor_bytes(value, target.data.shape)
                if source.shape != target.data.shape:
                    raise ValueError(
                        f"shape mismatch for {name}: {source.shape} != {target.data.shape}"
                    )
                target.data[...] = source
                patched[name] = hashlib.sha256(source.tobytes()).hexdigest()
        del reader

        verification = GGUFReader(temporary, "r")
        verified = {
            tensor.name: hashlib.sha256(tensor.data.tobytes()).hexdigest()
            for tensor in verification.tensors
            if tensor.name in patched
        }
        if verified != patched:
            raise ValueError("patched tensor verification failed")
        del verification
        os.replace(temporary, args.output)
        summary = {
            "output": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "patched_tensor_sha256": patched,
            "checkpoint_health": checkpoint_health,
        }
        ledger.artifact(
            run_id, "hybrid-bf16-gguf", args.output, bytes_count=args.output.stat().st_size
        )
        ledger.finish(run_id, "passed", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        ledger.finish(run_id, "failed", {}, error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
