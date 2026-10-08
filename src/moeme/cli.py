from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .architecture import inspect_config_file
from .automation import run_foundation, status_summary, verify_hub_checkpoint
from .autopilot import CheckpointAutopilot
from .layout import ExpertLayout
from .ledger import ExperimentLedger
from .parity import validate_layer_forward_parity
from .partition import PartitionedSwiGLU, dense_swiglu
from .runtime_contract import audit_llama_runtime
from .safetensors_stream import convert_checkpoint_streaming, validate_converted_shards


def _inspect(args: argparse.Namespace) -> int:
    facts = inspect_config_file(args.config)
    layout = ExpertLayout(intermediate_size=facts.intermediate_size, top_k=args.top_k)
    result = facts.to_dict()
    result["moeme_layout"] = {
        "groups": layout.groups,
        "shared_width": layout.shared_width,
        "routed_experts": layout.routed_experts,
        "expert_width": layout.expert_width,
        "top_k": layout.top_k,
        "stored_width": layout.stored_width,
        "active_width": layout.active_width,
        "active_fraction": float(layout.active_fraction),
    }
    print(json.dumps(result, indent=2))
    return 0


def _selfcheck(args: argparse.Namespace) -> int:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randn(args.tokens, args.hidden, generator=generator, dtype=torch.float64)
    gate = torch.randn(args.intermediate, args.hidden, generator=generator, dtype=torch.float64)
    up = torch.randn(args.intermediate, args.hidden, generator=generator, dtype=torch.float64)
    down = torch.randn(args.hidden, args.intermediate, generator=generator, dtype=torch.float64)
    layout = ExpertLayout(intermediate_size=args.intermediate)
    converted = PartitionedSwiGLU.from_dense(gate, up, down, layout)
    expected = dense_swiglu(x, gate, up, down)
    actual = converted.forward_all(x)
    error = (expected - actual).abs()
    report = {
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
        "dense_norm": expected.norm().item(),
        "exact_within_tolerance": torch.allclose(expected, actual, atol=args.atol, rtol=args.rtol),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["exact_within_tolerance"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moeme")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-config", help="validate a Qwen config")
    inspect_parser.add_argument("config", type=Path)
    inspect_parser.add_argument("--top-k", type=int, default=4)
    inspect_parser.set_defaults(run=_inspect)

    check_parser = subparsers.add_parser("selfcheck", help="prove exact toy SwiGLU partitioning")
    check_parser.add_argument("--hidden", type=int, default=64)
    check_parser.add_argument("--intermediate", type=int, default=128)
    check_parser.add_argument("--tokens", type=int, default=17)
    check_parser.add_argument("--seed", type=int, default=7)
    check_parser.add_argument("--atol", type=float, default=1e-10)
    check_parser.add_argument("--rtol", type=float, default=1e-10)
    check_parser.set_defaults(run=_selfcheck)

    run_parser = subparsers.add_parser("run", help="run or resume the automated foundation")
    run_parser.add_argument("--config", type=Path, default=Path("configs/qwen3.8-27b.config.json"))
    run_parser.add_argument("--state-dir", type=Path, default=Path(".moeme"))
    run_parser.add_argument("--force", action="store_true")
    run_parser.set_defaults(run=_run)

    status_parser = subparsers.add_parser("status", help="print compact saved phase state")
    status_parser.add_argument("--state-dir", type=Path, default=Path(".moeme"))
    status_parser.set_defaults(run=_status)

    hub_parser = subparsers.add_parser(
        "verify-hub", help="validate the real checkpoint from safetensors headers"
    )
    hub_parser.add_argument("--repo", default="Qwen/Qwen3.8-27B")
    hub_parser.add_argument("--revision", default="main")
    hub_parser.add_argument("--config", type=Path, default=Path("configs/qwen3.8-27b.config.json"))
    hub_parser.add_argument("--state-dir", type=Path, default=Path(".moeme"))
    hub_parser.add_argument("--force", action="store_true")
    hub_parser.set_defaults(run=_verify_hub)

    history_parser = subparsers.add_parser("history", help="show experiment attempts")
    history_parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    history_parser.add_argument("--limit", type=int, default=50)
    history_parser.set_defaults(run=_history)

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="locate the earliest recorded failed stage or capability gate"
    )
    diagnose_parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    diagnose_parser.set_defaults(run=_diagnose)

    runtime_parser = subparsers.add_parser(
        "audit-runtime", help="verify llama.cpp honors the MoEMe GGUF runtime contract"
    )
    runtime_parser.add_argument("source", type=Path)
    runtime_parser.set_defaults(run=_audit_runtime)

    convert_parser = subparsers.add_parser(
        "convert", help="stream a dense checkpoint into disjoint MoEMe tensors"
    )
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--top-k", type=int, default=4)
    convert_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="convert completed source shards now and resume the rest later",
    )
    convert_parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    convert_parser.set_defaults(run=_convert)

    validate_parser = subparsers.add_parser(
        "validate-conversion", help="byte-verify converted tensors against source shards"
    )
    validate_parser.add_argument("source", type=Path)
    validate_parser.add_argument("output", type=Path)
    validate_parser.add_argument("--top-k", type=int, default=4)
    validate_parser.add_argument("--allow-incomplete", action="store_true")
    validate_parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    validate_parser.set_defaults(run=_validate_conversion)

    autopilot_parser = subparsers.add_parser(
        "autopilot", help="convert and validate shards automatically as downloads complete"
    )
    autopilot_parser.add_argument("--source", type=Path, default=Path("checkpoints/qwen3.8-27b"))
    autopilot_parser.add_argument(
        "--output", type=Path, default=Path("artifacts/moeme-27b-initial")
    )
    autopilot_parser.add_argument("--state-dir", type=Path, default=Path(".moeme"))
    autopilot_parser.add_argument("--top-k", type=int, default=4)
    autopilot_parser.add_argument("--watch", action="store_true")
    autopilot_parser.add_argument("--interval", type=float, default=30.0)
    autopilot_parser.set_defaults(run=_autopilot)

    parity_parser = subparsers.add_parser(
        "validate-layer", help="execute dense and converted FFNs on identical inputs"
    )
    parity_parser.add_argument("source", type=Path)
    parity_parser.add_argument("converted", type=Path)
    parity_parser.add_argument("--layer", type=int, default=0)
    parity_parser.add_argument("--tokens", type=int, default=4)
    parity_parser.add_argument("--device", default="cuda")
    parity_parser.add_argument("--top-k", type=int, default=4)
    parity_parser.add_argument("--max-relative-l2", type=float, default=0.005)
    parity_parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    parity_parser.set_defaults(run=_validate_layer)
    return parser


def _run(args: argparse.Namespace) -> int:
    results, skipped = run_foundation(args.config, args.state_dir, force=args.force)
    ledger = ExperimentLedger(args.state_dir / "experiments.sqlite3")
    for result in results:
        ledger.import_phase(
            result.phase,
            result.input_digest,
            result.status,
            result.summary,
        )
    print(
        json.dumps(
            {
                "phases": {result.phase: result.status for result in results},
                "reused": skipped,
                "state": str(args.state_dir / "state.json"),
                "compact_context": str(args.state_dir / "CONTEXT.md"),
            },
            indent=2,
        )
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    print(json.dumps(status_summary(args.state_dir), indent=2))
    return 0


def _verify_hub(args: argparse.Namespace) -> int:
    result, reused = verify_hub_checkpoint(
        args.repo,
        args.revision,
        args.config,
        args.state_dir,
        force=args.force,
    )
    ExperimentLedger(args.state_dir / "experiments.sqlite3").import_phase(
        result.phase,
        result.input_digest,
        result.status,
        result.summary,
    )
    print(json.dumps({"status": result.status, "reused": reused, **result.summary}, indent=2))
    return 0


def _history(args: argparse.Namespace) -> int:
    records = ExperimentLedger(args.ledger).history(limit=args.limit)
    print(
        json.dumps(
            [
                {
                    "id": record.id,
                    "stage": record.stage,
                    "attempt": record.attempt,
                    "status": record.status,
                    "started_at": record.started_at,
                    "finished_at": record.finished_at,
                    "parent_run_id": record.parent_run_id,
                    "summary": record.summary,
                    "error": record.error,
                }
                for record in records
            ],
            indent=2,
        )
    )
    return 0


def _diagnose(args: argparse.Namespace) -> int:
    print(json.dumps(ExperimentLedger(args.ledger).diagnosis(), indent=2))
    return 0


def _audit_runtime(args: argparse.Namespace) -> int:
    result = audit_llama_runtime(args.source)
    print(json.dumps(result, indent=2))
    return 0 if result["compatible"] else 1


def _convert(args: argparse.Namespace) -> int:
    config_path = args.source / "config.json"
    facts = inspect_config_file(config_path)
    layout = ExpertLayout(intermediate_size=facts.intermediate_size, top_k=args.top_k)
    ledger = ExperimentLedger(args.ledger)
    configuration = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "top_k": args.top_k,
        "layout": {
            "groups": layout.groups,
            "shared_groups": layout.shared_groups,
            "routed_experts": layout.routed_experts,
        },
    }
    input_digest = json.dumps(configuration, sort_keys=True)
    previous = next(
        (record for record in ledger.history() if record.stage == "streaming-conversion"),
        None,
    )
    with ledger.run(
        "streaming-conversion",
        input_digest,
        configuration,
        parent_run_id=previous.id if previous is not None else None,
    ) as run_id:
        manifest = convert_checkpoint_streaming(
            args.source,
            args.output,
            layout,
            allow_incomplete=args.allow_incomplete,
        )
        ledger.artifact(
            run_id,
            "checkpoint-manifest",
            args.output / "moeme-manifest.json",
            bytes_count=(args.output / "moeme-manifest.json").stat().st_size,
        )
        if manifest["complete"]:
            ledger.metric(
                run_id,
                "storage",
                "size_ratio",
                manifest["total_size"] / 55_562_855_904,
                "ratio",
                gate="<=1.01",
                passed=manifest["total_size"] <= 55_562_855_904 * 1.01,
            )
        else:
            ledger.metric(
                run_id,
                "conversion",
                "shard_progress",
                manifest["converted_shards"] / manifest["expected_shards"],
                "fraction",
            )
        ledger.finish(run_id, "passed", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def _validate_conversion(args: argparse.Namespace) -> int:
    facts = inspect_config_file(args.source / "config.json")
    layout = ExpertLayout(intermediate_size=facts.intermediate_size, top_k=args.top_k)
    ledger = ExperimentLedger(args.ledger)
    configuration = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "top_k": args.top_k,
    }
    previous = next(
        (record for record in ledger.history() if record.stage == "conversion-parity"),
        None,
    )
    with ledger.run(
        "conversion-parity",
        json.dumps(configuration, sort_keys=True),
        configuration,
        parent_run_id=previous.id if previous is not None else None,
    ) as run_id:
        result = validate_converted_shards(
            args.source,
            args.output,
            layout,
            allow_incomplete=args.allow_incomplete,
        )
        ledger.metric(
            run_id,
            "conversion",
            "bytes_exact",
            result["bytes_checked"],
            "bytes",
            gate="all compared bytes equal",
            passed=result["valid"],
        )
        ledger.finish(run_id, "passed", result)
    print(json.dumps(result, indent=2))
    return 0


def _autopilot(args: argparse.Namespace) -> int:
    facts = inspect_config_file(args.source / "config.json")
    layout = ExpertLayout(facts.intermediate_size, top_k=args.top_k)
    autopilot = CheckpointAutopilot(args.source, args.output, args.state_dir, layout)
    result = autopilot.watch(args.interval) if args.watch else autopilot.advance_once()
    if not args.watch:
        print(json.dumps(asdict(result), indent=2))
    return 0


def _validate_layer(args: argparse.Namespace) -> int:
    facts = inspect_config_file(args.source / "config.json")
    layout = ExpertLayout(facts.intermediate_size, top_k=args.top_k)
    result = validate_layer_forward_parity(
        args.source,
        args.converted,
        layout,
        layer=args.layer,
        tokens=args.tokens,
        device=args.device,
    )
    ledger = ExperimentLedger(args.ledger)
    configuration = {
        "source": str(args.source.resolve()),
        "converted": str(args.converted.resolve()),
        "layer": args.layer,
        "tokens": args.tokens,
        "device": args.device,
    }
    with ledger.run(
        "layer-forward-parity", json.dumps(configuration, sort_keys=True), configuration
    ) as run_id:
        passed = result["finite"] and result["relative_l2_error"] <= args.max_relative_l2
        ledger.metric(
            run_id,
            "conversion",
            "relative_l2_error",
            result["relative_l2_error"],
            "ratio",
            gate=f"<={args.max_relative_l2}",
            passed=passed,
        )
        ledger.finish(run_id, "passed" if passed else "failed", result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


def main() -> int:
    args = build_parser().parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
