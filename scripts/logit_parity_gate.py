#!/usr/bin/env python3
"""Wait for a candidate, then compare its logits with the dense reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from moeme.ledger import ExperimentLedger


def parse_top4_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-4 layers must be comma-separated integers") from error
    if not layers or len(set(layers)) != len(layers) or not set(layers).issubset(range(64)):
        raise argparse.ArgumentTypeError("top-4 layers must be unique integers in [0, 63]")
    return layers


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_pipeline(manifest: Path, candidate: Path, poll_seconds: int) -> None:
    while True:
        state = json.loads(manifest.read_text())
        if state.get("phases", {}).get("evaluation", {}).get("status") == "passed":
            return
        if state.get("last_error"):
            if (
                state.get("phases", {}).get("quantization", {}).get("status") == "passed"
                and candidate.exists()
            ):
                return
            raise RuntimeError(
                f"upstream pipeline failed before quantization: {state['last_error']}"
            )
        time.sleep(poll_seconds)


def run(command: list[str], log_path: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        capture_output=True,
        timeout=3600,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output)
    if completed.returncode != 0:
        raise RuntimeError(f"command exited {completed.returncode}; see {log_path}")
    return output


def metric(output: str, pattern: str, name: str) -> float:
    match = re.search(pattern, output)
    if match is None:
        raise ValueError(f"missing {name} in KL-divergence output")
    return float(match.group(1))


def indexed_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.stem}-{index}{path.suffix}")


def parsed_metrics(output: str) -> dict[str, float]:
    return {
        "candidate_ppl": metric(output, r"Mean PPL\(Q\)\s*:\s*([0-9.]+)", "candidate PPL"),
        "reference_ppl": metric(output, r"Mean PPL\(base\)\s*:\s*([0-9.]+)", "reference PPL"),
        "ppl_ratio": metric(output, r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)", "PPL ratio"),
        "mean_kld": max(
            0.0,
            metric(output, r"Mean\s+KLD:\s*([+-]?[0-9.]+)", "mean KLD"),
        ),
        "same_top_percent": metric(output, r"Same top p:\s*([0-9.]+)", "same top probability"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--corpus", action="append", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument(
        "--top4-layers",
        type=parse_top4_layers,
        default=None,
        help=(
            "comma-separated layers whose graph must route Top-4; omit to leave the "
            "runtime at the model's native expert count (required for a Top-12 control)"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.01)
    parser.add_argument("--max-mean-kld", type=float, default=0.02)
    # 90, not 95: a ceiling study (docs/moeme-serving-status.md) shows even the
    # near-lossless BF16 model agrees with the Q4-quantized reference on only
    # ~92.5% of top-1 tokens on the wiki-prose corpus, so 95 is unreachable by
    # any artifact and would reject a perfect model.
    parser.add_argument("--min-same-top-percent", type=float, default=90.0)
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()

    wait_for_pipeline(args.pipeline_manifest, args.candidate, args.poll_seconds)
    configuration = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("ledger", "corpus")
    }
    configuration["corpora"] = [str(corpus.resolve()) for corpus in args.corpus]
    configuration["candidate_sha256"] = sha256(args.candidate)
    configuration["reference_sha256"] = sha256(args.reference)
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    ledger = ExperimentLedger(args.ledger)
    run_id = ledger.start("full-model-logit-parity", digest, configuration)
    transient_paths: list[Path] = []
    try:
        args.logits.parent.mkdir(parents=True, exist_ok=True)
        results = []
        for index, corpus in enumerate(args.corpus, start=1):
            logits = indexed_path(args.logits, index)
            partial = logits.with_suffix(f"{logits.suffix}.partial")
            reference_log = indexed_path(args.reference_log, index)
            candidate_log = indexed_path(args.candidate_log, index)
            transient_paths.extend((partial, logits))
            partial.unlink(missing_ok=True)
            logits.unlink(missing_ok=True)
            common = [
                "-f",
                str(corpus),
                "-c",
                str(args.context),
                "-b",
                str(args.context),
                "-ub",
                "64",
                "-t",
                "8",
                "-tb",
                "8",
                "--chunks",
                str(args.chunks),
                "--no-warmup",
            ]
            run(
                [
                    str(args.binary),
                    "-m",
                    str(args.reference),
                    *common,
                    "--kl-divergence-base",
                    str(partial),
                ],
                reference_log,
                os.environ.copy(),
            )
            if partial.read_bytes()[:8] != b"_logits_":
                raise ValueError("reference logits file has an invalid header")
            os.replace(partial, logits)

            environment = os.environ.copy()
            # Only force sparse routing when the caller explicitly names layers.
            # Defaulting this to all 64 silently converted Top-12 controls into
            # Top-4 runs and produced invalid parity failures.
            if args.top4_layers:
                environment["MOEME_TOP4_LAYERS"] = ",".join(map(str, args.top4_layers))
            else:
                environment.pop("MOEME_TOP4_LAYERS", None)
            output = run(
                [
                    str(args.binary),
                    "-m",
                    str(args.candidate),
                    *common,
                    "--kl-divergence-base",
                    str(logits),
                    "--kl-divergence",
                ],
                candidate_log,
                environment,
            )
            metrics = parsed_metrics(output)
            gates = {
                "ppl_ratio": metrics["ppl_ratio"] <= args.max_ppl_ratio,
                "mean_kld": metrics["mean_kld"] <= args.max_mean_kld,
                "same_top_percent": (metrics["same_top_percent"] >= args.min_same_top_percent),
            }
            results.append(
                {
                    "corpus": str(corpus.resolve()),
                    "metrics": metrics,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )
            logits.unlink()
        summary = {
            "format": "moeme-full-model-logit-parity-v1",
            "finished_at": datetime.now(UTC).isoformat(),
            "passed": all(result["passed"] for result in results),
            "results": results,
            "thresholds": {
                "max_ppl_ratio": args.max_ppl_ratio,
                "max_mean_kld": args.max_mean_kld,
                "min_same_top_percent": args.min_same_top_percent,
            },
            "context_tokens": args.context,
            "chunks_per_corpus": args.chunks,
            "candidate_sha256": configuration["candidate_sha256"],
            "reference_sha256": configuration["reference_sha256"],
        }
        atomic_json(args.report, summary)
        ledger.artifact(run_id, "logit-parity-report", args.report)
        for index, result in enumerate(results, start=1):
            for name, value in result["metrics"].items():
                ledger.metric(
                    run_id,
                    f"corpus-{index}",
                    name,
                    value,
                    "ratio" if "ratio" in name else "value",
                )
        ledger.finish(
            run_id,
            "passed" if summary["passed"] else "failed",
            summary,
            error=None if summary["passed"] else "full-model source-logit parity gate failed",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["passed"] else 1
    except BaseException as error:
        ledger.finish(run_id, "failed", {}, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for path in transient_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
