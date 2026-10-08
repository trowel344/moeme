#!/usr/bin/env python3
"""Measure end-to-end source-logit parity as a function of active expert width.

The Phase C per-layer gate (`relative_l2 <= 0.20` at Top-4) is a reconstruction
proxy, not the project's actual quality criterion. The real criterion is
end-to-end: source-model logits and capability. This harness varies the number
of routed experts a layer activates at runtime (via the patched llama.cpp
`MOEME_ACTIVE_K` hook) and reports the smallest K whose full-model logit parity
still holds. It is read-only with respect to the model: the candidate is the
shipped GGUF itself, so no release artifact is created or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from gguf import GGUFReader

try:
    from scripts.logit_parity_gate import parsed_metrics
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from logit_parity_gate import parsed_metrics

from moeme.ledger import ExperimentLedger


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_active_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "active K values must be comma-separated integers"
        ) from error
    if not ks or len(set(ks)) != len(ks) or any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("active K values must be unique positive integers")
    return ks


def model_layer_count(model: Path) -> int:
    reader = GGUFReader(model, "r")
    keys = {field.name: field for field in reader.fields.values()}
    block_field = keys.get("qwen35moe.block_count") or keys.get("llama.block_count")
    if block_field is None:
        raise ValueError("model does not declare a block_count")
    return int(block_field.parts[-1][0])


def active_layers(spec: str, layer_count: int) -> list[int]:
    """Resolve a layer spec to the concrete layers the override applies to."""
    if spec == "all":
        return list(range(layer_count))
    if spec.startswith("indices:"):
        try:
            layers = [int(item) for item in spec.removeprefix("indices:").split(",")]
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "explicit layer indices must be comma-separated integers"
            ) from error
        if not layers or len(layers) != len(set(layers)):
            raise argparse.ArgumentTypeError("explicit layer indices must be non-empty and unique")
        if any(layer < 0 or layer >= layer_count for layer in layers):
            raise argparse.ArgumentTypeError(
                f"explicit layer indices must each be in [0, {layer_count - 1}]"
            )
        return layers
    try:
        count = int(spec)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--layers must be 'all' or a positive count") from error
    if count < 1 or count > layer_count:
        raise argparse.ArgumentTypeError(f"--layers count must be in [1, {layer_count}]")
    return list(range(count))


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


def indexed_path(path: Path, index: int, suffix: str) -> Path:
    return path.with_name(f"{path.stem}-{index}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="native-width reference model; defaults to --model at its own expert count",
    )
    parser.add_argument("--corpus", action="append", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--ks", type=parse_active_ks, default="4,6,8,10,12")
    parser.add_argument(
        "--layers",
        default="all",
        help="'all', an integer prefix count, or exact indices such as 'indices:63'",
    )
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.01)
    parser.add_argument("--max-mean-kld", type=float, default=0.02)
    # 90.0 matches the release logit-parity gate: even a near-lossless model
    # agrees with the quantized reference on only ~92.5% of top-1 tokens.
    parser.add_argument("--min-same-top-percent", type=float, default=90.0)
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()

    reference = args.reference if args.reference is not None else args.model
    layer_count = model_layer_count(args.model)
    layers = active_layers(args.layers, layer_count)

    configuration = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("ledger", "corpus")
    }
    configuration["model_sha256"] = sha256(args.model)
    configuration["reference_sha256"] = sha256(reference)
    configuration["corpora"] = [str(corpus.resolve()) for corpus in args.corpus]
    configuration["layer_count"] = layer_count
    configuration["active_layers"] = layers
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    ledger = ExperimentLedger(args.ledger)
    run_id = ledger.start("active-width-end-to-end-gate", digest, configuration)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    transient: list[Path] = []
    try:
        common = [
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
        reference_logits: list[Path] = []
        for index, corpus in enumerate(args.corpus, start=1):
            logits = indexed_path(args.log_dir / args.report.stem, index, ".kld")
            partial = logits.with_suffix(f"{logits.suffix}.partial")
            transient.extend((logits, partial))
            logits.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            run(
                [
                    str(args.binary),
                    "-m",
                    str(reference),
                    "-f",
                    str(corpus),
                    *common,
                    "--kl-divergence-base",
                    str(partial),
                ],
                args.log_dir / f"{args.report.stem}-reference-{index}.log",
                os.environ.copy(),
            )
            if partial.read_bytes()[:8] != b"_logits_":
                raise ValueError("reference logits file has an invalid header")
            os.replace(partial, logits)
            reference_logits.append(logits)

        results = []
        for k in args.ks:
            corpus_results = []
            for index, (corpus, logits) in enumerate(
                zip(args.corpus, reference_logits, strict=True), start=1
            ):
                environment = os.environ.copy()
                environment["MOEME_ACTIVE_K"] = str(k)
                environment["MOEME_ACTIVE_K_LAYERS"] = ",".join(map(str, layers))
                environment.pop("MOEME_TOP4_LAYERS", None)
                output = run(
                    [
                        str(args.binary),
                        "-m",
                        str(args.model),
                        "-f",
                        str(corpus),
                        *common,
                        "--kl-divergence-base",
                        str(logits),
                        "--kl-divergence",
                    ],
                    args.log_dir / f"{args.report.stem}-k{k}-candidate-{index}.log",
                    environment,
                )
                metrics = parsed_metrics(output)
                gates = {
                    "ppl_ratio": metrics["ppl_ratio"] <= args.max_ppl_ratio,
                    "mean_kld": metrics["mean_kld"] <= args.max_mean_kld,
                    "same_top_percent": metrics["same_top_percent"] >= args.min_same_top_percent,
                }
                corpus_results.append(
                    {
                        "corpus": str(corpus.resolve()),
                        "metrics": metrics,
                        "gates": gates,
                        "passed": all(gates.values()),
                    }
                )
            results.append(
                {
                    "active_k": k,
                    # The shipped partition stores 16 F/16 groups: 4 always-active
                    # shared groups plus 12 routed experts. Active fraction is
                    # therefore (4 + K) / 16 -- e.g. Top-4 is 0.5F, Top-12 is F.
                    "active_fraction": (k + 4) / 16.0,
                    "results": corpus_results,
                    "passed": all(value["passed"] for value in corpus_results),
                }
            )

        passing = sorted(result["active_k"] for result in results if result["passed"])
        summary = {
            "format": "moeme-active-width-gate-v1",
            "finished_at": datetime.now(UTC).isoformat(),
            "model": str(args.model.resolve()),
            "reference": str(reference.resolve()),
            "layer_count": layer_count,
            "active_layers": layers,
            "smallest_passing_k": passing[0] if passing else None,
            "passed": bool(passing),
            "thresholds": {
                "max_ppl_ratio": args.max_ppl_ratio,
                "max_mean_kld": args.max_mean_kld,
                "min_same_top_percent": args.min_same_top_percent,
            },
            "context_tokens": args.context,
            "chunks_per_corpus": args.chunks,
            "model_sha256": configuration["model_sha256"],
            "reference_sha256": configuration["reference_sha256"],
            "results": results,
        }
        atomic_json(args.report, summary)
        ledger.artifact(run_id, "active-width-gate-report", args.report)
        for result in results:
            for corpus_index, corpus_result in enumerate(result["results"], start=1):
                for name, value in corpus_result["metrics"].items():
                    ledger.metric(
                        run_id,
                        f"k{result['active_k']}-corpus-{corpus_index}",
                        name,
                        value,
                        "ratio" if "ratio" in name else "value",
                    )
        ledger.finish(
            run_id,
            "passed" if summary["passed"] else "failed",
            summary,
            error=None if summary["passed"] else "no tested active width held the end-to-end gate",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["passed"] else 1
    except BaseException as error:
        ledger.finish(run_id, "failed", {}, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for path in transient:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
