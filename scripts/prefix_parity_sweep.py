#!/usr/bin/env python3
"""Find the first corrected prefix that violates strict source-logit parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from gguf import GGUFReader

try:
    from scripts.logit_parity_gate import parsed_metrics
except ModuleNotFoundError:
    from logit_parity_gate import parsed_metrics


TENSOR_SUFFIXES = (
    "ffn_down_exps.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_gate_inp.weight",
    "ffn_down_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
)


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


def layer_tensor_names(layer: int) -> tuple[str, ...]:
    return tuple(f"blk.{layer}.{suffix}" for suffix in TENSOR_SUFFIXES)


def patch_quantized_layer(trained: Path, candidate: Path, layer: int) -> dict[str, str]:
    source_reader = GGUFReader(trained, "r")
    target_reader = GGUFReader(candidate, "r+")
    source = {tensor.name: tensor for tensor in source_reader.tensors}
    target = {tensor.name: tensor for tensor in target_reader.tensors}
    hashes = {}
    for name in layer_tensor_names(layer):
        if name not in source or name not in target:
            raise KeyError(f"missing quantized layer tensor {name}")
        source_tensor = source[name]
        target_tensor = target[name]
        if (
            source_tensor.tensor_type != target_tensor.tensor_type
            or source_tensor.data.shape != target_tensor.data.shape
            or source_tensor.data.nbytes != target_tensor.data.nbytes
        ):
            raise ValueError(f"quantized tensor layout mismatch for {name}")
        target_tensor.data[...] = source_tensor.data
        digest = hashlib.sha256(source_tensor.data.tobytes()).hexdigest()
        if hashlib.sha256(target_tensor.data.tobytes()).hexdigest() != digest:
            raise ValueError(f"quantized tensor verification failed for {name}")
        hashes[name] = digest
    del target_reader
    del source_reader
    return hashes


def run(command: list[str], log: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        capture_output=True,
        timeout=3600,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(output)
    if completed.returncode != 0:
        raise RuntimeError(f"command exited {completed.returncode}; see {log}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained-prefix", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--corpus", action="append", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-prefix", type=int, default=6)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.01)
    parser.add_argument("--max-mean-kld", type=float, default=0.02)
    parser.add_argument("--min-same-top-percent", type=float, default=95.0)
    args = parser.parse_args()
    if args.max_prefix not in range(1, 65):
        parser.error("--max-prefix must be in [1, 64]")
    required_space = args.base.stat().st_size + 2_000_000_000
    if shutil.disk_usage(args.candidate.parent).free < required_space:
        raise OSError(f"prefix sweep requires {required_space:,} free bytes")

    configuration = {
        "base": str(args.base.resolve()),
        "base_sha256": sha256(args.base),
        "trained_prefix": str(args.trained_prefix.resolve()),
        "trained_prefix_sha256": sha256(args.trained_prefix),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "corpora": [str(path.resolve()) for path in args.corpus],
        "max_prefix": args.max_prefix,
        "context": args.context,
        "thresholds": {
            "max_ppl_ratio": args.max_ppl_ratio,
            "max_mean_kld": args.max_mean_kld,
            "min_same_top_percent": args.min_same_top_percent,
        },
    }
    state = {
        "format": "moeme-prefix-parity-sweep-v1",
        "configuration": configuration,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "results": [],
    }
    atomic_json(args.report, state)
    args.candidate.unlink(missing_ok=True)
    partial_candidate = args.candidate.with_name(f".{args.candidate.name}.partial")
    partial_candidate.unlink(missing_ok=True)
    reference_logits: list[Path] = []
    try:
        subprocess.run(["cp", "--sparse=auto", str(args.base), str(partial_candidate)], check=True)
        os.replace(partial_candidate, args.candidate)
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
            "1",
            "--no-warmup",
        ]
        for index, corpus in enumerate(args.corpus, start=1):
            logits = args.report.with_name(f".{args.report.stem}-reference-{index}.kld")
            partial = logits.with_suffix(f"{logits.suffix}.partial")
            logits.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            run(
                [
                    str(args.binary),
                    "-m",
                    str(args.reference),
                    "-f",
                    str(corpus),
                    *common,
                    "--kl-divergence-base",
                    str(partial),
                ],
                args.report.with_name(f"{args.report.stem}-reference-{index}.log"),
                os.environ.copy(),
            )
            if partial.read_bytes()[:8] != b"_logits_":
                raise ValueError("reference logits file has an invalid header")
            os.replace(partial, logits)
            reference_logits.append(logits)

        patched: dict[str, str] = {}
        for prefix in range(args.max_prefix + 1):
            if prefix:
                patched.update(
                    patch_quantized_layer(args.trained_prefix, args.candidate, prefix - 1)
                )
            candidate_sha = sha256(args.candidate)
            corpus_results = []
            for index, (corpus, logits) in enumerate(
                zip(args.corpus, reference_logits, strict=True), start=1
            ):
                environment = os.environ.copy()
                if prefix:
                    environment["MOEME_TOP4_LAYERS"] = ",".join(map(str, range(prefix)))
                else:
                    environment.pop("MOEME_TOP4_LAYERS", None)
                output = run(
                    [
                        str(args.binary),
                        "-m",
                        str(args.candidate),
                        "-f",
                        str(corpus),
                        *common,
                        "--kl-divergence-base",
                        str(logits),
                        "--kl-divergence",
                    ],
                    args.report.with_name(
                        f"{args.report.stem}-prefix-{prefix}-candidate-{index}.log"
                    ),
                    environment,
                )
                metrics = parsed_metrics(output)
                gates = {
                    "ppl_ratio": metrics["ppl_ratio"] <= args.max_ppl_ratio,
                    "mean_kld": metrics["mean_kld"] <= args.max_mean_kld,
                    "same_top_percent": (metrics["same_top_percent"] >= args.min_same_top_percent),
                }
                corpus_results.append(
                    {
                        "corpus": str(corpus.resolve()),
                        "metrics": metrics,
                        "gates": gates,
                        "passed": all(gates.values()),
                    }
                )
            result = {
                "prefix_layers": prefix,
                "top4_layers": list(range(prefix)),
                "candidate_sha256": candidate_sha,
                "patched_tensor_sha256": dict(patched),
                "results": corpus_results,
                "passed": all(value["passed"] for value in corpus_results),
            }
            state["results"].append(result)
            state["updated_at"] = datetime.now(UTC).isoformat()
            atomic_json(args.report, state)
            if not result["passed"]:
                state["status"] = "failed"
                state["first_failed_prefix"] = prefix
                break
        else:
            state["status"] = "passed"
            state["first_failed_prefix"] = None
        state["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(args.report, state)
        return 0 if state["status"] == "passed" else 1
    except BaseException as error:
        state["status"] = "error"
        state["error"] = f"{type(error).__name__}: {error}"
        state["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(args.report, state)
        raise
    finally:
        partial_candidate.unlink(missing_ok=True)
        args.candidate.unlink(missing_ok=True)
        for path in reference_logits:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
