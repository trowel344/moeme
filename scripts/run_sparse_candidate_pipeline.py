#!/usr/bin/env python3
"""Resume direct quantized-layer injection and full-model acceptance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from moeme.cloud_results import scientific_candidate_eligibility, verify_result_manifest

try:
    from scripts.receipt_imatrix import verify_imatrix_receipt
except ModuleNotFoundError:
    from receipt_imatrix import verify_imatrix_receipt

try:
    from scripts.run_all64_pipeline import (
        atomic_json,
        begin_phase,
        fail_phase,
        pass_phase,
        require_space,
        run_logged,
        sha256,
        wait_healthy,
    )
except ModuleNotFoundError:
    from run_all64_pipeline import (  # type: ignore[no-redef]
        atomic_json,
        begin_phase,
        fail_phase,
        pass_phase,
        require_space,
        run_logged,
        sha256,
        wait_healthy,
    )


ALLOWED_CURVE_STATUSES = {"promising", "target_reached"}


def training_eligibility(
    cloud_run: dict, curve_analysis: dict, training_report: dict, layer: int
) -> dict:
    scientific = scientific_candidate_eligibility(training_report, curve_analysis)
    checks = {
        "cloud_run_passed": cloud_run.get("status") == "passed",
        "curve_supports_escalation": curve_analysis.get("status") in ALLOWED_CURVE_STATUSES,
        "scientific_eligibility": scientific["passed"],
        "layer_matches": training_report.get("layer") == layer,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "curve_status": curve_analysis.get("status"),
        "best_relative_l2": (curve_analysis.get("best") or {}).get("relative_l2"),
        "scientific_eligibility": scientific,
    }


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def configuration_digest(configuration: dict) -> str:
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()


def completed_phase_valid(
    state: dict,
    phase: str,
    *,
    candidate: Path,
    server_report: Path,
    parity_report: Path,
) -> tuple[bool, str | None]:
    receipt = (state.get("phases") or {}).get(phase) or {}
    if receipt.get("status") != "passed":
        return False, "phase has no passing receipt"
    if phase == "quantization":
        if not candidate.is_file():
            return False, "quantized candidate is missing"
        if candidate.stat().st_size != receipt.get("bytes"):
            return False, "quantized candidate size changed"
        if sha256(candidate) != receipt.get("sha256"):
            return False, "quantized candidate hash changed"
    elif phase == "evaluation":
        if not server_report.is_file():
            return False, "server evaluation report is missing"
        try:
            report = load_json(server_report)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False, "server evaluation report is unreadable"
        candidate_digest = ((state.get("phases") or {}).get("quantization") or {}).get("sha256")
        if report.get("passed") is not True:
            return False, "server evaluation report is not passing"
        if receipt.get("candidate_sha256") != candidate_digest:
            return False, "evaluation receipt is bound to a different candidate"
    elif phase == "quantized_parity":
        if not parity_report.is_file():
            return False, "quantized parity report is missing"
        try:
            report = load_json(parity_report)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False, "quantized parity report is unreadable"
        candidate_digest = ((state.get("phases") or {}).get("quantization") or {}).get("sha256")
        if report.get("passed") is not True:
            return False, "quantized parity report is not passing"
        if report.get("candidate_sha256") != candidate_digest:
            return False, "quantized parity report is bound to a different candidate"
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=63)
    parser.add_argument("--source-quantized", type=Path, required=True)
    parser.add_argument("--candidate-quantized", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--imatrix-receipt", type=Path, required=True)
    parser.add_argument("--dense-reference", type=Path, required=True)
    parser.add_argument("--llama-bin-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quantized-parity-report", type=Path, required=True)
    parser.add_argument("--server-report", type=Path, required=True)
    parser.add_argument("--corpus", action="append", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--server-context", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=20)
    parser.add_argument("--max-median-ttft", type=float, default=6.0)
    parser.add_argument("--max-ttft", type=float, default=10.0)
    parser.add_argument("--max-long-context-ttft", type=float, default=30.0)
    parser.add_argument("--min-median-decode-rate", type=float, default=3.0)
    parser.add_argument("--min-long-decode-rate", type=float, default=2.5)
    args = parser.parse_args()

    if args.layer < 0 or args.layer >= 64:
        parser.error("--layer must be in [0, 63]")
    checkpoint = args.training_dir / f"layer-{args.layer}-top4.safetensors"
    required_inputs = [
        args.training_dir / "cloud-result-manifest.json",
        args.training_dir / "cloud-run.json",
        args.training_dir / "curve-analysis.json",
        args.training_dir / "report.json",
        checkpoint,
        args.source_quantized,
        args.imatrix,
        args.imatrix_receipt,
        args.dense_reference,
        args.llama_bin_dir / "llama-perplexity",
        args.llama_bin_dir / "llama-quantize",
        args.llama_bin_dir / "llama-server",
        *args.corpus,
    ]
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing candidate-pipeline input(s): " + ", ".join(missing))
    transfer_verification = verify_result_manifest(args.training_dir)
    if not transfer_verification["candidate_ready"]:
        raise ValueError("cloud result is intact but not eligible for candidate injection")
    imatrix_receipt = verify_imatrix_receipt(args.imatrix, args.imatrix_receipt)

    configuration = {
        "training_dir": str(args.training_dir.resolve()),
        "layer": args.layer,
        "checkpoint": str(checkpoint.resolve()),
        "cloud_result_manifest_sha256": sha256(args.training_dir / "cloud-result-manifest.json"),
        "source_quantized": str(args.source_quantized.resolve()),
        "source_quantized_sha256": sha256(args.source_quantized),
        "candidate_quantized": str(args.candidate_quantized.resolve()),
        "imatrix": str(args.imatrix.resolve()),
        "imatrix_receipt": str(args.imatrix_receipt.resolve()),
        "imatrix_sha256": imatrix_receipt["sha256"],
        "dense_reference": str(args.dense_reference.resolve()),
        "llama_bin_dir": str(args.llama_bin_dir.resolve()),
        "corpora": [str(path.resolve()) for path in args.corpus],
        "port": args.port,
        "server_context": args.server_context,
        "gpu_layers": args.gpu_layers,
        "performance_gates": {
            "max_median_ttft": args.max_median_ttft,
            "max_ttft": args.max_ttft,
            "max_long_context_ttft": args.max_long_context_ttft,
            "min_median_decode_rate": args.min_median_decode_rate,
            "min_long_decode_rate": args.min_long_decode_rate,
        },
    }
    digest = configuration_digest(configuration)
    if args.manifest.exists():
        state = load_json(args.manifest)
        if state.get("configuration_digest") != digest:
            raise ValueError("candidate pipeline manifest does not match this invocation")
    else:
        state = {
            "format": "moeme-sparse-candidate-pipeline-v2",
            "configuration": configuration,
            "configuration_digest": digest,
            "phases": {},
            "started_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(args.manifest, state)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"

    def completed(phase: str) -> bool:
        valid, reason = completed_phase_valid(
            state,
            phase,
            candidate=args.candidate_quantized,
            server_report=args.server_report,
            parity_report=args.quantized_parity_report,
        )
        if not valid and state["phases"].get(phase, {}).get("status") == "passed":
            state.setdefault("resume_repairs", []).append(
                {
                    "phase": phase,
                    "reason": reason,
                    "detected_at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_json(args.manifest, state)
        return valid

    try:
        if not completed("transfer_verification"):
            begin_phase(args.manifest, state, "transfer_verification")
            pass_phase(args.manifest, state, "transfer_verification", transfer_verification)

        if not completed("training_eligibility"):
            begin_phase(args.manifest, state, "training_eligibility")
            eligibility = training_eligibility(
                load_json(args.training_dir / "cloud-run.json"),
                load_json(args.training_dir / "curve-analysis.json"),
                load_json(args.training_dir / "report.json"),
                args.layer,
            )
            if not eligibility["passed"]:
                raise ValueError(f"training result is not eligible for injection: {eligibility}")
            pass_phase(args.manifest, state, "training_eligibility", eligibility)

        if not completed("quantization"):
            begin_phase(
                args.manifest,
                state,
                "quantization",
                {"method": "direct-layer-patch", "layer": args.layer},
            )
            require_space(
                args.manifest.parent,
                args.source_quantized.stat().st_size + 2_000_000_000,
                "direct quantized-layer patch",
            )
            args.candidate_quantized.unlink(missing_ok=True)
            args.candidate_quantized.with_name(f".{args.candidate_quantized.name}.partial").unlink(
                missing_ok=True
            )
            layer_patch_receipt = args.manifest.with_suffix(".layer-patch.json")
            layer_patch_receipt.unlink(missing_ok=True)
            run_logged(
                [
                    sys.executable,
                    "scripts/quantize_patch_gguf_layer.py",
                    "--binary",
                    str(args.llama_bin_dir / "llama-quantize"),
                    "--source-quantized",
                    str(args.source_quantized),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(args.candidate_quantized),
                    "--layer",
                    str(args.layer),
                    "--threads",
                    "8",
                    "--imatrix",
                    str(args.imatrix),
                    "--log",
                    str(args.manifest.with_suffix(".layer-quantize.log")),
                    "--receipt",
                    str(layer_patch_receipt),
                ],
                args.manifest.with_suffix(".quantize-wrapper.log"),
                environment,
            )
            pass_phase(
                args.manifest,
                state,
                "quantization",
                {
                    "bytes": args.candidate_quantized.stat().st_size,
                    "sha256": sha256(args.candidate_quantized),
                    "method": "direct-layer-patch",
                    "source_sha256": configuration["source_quantized_sha256"],
                    "layer_patch_receipt": load_json(layer_patch_receipt),
                },
            )

        if not completed("evaluation"):
            begin_phase(args.manifest, state, "evaluation", {"context": args.server_context})
            server_environment = environment.copy()
            server_environment["MOEME_TOP4_LAYERS"] = str(args.layer)
            url = f"http://127.0.0.1:{args.port}"
            server_log_path = args.manifest.with_suffix(".server.log")
            with server_log_path.open("a", encoding="utf-8") as server_log:
                server = subprocess.Popen(
                    [
                        str(args.llama_bin_dir / "llama-server"),
                        "-m",
                        str(args.candidate_quantized),
                        "-c",
                        str(args.server_context),
                        "-ngl",
                        str(args.gpu_layers),
                        "-t",
                        "8",
                        "-tb",
                        "8",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.port),
                        "--parallel",
                        "1",
                        "--cache-ram",
                        "0",
                    ],
                    env=server_environment,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    wait_healthy(url, server)
                    run_logged(
                        [
                            sys.executable,
                            "scripts/server_eval.py",
                            "--url",
                            url,
                            "--report",
                            str(args.server_report),
                            "--max-median-ttft",
                            str(args.max_median_ttft),
                            "--max-ttft",
                            str(args.max_ttft),
                            "--max-long-context-ttft",
                            str(args.max_long_context_ttft),
                            "--min-median-decode-rate",
                            str(args.min_median_decode_rate),
                            "--min-long-decode-rate",
                            str(args.min_long_decode_rate),
                        ],
                        args.manifest.with_suffix(".evaluation.log"),
                        environment,
                    )
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
            evaluation = load_json(args.server_report)
            evaluation["candidate_sha256"] = state["phases"]["quantization"]["sha256"]
            pass_phase(args.manifest, state, "evaluation", evaluation)

        if not completed("quantized_parity"):
            begin_phase(args.manifest, state, "quantized_parity", {"layer": args.layer})
            command = [
                sys.executable,
                "scripts/logit_parity_gate.py",
                "--pipeline-manifest",
                str(args.manifest),
                "--reference",
                str(args.dense_reference),
                "--candidate",
                str(args.candidate_quantized),
                "--binary",
                str(args.llama_bin_dir / "llama-perplexity"),
                "--logits",
                str(args.quantized_parity_report.with_suffix(".kld")),
                "--report",
                str(args.quantized_parity_report),
                "--reference-log",
                str(args.quantized_parity_report.with_suffix(".reference.log")),
                "--candidate-log",
                str(args.quantized_parity_report.with_suffix(".candidate.log")),
                "--context",
                "512",
                "--chunks",
                "1",
                "--top4-layers",
                str(args.layer),
            ]
            for corpus in args.corpus:
                command.extend(("--corpus", str(corpus)))
            run_logged(command, args.manifest.with_suffix(".quantized-parity.log"), environment)
            pass_phase(
                args.manifest,
                state,
                "quantized_parity",
                load_json(args.quantized_parity_report),
            )

        state["finished_at"] = datetime.now(UTC).isoformat()
        state["passed"] = True
        state.pop("current_phase", None)
        atomic_json(args.manifest, state)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        fail_phase(args.manifest, state, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
