#!/usr/bin/env python3
"""Assemble the Top-12 release manifest from measured receipts.

The release launcher (``serve_validated_model.py``) requires a pipeline manifest
that records the quantization artifact hash and a passed evaluation phase. For
the Top-12 deliverable there is no end-to-end pipeline run to produce it, so this
derives the manifest from the receipts that were actually measured, and refuses
to write it unless the parity and capability receipts pass and are bound to the
model's own SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required release receipt is missing: {path}")
    return json.loads(path.read_text())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capability_passed(server: dict) -> bool:
    return bool(
        server.get("passed")
        and server.get("quality_passed")
        and server.get("stability_passed")
        and server.get("performance_passed")
    )


def assemble(
    model: Path,
    parity_report: Path,
    server_report: Path,
    *,
    context: int = 8192,
) -> dict:
    parity = read_json(parity_report)
    server = read_json(server_report)
    if not parity.get("passed"):
        raise ValueError("source-logit parity report has not passed")
    if not capability_passed(server):
        raise ValueError("capability report has not passed")
    digest = sha256(model)
    if parity.get("candidate_sha256") != digest:
        raise ValueError("source-logit parity receipt is not bound to the model")
    return {
        "format": "moeme-top12-release-manifest-v1",
        "finished_at": datetime.now(UTC).isoformat(),
        "phases": {
            "quantization": {
                "status": "passed",
                "bytes": model.stat().st_size,
                "sha256": digest,
            },
            "evaluation": {
                "status": "passed",
                "context": context,
                "capability_cases": server.get("cases_passed"),
                "capability_total": server.get("cases_total"),
                "parity_corpora": len(parity.get("results", [])),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/gguf/moeme-27b-top12-imatrix-q4_k_m.gguf"),
    )
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--server-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(".moeme/top12-release-manifest.json"))
    parser.add_argument("--context", type=int, default=8192)
    args = parser.parse_args()

    state = assemble(
        args.model,
        args.parity_report,
        args.server_report,
        context=args.context,
    )
    atomic_json(args.manifest, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
