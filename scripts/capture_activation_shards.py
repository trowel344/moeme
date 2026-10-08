#!/usr/bin/env python3
"""Run a resumable sequence of atomic single-layer activation captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.capture_activations import ChildSignalForwarder
except ModuleNotFoundError:
    from capture_activations import ChildSignalForwarder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def shard_plan(total_chunks: int, chunks_per_shard: int) -> list[dict[str, int]]:
    if total_chunks <= 0 or chunks_per_shard <= 0:
        raise ValueError("chunk counts must be positive")
    result = []
    offset = 0
    index = 0
    while offset < total_chunks:
        chunks = min(chunks_per_shard, total_chunks - offset)
        result.append({"index": index, "from_chunk": offset, "chunks": chunks})
        offset += chunks
        index += 1
    return result


def reusable_shard(
    directory: Path,
    *,
    layer: int,
    from_chunk: int,
    chunks: int,
    ctx_size: int,
    corpus_sha256: str,
    model_sha256: str,
    binary_sha256: str,
) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    layer_receipt = manifest.get("layers", {}).get(str(layer), {})
    payload = directory / f"layer-{layer}.f32"
    contract_matches = (
        manifest.get("corpus_sha256") == corpus_sha256
        and manifest.get("from_chunk") == from_chunk
        and manifest.get("tokens_per_layer") == chunks * ctx_size
        and layer_receipt.get("tokens") == chunks * ctx_size
        and (manifest.get("source_model") or {}).get("sha256") == model_sha256
        and (manifest.get("capture_binary") or {}).get("sha256") == binary_sha256
        and payload.is_file()
    )
    if not contract_matches:
        return False
    if payload.stat().st_size != layer_receipt.get("bytes"):
        return False
    expected_digest = layer_receipt.get("sha256")
    return isinstance(expected_digest, str) and sha256(payload) == expected_digest


def bind_shard_source(
    directory: Path,
    *,
    model: Path,
    model_sha256: str,
    binary: Path,
    binary_sha256: str,
) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_model"] = {
        "path": str(model.resolve()),
        "bytes": model.stat().st_size,
        "sha256": model_sha256,
    }
    manifest["capture_binary"] = {
        "path": str(binary.resolve()),
        "bytes": binary.stat().st_size,
        "sha256": binary_sha256,
    }
    atomic_json(manifest_path, manifest)


def validate_existing_campaign(existing: dict, expected: dict) -> dict:
    immutable = (
        "format",
        "corpus_sha256",
        "layer",
        "ctx_size",
        "total_chunks",
        "chunks_per_shard",
        "total_tokens",
    )
    mismatched = [key for key in immutable if existing.get(key) != expected.get(key)]
    for key in ("source_model", "capture_binary"):
        existing_identity = existing.get(key) or {}
        expected_identity = expected.get(key) or {}
        if any(
            existing_identity.get(field) != expected_identity.get(field)
            for field in ("bytes", "sha256")
        ):
            mismatched.append(key)
    if mismatched:
        raise ValueError("activation shard campaign contract mismatch: " + ", ".join(mismatched))
    shards = existing.get("shards", {})
    if not isinstance(shards, dict):
        raise TypeError("activation shard campaign shards must be an object")
    return shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--total-chunks", type=int, required=True)
    parser.add_argument("--chunks-per-shard", type=int, default=100)
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--gpu-layers", type=int, default=18)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--disable-cuda-graphs", action="store_true")
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    args = parser.parse_args()

    plan = shard_plan(args.total_chunks, args.chunks_per_shard)
    corpus_digest = sha256(args.corpus)
    model_digest = sha256(args.model)
    binary_digest = sha256(args.binary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    campaign_path = args.output_dir / "campaign.json"
    campaign = {
        "format": "moeme-activation-shard-campaign-v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": corpus_digest,
        "source_model": {
            "path": str(args.model.resolve()),
            "bytes": args.model.stat().st_size,
            "sha256": model_digest,
        },
        "capture_binary": {
            "path": str(args.binary.resolve()),
            "bytes": args.binary.stat().st_size,
            "sha256": binary_digest,
        },
        "layer": args.layer,
        "ctx_size": args.ctx_size,
        "total_chunks": args.total_chunks,
        "chunks_per_shard": args.chunks_per_shard,
        "total_tokens": args.total_chunks * args.ctx_size,
        "shards": {},
    }
    existing = {}
    try:
        previous = json.loads(campaign_path.read_text())
        existing = validate_existing_campaign(previous, campaign)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    campaign["shards"] = existing
    atomic_json(campaign_path, campaign)

    repository = Path(__file__).resolve().parents[1]
    for shard in plan:
        key = str(shard["index"])
        shard_dir = args.output_dir / f"shard-{shard['index']:04d}"
        if reusable_shard(
            shard_dir,
            layer=args.layer,
            from_chunk=shard["from_chunk"],
            chunks=shard["chunks"],
            ctx_size=args.ctx_size,
            corpus_sha256=corpus_digest,
            model_sha256=model_digest,
            binary_sha256=binary_digest,
        ):
            campaign["shards"][key] = {
                **shard,
                "status": "passed",
                "directory": str(shard_dir.resolve()),
                "reused": True,
            }
            continue
        if shard_dir.exists():
            raise RuntimeError(f"existing shard is incomplete or mismatched: {shard_dir}")
        log = args.output_dir / "logs" / f"shard-{shard['index']:04d}.log"
        imatrix = args.output_dir / "imatrix" / f"shard-{shard['index']:04d}.gguf"
        command = [
            sys.executable,
            str(repository / "scripts/capture_activations.py"),
            "--binary",
            str(args.binary),
            "--model",
            str(args.model),
            "--corpus",
            str(args.corpus),
            "--output-dir",
            str(shard_dir),
            "--imatrix-output",
            str(imatrix),
            "--log",
            str(log),
            "--chunks",
            str(shard["chunks"]),
            "--from-chunk",
            str(shard["from_chunk"]),
            "--ctx-size",
            str(args.ctx_size),
            "--gpu-layers",
            str(args.gpu_layers),
            "--threads",
            str(args.threads),
            "--layer",
            str(args.layer),
            "--ledger",
            str(args.ledger),
        ]
        if args.disable_cuda_graphs:
            command.append("--disable-cuda-graphs")
        started_at = datetime.now(UTC).isoformat()
        process = subprocess.Popen(command, cwd=repository)
        with ChildSignalForwarder(process) as forwarder:
            returncode = process.wait()
        interrupted = forwarder.signum is not None or returncode in {
            128 + signal.SIGINT,
            128 + signal.SIGTERM,
        }
        if returncode == 0:
            bind_shard_source(
                shard_dir,
                model=args.model,
                model_sha256=model_digest,
                binary=args.binary,
                binary_sha256=binary_digest,
            )
        campaign["shards"][key] = {
            **shard,
            "status": "interrupted" if interrupted else "passed" if returncode == 0 else "failed",
            "directory": str(shard_dir.resolve()),
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "returncode": returncode,
            "reused": False,
        }
        campaign["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json(campaign_path, campaign)
        if returncode != 0:
            if interrupted:
                campaign["status"] = "interrupted"
                atomic_json(campaign_path, campaign)
            return returncode

    campaign["status"] = "passed"
    campaign["updated_at"] = datetime.now(UTC).isoformat()
    atomic_json(campaign_path, campaign)
    print(json.dumps(campaign, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
