#!/usr/bin/env python3
"""Build the pinned, patched llama.cpp runtime on a fresh cloud machine."""

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

REPOSITORY = "https://github.com/Lidenburg/llama.cpp.git"
COMMIT = "e85e4d90cd44a8c8332093b0c97a53fa13137f5e"


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


def run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{tail}")
    return completed.stdout


def ensure_source(source: Path) -> None:
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", REPOSITORY, str(source)])
    if not (source / ".git").is_dir():
        raise ValueError(f"runtime source is not a git checkout: {source}")
    run(["git", "fetch", "origin", COMMIT, "--depth", "1"], cwd=source)
    run(["git", "checkout", "--detach", COMMIT], cwd=source)


def apply_patch_once(source: Path, patch: Path) -> str:
    forward = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if forward.returncode == 0:
        run(["git", "apply", str(patch)], cwd=source)
        return "applied"
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if reverse.returncode == 0:
        return "already-applied"
    raise RuntimeError(
        "runtime patch does not apply cleanly in either direction:\n"
        + "\n".join(forward.stderr.splitlines()[-20:])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("cloud-runtime/llama.cpp"))
    parser.add_argument(
        "--patch", type=Path, default=Path("docs/patches/llama-moeme-e85e4d9.patch")
    )
    parser.add_argument("--receipt", type=Path, default=Path("cloud-runtime/runtime-receipt.json"))
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    for command in ("git", "cmake"):
        if shutil.which(command) is None:
            raise RuntimeError(f"required build command is missing: {command}")
    source = args.source.resolve()
    patch = args.patch.resolve()
    ensure_source(source)
    patch_status = apply_patch_once(source, patch)
    build = source / "build"
    run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DGGML_CUDA=ON",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "-j",
            str(args.jobs),
            "--target",
            "llama-server",
            "llama-imatrix",
        ]
    )
    binaries = {}
    for name in ("llama-server", "llama-imatrix"):
        path = build / "bin" / name
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"build did not produce an executable {name}: {path}")
        binaries[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    receipt = {
        "format": "moeme-cloud-runtime-v1",
        "built_at": datetime.now(UTC).isoformat(),
        "repository": REPOSITORY,
        "commit": run(["git", "rev-parse", "HEAD"], cwd=source).strip(),
        "patch": str(patch),
        "patch_sha256": sha256(patch),
        "patch_status": patch_status,
        "binaries": binaries,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
