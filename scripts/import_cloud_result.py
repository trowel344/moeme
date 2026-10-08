#!/usr/bin/env python3
"""Atomically import and hash-verify a downloaded MoEMe cloud result."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from moeme.cloud_results import atomic_json, verify_result_manifest


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate_directory_source(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"cloud result directory contains a symbolic link: {item}")
        if item.is_file():
            total += item.stat().st_size
    return total


def validate_zip(archive: zipfile.ZipFile) -> int:
    total = 0
    for member in archive.infolist():
        path = Path(member.filename)
        mode = member.external_attr >> 16
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe path in cloud result archive: {member.filename}")
        if stat.S_ISLNK(mode):
            raise ValueError(f"symbolic link in cloud result archive: {member.filename}")
        total += member.file_size
    return total


def locate_result_directory(root: Path) -> Path:
    manifests = sorted(root.rglob("cloud-result-manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected exactly one cloud-result-manifest.json, found {len(manifests)}")
    return manifests[0].parent


def require_import_space(target: Path, bytes_required: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free
    required = bytes_required + 1_000_000_000
    if free < required:
        raise OSError(
            f"cloud result import requires {required:,} free bytes; only {free:,} available"
        )


def import_result(source: Path, target: Path, receipt: Path) -> dict:
    source = source.resolve()
    target = target.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        if source == target:
            result = verify_result_manifest(target)
            atomic_json(receipt, result)
            return result
        raise FileExistsError(f"refusing to overlay existing cloud result: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.import.", dir=target.parent) as raw:
        temporary = Path(raw)
        if source.is_dir():
            total = validate_directory_source(source)
            require_import_space(target, total)
            candidate = temporary / "result"
            shutil.copytree(source, candidate)
        elif zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                total = validate_zip(archive)
                require_import_space(target, total)
                extracted = temporary / "extracted"
                extracted.mkdir()
                archive.extractall(extracted)
            located = locate_result_directory(extracted)
            candidate = temporary / "result"
            os.replace(located, candidate)
        else:
            raise ValueError("cloud result source must be a directory or ZIP archive")

        result = verify_result_manifest(candidate)
        imported_bytes = directory_bytes(candidate)
        os.replace(candidate, target)
    result = {
        **result,
        "source": str(source),
        "target": str(target),
        "imported_bytes": imported_bytes,
        "atomic_publish": True,
        "source_preserved": True,
    }
    atomic_json(receipt, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=Path("cloud-results/layer63-200k"))
    parser.add_argument("--receipt", type=Path, default=Path(".moeme/layer63-cloud-transfer.json"))
    args = parser.parse_args()
    result = import_result(args.source, args.target, args.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
