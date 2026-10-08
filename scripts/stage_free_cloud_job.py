#!/usr/bin/env python3
"""Atomically stage one upload-ready, zero-copy free-cloud training folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from moeme.activations import activation_capture_paths, inspect_activation_capture

SOURCE_FILES = (
    "pyproject.toml",
    "README.md",
    "scripts/analyze_sparse_curve.py",
    "scripts/cloud_bootstrap.py",
    "scripts/launch_free_cloud_job.py",
    "scripts/oracle_router_analysis.py",
    "scripts/run_free_cloud_training.py",
    "scripts/stage_free_cloud_job.py",
    "scripts/train_sparse_layer.py",
    "scripts/verify_cloud_result.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("cloud configuration must be a JSON object")
    for key in ("activations", "checkpoint", "seed_checkpoint", "layer", "output_dir"):
        if key not in value:
            raise ValueError(f"cloud configuration is missing {key}")
    return value


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def json_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def verify_expected_file(path: Path, receipt: dict, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    size = path.stat().st_size
    if size != int(receipt.get("bytes", -1)):
        raise ValueError(f"{label} size does not match its receipt")
    digest = sha256(path)
    if digest != receipt.get("sha256"):
        raise ValueError(f"{label} SHA-256 does not match its receipt")
    return {"bytes": size, "sha256": digest}


def input_files(config_path: Path, config: dict) -> tuple[list[Path], dict[Path, dict], dict]:
    activations = resolve_config_path(config_path, config["activations"])
    checkpoint = resolve_config_path(config_path, config["checkpoint"])
    seed = resolve_config_path(config_path, config["seed_checkpoint"])
    layer = int(config["layer"])
    required_directories = (activations, checkpoint)
    for directory in required_directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"cloud input directory is not ready: {directory}")
    if not seed.is_file():
        raise FileNotFoundError(f"cloud seed is not ready: {seed}")
    verified: dict[Path, dict] = {}
    activation_tokens = 0
    activation_width = None
    for path in activation_capture_paths(activations, layer):
        info = inspect_activation_capture(path)
        manifest = json_object(path.parent / "manifest.json")
        receipt = (manifest.get("layers") or {}).get(str(layer))
        if not isinstance(receipt, dict):
            raise TypeError(f"activation manifest has no layer-{layer} receipt: {path.parent}")
        if info["tokens"] != manifest.get("tokens_per_layer"):
            raise ValueError("activation token count does not match its manifest")
        verified[path.resolve()] = verify_expected_file(path, receipt, "activation capture")
        activation_tokens += int(info["tokens"])
        width = int(info["width"])
        if activation_width is None:
            activation_width = width
        elif activation_width != width:
            raise ValueError("activation shards have different widths")
    checkpoint_manifest = json_object(checkpoint / "moeme-manifest.json")
    checkpoint_receipt = json_object(checkpoint / "receipt.json")
    checkpoint_index = checkpoint / "model.safetensors.index.json"
    if not checkpoint_index.is_file():
        raise FileNotFoundError(f"portable checkpoint index is missing: {checkpoint_index}")
    checkpoint_tensor = checkpoint / str(checkpoint_receipt["tensor_file"])
    if int(checkpoint_manifest.get("layer", -1)) != layer:
        raise ValueError("portable checkpoint is for a different layer")
    verified[checkpoint_tensor.resolve()] = verify_expected_file(
        checkpoint_tensor, checkpoint_receipt, "portable checkpoint"
    )
    seed_receipt = json_object(seed.parent / "receipt.json")
    if int(seed_receipt.get("layer", -1)) != layer:
        raise ValueError("training seed is for a different layer")
    verified[seed.resolve()] = verify_expected_file(seed, seed_receipt, "training seed")
    files = [
        path
        for directory in required_directories
        for path in directory.rglob("*")
        if path.is_file()
    ]
    seed_receipt_path = seed.parent / "receipt.json"
    files.extend((seed, seed_receipt_path))
    summary = {
        "layer": layer,
        "activation_tokens": activation_tokens,
        "activation_width": activation_width,
        "activation_files": len(activation_capture_paths(activations, layer)),
        "critical_hashes_verified": len(verified),
    }
    return sorted({path.resolve() for path in files}), verified, summary


def stage_file(source: Path, destination: Path, *, hardlink: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(source, destination)
        except OSError as error:
            raise RuntimeError(
                f"cannot zero-copy stage {source}; output must be on the same filesystem"
            ) from error
    else:
        shutil.copy2(source, destination)


def file_receipt(path: Path, role: str, hardlinked: bool, verified: dict | None = None) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": (verified or {}).get("sha256") or sha256(path),
        "role": role,
        "hardlinked": hardlinked,
    }


def verify_stage(directory: Path) -> dict:
    manifest_path = directory / "upload-manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("upload manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "moeme-free-cloud-upload-v1":
        raise ValueError("unsupported cloud upload manifest")
    expected_files = set(manifest.get("files", {}))
    symlinks = sorted(
        str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_symlink()
    )
    if symlinks:
        raise ValueError("staged cloud job contains symlink(s): " + ", ".join(symlinks))
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    missing = sorted(expected_files - actual_files)
    unexpected = sorted(actual_files - expected_files)
    if missing or unexpected:
        raise ValueError(
            f"staged cloud inventory mismatch; missing={missing}, unexpected={unexpected}"
        )
    verified_bytes = 0
    for name, receipt in manifest.get("files", {}).items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"staged cloud file is missing: {name}")
        if path.stat().st_size != receipt.get("bytes"):
            raise ValueError(f"staged cloud file size mismatch: {name}")
        if sha256(path) != receipt.get("sha256"):
            raise ValueError(f"staged cloud file hash mismatch: {name}")
        verified_bytes += path.stat().st_size
    return {
        "format": "moeme-free-cloud-upload-verification-v1",
        "directory": str(directory.resolve()),
        "file_count": len(manifest.get("files", {})),
        "bytes": verified_bytes,
        "verified": True,
    }


def build_stage(
    repository: Path, config_path: Path, output: Path, *, rebuild: bool = False
) -> dict:
    repository = repository.resolve()
    config_path = config_path.resolve()
    output = output.resolve()
    config = load_config(config_path)
    inputs, verified_inputs, input_validation = input_files(config_path, config)
    sources = [repository / name for name in SOURCE_FILES]
    sources.append(config_path)
    sources.extend(sorted((repository / "src/moeme").glob("*.py")))
    missing_sources = [str(path) for path in sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError("cloud source file(s) missing: " + ", ".join(missing_sources))
    if output.exists() and not rebuild:
        return verify_stage(output)
    if output.exists():
        verify_stage(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        receipts = {}
        for source in sources:
            relative = source.relative_to(repository)
            destination = temporary / relative
            stage_file(source, destination, hardlink=False)
            receipts[str(relative)] = file_receipt(destination, "source", False)
        for source in inputs:
            relative = source.relative_to(repository)
            destination = temporary / relative
            stage_file(source, destination, hardlink=True)
            receipts[str(relative)] = file_receipt(
                destination,
                "training-input",
                True,
                verified_inputs.get(source.resolve()),
            )
        runbook = temporary / "RUN.md"
        runbook.write_text(
            "# MoEMe free-cloud layer-63 job\n\n"
            "## Kaggle (selected by the measured free-tier budget)\n\n"
            "```bash\n"
            "STAGE=/kaggle/input/<private-dataset>\n"
            'python3 "$STAGE/scripts/cloud_bootstrap.py" '
            '--stage "$STAGE" --provider kaggle\n'
            "```\n\n"
            "The local launch_kaggle_kernel.py helper normally generates and submits "
            "this private job. The command above is the manual fallback.\n\n"
            "## Lightning adapter (only if its current free storage fits)\n\n"
            "```bash\n"
            "python3 scripts/cloud_bootstrap.py --provider lightning\n"
            "```\n\n"
            "Kaggle inputs remain under `/kaggle/input`; runtime configuration, exact "
            "restart state and results are redirected to `/kaggle/working/moeme`. Attach "
            "a prior saved result directory with `--resume-from /kaggle/input/<prior>/` "
            "when resuming in a new session.\n"
        )
        receipts["RUN.md"] = file_receipt(runbook, "runbook", False)
        manifest = {
            "format": "moeme-free-cloud-upload-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "configuration": str(config_path.relative_to(repository)),
            "layer": int(config["layer"]),
            "input_validation": input_validation,
            "files": dict(sorted(receipts.items())),
            "logical_bytes": sum(item["bytes"] for item in receipts.values()),
            "large_inputs_are_hardlinked": True,
        }
        (temporary / "upload-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            backup = output.parent / f".{output.name}.rebuild-backup"
            if backup.exists():
                raise FileExistsError(f"stale rebuild backup requires inspection: {backup}")
            os.replace(output, backup)
            try:
                os.replace(temporary, output)
                verify_stage(output)
            except BaseException:
                if output.exists():
                    shutil.rmtree(output)
                os.replace(backup, output)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, output)
        return verify_stage(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/free-cloud-layer63-200k.json"))
    parser.add_argument("--output", type=Path, default=Path("cloud-jobs/layer63-200k"))
    parser.add_argument("--verify-stage", type=Path)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="atomically replace a verified generated stage with current source/configuration",
    )
    args = parser.parse_args()
    if args.verify_stage is not None and args.rebuild:
        parser.error("--rebuild cannot be combined with --verify-stage")
    result = (
        verify_stage(args.verify_stage.resolve())
        if args.verify_stage is not None
        else build_stage(args.repository, args.config, args.output, rebuild=args.rebuild)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
