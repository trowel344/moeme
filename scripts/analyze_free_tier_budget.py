#!/usr/bin/env python3
"""Use the full-shape local smoke to prove provider storage fit before upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LIGHTNING_UNBILLED_BYTES = 10_000_000_000
DEFAULT_KAGGLE_WORKING_BYTES = 20_000_000_000
DEFAULT_RESERVE_BYTES = 256 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def result_checkpoint(smoke: Path) -> Path:
    checkpoints = sorted(smoke.glob("layer-*-top*.safetensors"))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one final smoke checkpoint, found {len(checkpoints)}")
    return checkpoints[0]


def budget_report(
    stage: Path,
    smoke: Path,
    *,
    production_tokens: int,
    lightning_unbilled_bytes: int = DEFAULT_LIGHTNING_UNBILLED_BYTES,
    kaggle_working_bytes: int = DEFAULT_KAGGLE_WORKING_BYTES,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
) -> dict:
    manifest = json.loads((stage / "upload-manifest.json").read_text())
    if manifest.get("format") != "moeme-free-cloud-upload-v1":
        raise ValueError("unsupported upload manifest")
    preflight = json.loads((smoke / "cloud-preflight.json").read_text())
    smoke_tokens = int(preflight["activation"]["tokens"])
    if smoke_tokens <= 0 or production_tokens <= 0:
        raise ValueError("activation token counts must be positive")
    state = smoke / "training-state.pt"
    checkpoint = result_checkpoint(smoke)
    if not state.is_file():
        raise FileNotFoundError(f"smoke restart state is missing: {state}")
    labels = smoke / "oracle-labels"
    smoke_label_bytes = directory_bytes(labels) if labels.is_dir() else 0
    scaled_label_bytes = (smoke_label_bytes * production_tokens + smoke_tokens - 1) // smoke_tokens
    total_smoke_bytes = directory_bytes(smoke)
    fixed_smoke_bytes = max(
        0,
        total_smoke_bytes - state.stat().st_size - checkpoint.stat().st_size - smoke_label_bytes,
    )
    # Atomic restart publication briefly retains the prior state and its complete
    # replacement. The final BF16 checkpoint, scaled oracle cache, receipts/logs,
    # and an explicit reserve remain alongside them.
    output_atomic_peak = (
        2 * state.stat().st_size
        + checkpoint.stat().st_size
        + scaled_label_bytes
        + fixed_smoke_bytes
        + reserve_bytes
    )
    output_steady = (
        state.stat().st_size
        + checkpoint.stat().st_size
        + scaled_label_bytes
        + fixed_smoke_bytes
        + reserve_bytes
    )
    stage_bytes = int(manifest["logical_bytes"])
    lightning_peak = stage_bytes + output_atomic_peak
    lightning_fits = lightning_peak <= lightning_unbilled_bytes
    kaggle_fits = output_atomic_peak <= kaggle_working_bytes
    recommended = "lightning" if lightning_fits else "kaggle" if kaggle_fits else None
    return {
        "format": "moeme-free-tier-budget-v1",
        "analyzed_at": datetime.now(UTC).isoformat(),
        "limit_assumptions": {
            "checked_at": "2026-10-08",
            "scope": "incremental job footprint; account-wide existing usage is not observable locally",
            "lightning_source": "https://api.lightning.ai/docs/platform/overview/faq/billing",
            "kaggle_source": "https://www.kaggle.com/docs/notebooks",
            "requires_account_usage_check": True,
        },
        "measured_from": {
            "stage_manifest": str((stage / "upload-manifest.json").resolve()),
            "stage_manifest_sha256": sha256(stage / "upload-manifest.json"),
            "smoke": str(smoke.resolve()),
            "smoke_activation_tokens": smoke_tokens,
            "production_activation_tokens": production_tokens,
        },
        "components": {
            "stage_logical_bytes": stage_bytes,
            "restart_state_bytes": state.stat().st_size,
            "final_checkpoint_bytes": checkpoint.stat().st_size,
            "smoke_oracle_label_bytes": smoke_label_bytes,
            "scaled_oracle_label_bytes": scaled_label_bytes,
            "fixed_smoke_result_bytes": fixed_smoke_bytes,
            "reserve_bytes": reserve_bytes,
            "estimated_output_steady_bytes": output_steady,
            "estimated_output_atomic_peak_bytes": output_atomic_peak,
        },
        "providers": {
            "lightning": {
                "limit_bytes": lightning_unbilled_bytes,
                "estimated_peak_bytes": lightning_peak,
                "fits_unbilled_storage": lightning_fits,
                "fit_is_incremental": True,
            },
            "kaggle": {
                "limit_bytes": kaggle_working_bytes,
                "estimated_peak_bytes": output_atomic_peak,
                "fits_working_storage": kaggle_fits,
                "stage_is_read_only_input": True,
            },
        },
        "recommended_provider": recommended,
        "passed": recommended is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=Path("cloud-jobs/layer63-200k"))
    parser.add_argument("--smoke", type=Path, default=Path("cloud-results/local-seeded-smoke"))
    parser.add_argument("--production-tokens", type=int, default=204800)
    parser.add_argument("--output", type=Path, default=Path(".moeme/free-tier-budget.json"))
    parser.add_argument(
        "--lightning-unbilled-bytes", type=int, default=DEFAULT_LIGHTNING_UNBILLED_BYTES
    )
    parser.add_argument("--kaggle-working-bytes", type=int, default=DEFAULT_KAGGLE_WORKING_BYTES)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    args = parser.parse_args()
    report = budget_report(
        args.stage.resolve(),
        args.smoke.resolve(),
        production_tokens=args.production_tokens,
        lightning_unbilled_bytes=args.lightning_unbilled_bytes,
        kaggle_working_bytes=args.kaggle_working_bytes,
        reserve_bytes=args.reserve_bytes,
    )
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
