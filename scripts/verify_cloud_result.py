#!/usr/bin/env python3
"""Verify a downloaded cloud result before local candidate injection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moeme.cloud_results import atomic_json, verify_result_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--require-candidate-ready", action="store_true")
    args = parser.parse_args()
    result = verify_result_manifest(args.directory)
    if args.require_candidate_ready and not result["candidate_ready"]:
        raise ValueError("cloud result is intact but not eligible for candidate injection")
    if args.receipt is not None:
        atomic_json(args.receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
