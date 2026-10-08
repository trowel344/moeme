#!/usr/bin/env python3
"""Build a pinned, reproducible corpus for the sparse-training de-risk."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILES = (
    "wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "wikitext-103-raw-v1/train-00001-of-00002.parquet",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/moeme-27b-48e"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/training/layer63-de-risk-200k.txt")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/training/layer63-de-risk-200k.json")
    )
    parser.add_argument("--target-tokens", type=int, default=205_000)
    args = parser.parse_args()
    if args.target_tokens <= 0:
        parser.error("--target-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    sections = []
    project_paths = sorted(Path("src/moeme").glob("*.py")) + sorted(Path("docs").glob("*.md"))
    for path in project_paths:
        sections.append(f"Document: {path}\n\n{path.read_text(encoding='utf-8')}")

    source_files = []
    token_count = len(tokenizer.encode("\n\n".join(sections), add_special_tokens=False))
    for filename in WIKITEXT_FILES:
        parquet = Path(
            hf_hub_download(
                WIKITEXT_REPO,
                filename,
                repo_type="dataset",
                revision=WIKITEXT_REVISION,
            )
        )
        source_files.append(
            {
                "repo_file": filename,
                "cached_path": str(parquet),
                "sha256": sha256_file(parquet),
            }
        )
        table = pq.read_table(parquet, columns=["text"])
        for value in table.column("text").to_pylist():
            text = value.strip()
            if not text:
                continue
            sections.append(text)
            token_count += len(tokenizer.encode(text + "\n\n", add_special_tokens=False))
            if token_count >= args.target_tokens:
                break
        if token_count >= args.target_tokens:
            break

    if token_count < args.target_tokens:
        raise RuntimeError(f"source corpus yielded only {token_count} tokens")
    content = "\n\n".join(sections) + "\n"
    exact_tokens = len(tokenizer.encode(content, add_special_tokens=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    manifest = {
        "format": "moeme-sparse-training-corpus-v1",
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_bytes(content.encode()),
        "tokens": exact_tokens,
        "target_tokens": args.target_tokens,
        "tokenizer": str(args.tokenizer.resolve()),
        "wikitext": {
            "repo": WIKITEXT_REPO,
            "revision": WIKITEXT_REVISION,
            "files": source_files,
        },
        "project_files": [str(path) for path in project_paths],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
