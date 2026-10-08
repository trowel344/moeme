#!/usr/bin/env python3
"""Build a small, pinned text/code/math corpus for router calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/calibration/router-v1.txt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/calibration/router-v1.json"))
    parser.add_argument("--wiki-lines", type=int, default=800)
    args = parser.parse_args()

    parquet = hf_hub_download(
        WIKITEXT_REPO,
        WIKITEXT_FILE,
        repo_type="dataset",
        revision=WIKITEXT_REVISION,
    )
    table = pq.read_table(parquet, columns=["text"])
    wiki = []
    for value in table.column("text").to_pylist():
        text = value.strip()
        if text:
            wiki.append(text)
        if len(wiki) >= args.wiki_lines:
            break

    project_files = sorted(Path("src/moeme").glob("*.py")) + sorted(Path("docs").glob("*.md"))
    code_and_docs = [path.read_text(encoding="utf-8") for path in project_files]
    math = [
        f"Problem: Calculate {a} multiplied by {b}. Answer: {a * b}."
        for a in range(11, 40, 3)
        for b in range(7, 30, 5)
    ]
    instructions = [
        "User: Summarize the following material accurately and list unresolved risks. Assistant:",
        "User: Inspect this Python function for correctness, edge cases, and unsafe assumptions. Assistant:",
        "User: Return a JSON object with keys status, evidence, and next_action. Assistant:",
        "User: Explain the tradeoff, then give a concrete recommendation. Assistant:",
    ]
    sections = wiki + code_and_docs + math + instructions
    content = "\n\n".join(sections) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode()).hexdigest()
    manifest = {
        "format": "moeme-router-calibration-v1",
        "output": str(args.output.resolve()),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "wikitext": {
            "repo": WIKITEXT_REPO,
            "revision": WIKITEXT_REVISION,
            "file": WIKITEXT_FILE,
            "lines": len(wiki),
        },
        "project_files": [str(path) for path in project_files],
        "math_examples": len(math),
        "instruction_examples": len(instructions),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
