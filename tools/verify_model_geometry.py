#!/usr/bin/env python3
"""
verify_model_geometry.py -- exact parameter accounting from a Hugging Face repo.

Reads ONLY the safetensors file headers (a few hundred KB) via HTTP range
requests, so it is cheap even for a 500 GB checkpoint. No auth needed for
public repos.

This exists because the Qwen3.8-27B handoff contained parameter splits
("~17.1B FFN params, ~9.8B other") that must be independently verified from
the real artifact rather than trusted from a prior chat.

Usage:
    python3 tools/verify_model_geometry.py Qwen/Qwen3.8-27B
    python3 tools/verify_model_geometry.py Qwen/Qwen3.8-Flash-Next --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from collections import defaultdict

API = "https://huggingface.co/api/models/{repo}"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{name}"


def curl_bytes(url: str, byte_range: str | None = None) -> bytes:
    cmd = ["curl", "-sSL", "--fail", "--retry", "3", "--retry-delay", "2"]
    if byte_range:
        cmd += ["-r", byte_range]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {res.stderr.decode()[:400]}")
    return res.stdout


def curl_text(url: str) -> str:
    return curl_bytes(url).decode("utf-8")


def list_safetensors(repo: str) -> tuple[list[str], dict]:
    meta = json.loads(curl_text(API.format(repo=repo)))
    files = [
        s["rfilename"]
        for s in meta.get("siblings", [])
        if s["rfilename"].endswith(".safetensors")
    ]
    if not files:
        raise SystemExit(f"no .safetensors files found in {repo}")
    return sorted(files), meta


def read_header(repo: str, name: str) -> dict:
    """Read and parse a safetensors header using two range requests."""
    url = RESOLVE.format(repo=repo, name=name)
    n = struct.unpack("<Q", curl_bytes(url, "0-7"))[0]
    if n <= 0 or n > 500_000_000:
        raise RuntimeError(f"implausible header length {n} for {name}")
    raw = curl_bytes(url, f"8-{8 + n - 1}")
    header = json.loads(raw.decode("utf-8"))
    header.pop("__metadata__", None)
    return header


# --- classification ---------------------------------------------------------

def classify(name: str) -> tuple[str, str]:
    """Return (component, detail) for a tensor name."""
    if name.startswith(("visual.", "model.visual.", "vision_model.", "model.vision")):
        return "vision_tower", "vision_tower"
    if re.search(r"(^|\.)mtp", name) or ".mtp." in name:
        return "mtp_head", "mtp_head"
    if "embed_tokens" in name:
        return "token_embedding", "token_embedding"
    if name.startswith("lm_head") or ".lm_head" in name:
        return "lm_head", "lm_head"
    if "norm" in name:
        return "norms", "norms"

    m = re.search(r"layers\.(\d+)\.(.+)$", name)
    if not m:
        return "other", name

    layer, tail = int(m.group(1)), m.group(2)
    if "mlp" in tail or "ffn" in tail:
        comp = "ffn"
    elif "self_attn" in tail:
        comp = "attention"
    elif "linear_attn" in tail or "mamba" in tail:
        comp = "linear_attention"
    else:
        comp = "other_layer"
    # collapse to the projection name; only tag the attention kind, because the
    # FFN block lives in *both* DeltaNet and full-attention layers and tagging it
    # with the attention kind is misleading.
    detail = tail.rsplit(".", 1)[0]
    label = detail if comp == "ffn" else f"{comp}:{detail}"
    return comp, label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--json", help="also dump raw accounting to this path")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="debug: only read the first N shards")
    args = ap.parse_args()

    files, meta = list_safetensors(args.repo)
    if args.limit_shards:
        files = files[: args.limit_shards]

    by_component: dict[str, int] = defaultdict(int)
    by_detail: dict[str, int] = defaultdict(int)
    by_layer: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dtypes: dict[str, int] = defaultdict(int)
    total = 0
    ntensors = 0

    for f in files:
        header = read_header(args.repo, f)
        for name, info in header.items():
            shape = info.get("shape", [])
            n = 1
            for d in shape:
                n *= d
            comp, detail = classify(name)
            by_component[comp] += n
            by_detail[detail] += n
            dtypes[info.get("dtype", "?")] += n
            total += n
            ntensors += 1
            lm = re.search(r"layers\.(\d+)\.", name)
            if lm and comp in ("ffn", "attention", "linear_attention", "other_layer", "norms"):
                by_layer[int(lm.group(1))][comp] += n
            elif not lm:
                by_layer[-1][comp] += n

    print(f"# Model geometry: {args.repo}")
    print(f"# shards={'ALL' if not args.limit_shards else files}")
    print(f"# tensors={ntensors}  total_params={total:,}  ({total/1e9:.4f} B)")
    print(f"# dtypes={dict(dtypes)}")
    print()
    print("## Parameters by component")
    for c, n in sorted(by_component.items(), key=lambda kv: -kv[1]):
        print(f"  {c:>18}: {n:>15,}  ({100*n/total:5.2f}%)")
    print()
    print("## Parameters by module detail (top 25)")
    for c, n in sorted(by_detail.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {c:>46}: {n:>14,}  ({100*n/total:5.2f}%)")

    lm_layers = [k for k in by_layer if k >= 0]
    if lm_layers:
        print()
        print(f"## Per-layer breakdown (layers 0..{max(lm_layers)})")
        nonlayer = {k: v for k, v in by_layer.get(-1, {}).items()}
        agg: dict[str, int] = defaultdict(int)
        for k in lm_layers:
            for c, n in by_layer[k].items():
                agg[c] += n
        for c, n in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {c:>18}: {n:>15,} total over {len(lm_layers)} layers"
                  f"  ({n/len(lm_layers):>12,.0f}/layer)")
        if nonlayer:
            print("  -- non-layer tensors --")
            for c, n in sorted(nonlayer.items(), key=lambda kv: -kv[1]):
                print(f"  {c:>18}: {n:>15,}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "repo": args.repo,
                    "total_params": total,
                    "tensors": ntensors,
                    "dtypes": dict(dtypes),
                    "by_component": dict(by_component),
                    "by_detail": dict(by_detail),
                    "by_layer": {str(k): dict(v) for k, v in by_layer.items()},
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
