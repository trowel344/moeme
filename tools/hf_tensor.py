#!/usr/bin/env python3
"""
hf_tensor.py -- fetch individual tensors out of a Hugging Face safetensors repo.

Why this exists: the full Qwen3.8-27B checkpoint is 55.6 GB. To verify that the
FFN partitioning is exactly dense-equivalent on the *real* weights we only need
one layer's gate/up/down matrices (~535 MB). Safetensors is laid out so that
this is possible with HTTP range requests: read the header, look up the byte
offset of the tensor we want, fetch just that range.

Usage (as a library):
    from hf_tensor import load_tensors, index_for
    idx = index_for("Qwen/Qwen3.8-27B")
    tensors = load_tensors("Qwen/Qwen3.8-27B", idx,
                           ["model.language_model.layers.1.mlp.gate_proj.weight"])

Usage (to inspect):
    python3 tools/hf_tensor.py Qwen/Qwen3.8-27B --pattern 'layers\\.1\\.mlp'
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys

CACHE = os.path.join(".cache", "moeme")
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{name}"


def _curl(url: str, byte_range: str | None = None, retries: int = 3) -> bytes:
    cmd = ["curl", "-sSL", "--fail", "--retry", str(retries), "--retry-delay", "2"]
    if byte_range:
        cmd += ["-r", byte_range]
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {res.stderr.decode()[:300]}")
    return res.stdout


def index_for(repo: str) -> dict:
    """Return the safetensors index (weight_map + metadata)."""
    return json.loads(
        _curl(f"https://huggingface.co/{repo}/raw/main/model.safetensors.index.json")
    )


def read_shard_header(repo: str, shard: str) -> tuple[dict, int]:
    """Parse a shard's header; also return the header's byte length."""
    url = RESOLVE.format(repo=repo, name=shard)
    n = struct.unpack("<Q", _curl(url, "0-7"))[0]
    raw = _curl(url, f"8-{8 + n - 1}")
    header = json.loads(raw.decode("utf-8"))
    header.pop("__metadata__", None)
    return header, n


def get_tensor(repo: str, name: str, index: dict | None = None,
               use_cache: bool = True):
    """Download a single tensor and return (torch.Tensor, header_entry)."""
    import numpy as np
    import torch

    index = index or index_for(repo)
    try:
        shard = index["weight_map"][name]
    except KeyError:
        raise KeyError(f"{name} not in {repo}; check the exact tensor name")

    os.makedirs(CACHE, exist_ok=True)
    safe = name.replace("/", "_")
    cache_path = os.path.join(CACHE, f"{repo.replace('/', '__')}__{safe}.bin")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            raw = fh.read()
        meta_path = cache_path + ".json"
        with open(meta_path) as fh:
            info = json.load(fh)
    else:
        header, hlen = read_shard_header(repo, shard)
        info = header[name]
        begin, end = info["data_offsets"]
        base = 8 + hlen
        raw = _curl(RESOLVE.format(repo=repo, name=shard),
                    f"{base + begin}-{base + end - 1}")
        if len(raw) != end - begin:
            raise RuntimeError(
                f"short read for {name}: got {len(raw)} want {end - begin}"
            )
        with open(cache_path, "wb") as fh:
            fh.write(raw)
        with open(cache_path + ".json", "w") as fh:
            json.dump(info, fh)

    np_dtype = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32,
                "I8": np.int8, "U8": np.uint8}[info["dtype"]]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(info["shape"])
    t = torch.from_numpy(arr.copy())
    torch_dtype = {"BF16": torch.bfloat16, "F16": torch.float16,
                   "F32": torch.float32, "I8": torch.int8,
                   "U8": torch.uint8}[info["dtype"]]
    return t.view(torch_dtype), info


def load_tensors(repo: str, index: dict, names: list[str], **kw):
    out = {}
    for n in names:
        out[n], _ = get_tensor(repo, n, index, **kw)
    return out


def main() -> int:
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--pattern", default=".", help="regex over tensor names")
    ap.add_argument("--fetch", action="append", default=[],
                    help="tensor name to actually download")
    args = ap.parse_args()

    idx = index_for(args.repo)
    names = sorted(n for n in idx["weight_map"] if re.search(args.pattern, n))
    print(f"{len(names)} tensors matching {args.pattern!r}:")
    for n in names[:40]:
        print("  ", n)
    if len(names) > 40:
        print(f"   ... and {len(names) - 40} more")

    for n in args.fetch:
        t, info = get_tensor(args.repo, n, idx)
        print(f"\nfetched {n}: shape={tuple(t.shape)} dtype={t.dtype} "
              f"nbytes={t.numel() * t.element_size()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
