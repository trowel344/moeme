import hashlib
import json
from pathlib import Path

import pytest

from scripts.capture_activation_shards import (
    bind_shard_source,
    reusable_shard,
    shard_plan,
    validate_existing_campaign,
)


def test_shard_plan_covers_total_without_overlap() -> None:
    assert shard_plan(9, 4) == [
        {"index": 0, "from_chunk": 0, "chunks": 4},
        {"index": 1, "from_chunk": 4, "chunks": 4},
        {"index": 2, "from_chunk": 8, "chunks": 1},
    ]


def test_only_matching_complete_shard_is_reusable(tmp_path: Path) -> None:
    shard = tmp_path / "shard-0000"
    shard.mkdir()
    payload = shard / "layer-63.f32"
    payload.write_bytes(b"capture")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (shard / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_sha256": "abc",
                "from_chunk": 100,
                "tokens_per_layer": 25600,
                "layers": {
                    "63": {
                        "tokens": 25600,
                        "bytes": payload.stat().st_size,
                        "sha256": digest,
                    }
                },
                "source_model": {"sha256": "model"},
                "capture_binary": {"sha256": "binary"},
            }
        )
    )
    assert reusable_shard(
        shard,
        layer=63,
        from_chunk=100,
        chunks=100,
        ctx_size=256,
        corpus_sha256="abc",
        model_sha256="model",
        binary_sha256="binary",
    )
    assert not reusable_shard(
        shard,
        layer=63,
        from_chunk=0,
        chunks=100,
        ctx_size=256,
        corpus_sha256="abc",
        model_sha256="model",
        binary_sha256="binary",
    )
    payload.write_bytes(b"corrupt")
    assert not reusable_shard(
        shard,
        layer=63,
        from_chunk=100,
        chunks=100,
        ctx_size=256,
        corpus_sha256="abc",
        model_sha256="model",
        binary_sha256="binary",
    )


def test_source_binding_and_campaign_contract_are_content_bound(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "manifest.json").write_text('{"format":"capture"}')
    model = tmp_path / "model.gguf"
    binary = tmp_path / "llama-imatrix"
    model.write_bytes(b"model")
    binary.write_bytes(b"binary")
    bind_shard_source(
        shard,
        model=model,
        model_sha256="model-sha",
        binary=binary,
        binary_sha256="binary-sha",
    )
    manifest = json.loads((shard / "manifest.json").read_text())
    assert manifest["source_model"]["bytes"] == 5
    assert manifest["capture_binary"]["sha256"] == "binary-sha"

    expected = {
        "format": "campaign",
        "corpus_sha256": "corpus",
        "source_model": manifest["source_model"],
        "capture_binary": manifest["capture_binary"],
        "layer": 63,
        "ctx_size": 256,
        "total_chunks": 10,
        "chunks_per_shard": 5,
        "total_tokens": 2560,
        "shards": {"0": {"status": "passed"}},
    }
    assert validate_existing_campaign(expected, expected) == expected["shards"]
    changed = {**expected, "source_model": {"sha256": "different"}}
    with pytest.raises(ValueError, match="source_model"):
        validate_existing_campaign(changed, expected)
