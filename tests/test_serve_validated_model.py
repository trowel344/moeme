import hashlib
import json
from pathlib import Path

import pytest

from scripts.serve_validated_model import (
    release_environment,
    server_command,
    validate_release,
)


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def receipts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"qualified model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    pipeline = tmp_path / "pipeline.json"
    server = tmp_path / "server.json"
    parity = tmp_path / "parity.json"
    write(
        pipeline,
        {
            "phases": {
                "quantization": {"status": "passed", "sha256": digest},
                "evaluation": {"status": "passed"},
            }
        },
    )
    write(
        server,
        {
            "passed": True,
            "quality_passed": True,
            "stability_passed": True,
            "performance_passed": True,
            "suite_digest": "suite",
        },
    )
    write(
        parity,
        {
            "passed": True,
            "candidate_sha256": digest,
            "results": [{}, {}, {}],
        },
    )
    return model, pipeline, server, parity


def test_validated_release_requires_all_receipts_and_hash(tmp_path: Path) -> None:
    model, pipeline, server, parity = receipts(tmp_path)
    result = validate_release(model, pipeline, server, parity)
    assert result["parity_corpora"] == 3
    assert result["bytes"] == len(b"qualified model")


def test_validated_release_rejects_failed_parity(tmp_path: Path) -> None:
    model, pipeline, server, parity = receipts(tmp_path)
    write(parity, {"passed": False})
    with pytest.raises(ValueError, match="source-logit parity"):
        validate_release(model, pipeline, server, parity)


def test_validated_release_rejects_stale_parity_receipt(tmp_path: Path) -> None:
    model, pipeline, server, parity = receipts(tmp_path)
    write(parity, {"passed": True, "candidate_sha256": "old", "results": []})
    with pytest.raises(ValueError, match="not bound"):
        validate_release(model, pipeline, server, parity)


def test_validated_release_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    model, pipeline, server, parity = receipts(tmp_path)
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_release(model, pipeline, server, parity)


def test_release_environment_leaves_routing_native_by_default() -> None:
    environment = release_environment({"MOEME_TOP4_LAYERS": "0,1"}, None)
    assert "MOEME_TOP4_LAYERS" not in environment


def test_release_environment_forces_only_explicit_top4_layers() -> None:
    environment = release_environment({}, (0, 1, 63))
    assert environment["MOEME_TOP4_LAYERS"] == "0,1,63"


def test_release_environment_empty_layer_list_keeps_native_routing() -> None:
    environment = release_environment({}, ())
    assert "MOEME_TOP4_LAYERS" not in environment


def test_server_command_uses_8k_context_and_bounded_prompt_cache() -> None:
    command = server_command(
        Path("llama-server"),
        Path("model.gguf"),
        "127.0.0.1",
        18080,
        8192,
        8,
        0,
    )
    assert command[command.index("-c") + 1] == "8192"
    assert command[command.index("--cache-ram") + 1] == "0"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--fit") + 1] == "off"
    assert command[command.index("-ngl") + 1] == "20"
    assert command[command.index("-fa") + 1] == "on"


def test_server_command_keeps_upstream_batch_sizes_and_bounds_context_checkpoints() -> None:
    """Batch sizes stay at llama.cpp's defaults; checkpoints are bounded.

    Lowering the batch sizes aborted the model with a cuBLAS error on long
    prompts and cost ~60% of prefill throughput, so the defaults are kept.
    Context checkpoints are still bounded: the default 32 reserves up to
    ~4.8 GB of RAM on this host and was measured to collapse served throughput.
    """
    command = server_command(
        Path("llama-server"),
        Path("model.gguf"),
        "127.0.0.1",
        18080,
        8192,
        8,
        0,
    )
    assert command[command.index("-b") + 1] == "2048"
    assert command[command.index("-ub") + 1] == "512"
    assert command[command.index("--ctx-checkpoints") + 1] == "2"


def test_server_command_honors_explicit_batch_and_checkpoint_settings() -> None:
    command = server_command(
        Path("llama-server"),
        Path("model.gguf"),
        "127.0.0.1",
        18080,
        8192,
        8,
        0,
        batch_size=512,
        ubatch_size=128,
        context_checkpoints=0,
    )
    assert command[command.index("-b") + 1] == "512"
    assert command[command.index("-ub") + 1] == "128"
    assert command[command.index("--ctx-checkpoints") + 1] == "0"


def test_server_command_honors_gpu_layers_and_disables_flash_attention() -> None:
    command = server_command(
        Path("llama-server"),
        Path("model.gguf"),
        "127.0.0.1",
        18080,
        8192,
        8,
        0,
        gpu_layers=0,
        flash_attention=False,
    )
    assert command[command.index("-ngl") + 1] == "0"
    assert "-fa" not in command
