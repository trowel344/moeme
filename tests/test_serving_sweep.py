from pathlib import Path

from scripts.serving_sweep import server_command, sha256


def test_kaggle_residency_command_keeps_full_offload_flags(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    config = {
        "name": "t4x2",
        "context": 8192,
        "gpu_layers": 99,
        "batch": 2048,
        "ubatch": 512,
        "extra": ["--split-mode", "layer", "--ctx-checkpoints", "2"],
    }
    command = server_command(Path("llama-server"), model, config, 18099, 4)
    assert command[command.index("-ngl") + 1] == "99"
    assert command[command.index("--split-mode") + 1] == "layer"
    assert command[command.index("--ctx-checkpoints") + 1] == "2"
    assert len(sha256(model)) == 64
