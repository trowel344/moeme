import json
import sys
from pathlib import Path

import pytest

from scripts.launch_sparse_candidate import candidate_command, launch_status, load_configuration


def candidate_config() -> dict:
    return {
        "training_dir": "results",
        "layer": 63,
        "source_quantized": "source.gguf",
        "candidate_quantized": "candidate.gguf",
        "imatrix": "imatrix.gguf",
        "imatrix_receipt": "imatrix.json",
        "dense_reference": "dense.gguf",
        "llama_bin_dir": "llama-bin",
        "manifest": "pipeline.json",
        "quantized_parity_report": "parity.json",
        "server_report": "server.json",
        "corpora": ["one.txt", "two.txt", "three.txt"],
        "port": 1234,
        "server_context": 4096,
        "gpu_layers": 12,
    }


def test_candidate_config_resolves_every_path_and_builds_one_command(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate_config()))
    configuration = load_configuration(path, tmp_path)
    command = candidate_command(configuration, tmp_path)
    assert command[:2] == [
        sys.executable,
        str(tmp_path / "scripts/run_sparse_candidate_pipeline.py"),
    ]
    assert command[command.index("--source-quantized") + 1] == str(tmp_path / "source.gguf")
    assert command[command.index("--port") + 1] == "1234"
    assert command[command.index("--max-median-ttft") + 1] == "6.0"
    assert command[command.index("--max-ttft") + 1] == "10.0"
    assert command[command.index("--max-long-context-ttft") + 1] == "30.0"
    assert command[command.index("--min-median-decode-rate") + 1] == "3.0"
    assert command[command.index("--min-long-decode-rate") + 1] == "2.5"
    assert [command[index + 1] for index, item in enumerate(command) if item == "--corpus"] == [
        str(tmp_path / "one.txt"),
        str(tmp_path / "two.txt"),
        str(tmp_path / "three.txt"),
    ]


def test_candidate_config_fails_closed_on_missing_or_empty_corpora(tmp_path: Path) -> None:
    value = candidate_config()
    value.pop("imatrix_receipt")
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="imatrix_receipt"):
        load_configuration(path, tmp_path)

    value = candidate_config()
    value["corpora"] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="at least one corpus"):
        load_configuration(path, tmp_path)


@pytest.mark.parametrize(
    ("returncode", "status"),
    [(0, "passed"), (130, "interrupted"), (143, "interrupted"), (1, "failed")],
)
def test_candidate_launch_status_distinguishes_resumable_interruption(
    returncode: int, status: str
) -> None:
    assert launch_status(returncode) == status
