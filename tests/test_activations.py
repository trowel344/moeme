import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from moeme.activations import (
    MAGIC,
    VERSION,
    ActivationTensorDataset,
    IndexedActivationDataset,
    activation_block_split_indices,
    activation_capture_paths,
    activation_capture_statistics,
    inspect_activation_capture,
    load_activation_capture,
)


def write_capture(path: Path, chunks: list[np.ndarray]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", MAGIC, VERSION, chunks[0].shape[1]))
        for values in chunks:
            handle.write(struct.pack("<I", len(values)))
            handle.write(values.astype(np.float32).tobytes())


def test_chunked_activation_capture_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "layer-0.f32"
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = np.arange(3, dtype=np.float32).reshape(1, 3) + 10
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", MAGIC, VERSION, 3))
        for values in (first, second):
            handle.write(struct.pack("<I", len(values)))
            handle.write(values.tobytes())

    info = inspect_activation_capture(path)
    assert info["tokens"] == 3
    assert info["chunk_count"] == 2
    np.testing.assert_array_equal(load_activation_capture(path), np.concatenate((first, second)))
    statistics = activation_capture_statistics(path)
    combined = np.concatenate((first, second))
    assert statistics["finite"] is True
    assert statistics["mean"] == pytest.approx(float(combined.mean()))
    assert statistics["std"] == pytest.approx(float(combined.std()))


def test_truncated_activation_capture_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.f32"
    path.write_bytes(struct.pack("<IIII", MAGIC, VERSION, 3, 2) + b"short")
    with pytest.raises(ValueError, match="truncated activation chunk payload"):
        inspect_activation_capture(path)


def test_tensor_dataset_indexes_across_chunks_and_files(tmp_path: Path) -> None:
    first_path = tmp_path / "first.f32"
    second_path = tmp_path / "second.f32"
    write_capture(first_path, [np.arange(6, dtype=np.float32).reshape(2, 3)])
    write_capture(second_path, [np.arange(6, 15, dtype=np.float32).reshape(3, 3)])
    dataset = ActivationTensorDataset([first_path, second_path])
    assert dataset.shape == (5, 3)
    torch.testing.assert_close(dataset[0], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(
        dataset[torch.tensor([4, 1, 2])],
        torch.tensor([[12.0, 13.0, 14.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
    )
    view = IndexedActivationDataset(dataset, torch.tensor([4, 0, 2]))
    torch.testing.assert_close(view[1:], torch.tensor([[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]))


def test_activation_paths_require_complete_portable_shard_campaign(tmp_path: Path) -> None:
    root = tmp_path / "shards"
    for index in range(2):
        shard = root / f"shard-{index:04d}"
        shard.mkdir(parents=True)
        write_capture(shard / "layer-63.f32", [np.ones((1, 2), dtype=np.float32)])
    campaign = {
        "status": "passed",
        "shards": {"0": {"status": "passed"}, "1": {"status": "passed"}},
    }
    (root / "campaign.json").write_text(json.dumps(campaign))
    assert activation_capture_paths(root, 63) == [
        root / "shard-0000/layer-63.f32",
        root / "shard-0001/layer-63.f32",
    ]
    campaign["status"] = "running"
    (root / "campaign.json").write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="not complete"):
        activation_capture_paths(root, 63)


def test_block_split_keeps_capture_chunks_wholly_on_one_side(tmp_path: Path) -> None:
    path = tmp_path / "capture.f32"
    chunks = [np.full((2, 3), index, dtype=np.float32) for index in range(4)]
    write_capture(path, chunks)
    dataset = ActivationTensorDataset([path])
    validation, training = activation_block_split_indices(dataset, 0.25, 7)
    assert len(validation) == 2
    assert len(training) == 6
    assert set(validation.tolist()).isdisjoint(training.tolist())
    assert sorted([*validation.tolist(), *training.tolist()]) == list(range(8))
    # Every two-row source chunk stays intact.
    for start in range(0, 8, 2):
        in_validation = [row in validation.tolist() for row in (start, start + 1)]
        assert len(set(in_validation)) == 1
