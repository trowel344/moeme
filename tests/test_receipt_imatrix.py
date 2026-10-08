from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

from scripts.receipt_imatrix import receipt_imatrix, verify_imatrix_receipt


def test_imatrix_receipt_structurally_validates_and_hash_binds_gguf(tmp_path: Path) -> None:
    imatrix = tmp_path / "imatrix.gguf"
    receipt_path = tmp_path / "imatrix.receipt.json"
    writer = GGUFWriter(imatrix, "qwen35moe")
    writer.add_uint32("imatrix.chunk_count", 2)
    writer.add_uint32("imatrix.chunk_size", 256)
    writer.add_array("imatrix.datasets", ["test.txt"])
    writer.add_tensor("blk.0.ffn_gate.weight.in_sum2", np.ones(4, dtype=np.float32))
    writer.add_tensor("blk.0.ffn_gate.weight.counts", np.ones(1, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    receipt = receipt_imatrix(imatrix, receipt_path)
    assert receipt["bytes"] == imatrix.stat().st_size
    assert receipt["field_count"] > 0
    assert receipt["tensor_pair_count"] == 1
    assert receipt["chunk_count"] == 2
    assert receipt["chunk_size"] == 256
    assert verify_imatrix_receipt(imatrix, receipt_path)["sha256"] == receipt["sha256"]
    with imatrix.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="size does not match"):
        verify_imatrix_receipt(imatrix, receipt_path)
