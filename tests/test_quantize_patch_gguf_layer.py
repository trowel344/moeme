from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

from scripts.quantize_patch_gguf_layer import (
    TENSOR_NAMES,
    checkpoint_array,
    patch_quantized_copy,
    quantize_layer_carrier,
    write_layer_carrier,
)


def write_test_model(path: Path, value: int, *, include_extra: bool = True) -> None:
    writer = GGUFWriter(path, "qwen35moe")
    writer.add_name("layer patch test")
    for suffix in TENSOR_NAMES.values():
        name = f"blk.1.{suffix}"
        if suffix == "ffn_gate_inp.weight":
            array = np.full((2, 4), float(value), dtype=np.float32)
            raw_dtype = GGMLQuantizationType.F32
        else:
            array = np.full((2, 4), value, dtype=np.uint16)
            raw_dtype = GGMLQuantizationType.BF16
        writer.add_tensor(name, array, raw_dtype=raw_dtype)
    if include_extra:
        writer.add_tensor("blk.0.attn_norm.weight", np.arange(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def tensor_bytes(path: Path) -> dict[str, bytes]:
    reader = GGUFReader(path, "r")
    result = {tensor.name: tensor.data.tobytes() for tensor in reader.tensors}
    del reader
    return result


def test_checkpoint_array_preserves_bf16_bits_and_f32() -> None:
    bf16 = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    array, kind = checkpoint_array(bf16)
    assert kind == GGMLQuantizationType.BF16
    assert array.dtype == np.uint16
    assert np.array_equal(array, bf16.view(torch.uint16).numpy())

    f32 = torch.tensor([[3.0]], dtype=torch.float32)
    array, kind = checkpoint_array(f32)
    assert kind == GGMLQuantizationType.F32
    assert array.dtype == np.float32

    with pytest.raises(TypeError, match="BF16 or F32"):
        checkpoint_array(torch.ones(1, dtype=torch.float16))


def test_write_layer_carrier_uses_target_layout_and_types(tmp_path: Path) -> None:
    source_path = tmp_path / "source.gguf"
    carrier_path = tmp_path / "carrier.gguf"
    write_test_model(source_path, 1)
    state = {}
    for key in TENSOR_NAMES:
        dtype = torch.float32 if key == "router" else torch.bfloat16
        state[key] = torch.full((2, 4), 2.0, dtype=dtype)
    source = GGUFReader(source_path, "r")
    quant_types = write_layer_carrier(source, state, 1, carrier_path)
    del source

    carrier = GGUFReader(carrier_path, "r")
    assert len(carrier.tensors) == 7
    assert set(quant_types.values()) == {"BF16", "F32"}
    assert {tensor.name for tensor in carrier.tensors} == {
        f"blk.1.{suffix}" for suffix in TENSOR_NAMES.values()
    }


def test_patch_quantized_copy_changes_only_named_layer(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    layer = tmp_path / "layer.gguf"
    output = tmp_path / "candidate.gguf"
    write_test_model(source, 1)
    write_test_model(layer, 7, include_extra=False)
    before = tensor_bytes(source)
    layer_values = tensor_bytes(layer)

    hashes = patch_quantized_copy(source, layer, output, 1)
    after = tensor_bytes(output)

    assert tensor_bytes(source) == before
    assert after["blk.0.attn_norm.weight"] == before["blk.0.attn_norm.weight"]
    for suffix in TENSOR_NAMES.values():
        name = f"blk.1.{suffix}"
        assert after[name] == layer_values[name]
        assert name in hashes


def test_quantizer_includes_every_imatrix_tensor_by_exact_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

    monkeypatch.setattr("scripts.quantize_patch_gguf_layer.subprocess.run", fake_run)
    names = {
        "blk.63.ffn_gate_exps.weight": "Q5_K",
        "blk.63.ffn_up_exps.weight": "Q5_K",
    }
    quantize_layer_carrier(
        Path("llama-quantize"),
        Path("carrier.gguf"),
        Path("quantized.gguf"),
        Path("imatrix.gguf"),
        63,
        names,
        8,
        tmp_path / "quantize.log",
    )
    command = captured["command"]
    included = [
        command[index + 1] for index, item in enumerate(command) if item == "--include-weights"
    ]
    assert included == sorted(names)
