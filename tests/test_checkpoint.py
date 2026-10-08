from moeme.checkpoint import TensorSpec, validate_mlp_tensors


def tensors(layers: int = 2) -> dict[str, TensorSpec]:
    result = {}
    for layer in range(layers):
        prefix = f"model.language_model.layers.{layer}.mlp"
        result[f"{prefix}.gate_proj.weight"] = TensorSpec((32, 16), "BF16")
        result[f"{prefix}.up_proj.weight"] = TensorSpec((32, 16), "BF16")
        result[f"{prefix}.down_proj.weight"] = TensorSpec((16, 32), "BF16")
    return result


def test_complete_checkpoint_passes() -> None:
    result = validate_mlp_tensors(
        tensors(), layers=2, hidden_size=16, intermediate_size=32, expected_dtype="BF16"
    )
    assert result["valid"] is True
    assert result["mlp_tensors_checked"] == 6


def test_missing_shape_and_dtype_are_reported() -> None:
    value = tensors()
    del value["model.language_model.layers.1.mlp.up_proj.weight"]
    value["model.language_model.layers.0.mlp.down_proj.weight"] = TensorSpec((32, 16), "F16")
    result = validate_mlp_tensors(
        value, layers=2, hidden_size=16, intermediate_size=32, expected_dtype="BF16"
    )
    assert result["valid"] is False
    assert len(result["missing"]) == 1
    assert len(result["mismatched"]) == 1
    assert "shape" in result["mismatched"][0]
    assert "dtype" in result["mismatched"][0]
