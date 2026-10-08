from scripts.prefix_parity_sweep import layer_tensor_names


def test_layer_tensor_names_cover_all_trained_projections() -> None:
    names = layer_tensor_names(7)
    assert len(names) == 7
    assert "blk.7.ffn_gate_inp.weight" in names
    assert "blk.7.ffn_down_exps.weight" in names
    assert "blk.7.ffn_up_shexp.weight" in names
