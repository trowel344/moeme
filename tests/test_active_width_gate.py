import pytest
from gguf import GGUFWriter

from scripts.active_width_gate import active_layers, model_layer_count, parse_active_ks


def test_parse_active_ks_is_explicit_and_fail_closed() -> None:
    assert parse_active_ks("4,6,8") == (4, 6, 8)
    with pytest.raises(Exception, match="unique positive"):
        parse_active_ks("4,4")
    with pytest.raises(Exception, match="unique positive"):
        parse_active_ks("4,0")


def test_active_layers_expands_all_and_prefix() -> None:
    assert active_layers("all", 64) == list(range(64))
    assert active_layers("3", 64) == [0, 1, 2]
    assert active_layers("indices:63", 64) == [63]
    assert active_layers("indices:1,9,63", 64) == [1, 9, 63]
    with pytest.raises(Exception, match="count must be in"):
        active_layers("65", 64)
    with pytest.raises(Exception, match="count must be in"):
        active_layers("0", 64)
    with pytest.raises(Exception, match="non-empty and unique"):
        active_layers("indices:1,1", 64)
    with pytest.raises(Exception, match="each be in"):
        active_layers("indices:64", 64)


def test_model_layer_count_reads_block_count(tmp_path) -> None:
    model = tmp_path / "tiny.gguf"
    writer = GGUFWriter(model, "qwen35moe")
    writer.add_block_count(7)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert model_layer_count(model) == 7
