import pytest

from scripts.logit_parity_gate import (
    indexed_path,
    metric,
    parse_top4_layers,
    parsed_metrics,
)


def test_metric_parses_llama_perplexity_statistics() -> None:
    output = """
Mean PPL(Q)/PPL(base)         :   1.042000 +/- 0.01
Mean    KLD:   0.071000 +/- 0.01
Same top p: 84.500 +/- 1.0 %
"""
    assert metric(output, r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)", "ratio") == 1.042
    assert metric(output, r"Mean\s+KLD:\s*([0-9.]+)", "KLD") == 0.071
    assert metric(output, r"Same top p:\s*([0-9.]+)", "same top") == 84.5


def test_metric_rejects_missing_output() -> None:
    with pytest.raises(ValueError, match="missing KLD"):
        metric("no statistics", r"KLD: ([0-9.]+)", "KLD")


def test_indexed_path_preserves_parent_and_suffix(tmp_path) -> None:
    assert indexed_path(tmp_path / "reference.log", 3) == tmp_path / "reference-3.log"


def test_parsed_metrics_extracts_all_acceptance_values() -> None:
    output = """
Mean PPL(Q) : 6.2
Mean PPL(base) : 6.1
Mean PPL(Q)/PPL(base) : 1.016
Mean KLD: 0.019
Same top p: 95.2
"""
    assert parsed_metrics(output) == {
        "candidate_ppl": 6.2,
        "reference_ppl": 6.1,
        "ppl_ratio": 1.016,
        "mean_kld": 0.019,
        "same_top_percent": 95.2,
    }


def test_parsed_metrics_clamps_negative_kl_roundoff() -> None:
    output = """
Mean PPL(Q) : 6.0
Mean PPL(base) : 6.0
Mean PPL(Q)/PPL(base) : 1.0
Mean KLD: -0.000001
Same top p: 100.0
"""
    assert parsed_metrics(output)["mean_kld"] == 0.0


def test_top4_layer_parser_is_explicit_and_fail_closed() -> None:
    assert parse_top4_layers("0,1,5") == (0, 1, 5)
    with pytest.raises(Exception, match="unique integers"):
        parse_top4_layers("0,0")
    with pytest.raises(Exception, match="unique integers"):
        parse_top4_layers("64")
