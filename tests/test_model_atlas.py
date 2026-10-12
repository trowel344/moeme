"""
Tests for `tools/model_atlas.py`, the 3D atlas page.

Two properties are worth pinning, and neither is about pixels:

1. **Every number in the page is derived from an artifact, not typed into it.**
   The atlas is a *view* of this repo's measurements, so if it ever disagrees
   with `src/moeme/accounting.py`, `results/branch_poc_08b.json` or
   `configs/tiers.json`, it is wrong and should fail rather than quietly
   mislead. The assertions below re-derive the load-bearing values from those
   files and compare.
2. **The page is self-contained.** It has to open offline by double-click on a
   machine with no network, so the rendered HTML must reference nothing
   external. That is a contract a test can hold.

Also checked: the committed `docs/model_atlas.html` is what the generator
currently produces, so editing the template without regenerating fails here
rather than shipping a page that no longer matches its own tool.
"""

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import model_atlas  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return model_atlas.derive()


# ---------------------------------------------------------------------------
# the derived numbers must agree with the artifacts
# ---------------------------------------------------------------------------

def test_geometry_reconciles_against_the_accounting_module(data):
    """The tower must add up: 64 layers x per-layer FFN = total FFN, and the
    components must sum to the measured checkpoint total."""
    g = data["geometry"]
    assert g["floor"] == 9_783_238_144          # the irreducible backbone
    assert (g["linear_attention"] + g["attention"] + g["lm_head"]
            + g["embedding"] + g["norms"]) == g["floor"]
    assert g["ffn_per_layer"] * g["layers"] == g["ffn"]
    assert g["text_tower"] + g["vision"] + g["mtp"] == g["params"]
    assert g["params"] == 27_781_427_952
    # the second accounting convention the design doc says must be stated
    assert g["floor_matmul_only"] == g["floor"] - g["embedding"]


def test_active_targets_match_the_arithmetic_in_the_report(data):
    """9B is impossible and 12B is possible, at exactly the documented cost."""
    t = {x["target_b"]: x for x in data["targets"]}
    assert t[9]["possible"] is False
    assert t[9]["ffn_active_b"] < 0                       # below the floor
    assert t[12]["possible"] is True
    assert t[12]["ffn_active_b"] == pytest.approx(
        (12e9 - 9_783_238_144) / 1e9, abs=1e-9)
    assert t[12]["ffn_fraction_pct"] == pytest.approx(12.96, abs=0.05)
    assert data["geometry"]["floor"] / 9e9 > 1.08


def test_oracle_curve_is_the_k1_family_from_the_artifact(data):
    """The loss curve must be read out of the branch PoC, not approximated.

    Only the k=1 rows qualify: with k>1, `residual_best_single_expert` is one
    slice rather than the best set, so using those would understate the oracle.
    """
    with open(os.path.join(ROOT, "results", "branch_poc_08b.json")) as fh:
        runs = json.load(fh)["runs"]
    want = sorted(
        (r["untrained"]["active_width_fraction"],
         r["untrained"]["residual_oracle_topk"])
        for r in runs
        if "tile" in r["name"] and "trained" not in r["name"]
        and not r.get("btx") and r["config"]["top_k"] == 1
    )
    got = [(p["active_fraction"], p["oracle_loss"]) for p in data["oracle_curve"]]
    assert got == want
    assert [frac for frac, _ in got] == [0.125, 0.25, 0.5, 1.0]
    assert got[-1][1] == 0.0            # full width is the dense layer: no loss
    # monotone: reading more of the FFN is never worse
    losses = [p[1] for p in got]
    assert losses == sorted(losses, reverse=True)


def test_ladder_places_the_residency_cliff_where_the_table_says(data):
    """Q3_K_M fits and Q4_K_M does not, and the threshold is ~4.11 bpw."""
    q = {r["name"]: r for r in data["ladder"]}
    assert q["Q3_K_M"]["resident"] is True
    assert q["Q4_K_S"]["resident"] is False
    assert q["Q4_K_M"]["resident"] is False
    assert q["Q4_K_M"]["streamed_gb"] > 0
    assert data["thresholds"]["threshold_bpw"] == pytest.approx(4.1122, abs=5e-4)
    assert data["thresholds"]["resident_gb"] == pytest.approx(14.28, abs=0.01)
    # the page claims a mixed recipe lands resident: it must actually do so
    assert all(m["resident"] for m in data["mixed"])


def test_measured_points_come_from_the_tier_config(data):
    """The page plots real runs; it must not show a measured point that is not
    in the registry."""
    with open(os.path.join(ROOT, "configs", "tiers.json")) as fh:
        blob = json.dumps(json.load(fh))
    assert len(data["measured"]) >= 3
    for m in data["measured"]:
        assert m["tag"] in blob
        assert m["tok_s"] > 0
    dense = next(m for m in data["measured"] if m["tag"] == "qwen3.8:latest")
    assert dense["tok_s"] == pytest.approx(0.561, abs=1e-3)   # measured today
    moe = next(m for m in data["measured"] if "q2_k" in m["tag"])
    assert moe["resident"] is True
    assert moe["tok_s"] == pytest.approx(19.356, abs=1e-3)


def test_quality_block_keeps_the_unbacked_pair_labelled(data):
    """The Q2-vs-Q4 perplexity pair is flagged in its own source as not backed
    by a committed artifact. The atlas must not present it as measured."""
    q = data["quality"]
    assert q["ppl_pair"]["_provenance"].lower().startswith("measured in-session")
    assert "NOT backed by a committed" in q["ppl_pair"]["_provenance"]
    # what *is* committed: the 35B's own Q2_K run, with its window recorded
    assert q["ppl_35b_quant"] == "Q2_K"
    assert q["ppl_35b_q2k"] == pytest.approx(12.7113, abs=1e-3)
    assert q["ppl_chunks"] and q["ppl_ctx"]


# ---------------------------------------------------------------------------
# the page itself
# ---------------------------------------------------------------------------

def test_rendered_page_is_self_contained(data):
    """No network, or it will not open on the machine it exists for."""
    html = model_atlas.render(data)
    assert "__DATA__" not in html                  # the placeholder was replaced
    assert json.dumps(data, separators=(",", ":")) in html
    for pattern in (r'src\s*=\s*["\']https?:', r'href\s*=\s*["\']https?:',
                    r'@import', r'<link[^>]+stylesheet'):
        assert not re.search(pattern, html, re.I), f"external ref: {pattern}"
    assert "<canvas" in html and "SELFTEST" in html


def test_committed_page_is_what_the_generator_produces(data):
    """The page on disk must match its tool, or one of the two is stale."""
    path = os.path.join(ROOT, "docs", "model_atlas.html")
    assert os.path.exists(path), "docs/model_atlas.html missing"
    with open(path) as fh:
        on_disk = fh.read()
    assert on_disk == model_atlas.render(data), (
        "docs/model_atlas.html is stale relative to tools/model_atlas.py; "
        "re-run: python3 tools/model_atlas.py"
    )
