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


def test_layer_section_sums_match_the_block_totals(data):
    """The sectioned layer is drawn from these part lists, so they have to
    account for the whole block -- every projection, none counted twice.

    A part with `n=2` means two matrices of the same shape (in_proj_a and
    in_proj_b): counting it once understated the Gated DeltaNet layer by
    245,760 parameters, which is exactly the kind of quiet error a drawing
    turns into a lie.
    """
    S = data["section"]
    for name, blk in S.items():
        assert blk["parts_total"] == blk["params"], name
        assert sum(p["rows"] * p["cols"] * p["n"]
                   for p in blk["parts"]) == blk["params"]
        assert all(p["n"] >= 1 for p in blk["parts"])
    g = data["geometry"]
    assert S["gdn"]["parts_total"] == g["gdn_layer_params"] == 115_875_936
    assert S["attention"]["parts_total"] == g["fa_layer_params"] == 104_857_600
    assert S["ffn"]["parts_total"] == g["ffn_per_layer"] == 267_386_880
    paired = next(p for p in S["gdn"]["parts"] if "in_proj_a" in p["name"])
    assert paired["n"] == 2


def test_geometry_supports_the_ratios_the_drawing_claims(data):
    """View A draws the FFN block 3.40x the height of the backbone block and
    colours every 4th column as attention. Both must be true of the model."""
    g = data["geometry"]
    assert g["intermediate"] / g["hidden"] == pytest.approx(3.40, abs=0.005)
    assert g["gdn_layers"] + g["fa_layers"] == g["layers"] == 64
    assert g["gdn_layers"] == 3 * g["fa_layers"]
    # the drawing labels this share, so it must be the measured one
    assert g["ffn"] / g["params"] == pytest.approx(0.616, abs=0.001)


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
# both sheets are readable -- a dark theme's failure mode is text you cannot see
# ---------------------------------------------------------------------------

def test_both_themes_meet_their_contrast_minimums():
    """Every meaningful pair (text on the sheet, text on a block, chrome) has to
    clear its WCAG minimum in *both* themes.

    This is the check the first dark version of this page would have failed: it
    was dark, but the text sat at whatever brightness the palette happened to
    have. Asserted from the palette, so it needs no browser.
    """
    report = model_atlas.contrast_report()
    assert set(report) == {"light", "dark"}
    for theme, rows in report.items():
        assert rows, f"no contrast requirements checked for {theme}"
        bad = [r for r in rows if not r["ok"]]
        assert not bad, f"{theme} theme fails contrast: {bad}"
    # and the requirement list has to be asking for something real
    assert any(r["min"] >= 7.0 for r in report["dark"])
    assert any(r["min"] >= 7.0 for r in report["light"])


def test_the_two_palettes_define_the_same_keys():
    """A colour present in one theme and missing from the other would render as
    undefined (invisible or black) on exactly one sheet."""
    light, dark = model_atlas.PALETTES["light"], model_atlas.PALETTES["dark"]
    assert set(light) == set(dark)
    for key, _var in model_atlas.CSS_VARS:
        assert key in light, f"CSS variable source {key} missing from light"
        assert key in dark, f"CSS variable source {key} missing from dark"
    for theme, pal in model_atlas.PALETTES.items():
        for key, value in pal.items():
            assert re.match(r"^(#[0-9a-f]{6}|rgba\([\d., ]+\))$", value), \
                f"{theme}.{key} is not a plain colour: {value}"


def test_page_ships_both_themes_and_a_switch(data):
    html = model_atlas.render(data)
    assert 'data-theme="dark"' in html                 # opens dark
    assert 'html[data-theme="dark"]' in html           # dark overrides generated
    assert ":root{" in html                             # light stays the base
    assert 'id="theme-btn"' in html and "setTheme" in html
    # the palette must reach the canvas too, not only the CSS
    assert '"palette"' in html
    assert "contrastRatio" in html and "CONTRAST_PAIRS" in html


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


def test_every_tab_declares_a_fit_box_and_the_layout_checks(data):
    """Coherence between the tabs and the sheet layout.

    The drawing is only readable if it is laid out to fit the frame: an earlier
    version had no fit at all and ran off the paper on every side. Each view
    must therefore declare the volume it draws into, and the page must keep the
    two checks that catch the readable-but-broken cases (clipping, and two
    labels printed on top of each other). Those are browser-only properties, so
    they are asserted in-page and read back by the render verification.
    """
    tmpl = model_atlas.TEMPLATE
    tabs = set(re.findall(r'data-view="([a-z]+)"', tmpl))
    assert tabs == {"stack", "layer", "target", "envelope"}
    boxes = set(re.findall(r"^\s{2}([a-z]+):\s+\{ min:\[", tmpl, re.M))
    assert tabs <= boxes, f"views with no fit box: {sorted(tabs - boxes)}"
    assert "drawing is inside the frame" in tmpl
    assert "no two labels overlap" in tmpl
    assert "theme contrast" in tmpl
    assert "__atlasSelfTest" in tmpl and "measureInk" in tmpl


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
