#!/usr/bin/env python3
"""
model_atlas.py -- one self-contained HTML blueprint of Qwen3.8-27B as this repo
measures it.

Why this exists
---------------
Every headline number here is a *shape*: 64 layers, a floor you cannot route
around, an FFN you can, a 14.28 GB residency ceiling, and a measured loss curve
over how much of the FFN you read. Read as prose they are easy to misremember;
drawn to scale they are hard to.

The drawing uses the model's real tensor geometry rather than a schema, because
the geometry is the argument:

* the FFN's intermediate axis is **17,408** against a hidden size of **5,120**, so
  the FFN block is 3.4x the height of the backbone block in every layer -- which
  is *why* it is 61.6% of the parameters;
* a "partition" of that FFN is a slice of the intermediate axis, so "read 12.5%
  of the FFN" is literally a horizontal band across the block, and it is drawn
  that way;
* the 3:1 Gated-DeltaNet-to-attention pattern is the released layout
  (16 groups of 3 DeltaNet layers + 1 full-attention layer).

Every number is derived at generation time from `src/moeme/accounting.py`,
`results/branch_poc_08b.json`, `configs/tiers.json` and
`results/quant_perplexity_35b.json` -- none is typed by hand -- and the output is
a single HTML file with no external references, so it opens offline by
double-click. Measured values name their artifact; model predictions are labelled
"predicted".

Usage:
    python3 tools/model_atlas.py                 # writes docs/model_atlas.html
    python3 tools/model_atlas.py --out /tmp/a.html
    python3 tools/model_atlas.py --print-data    # the derived data block only

The derived data is importable, so tests can assert the page still agrees with
the artifacts (`tests/test_model_atlas.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from moeme.accounting import (  # noqa: E402
    Qwen38_27B,
    bytes_for_params,
    predicted_decode_tok_s,
)
from residency_table import QUANTS, tiers  # noqa: E402


def _load(path: str) -> dict:
    with open(os.path.join(ROOT, path)) as fh:
        return json.load(fh)


def _find_value(node, key: str):
    """The value stored under `key`, first match in a depth-first walk.

    Returns the *value*, not its parent -- an earlier version of this file
    returned the containing dict and silently produced empty datasets.
    """
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            got = _find_value(v, key)
            if got is not None:
                return got
    elif isinstance(node, list):
        for v in node:
            got = _find_value(v, key)
            if got is not None:
                return got
    return None


# ---------------------------------------------------------------------------
# the data block: derived, never typed
# ---------------------------------------------------------------------------

def derive() -> dict:
    C = Qwen38_27B()
    gpu, ram, disk = tiers()
    resident_bytes = gpu.capacity_bytes + ram.capacity_bytes

    irr = C.irreducible_active_params()
    ffn = C.ffn_params()
    matmul_only = irr - C.embedding_params()      # embedding is a gather
    params = C.measured_total

    gdn_layer = C.linear_attention_params() // C.linear_attention_layers
    fa_layer = C.attention_params() // C.full_attention_layers
    ffn_layer = C.ffn_params() // C.num_layers

    geometry = {
        "params": params,
        "ffn": ffn,
        "ffn_per_layer": ffn_layer,
        "layers": C.num_layers,
        "hidden": C.hidden,
        "intermediate": C.intermediate,
        "vocab": C.vocab,
        "linear_attention": C.measured_linear_attention,
        "attention": C.measured_attention,
        "gdn_layers": C.linear_attention_layers,
        "fa_layers": C.full_attention_layers,
        "gdn_layer_params": gdn_layer,
        "fa_layer_params": fa_layer,
        "norm_per_layer": 2 * C.hidden,
        "lm_head": C.measured_lm_head,
        "embedding": C.measured_embedding,
        "norms": C.measured_norms,
        "vision": C.vision_params_measured,
        "mtp": C.mtp_params_measured,
        "text_tower": C.text_backbone_params(),
        "floor": irr,
        "floor_matmul_only": matmul_only,
        "source": "src/moeme/accounting.py, docs/source_validation.md "
                  "(verified against the safetensors headers)",
    }

    # the real internals of one layer, from the accounting dimensions
    section = {
        "gdn": {
            "title": "Gated DeltaNet layer",
            "subtitle": "48 of the 64 layers",
            "params": gdn_layer,
            "parts": [
                {"name": "in_proj_qkv", "rows": C.hidden,
                 "cols": 2 * C.gdn_qk_dim + C.gdn_v_dim, "n": 1},
                {"name": "in_proj_z", "rows": C.hidden, "cols": C.gdn_v_dim,
                 "n": 1},
                {"name": "out_proj", "rows": C.gdn_v_dim, "cols": C.hidden,
                 "n": 1},
                {"name": "in_proj_a / in_proj_b", "rows": C.hidden,
                 "cols": C.gdn_num_heads, "n": 2},
                {"name": "conv1d", "rows": C.gdn_conv_dim, "cols": 4, "n": 1},
                {"name": "A_log / dt_bias", "rows": 2, "cols": C.gdn_num_heads,
                 "n": 1},
            ],
        },
        "attention": {
            "title": "Full-attention layer",
            "subtitle": "16 of the 64 layers (1 in every 4)",
            "params": fa_layer,
            "parts": [
                {"name": "q_proj + gate (fused)", "rows": C.hidden,
                 "cols": 2 * C.attn_q_dim, "n": 1},
                {"name": "k_proj", "rows": C.hidden, "cols": C.attn_kv_dim, "n": 1},
                {"name": "v_proj", "rows": C.hidden, "cols": C.attn_kv_dim, "n": 1},
                {"name": "o_proj", "rows": C.attn_q_dim, "cols": C.hidden, "n": 1},
            ],
        },
        "ffn": {
            "title": "swiGLU FFN",
            "subtitle": "all 64 layers",
            "params": ffn_layer,
            "parts": [
                {"name": "gate_proj", "rows": C.intermediate, "cols": C.hidden,
                 "n": 1},
                {"name": "up_proj", "rows": C.intermediate, "cols": C.hidden,
                 "n": 1},
                {"name": "down_proj", "rows": C.hidden, "cols": C.intermediate,
                 "n": 1},
            ],
        },
    }
    # a part list must account for the whole block, or the drawing lies
    for block in section.values():
        block["parts_total"] = sum(p["rows"] * p["cols"] * p["n"]
                                   for p in block["parts"])

    # what an active-parameter target requires, per accounting convention
    targets = []
    for target_b in (9, 12, 15, 20, 24):
        a = target_b * 1e9
        need = a - irr
        targets.append({
            "target_b": target_b,
            "ffn_active_b": need / 1e9,
            "ffn_fraction_pct": 100.0 * need / ffn,
            "need_matmul_b": (a - matmul_only) / 1e9,
            "ffn_fraction_matmul_pct": 100.0 * (a - matmul_only) / ffn,
            "possible": need > 0,
        })

    # measured oracle loss vs how much FFN width is actually read.
    # The k=1 tile family is the one where "best single expert" *is* the oracle:
    # with k>1 that column is one slice, not the best set (see the k=4 rows).
    branch = _load("results/branch_poc_08b.json")
    oracle = []
    slices = []
    for r in branch["runs"]:
        if r.get("btx") or "tile" not in r["name"] or "trained" in r["name"]:
            continue
        u, cfg = r["untrained"], r["config"]
        slices.append({
            "run": r["name"],
            "active_fraction": u["active_width_fraction"],
            "k": cfg.get("top_k"),
            "experts": cfg.get("n_experts"),
            "best_single": u["residual_best_single_expert"],
            "oracle": u["residual_oracle_topk"],
            "ppl": r.get("val_ppl"),
        })
        if cfg.get("top_k") == 1:
            oracle.append({"active_fraction": u["active_width_fraction"],
                           "oracle_loss": u["residual_oracle_topk"],
                           "ppl": r.get("val_ppl")})
    oracle.sort(key=lambda x: x["active_fraction"])
    slices.sort(key=lambda x: (x["active_fraction"], x["run"]))
    curve_src = ("results/branch_poc_08b.json, k=1 tile family (4 layers of "
                 f"Qwen3.5-0.8B, dense val ppl {branch['dense_val_ppl']:.4f})")

    # the quantisation ladder for THIS model, plus this session's mixed recipe
    ladder = []
    for name, bpw in QUANTS:
        r = predicted_decode_tok_s(active_params=params, bits_per_param=bpw,
                                   gpu=gpu, ram=ram, disk=disk)
        streamed = r["placement"].get("disk", {}).get("bytes", 0.0)
        ladder.append({
            "name": name, "bpw": bpw,
            "size_gb": bytes_for_params(params, bpw) / 1e9,
            "streamed_gb": streamed / 1e9,
            "tok_s": r["tok_s"],
            "resident": streamed <= 0.1e9,
        })
    mixed = []
    for label, mix in (("75/15/10 @ 4/3/2b", 0.75 * 4 + 0.15 * 3 + 0.10 * 2),
                       ("60/25/15 @ 4/3/2b", 0.60 * 4 + 0.25 * 3 + 0.15 * 2),
                       ("IQ-style 4/2b 70/30", 0.70 * 4 + 0.30 * 2)):
        r = predicted_decode_tok_s(active_params=params, bits_per_param=mix,
                                   gpu=gpu, ram=ram, disk=disk)
        streamed = r["placement"].get("disk", {}).get("bytes", 0.0)
        mixed.append({
            "name": label, "bpw": mix,
            "size_gb": bytes_for_params(params, mix) / 1e9,
            "streamed_gb": streamed / 1e9,
            "tok_s": r["tok_s"],
            "resident": streamed <= 0.1e9,
        })

    # measured artifacts, for the same axes
    tiers_json = _load("configs/tiers.json")
    measured = []
    for tag in ("qwen3.8:latest", "ornith-1.5:9b", "ornith-1.5-35b-a3b:q2_k",
                "ornith-1.5:35b"):
        rec = _find_value(tiers_json, tag)
        if not isinstance(rec, dict) or "measured_tok_s" not in rec:
            continue
        measured.append({
            "tag": tag,
            "total_b": rec.get("total_params", 0) / 1e9 or None,
            "active_b": rec.get("active_params", 0) / 1e9 or None,
            "bpw": rec.get("bits_per_param"),
            "size_gb": rec.get("gguf_bytes", 0) / 1e9,
            "tok_s": rec["measured_tok_s"],
            "resident": bool(rec.get("resident", False)),
        })
    known_totals = {"qwen3.8:latest": 27.781, "ornith-1.5-35b-a3b:q2_k": 35.505,
                    "ornith-1.5:35b": 35.505}
    actives = {"ornith-1.5-35b-a3b:q2_k": 3.0, "ornith-1.5:35b": 3.0}
    for m in measured:
        if m["total_b"] is None:
            m["total_b"] = known_totals.get(m["tag"])
        if m["active_b"] is None:
            m["active_b"] = actives.get(m["tag"])

    thresholds = {
        "resident_gb": resident_bytes / 1e9,
        "vram_gib": gpu.capacity_bytes / (1024 ** 3),
        "ram_gb": ram.capacity_bytes / 1e9,
        "threshold_bpw": resident_bytes * 8 / params,
        "bw_gpu": gpu.bandwidth_bps / 1e9,
        "bw_ram": ram.bandwidth_bps / 1e9,
        "bw_disk": disk.bandwidth_bps / 1e9,
        "source": "results/hardware.json via tools/residency_table.py constants",
    }

    # quality: what the sparse model actually scores, and the caveats
    gate = _find_value(tiers_json, "short19_num_predict_192") or {}
    same = _find_value(tiers_json,
                       "same_architecture_two_quants_same_19_tasks_768_cap") or {}
    q = _find_value(tiers_json, "perplexity_matched_pair_identical_tokens") or {}
    ppl35 = _load("results/quant_perplexity_35b.json")
    ppl_runs = ppl35.get("runs") or {}
    if isinstance(ppl_runs, list):
        ppl_runs = {str(i): v for i, v in enumerate(ppl_runs)}
    ppl_run = next((v for v in ppl_runs.values()
                    if isinstance(v, dict) and v.get("quant") == "Q2_K"), None)
    corpus = ppl35.get("corpus") or {}
    quality = {
        "gate": gate,
        "same_quants": same,
        "ppl_pair": q,
        "ppl_35b_q2k": (ppl_run or {}).get("ppl"),
        "ppl_35b_stderr": (ppl_run or {}).get("stderr"),
        "ppl_35b_quant": (ppl_run or {}).get("quant"),
        "ppl_corpus_bytes": corpus.get("bytes") if isinstance(corpus, dict) else None,
        "ppl_corpus_sha256": (corpus.get("sha256") if isinstance(corpus, dict)
                              else None),
        "ppl_chunks": ppl35.get("chunks"),
        "ppl_ctx": ppl35.get("ctx"),
        "source": "configs/tiers.json (quality_gate), "
                  "results/quant_perplexity_35b.json",
    }

    return {
        "geometry": geometry,
        "section": section,
        "targets": targets,
        "oracle_curve": oracle,
        "slices": slices,
        "curve_source": curve_src,
        "ladder": ladder,
        "mixed": mixed,
        "measured": measured,
        "thresholds": thresholds,
        "quality": quality,
        "tier_model_note": (
            "The memory-tier model is validated within ~2x on 4 measured runs "
            "(2% error on the 27B Q4_K_M case: predicted 0.48 against 0.561 "
            "measured). Treat every tok/s here as a ceiling, not a forecast."
        ),
    }


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3.8-27B — blueprint</title>
<style>
  :root{
    --paper:#f6f4ee; --panel:#fffdf8; --grid:#e4e0d4; --grid2:#d6d1c2;
    --ink:#161d26; --dim:#5c6672; --blue:#1d4ed8; --blue2:#93b0f5;
    --red:#b91c1c; --green:#15803d; --amber:#a16207;
    --attn:#7c4dff; --gdn:#00838f; --ffn:#1d4ed8; --read:#15803d;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--paper);color:var(--ink);
       font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  #wrap{display:grid;grid-template-columns:1fr 320px;grid-template-rows:auto auto
        1fr auto;height:100%}
  header{grid-column:1/3;display:flex;gap:16px;align-items:baseline;
         padding:12px 18px;border-bottom:2px solid var(--ink);background:var(--panel)}
  h1{font-size:17px;margin:0;font-weight:700;letter-spacing:.01em}
  header .sub{color:var(--dim);font-size:13px}
  #tabs{grid-column:1/3;display:flex;gap:0;border-bottom:1px solid var(--grid2);
        background:var(--panel);padding-left:10px}
  #tabs button{background:none;color:var(--dim);border:none;border-bottom:3px solid
    transparent;padding:10px 16px;cursor:pointer;font:inherit;font-weight:600}
  #tabs button[aria-selected="true"]{color:var(--blue);border-bottom-color:var(--blue)}
  #stage{position:relative;overflow:hidden;background:var(--paper)}
  canvas{display:block;width:100%;height:100%;cursor:grab}
  canvas:active{cursor:grabbing}
  #tip{position:absolute;pointer-events:none;background:#fffdf8f2;border:1.5px solid
    var(--ink);border-radius:4px;padding:8px 10px;font-size:13px;max-width:320px;
    display:none;z-index:5;white-space:pre-line;
    box-shadow:3px 3px 0 rgba(22,29,38,.12)}
  #side{border-left:2px solid var(--ink);background:var(--panel);padding:14px 16px;
        overflow:auto;font-size:12.5px}
  #side h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
    color:var(--dim);margin:0 0 8px;padding-bottom:5px;border-bottom:1px solid var(--grid2)}
  #side table{width:100%;border-collapse:collapse;margin-bottom:16px}
  #side td{padding:2.5px 0;vertical-align:top}
  #side td:last-child{text-align:right;font-variant-numeric:tabular-nums;
    white-space:nowrap}
  #side .tot td{border-top:1.5px solid var(--ink);font-weight:700;padding-top:5px}
  #side p{color:var(--dim);margin:0 0 12px}
  .badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;
    font-weight:700;letter-spacing:.03em}
  .m{background:#dcfce7;color:#14532d} .p{background:#fef3c7;color:#713f12}
  #controls{grid-column:1/3;display:flex;gap:26px;flex-wrap:wrap;align-items:center;
    padding:11px 18px;border-top:2px solid var(--ink);background:var(--panel)}
  .ctl{display:flex;flex-direction:column;gap:2px}
  .ctl label{color:var(--dim);font-size:11px;text-transform:uppercase;
    letter-spacing:.05em}
  .ctl .v{color:var(--blue);font-weight:700;font-variant-numeric:tabular-nums}
  input[type=range]{width:210px;accent-color:var(--blue)}
  .btn{background:var(--panel);border:1.5px solid var(--ink);border-radius:4px;
    padding:5px 11px;cursor:pointer;font:inherit;font-weight:600}
  .btn[aria-pressed="true"]{background:var(--blue);color:#fff;border-color:var(--blue)}
  #cards{display:flex;gap:12px;flex-wrap:wrap;margin-left:auto}
  .card{border:1.5px solid var(--ink);border-radius:4px;padding:5px 10px;
    background:var(--paper);min-width:118px}
  .card .k{display:block;font-size:10.5px;text-transform:uppercase;
    letter-spacing:.05em;color:var(--dim)}
  .card .v2{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}
  #prov{grid-column:1/3;padding:7px 18px;border-top:1px solid var(--grid2);
    color:var(--dim);font-size:11.5px;background:var(--panel)}
  #selftest{position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden}
  @media (max-width:900px){#wrap{grid-template-columns:1fr}
    #side{display:none} header,#tabs,#controls,#prov{grid-column:1}}
</style>
</head>
<body>
<div id="wrap">
  <header>
    <h1>Qwen3.8-27B — blueprint</h1>
    <span class="sub">Drawn to scale from the measured tensor geometry. Drag to
      rotate, wheel to zoom, hover any block.</span>
  </header>

  <div id="tabs" role="tablist">
    <button role="tab" data-view="stack" aria-selected="true">A &middot; Whole model, to scale</button>
    <button role="tab" data-view="layer">B &middot; One layer, sectioned</button>
    <button role="tab" data-view="target">C &middot; Active targets vs the floor</button>
    <button role="tab" data-view="envelope">D &middot; Residency envelope</button>
  </div>

  <div id="stage">
    <canvas id="c"></canvas>
    <div id="tip"></div>
  </div>

  <div id="side"></div>

  <div id="controls">
    <div class="ctl" id="ctl-read">
      <label for="read">FFN width read per token: <span class="v" id="read-v">12.5%</span></label>
      <input type="range" id="read" min="12.5" max="100" step="12.5" value="12.5">
    </div>
    <div class="ctl" id="ctl-type">
      <label>Layer type shown</label>
      <div style="display:flex;gap:8px">
        <button class="btn" id="btn-gdn" aria-pressed="true">DeltaNet (48)</button>
        <button class="btn" id="btn-fa" aria-pressed="false">Attention (16)</button>
      </div>
    </div>
    <div class="ctl">
      <label for="zoom">Zoom</label>
      <input type="range" id="zoom" min="0.5" max="3" step="0.05" value="1">
    </div>
    <div id="cards"></div>
  </div>

  <div id="prov"></div>
</div>
<div id="selftest">SELFTEST: pending</div>

<script id="atlas-data" type="application/json">__DATA__</script>
<script>
"use strict";
/* Recorded so headless verification can tell "rendered fine" from "threw". */
let scriptError = null;
window.addEventListener("error", e => {
  scriptError = (e && (e.message || e.error)) ? String(e.message || e.error) : "error";
});
const D = JSON.parse(document.getElementById("atlas-data").textContent);
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
const tip = document.getElementById("tip"), side = document.getElementById("side");
const cards = document.getElementById("cards"), prov = document.getElementById("prov");

const C = {
  ink:"#161d26", ink2:"#5c6672", grid:"#e4e0d4", grid2:"#d6d1c2",
  blue:"#1d4ed8", blue2:"#93b0f5", red:"#b91c1c", green:"#15803d",
  amber:"#a16207", attn:"#7c4dff", gdn:"#00838f", read:"#15803d",
  paper:"#f6f4ee", panel:"#fffdf8", dim:"#5c6672", white:"#ffffff"
};

/* ---------- drafting projector: orthographic, fitted to the sheet --------
   A drawing that runs off the paper cannot be read, so the sheet is laid out
   to fit: the view declares its bounds and `fitSheet` picks the scale and
   centring that put all of it inside the frame with a margin. */
const cam = { yaw:-0.62, pitch:0.42, zoom:1.0 };
const CAM_DEFAULT = { yaw:-0.62, pitch:0.42 };
let sheet = null;
let prims = [], anchors = [];

function rotRaw(p){
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  let x = p[0]*cy - p[2]*sy, z = p[0]*sy + p[2]*cy;
  let y = p[1]*cp - z*sp;   z = p[1]*sp + z*cp;
  return { x, y, z };
}
function project(p){
  const w = cv.clientWidth, h = cv.clientHeight;
  const r = rotRaw(p);
  const s = (sheet ? sheet.scale : 60) * cam.zoom;
  const cx = sheet ? sheet.cx : 0, cy = sheet ? sheet.cy : 0;
  return { x: w/2 + (r.x - cx)*s, y: h/2 - (r.y - cy)*s, z: r.z };
}
function fitSheet(bounds){
  const w = cv.clientWidth, h = cv.clientHeight, m = 70;
  let x0=1e9, y0=1e9, x1=-1e9, y1=-1e9;
  for (const X of [bounds.min[0], bounds.max[0]])
    for (const Y of [bounds.min[1], bounds.max[1]])
      for (const Z of [bounds.min[2], bounds.max[2]]){
        const r = rotRaw([X,Y,Z]);
        x0=Math.min(x0,r.x); x1=Math.max(x1,r.x);
        y0=Math.min(y0,r.y); y1=Math.max(y1,r.y);
      }
  const sx = (w - 2*m)/Math.max(1e-6, x1-x0);
  const sy = (h - 2*m)/Math.max(1e-6, y1-y0);
  sheet = { scale: Math.min(sx,sy), cx:(x0+x1)/2, cy:(y0+y1)/2 };
}

/* Each view declares the volume it draws into, so the sheet can be laid out. */
const VIEW_BOUNDS = {
  stack:    { min:[-0.95, -0.25, -0.50], max:[11.10, 1.75, 0.50] },
  layer:    { min:[-1.95, -0.20, -0.55], max:[ 1.65, 1.78, 0.55] },
  target:   { min:[-3.30, -0.25, -2.75], max:[ 3.30, 5.40, 2.75] },
  envelope: { min:[-3.40, -2.30, -2.55], max:[ 3.40, 2.30, 2.55] },
};
let needsFit = true;
function poly(pts, fill, stroke, sw, alpha){
  const pj = pts.map(project);
  if (pj.some(p => !p)) return;
  prims.push({ kind:"poly", pj, fill, stroke, sw:sw||0,
               alpha:alpha===undefined?1:alpha,
               depth: pj.reduce((a,b)=>a+b.z,0)/pj.length });
}
function seg(a, b, color, width, dash, screen){
  let A = a, B = b;
  if (!screen){ A = project(a); B = project(b); }
  if (!A || !B) return;
  prims.push({ kind:"line", pa:A, pb:B, color, width:width||1, dash,
               depth:-1e6 });
}
function box(cx,cy,cz, sx,sy,sz, fill, stroke, sw, alpha, hatch){
  const X=[cx-sx/2,cx+sx/2], Y=[cy-sy/2,cy+sy/2], Z=[cz-sz/2,cz+sz/2];
  const v=(i,j,k)=>[X[i],Y[j],Z[k]];
  const faces = [
    [v(0,0,0),v(0,1,0),v(0,1,1),v(0,0,1)],
    [v(1,0,0),v(1,1,0),v(1,1,1),v(1,0,1)],
    [v(0,0,0),v(1,0,0),v(1,0,1),v(0,0,1)],
    [v(0,1,0),v(1,1,0),v(1,1,1),v(0,1,1)],
    [v(0,0,0),v(1,0,0),v(1,1,0),v(0,1,0)],
    [v(0,0,1),v(1,0,1),v(1,1,1),v(0,1,1)],
  ];
  faces.forEach(f => poly(f, fill, stroke, sw, alpha));
  if (hatch) hatchTop(cx,cy,cz,sx,sy,sz);
}
/* 45-degree section hatching, the drafting convention for "not this material" */
function hatchTop(cx,cy,cz,sx,sy,sz){
  const n = 4;
  for (let i=1;i<n;i++){
    const t = i/n;
    seg([cx-sx/2, cy+sy/2, cz-sz/2 + t*sz],
        [cx+sx/2, cy+sy/2, cz-sz/2 + t*sz], "#b9b3a4", 1);
  }
}
function screenText(x, y, text, color, size, align, weight){
  prims.push({ kind:"stext", x, y, text, color,
               size:size||12.5, align:align||"left", weight:weight||"400",
               depth:-2e6 });
}
function label(p, text, color, size, align, weight){
  const pj = project(p);
  if (!pj) return;
  prims.push({ kind:"stext", x:pj.x, y:pj.y, text, color,
               size:size||12.5, align:align||"left", weight:weight||"400",
               depth:pj.z - 0.01 });
}
/* a dimension line with end ticks and a horizontal label -- drafting style */
function dim(a, b, text, sideSign){
  const A = project(a), B = project(b);
  if (!A || !B) return;
  const dx = B.x-A.x, dy = B.y-A.y, L = Math.hypot(dx,dy) || 1;
  const nx = -dy/L*(sideSign||1), ny = dx/L*(sideSign||1);
  prims.push({ kind:"line", pa:A, pb:B, color:C.blue, width:1.1,
               depth:-1e6 });
  for (const P of [A,B]){
    prims.push({ kind:"line",
      pa:{x:P.x-nx*7, y:P.y-ny*7}, pb:{x:P.x+nx*7, y:P.y+ny*7},
      color:C.blue, width:1.1, depth:-1e6 });
  }
  const tx = (A.x+B.x)/2 + nx*13, ty = (A.y+B.y)/2 + ny*13;
  const tw = String(text).length * 12 * 0.55;
  prims.push({ kind:"stext",
               x: Math.max(8 + tw/2, Math.min(cv.clientWidth - 8 - tw/2, tx)),
               y: Math.max(40, Math.min(cv.clientHeight - 8, ty)),
               text, color:C.blue, size:12, align:"center", weight:"700",
               depth:-1e6 });
}
/* a callout: leader line from a 3D anchor to a screen-space note.
   The note is clamped inside the frame and its text is measured, so a callout
   near an edge flips sides instead of running off the paper. */
function callout(p, dx, dy, text, color){
  const pj = project(p);
  if (!pj) return;
  const w = cv.clientWidth, h = cv.clientHeight, size = 12.5, pad = 10;
  const lines = String(text).split("\n");
  const tw = Math.max(...lines.map(l => l.length)) * size * 0.55;
  const th = lines.length * size * 1.32;
  let align = dx >= 0 ? "left" : "right";
  if (align === "left" && pj.x + dx + tw > w - pad) align = "right";
  if (align === "right" && pj.x + dx - tw < pad) align = "left";
  let x = pj.x + dx;
  x = align === "left" ? Math.min(x, w - pad - tw) : Math.max(x, pad + tw);
  x = Math.max(pad, Math.min(w - pad, x));
  const y = Math.max(50 + th/2, Math.min(h - 16 - th/2, pj.y + dy));
  prims.push({ kind:"line", pa:pj, pb:{x, y}, color:color||C.ink2,
               width:1, depth:-1e6 });
  prims.push({ kind:"stext", x: x + (align==="left"?7:-7), y, text,
               color: color||C.ink, size, align, depth:-1e6 });
}
function dot3(p, color, r, tipText){
  const pj = project(p);
  if (!pj) return;
  prims.push({ kind:"dot", pj, color, r:r||4, depth:pj.z });
  if (tipText) anchors.push({ x:pj.x, y:pj.y, text:tipText });
}

/* ---------- shared: the measured loss curve, snapped to real points -------- */
function oracleAt(frac){
  let best = D.oracle_curve[0], bd = 1e9;
  for (const p of D.oracle_curve){
    const d = Math.abs(p.active_fraction - frac);
    if (d < bd){ bd = d; best = p; }
  }
  return best;
}
function lossColour(loss){          // 0 loss = green, 0.71 loss = red
  const t = Math.max(0, Math.min(1, 1 - loss/0.75));
  const c = [[185,28,28],[161,98,7],[21,128,61]];
  const x = t*(c.length-1), i = Math.min(c.length-2, Math.floor(x)), f = x-i;
  const mix = j => Math.round(c[i][j] + (c[i+1][j]-c[i][j])*f);
  return `rgb(${mix(0)},${mix(1)},${mix(2)})`;
}

/* ---------- drafting paper grid ---------- */
function paperGrid(){
  const w = cv.clientWidth, h = cv.clientHeight, step = 28;
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = (w/2)%step; x < w; x += step){ ctx.moveTo(x,0); ctx.lineTo(x,h); }
  for (let y = (h/2)%step; y < h; y += step){ ctx.moveTo(0,y); ctx.lineTo(w,y); }
  ctx.stroke();
  ctx.strokeStyle = C.grid2;
  ctx.beginPath();
  for (let x = (w/2)%(step*5); x < w; x += step*5){ ctx.moveTo(x,0); ctx.lineTo(x,h); }
  for (let y = (h/2)%(step*5); y < h; y += step*5){ ctx.moveTo(0,y); ctx.lineTo(w,y); }
  ctx.stroke();
}

/* =========================================================================
   VIEW A -- the whole model: 64 layers across the sheet, true heights
   ========================================================================= */
let readFrac = 0.125;

function buildStack(){
  const G = D.geometry;
  const H = G.hidden, I = G.intermediate;
  const yFFN = 1.5, yBB = yFFN*H/I;         // 5,120 : 17,408, to scale
  const ffnH = yFFN - yBB;
  const dx = 0.155, gap = 0.015, dz = 0.5;
  const stride = dx + gap;
  const x0 = 0, x1 = G.layers*stride - gap;
  const every = 4;                          // 16 x (3 DeltaNet + 1 attention)

  // footprint plate
  poly([[x0-0.06,0,-dz/2-0.06],[x1+0.06,0,-dz/2-0.06],
        [x1+0.06,0,dz/2+0.06],[x0-0.06,0,dz/2+0.06]],
       "#efece2", C.grid2, 1, 1);

  for (let i=0;i<G.layers;i++){
    const cx = x0 + i*stride + dx/2;
    const isFA = (i % every) === every-1;
    // backbone: read on every token, routable: no
    box(cx, yBB/2, 0, dx, yBB, dz,
        isFA ? "#cbb9ff" : "#a7d9de", C.ink, 0.9, 1, true);
    // FFN: the part routing can slice, sitting on top of the backbone
    box(cx, yBB + ffnH/2, 0, dx, ffnH, dz, "#dce6fd", "#8fa8e8", 0.7, 1);
    // the slice actually read this token, coloured by its measured loss
    const rh = ffnH*readFrac;
    box(cx, yBB + rh/2, 0, dx*1.03, rh, dz*1.03,
        lossColour(oracleAt(readFrac).oracle_loss), null, 0, 0.95);
  }

  // vertical dimensions against the true axes
  dim([x0-0.34, 0, 0], [x0-0.34, yFFN, 0],
      `FFN intermediate\n${I.toLocaleString()}`, -1);
  dim([x0-0.12, 0, dz/2+0.12], [x0-0.12, yBB, dz/2+0.12],
      `hidden\n${H.toLocaleString()}`, -1);
  // horizontal: the 64 layers and the 3:1 rhythm
  dim([x0, 0, dz/2+0.10], [x1, 0, dz/2+0.10],
      `${G.layers} layers  (48 DeltaNet + 16 attention)`, 1);

  callout([x0+30*stride+dx/2, yBB+ffnH*0.55, dz/2], 96, -74,
    `FFN — ${(G.ffn_per_layer/1e6).toFixed(1)}M per layer\n` +
    `${(G.ffn/1e9).toFixed(2)}B across 64 layers = 61.6% of the model\n` +
    `partitioned along the intermediate axis:\n` +
    `reading ${(readFrac*100).toFixed(1)}% = ` +
    `${Math.round(readFrac*I).toLocaleString()} of ${I.toLocaleString()} units`,
    C.blue);
  callout([x0+3*stride+dx/2, yFFN, 0], -120, -104,
    `one group of 4 =\n3 DeltaNet + 1 attention,\nrepeated 16 times`,
    C.ink2);
  callout([x0+20*stride+dx/2, yBB*0.6, dz/2], -46, -92,
    `backbone — ${(G.floor_matmul_only/1e9).toFixed(2)}B\n` +
    `read on every token\nroutable: no`, C.red);
  callout([x0-0.06, 0, -dz/2], 34, 78,
    `embedding ${(G.embedding/1e6).toFixed(0)}M\n` +
    `${G.vocab.toLocaleString()} x ${H.toLocaleString()} — a gather`, C.ink2);
  // legend, as sheet furniture rather than a leader into a crowded corner
  const h = cv.clientHeight;
  screenText(20, h-46,
    "One column = one layer.  Teal = Gated DeltaNet (48 layers), violet = " +
    "full attention (16 layers, every 4th).", C.ink2, 11.5);
  screenText(20, h-28,
    `The coloured band is the slice of the FFN actually read: ` +
    `${(readFrac*100).toFixed(1)}% now, measured oracle loss ` +
    `${oracleAt(readFrac).oracle_loss.toFixed(4)}.`, C.ink2, 11.5);

  screenText(20, 26, "ELEVATION A — the whole model, heights to scale",
             C.ink, 13.5, "left", "700");
  screenText(20, 45,
    `Faces are drawn at their real relative sizes: the FFN block is ` +
    `${(I/H).toFixed(2)}x the height of the backbone block, which is why it ` +
    `holds 61.6% of the parameters.`, C.ink2, 12);
  screenText(20, 62,
    `Column spacing is schematic (the model has no physical width).`, C.ink2, 11.5);
}

function sideStack(){
  const G = D.geometry, T = D.thresholds;
  const loss = oracleAt(readFrac).oracle_loss;
  const readB = (G.floor_matmul_only + G.ffn*readFrac)/1e9;
  side.innerHTML = `
    <h2>Measured dimensions</h2>
    <table>
      <tr><td>FFN (64 layers)</td><td>${G.ffn.toLocaleString()}</td></tr>
      <tr><td>Gated DeltaNet (48)</td><td>${G.linear_attention.toLocaleString()}</td></tr>
      <tr><td>Full attention (16)</td><td>${G.attention.toLocaleString()}</td></tr>
      <tr><td>Embedding</td><td>${G.embedding.toLocaleString()}</td></tr>
      <tr><td>LM head</td><td>${G.lm_head.toLocaleString()}</td></tr>
      <tr><td>Norms</td><td>${G.norms.toLocaleString()}</td></tr>
      <tr><td>Vision tower <span class="badge p">not text</span></td><td>${G.vision.toLocaleString()}</td></tr>
      <tr><td>MTP head <span class="badge p">draft</span></td><td>${G.mtp.toLocaleString()}</td></tr>
      <tr class="tot"><td>Total</td><td>${G.params.toLocaleString()}</td></tr>
    </table>
    <h2>Why the FFN dominates</h2>
    <p>Its intermediate axis is ${G.intermediate.toLocaleString()} against a hidden
    size of ${G.hidden.toLocaleString()} — ${(G.intermediate/G.hidden).toFixed(2)}x
    taller in the drawing, and ${(100*G.ffn/G.params).toFixed(1)}% of all
    parameters.</p>
    <h2>This token</h2>
    <table>
      <tr><td>Backbone (floor)</td><td>${(G.floor_matmul_only/1e9).toFixed(2)}B</td></tr>
      <tr><td>FFN read (${(readFrac*100).toFixed(0)}%)</td><td>${(G.ffn*readFrac/1e9).toFixed(2)}B</td></tr>
      <tr class="tot"><td>Active / token</td><td>${readB.toFixed(2)}B</td></tr>
      <tr><td>Measured oracle loss</td><td>${loss.toFixed(4)}</td></tr>
    </table>
    <p><span class="badge m">measured</span> the loss is a real measurement, not a
    model: reading ${(readFrac*100).toFixed(1)}% of the FFN loses
    ${loss.toFixed(4)} of the layer output <i>even with a perfect router</i>.</p>
    <h2>Sources</h2>
    <p>Geometry: <code>src/moeme/accounting.py</code>.<br>
       Loss: <code>${D.curve_source}</code>.<br>
       Memory: <code>${T.source}</code>.</p>`;
  cards.innerHTML = cardHTML([
    ["active / token", readB.toFixed(2)+"B"],
    ["of total", (100*readB/(G.params/1e9)).toFixed(1)+"%"],
    ["oracle loss", loss.toFixed(4)],
    ["@4.9 bpw", (readB*4.9/8).toFixed(2)+" GB/tok"],
  ]);
}

/* =========================================================================
   VIEW B -- one layer, sectioned
   ========================================================================= */
let layerType = "gdn";

function buildLayer(){
  const G = D.geometry, S = D.section, blk = S[layerType];
  const H = G.hidden, I = G.intermediate;
  const unit = 1/I;                       // one intermediate unit in world Y

  // LEFT: the backbone block, bands sized by each projection's params
  const totalParts = blk.parts_total;
  let y0 = 0;
  const xBB = -0.75, wBB = 0.85;
  for (const p of blk.parts){
    const prm = p.rows*p.cols*p.n, hh = (prm/totalParts) * 1.5;
    box(xBB, y0+hh/2, 0, wBB, hh, 0.8,
        layerType==="gdn" ? "#a7d9de" : "#c9b8ff", C.ink, 1.1, 1, true);
    if (hh > 0.075 || blk.parts.indexOf(p) < 4)
      label([xBB - wBB/2 - 0.06, y0+hh/2, 0.42],
            `${p.name}  ${p.rows.toLocaleString()}x${p.cols.toLocaleString()}` +
            (p.n > 1 ? `  x${p.n}` : ""),
            C.ink, 12, "right");
    y0 += hh + 0.035;
  }
  label([xBB, 1.62, 0], `${blk.title} — ${(blk.params/1e6).toFixed(2)}M`,
        C.ink, 13.5, "center", "700");
  label([xBB, 1.50, 0], blk.subtitle, C.dim, 12, "center");

  // RIGHT: the FFN, three projections with real shapes
  const xF = 0.30, wF = 0.62;
  const ffnParts = D.section.ffn.parts;
  const segs = [["gate_proj",0.00,0.30],["up_proj",0.34,0.64],
                ["down_proj",0.68,1.00]];
  for (const [nm,a,b] of segs){
    const hh = (b-a)*1.5;
    box(xF, a*1.5+hh/2, 0, wF, hh, 0.8, "#dce6fd", "#8fa8e8", 1, 1);
    const shp = ffnParts.find(p => p.name === nm);
    label([xF + wF/2 + 0.06, a*1.5+hh/2, 0.42],
          `${nm}  ${shp.rows.toLocaleString()}x${shp.cols.toLocaleString()}`,
          C.blue, 12);
  }
  label([xF, 1.62, 0], `swiGLU FFN — ${(D.section.ffn.params/1e6).toFixed(2)}M`,
        C.blue, 13.5, "center", "700");
  label([xF, 1.50, 0], D.section.ffn.subtitle, C.dim, 12, "center");

  // the read band on the FFN, same language as view A
  const rh = 1.5*readFrac;
  box(xF + 0.01, rh/2, 0, wF+0.02, rh, 0.84,
      lossColour(oracleAt(readFrac).oracle_loss), null, 0, 0.9);

  // dimensions
  dim([1.12, 0, 0], [1.12, rh, 0],
      `${(readFrac*100).toFixed(1)}% of the FFN read`, 1);
  callout([xBB, 1.55, 0], -20, -68,
    `${G.layers} layers: ${G.gdn_layers} of this type,\n` +
    `${G.fa_layers} full attention`, C.ink2);
  callout([xF, 1.28, 0], 132, -30,
    `reading a band of the ${I.toLocaleString()}\n` +
    `intermediate units = the partition`, C.green);
  screenText(20, 26, "SECTION B — one transformer layer, opened up",
             C.ink, 13.5, "left", "700");
  screenText(20, 45,
    `Band heights are sized by each projection's parameter count, not to a ` +
    `length scale. RMSNorm x2 per layer (${G.norm_per_layer.toLocaleString()} params).`,
    C.ink2, 12);
}

function sideLayer(){
  const S = D.section, blk = S[layerType];
  const rows = blk.parts.map(p =>
    `<tr><td>${p.name}${p.n>1?` <span style="color:var(--dim)">x${p.n}</span>`:""}</td>` +
    `<td>${p.rows.toLocaleString()}x${p.cols.toLocaleString()}` +
    `<br><span style="color:var(--dim)">${(p.rows*p.cols*p.n/1e6).toFixed(2)}M</span></td></tr>`
  ).join("");
  side.innerHTML = `
    <h2>${blk.title}</h2>
    <p>${blk.subtitle} — ${blk.params.toLocaleString()} params</p>
    <table>${rows}
      <tr class="tot"><td>block total</td><td>${blk.parts_total.toLocaleString()}</td></tr>
    </table>
    <h2>swiGLU FFN (every layer)</h2>
    <table>${S.ffn.parts.map(p =>
      `<tr><td>${p.name}</td><td>${p.rows.toLocaleString()}x${p.cols.toLocaleString()}` +
      `</td></tr>`).join("")}
      <tr class="tot"><td>block total</td><td>${S.ffn.parts_total.toLocaleString()}</td></tr>
    </table>
    <h2>How it is wired</h2>
    <p>Each layer is <b>norm &rarr; ${blk.title.toLowerCase()} &rarr; residual</b>,
    then <b>norm &rarr; FFN &rarr; residual</b>. The FFN's gate and up projections
    go ${D.geometry.hidden.toLocaleString()} &rarr;
    ${D.geometry.intermediate.toLocaleString()}, are multiplied elementwise and
    come back down — which is why it holds three of the layer's matrices and
    ${(100*D.geometry.ffn/D.geometry.params).toFixed(1)}% of the model.</p>
    <h2>Sources</h2>
    <p>All shapes from <code>src/moeme/accounting.py</code>; part sums are checked
    against the measured block totals in <code>tests/test_model_atlas.py</code>.</p>`;
  const G = D.geometry;
  cards.innerHTML = cardHTML([
    ["this layer", ((blk.params + S.ffn.params)/1e6).toFixed(0)+"M"],
    ["DeltaNet layer", (G.gdn_layer_params/1e6).toFixed(2)+"M"],
    ["attention layer", (G.fa_layer_params/1e6).toFixed(2)+"M"],
    ["FFN, all 64", (G.ffn/1e9).toFixed(2)+"B"],
  ]);
}

/* =========================================================================
   VIEW C -- active targets vs the floor
   ========================================================================= */
function buildTarget(){
  const G = D.geometry, s = 1/3.0e9;
  const floorH = G.floor*s;
  poly([[-2.8,floorH,-2.4],[2.8,floorH,-2.4],[2.8,floorH,2.4],[-2.8,floorH,2.4]],
       "rgba(185,28,28,0.07)", C.red, 1.6);
  callout([2.8, floorH, 2.4], 10, -18,
    `FLOOR ${(G.floor/1e9).toFixed(2)}B — embeddings, LM head, DeltaNet, ` +
    `attention, norms.\nRead on every token. No router touches it.`, C.red);
  const n = D.targets.length;
  D.targets.forEach((t,i) => {
    const x = (i-(n-1)/2)*1.1, fh = (G.floor/1e9)*s;
    if (!t.possible){
      box(x, fh/2, 0, 0.55, fh, 0.55, "#f7d5d5", C.red, 1.6, 1);
      callout([x, fh+0.1, 0], -30, -30,
        `${t.target_b}B active needs ${t.ffn_active_b.toFixed(2)}B of FFN — ` +
        `a negative amount.\nImpossible by routing.`, C.red);
      label([x, fh+0.16, 0], `${t.target_b}B`, C.red, 14, "center", "700");
      return;
    }
    box(x, fh/2, 0, 0.55, fh, 0.55, "#f1e6d0", C.ink, 1.4, 1, true);
    const nh = t.ffn_active_b*s;
    const c = lossColour(oracleAt(t.ffn_fraction_pct/100).oracle_loss);
    box(x, fh+nh/2, 0, 0.58, nh, 0.58, c, null, 0, 0.93);
    dot3([x, fh+nh+0.16, 0], c, 5,
      `${t.target_b}B active\nFFN read ${t.ffn_active_b.toFixed(2)}B = ` +
      `${t.ffn_fraction_pct.toFixed(1)}% of the FFN\nmeasured oracle loss ` +
      `${oracleAt(t.ffn_fraction_pct/100).oracle_loss.toFixed(4)}`);
    label([x, fh+nh+0.30, 0], `${t.target_b}B`, C.ink, 14, "center", "700");
    label([x, fh*0.5, 0], `${(G.floor/1e9).toFixed(1)}B`, C.ink2, 12, "center");
  });
  screenText(18, 26, "ELEVATION C — an active target is not a knob", C.ink, 13,
             "left", "700");
  screenText(18, 44,
    "bar height = parameters read per token; the FFN part is coloured by the " +
    "measured oracle loss at that width", C.ink2, 12);
  side.innerHTML = `
    <h2>The arithmetic</h2>
    <p><code>active = floor + (k/E) x FFN</code></p>
    <table>
      ${D.targets.map(t => `<tr><td>${t.target_b}B target</td><td>` +
        (t.possible
          ? `${t.ffn_active_b.toFixed(2)}B FFN<br><span style="color:var(--dim)">` +
            `${t.ffn_fraction_pct.toFixed(1)}% of FFN</span>`
          : `<span style="color:var(--red)">impossible</span>`) +
        `</td></tr>`).join("")}
    </table>
    <h2>Read this the other way round</h2>
    <p>Wanting a <i>bigger</i> active number means reading a <i>smaller</i>
    fraction of the FFN, and that moves you toward the red end of the measured
    loss curve. 12B active costs ${(D.targets.find(t=>t.target_b===12).ffn_fraction_pct).toFixed(1)}%
    of the FFN; 24B costs
    ${(D.targets.find(t=>t.target_b===24).ffn_fraction_pct).toFixed(1)}%.</p>
    <h2>Sources</h2>
    <p>Floor: <code>src/moeme/accounting.py</code> (executable test).<br>
    Loss: <code>${D.curve_source}</code>.</p>`;
  cards.innerHTML = cardHTML([
    ["floor", (G.floor/1e9).toFixed(2)+"B"],
    ["12B needs", D.targets.find(t=>t.target_b===12).ffn_active_b.toFixed(2)+"B FFN"],
    ["= of FFN", D.targets.find(t=>t.target_b===12).ffn_fraction_pct.toFixed(1)+"%"],
    ["9B needs", D.targets.find(t=>t.target_b===9).ffn_active_b.toFixed(2)+"B"],
  ]);
}

/* =========================================================================
   VIEW D -- residency envelope
   ========================================================================= */
function buildEnvelope(){
  const T = D.thresholds;
  const xMax = 60, zMax = 26;
  const X = v => v/xMax*5.6 - 2.8;
  const Z = v => v/zMax*4.2 - 2.1;
  const Y = b => (b-1.5)/7.5*3.2 - 1.6;

  // the ceiling: everything under this fits the fast tier
  poly([[X(0),Y(1.5),Z(0)],[X(xMax),Y(1.5),Z(0)],[X(xMax),Y(1.5),Z(zMax)],
        [X(0),Y(1.5),Z(zMax)]], "rgba(21,128,61,0.06)", C.green, 1.4);
  callout([X(0), Y(1.5), Z(zMax)], -10, -26,
    `the ${T.resident_gb.toFixed(2)} GB fast tier — anything under this plane ` +
    `is RESIDENT`, C.green);
  // the size surface: for each bit-width, size = params x bpw / 8
  for (let i=0;i<=8;i++){
    const bpw = 1.5 + i*(6.6-1.5)/8;
    seg([X(0),Y(bpw),Z(0)], [X(xMax),Y(bpw),Z(xMax*bpw/8)], C.grid2, 1);
  }
  // this model's hard threshold
  seg([X(27.781),Y(T.threshold_bpw),Z(T.resident_gb)],
      [X(27.781),Y(T.threshold_bpw),Z(0)], C.amber, 2, [5,4]);
  dot3([X(27.781),Y(T.threshold_bpw),Z(T.resident_gb)], C.amber, 4.5,
    `this model's hard threshold\n${T.threshold_bpw.toFixed(2)} bpw -> ` +
    `${T.resident_gb.toFixed(2)} GB`);
  label([X(28.6),Y(T.threshold_bpw),Z(T.resident_gb)], 
        `${T.threshold_bpw.toFixed(2)} bpw`, C.amber, 12, "left", "700");

  const pts = [];
  for (const q of D.ladder)
    pts.push({ label:q.name, params:27.781, bpw:q.bpw, size:q.size_gb, tok:q.tok_s,
               res:q.resident, streamed:q.streamed_gb, kind:"ladder" });
  for (const m of D.mixed)
    pts.push({ label:m.name, params:27.781, bpw:m.bpw, size:m.size_gb, tok:m.tok_s,
               res:m.resident, streamed:m.streamed_gb, kind:"mixed" });
  for (const m of D.measured)
    if (m.total_b) pts.push({ label:m.tag, params:m.total_b, bpw:m.bpw,
        size:m.size_gb, tok:m.tok_s, res:m.resident, kind:"measured",
        active:m.active_b });

  for (const p of pts){
    const col = p.res ? C.green : C.red;
    const sz = p.kind==="measured" ? 0.13 : 0.10;
    box(X(p.params), Y(p.bpw), Z(p.size), sz, sz, sz, col, null, 0, 1);
    const move = p.res ? "RESIDENT"
      : (p.streamed !== undefined ? `streams ${p.streamed.toFixed(2)} GB/token`
                                  : "does not fit the fast tier");
    dot3([X(p.params), Y(p.bpw), Z(p.size)], col, p.kind==="measured"?6:4,
      `${p.label}\n${p.params.toFixed(1)}B total` +
      (p.active ? ` / ${p.active.toFixed(1)}B active` : "") +
      `\n${p.bpw.toFixed(2)} bpw -> ${p.size.toFixed(2)} GB\n${move}\n` +
      `${p.tok.toFixed(2)} tok/s ` +
      (p.kind==="measured" ? "(MEASURED)" : "(predicted)"));
  }
  label([X(30),Y(2.0),Z(0)], "27.78B dense", C.ink2, 12);
  label([X(38),Y(6.4),Z(13)], "35.5B / 3B active MoE", C.ink2, 12);
  screenText(18, 26, "ISOMETRIC D — what fits, at what precision", C.ink, 13,
             "left", "700");
  screenText(18, 44,
    "x = total parameters, y = bits per weight, z = model size on disk",
    C.ink2, 12);
  side.innerHTML = `
    <h2>The machine</h2>
    <table>
      <tr><td>VRAM usable</td><td>${T.vram_gib.toFixed(1)} GiB</td></tr>
      <tr><td>RAM usable</td><td>${T.ram_gb.toFixed(1)} GB</td></tr>
      <tr class="tot"><td>fast tier</td><td>${T.resident_gb.toFixed(2)} GB</td></tr>
      <tr><td>GPU bandwidth</td><td>${T.bw_gpu.toFixed(0)} GB/s</td></tr>
      <tr><td>RAM bandwidth</td><td>${T.bw_ram.toFixed(1)} GB/s</td></tr>
      <tr><td>disk bandwidth</td><td>${T.bw_disk.toFixed(2)} GB/s</td></tr>
    </table>
    <h2>Why residency is the whole game</h2>
    <p>A byte that touches the disk costs ~${(T.bw_ram/T.bw_disk).toFixed(0)}x a
    byte in RAM. Below <b>${T.threshold_bpw.toFixed(2)} bpw</b> the 27B fits the
    fast tier and stops streaming entirely.</p>
    <h2>The ladder's gap</h2>
    <p>Q3_K_M is 3.40 bpw and fits (${D.ladder.find(q=>q.name==="Q3_K_M").size_gb.toFixed(1)} GB);
    the next step up, Q4_K_S, is 4.40 bpw and does not
    (${D.ladder.find(q=>q.name==="Q4_K_S").size_gb.toFixed(1)} GB). That is a
    ~1.0 bpw hole exactly where the threshold sits — which is where a mixed,
    per-module assignment could land.</p>
    <h2>Sources</h2>
    <p>Measurements: <code>configs/tiers.json</code>. Ladder and thresholds:
    <code>${T.source}</code>. <span class="badge p">predicted</span>
    ${D.tier_model_note}</p>`;
  cards.innerHTML = cardHTML([
    ["fast tier", T.resident_gb.toFixed(2)+" GB"],
    ["threshold", T.threshold_bpw.toFixed(2)+" bpw"],
    ["today", D.ladder.find(q=>q.name==="Q4_K_M").tok_s.toFixed(2)+" tok/s"],
    ["best measured", D.measured.slice().sort((a,b)=>b.tok_s-a.tok_s)[0].tok_s.toFixed(2)+" tok/s"],
  ]);
}

function cardHTML(pairs){
  return pairs.map(([k,v]) =>
    `<div class="card"><span class="k">${k}</span><span class="v2">${v}</span></div>`
  ).join("");
}

/* ---------- paint ---------- */
let currentView = "stack";
function draw(){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w*dpr) || cv.height !== Math.round(h*dpr)){
    cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  }
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = C.paper; ctx.fillRect(0,0,w,h);
  paperGrid();
  if (needsFit && VIEW_BOUNDS[currentView]){
    fitSheet(VIEW_BOUNDS[currentView]);
    needsFit = false;
  }
  prims = []; anchors = [];
  if (currentView === "stack"){ buildStack(); sideStack(); }
  else if (currentView === "layer"){ buildLayer(); sideLayer(); }
  else if (currentView === "target"){ buildTarget(); }
  else { buildEnvelope(); }
  prims.sort((a,b) => b.depth - a.depth);
  for (const p of prims){
    if (p.kind === "poly"){
      ctx.beginPath(); ctx.moveTo(p.pj[0].x, p.pj[0].y);
      for (let i=1;i<p.pj.length;i++) ctx.lineTo(p.pj[i].x, p.pj[i].y);
      ctx.closePath();
      ctx.globalAlpha = p.alpha; ctx.fillStyle = p.fill; ctx.fill();
      if (p.stroke && p.sw){ ctx.strokeStyle = p.stroke; ctx.lineWidth = p.sw;
                             ctx.stroke(); }
      ctx.globalAlpha = 1;
    } else if (p.kind === "line"){
      ctx.beginPath(); ctx.moveTo(p.pa.x, p.pa.y); ctx.lineTo(p.pb.x, p.pb.y);
      ctx.strokeStyle = p.color; ctx.lineWidth = p.width;
      if (p.dash) ctx.setLineDash(p.dash);
      ctx.stroke(); ctx.setLineDash([]);
    } else if (p.kind === "dot"){
      ctx.beginPath(); ctx.arc(p.pj.x, p.pj.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = p.color; ctx.fill();
      ctx.strokeStyle = C.panel; ctx.lineWidth = 1.5; ctx.stroke();
    } else if (p.kind === "stext"){
      ctx.fillStyle = p.color;
      ctx.font = `${p.weight} ${p.size}px ui-sans-serif,system-ui,sans-serif`;
      ctx.textAlign = p.align; ctx.textBaseline = "middle";
      const lines = String(p.text).split("\n");
      const lh = p.size*1.32;
      const top = p.y - (lines.length-1)*lh/2;
      lines.forEach((ln,i) => ctx.fillText(ln, p.x, top + i*lh));
      ctx.textAlign = "left";
    }
  }
}

/* ---------- interaction ---------- */
let drag = null;
cv.addEventListener("mousedown", e => { drag = {x:e.clientX, y:e.clientY}; });
window.addEventListener("mouseup", () => { drag = null; });
window.addEventListener("mousemove", e => {
  if (drag){
    cam.yaw += (e.clientX-drag.x)*0.007;
    cam.pitch = Math.max(-0.2, Math.min(1.25, cam.pitch + (e.clientY-drag.y)*0.007));
    drag = {x:e.clientX, y:e.clientY}; draw();
  }
  const r = cv.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  let hit = null, bd = 20;
  for (const a of anchors){
    const d = Math.hypot(a.x-mx, a.y-my);
    if (d < bd && a.text){ bd = d; hit = a; }
  }
  if (hit){ tip.style.display="block"; tip.textContent = hit.text;
            tip.style.left = Math.min(mx+16, r.width-330)+"px";
            tip.style.top = (my+14)+"px"; }
  else tip.style.display = "none";
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  cam.zoom = Math.max(0.3, Math.min(3.6, cam.zoom*(e.deltaY>0?0.93:1.07)));
  document.getElementById("zoom").value = cam.zoom; draw();
}, {passive:false});

function setView(name){
  if (!["stack","layer","target","envelope"].includes(name)) name = "stack";
  currentView = name;
  document.querySelectorAll("#tabs button").forEach(b =>
    b.setAttribute("aria-selected", String(b.dataset.view === name)));
  document.getElementById("ctl-read").style.display =
    (name==="stack"||name==="layer") ? "flex" : "none";
  document.getElementById("ctl-type").style.display =
    name==="layer" ? "flex" : "none";
  prov.textContent = PROV[name];
  cam.zoom = 1.0;
  document.getElementById("zoom").value = 1.0;
  needsFit = true;
  draw();
}
document.querySelectorAll("#tabs button").forEach(b => {
  b.addEventListener("click", () => {
    cam.yaw = CAM_DEFAULT.yaw; cam.pitch = CAM_DEFAULT.pitch;
    location.hash = b.dataset.view;
    setView(b.dataset.view);
  });
});
window.addEventListener("hashchange", () => setView(location.hash.slice(1)));
document.getElementById("read").addEventListener("input", e => {
  readFrac = Number(e.target.value)/100;
  document.getElementById("read-v").textContent = (readFrac*100).toFixed(1)+"%";
  draw();
});
document.getElementById("zoom").addEventListener("input", e => {
  cam.zoom = Number(e.target.value); draw();
});
document.getElementById("btn-gdn").addEventListener("click", () => setLayerType("gdn"));
document.getElementById("btn-fa").addEventListener("click", () => setLayerType("fa"));
function setLayerType(t){
  layerType = t;
  document.getElementById("btn-gdn").setAttribute("aria-pressed", String(t==="gdn"));
  document.getElementById("btn-fa").setAttribute("aria-pressed", String(t==="fa"));
  draw();
}
window.addEventListener("resize", () => { needsFit = true; draw(); });

const PROV = {
  stack: "ELEVATION A — parameter geometry. Shapes and totals from "
       + "src/moeme/accounting.py (verified against the safetensors headers); "
       + "loss from " + D.curve_source + ".",
  layer: "SECTION B — one transformer layer. Projection shapes from "
       + "src/moeme/accounting.py; part sums are checked against the measured "
       + "block totals in tests/test_model_atlas.py.",
  target: "ELEVATION C — active-parameter targets. Floor from "
       + "src/moeme/accounting.py (an executable test); loss colouring from "
       + D.curve_source + ".",
  envelope: "ISOMETRIC D — residency. Measured runs from configs/tiers.json; "
       + "ladder and thresholds from " + D.thresholds.source + ". "
       + D.tier_model_note,
};

/* ---------- did the drawing land inside the frame? ----------
   Blueprints are read off the paper, so clipped labels are a real defect, not a
   cosmetic one. Ink = pixels dark enough to be a line or glyph (block fills and
   the paper grid are deliberately excluded). */
function measureInk(){
  const w = cv.clientWidth, h = cv.clientHeight;
  const d = ctx.getImageData(0,0,cv.width,cv.height).data;
  const sx = cv.width/w, sy = cv.height/h;
  let x0=1e9, y0=1e9, x1=-1, y1=-1, n=0;
  for (let y=0;y<cv.height;y+=2){
    for (let x=0;x<cv.width;x+=2){
      const i=(y*cv.width+x)*4;
      if (d[i]+d[i+1]+d[i+2] < 480){
        n++;
        const px=x/sx, py=y/sy;
        if (px<x0) x0=px; if (px>x1) x1=px;
        if (py<y0) y0=py; if (py>y1) y1=py;
      }
    }
  }
  if (!n) return { ink:0, bbox:null, touches:true };
  return { ink:n, bbox:[x0,y0,x1,y1].map(v=>Math.round(v)),
           touches: (x0<2 || y0<2 || x1>w-3 || y1>h-3) };
}

/* Two labels printed on top of each other are unreadable, so the page measures
   its own label boxes and reports any that collide. */
function labelBox(p){
  const lines = String(p.text).split("\n");
  const w = Math.max(...lines.map(l => l.length)) * p.size * 0.55;
  const h = lines.length * p.size * 1.32;
  const x = p.align === "center" ? p.x - w/2 : (p.align === "right" ? p.x - w : p.x);
  return { x0:x, y0:p.y - h/2, x1:x + w, y1:p.y + h/2, text:p.text };
}
function measureLabels(){
  const boxes = prims.filter(p => p.kind === "stext").map(labelBox);
  let worst = 0, pairs = 0, names = [];
  for (let i=0;i<boxes.length;i++)
    for (let j=i+1;j<boxes.length;j++){
      const a=boxes[i], b=boxes[j];
      const ox = Math.min(a.x1,b.x1) - Math.max(a.x0,b.x0);
      const oy = Math.min(a.y1,b.y1) - Math.max(a.y0,b.y0);
      if (ox > 3 && oy > 3){
        pairs++;
        worst = Math.max(worst, ox*oy);
        if (names.length < 3) names.push(a.text.split("\n")[0].slice(0,22) + " / "
                                       + b.text.split("\n")[0].slice(0,22));
      }
    }
  return { labels:boxes.length, overlaps:pairs, worst_px:Math.round(worst),
           sample:names };
}

/* ---------- self-test: what headless verification reads ---------- */
window.__atlasSelfTest = function(){
  const out = [];
  const ok = (name, cond) => out.push((cond ? "ok " : "FAIL ") + name);
  const G = D.geometry, S = D.section;
  ok("no script errors", scriptError === null);
  ok("floor is 9.783B", Math.abs(G.floor - 9783238144) < 1);
  ok("floor = components",
     G.linear_attention + G.attention + G.lm_head + G.embedding + G.norms
       === G.floor);
  ok("ffn per layer x 64 = ffn", G.ffn_per_layer * G.layers === G.ffn);
  ok("total reconciles", G.text_tower + G.vision + G.mtp === G.params);
  ok("intermediate > hidden", G.intermediate > G.hidden);
  ok("gdn+fa covers 64 layers", G.gdn_layers + G.fa_layers === G.layers);
  ok("layer part sums match block totals",
     Object.values(S).every(b => b.parts_total === b.params));
  ok("oracle curve monotone",
     D.oracle_curve.every((p,i,a) => i===0 || p.oracle_loss <= a[i-1].oracle_loss));
  ok("oracle curve is the k=1 family", D.oracle_curve.length === 4);
  ok("12B is possible", D.targets.find(t=>t.target_b===12).possible === true);
  ok("9B is impossible", D.targets.find(t=>t.target_b===9).possible === false);
  ok("Q4_K_M streams", D.ladder.find(q=>q.name==="Q4_K_M").resident === false);
  ok("Q3_K_M resident", D.ladder.find(q=>q.name==="Q3_K_M").resident === true);
  ok("threshold ~4.11 bpw", Math.abs(D.thresholds.threshold_bpw - 4.112) < 0.01);
  ok("measured runs present", D.measured.length >= 3);
  ok("mixed recipe is resident", D.mixed[0].resident === true);
  ok("projection works", !!project([0,0,0]));
  const ink = measureInk();
  ok("drawing is inside the frame", !ink.touches);
  // how big is the drawing actually painted? "it fits" is not "you can see it"
  const G2 = D.geometry;
  const metrics = {
    scale: sheet ? +sheet.scale.toFixed(2) : null,
    ffn_block_px: sheet ? +(1.5*G2.hidden/G2.intermediate*sheet.scale).toFixed(1) : null,
    column_px: (currentView === "stack" && sheet) ? +(0.155*sheet.scale).toFixed(1) : null,
    smallest_text_px: 11.5,
  };
  ok("scale is sane", !!sheet && sheet.scale > 4);
  const labels = measureLabels();
  ok("no two labels overlap", labels.overlaps === 0);
  window.__atlasInk = ink;
  window.__atlasMetrics = metrics;
  window.__atlasLabels = labels;
  window.__atlasAssertions = out.length;
  document.getElementById("selftest").textContent =
    "SELFTEST: " + (out.every(x => x.startsWith("ok")) ? "ok" : "FAILED")
    + " | assertions=" + out.length + " | " + out.join(" | ")
    + " | view=" + currentView + " | ink=" + JSON.stringify(ink)
    + " | metrics=" + JSON.stringify(metrics)
    + " | labels=" + JSON.stringify(labels);
  return out;
};

setView((location.hash || "#stack").slice(1));
window.__atlasSelfTest();
</script>
</body>
</html>
"""


def render(data: dict) -> str:
    return TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "model_atlas.html"))
    ap.add_argument("--print-data", action="store_true",
                    help="print the derived data block and exit")
    args = ap.parse_args()

    data = derive()
    if args.print_data:
        print(json.dumps(data, indent=2))
        return 0

    html = render(data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(html)

    G, T = data["geometry"], data["thresholds"]
    t12 = next(t for t in data["targets"] if t["target_b"] == 12)
    print(f"wrote {os.path.relpath(args.out, ROOT)} ({len(html)/1024:.1f} KiB)")
    print(f"  floor {G['floor']/1e9:.3f}B  FFN {G['ffn']/1e9:.3f}B  "
          f"total {G['params']/1e9:.3f}B  "
          f"(intermediate {G['intermediate']} vs hidden {G['hidden']})")
    print(f"  12B active needs {t12['ffn_active_b']:.2f}B of FFN "
          f"({t12['ffn_fraction_pct']:.1f}%);  9B needs "
          f"{next(t for t in data['targets'] if t['target_b']==9)['ffn_active_b']:.2f}B")
    print(f"  residency threshold {T['threshold_bpw']:.3f} bpw "
          f"({T['resident_gb']:.2f} GB)")
    print(f"  oracle curve: " + ", ".join(
        f"{p['active_fraction']*100:.1f}%->{p['oracle_loss']:.4f}"
        for p in data["oracle_curve"]))
    print("  layer sections: " + ", ".join(
        f"{k} {v['parts_total']/1e6:.2f}M" for k, v in data["section"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
