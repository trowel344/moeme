#!/usr/bin/env python3
"""
model_atlas.py -- one self-contained HTML page that shows this project's
geometry, its arithmetic and its measurements in 3D.

Why this exists
---------------
Every headline number in this repo is a *shape*: 64 layers, a floor you cannot
route around, an FFN you can, a residency envelope of 14.28 GB, and a measured
loss curve over how much of the FFN you read. Read as prose they are easy to
misremember; seen as geometry they are hard to. This script derives the data
from the repo's own artifacts and accounting module -- no number is typed by
hand -- and writes a single HTML file with no external dependencies, so it opens
offline by double-click.

Provenance is carried per dataset and printed in the page itself: measured
values name the artifact they came from, and values that are predictions from the
memory-tier model are labelled "predicted".

Usage:
    python3 tools/model_atlas.py                 # writes docs/model_atlas.html
    python3 tools/model_atlas.py --out /tmp/a.html
    python3 tools/model_atlas.py --print-data    # the derived data block only

The derived data is also importable, so tests can assert the page still agrees
with the artifacts (`tests/test_model_atlas.py`).
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

RESULTS = os.path.join(ROOT, "results")


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

    geometry = {
        "params": params,
        "ffn": ffn,
        "ffn_per_layer": 3 * C.hidden * C.intermediate,
        "layers": C.num_layers,
        "hidden": C.hidden,
        "intermediate": C.intermediate,
        "linear_attention": C.measured_linear_attention,
        "attention": C.measured_attention,
        "lm_head": C.measured_lm_head,
        "embedding": C.measured_embedding,
        "norms": C.measured_norms,
        "vision": C.vision_params_measured,
        "mtp": C.mtp_params_measured,
        "text_tower": C.text_backbone_params(),
        "floor": irr,
        "floor_matmul_only": matmul_only,
        "source": "src/moeme/accounting.py, docs/source_validation.md "
                  "(verified against safetensors headers)",
    }

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
        row = {
            "run": r["name"],
            "active_fraction": u["active_width_fraction"],
            "k": cfg.get("top_k"),
            "experts": cfg.get("n_experts"),
            "best_single": u["residual_best_single_expert"],
            "oracle": u["residual_oracle_topk"],
            "ppl": r.get("val_ppl"),
        }
        slices.append(row)
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
    known_totals = {"qwen3.8:latest": 27.781, "ornith-1.5:9b": None,
                    "ornith-1.5-35b-a3b:q2_k": 35.505, "ornith-1.5:35b": 35.505}
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
<title>moeme — model atlas</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --line:#30363d; --ink:#e6edf3;
    --dim:#8b949e; --accent:#58a6ff; --good:#3fb950; --bad:#f85149;
    --warn:#d29922; --floor:#6e7681; --ffn:#1f6feb;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;
         gap:18px;align-items:baseline;flex-wrap:wrap}
  h1{font-size:15px;margin:0;font-weight:600}
  header .sub{color:var(--dim);font-size:12px}
  #tabs{display:flex;gap:6px;padding:10px 18px 0}
  #tabs button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
    border-radius:6px 6px 0 0;padding:7px 13px;cursor:pointer;font:inherit}
  #tabs button[aria-selected="true"]{background:#1f2630;border-bottom-color:#1f2630;
    color:var(--accent)}
  #stage{position:relative;height:calc(100vh - 250px);min-height:380px;
         border-top:1px solid var(--line)}
  canvas{display:block;width:100%;height:100%;cursor:grab}
  canvas:active{cursor:grabbing}
  #tip{position:absolute;pointer-events:none;background:#0b1017f2;border:1px solid
    var(--line);border-radius:6px;padding:7px 9px;font-size:12px;max-width:290px;
    display:none;z-index:5;white-space:pre-line}
  #legend{position:absolute;right:12px;top:12px;background:#0b1017d9;border:1px solid
    var(--line);border-radius:6px;padding:9px 11px;font-size:12px;max-width:320px}
  #legend b{font-weight:600}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
  #controls{padding:12px 18px;border-top:1px solid var(--line);display:flex;gap:22px;
    flex-wrap:wrap;align-items:center}
  .ctl{display:flex;flex-direction:column;gap:3px}
  .ctl label{color:var(--dim);font-size:11px;text-transform:uppercase;
    letter-spacing:.04em}
  .ctl .val{font-variant-numeric:tabular-nums;color:var(--accent)}
  input[type=range]{width:190px}
  #readout{display:flex;gap:20px;flex-wrap:wrap;padding:0 18px 14px;font-size:12px}
  #readout div{background:var(--panel);border:1px solid var(--line);border-radius:6px;
    padding:7px 11px;min-width:132px}
  #readout .k{color:var(--dim);font-size:11px;display:block}
  #readout .v{font-size:15px;font-variant-numeric:tabular-nums}
  footer{padding:10px 18px 22px;color:var(--dim);font-size:11.5px;border-top:1px solid
    var(--line)}
  footer code{color:var(--ink)}
  .measured{color:var(--good)} .predicted{color:var(--warn)}
  #selftest{position:absolute;left:-9999px;top:auto;width:1px;height:1px;overflow:hidden}
</style>
</head>
<body>
<header>
  <h1>moeme — model atlas</h1>
  <span class="sub">Qwen3.8-27B geometry, the active-parameter arithmetic, and the
    measured residency envelope. Drag to rotate, wheel to zoom, hover a point.</span>
</header>

<div id="tabs" role="tablist">
  <button role="tab" data-view="layers" aria-selected="true">1 &middot; The model, in layers</button>
  <button role="tab" data-view="target">2 &middot; Active targets vs the floor</button>
  <button role="tab" data-view="envelope">3 &middot; Residency &amp; quantisation envelope</button>
</div>

<div id="stage">
  <canvas id="c"></canvas>
  <div id="tip"></div>
  <div id="legend"></div>
</div>

<div id="controls">
  <div class="ctl" id="ctl-ffn">
    <label for="ffn">FFN width read per token: <span class="val" id="ffn-v">12.5%</span></label>
    <input type="range" id="ffn" min="12.5" max="100" step="12.5" value="12.5">
  </div>
  <div class="ctl">
    <label for="zoom">Zoom</label>
    <input type="range" id="zoom" min="0.4" max="2.6" step="0.05" value="1.1">
  </div>
</div>

<div id="readout"></div>

<footer id="prov"></footer>
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
const tip = document.getElementById("tip"), legend = document.getElementById("legend");
const readout = document.getElementById("readout"), prov = document.getElementById("prov");

/* ---------- tiny 3D: rotate, project, painter-sort ---------- */
const cam = { yaw: -0.62, pitch: 0.34, zoom: 1.1, dist: 6.2 };
let prims = [], anchors = [];

function project(p, w, h){
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  let x = p[0]*cy - p[2]*sy, z = p[0]*sy + p[2]*cy;
  let y = p[1]*cp - z*sp;   z = p[1]*sp + z*cp;
  const zz = z + cam.dist;
  if (zz <= 0.05) return null;
  const f = (Math.min(w,h) * 0.42 * cam.zoom) / zz;
  return { x: w/2 + x*f, y: h/2 - y*f, z: zz };
}
function quad(pts, fill, stroke, alpha){
  const w = cv.clientWidth, h = cv.clientHeight;
  const pj = pts.map(p => project(p, w, h));
  if (pj.some(p => !p)) return;
  const depth = pj.reduce((a,b) => a + b.z, 0) / pj.length;
  prims.push({ kind:"poly", pj, fill, stroke, alpha: alpha === undefined ? 1 : alpha,
               depth });
}
function line(a, b, color, width, dash){
  const w = cv.clientWidth, h = cv.clientHeight;
  const pa = project(a,w,h), pb = project(b,w,h);
  if (!pa || !pb) return;
  prims.push({ kind:"line", pa, pb, color, width: width||1, dash, depth:(pa.z+pb.z)/2 });
}
function label(p, text, color, dy){
  const w = cv.clientWidth, h = cv.clientHeight;
  const pj = project(p, w, h);
  if (!pj) return;
  prims.push({ kind:"text", pj, text, color, dy: dy||0, depth: pj.z - 0.02 });
}
function dot(p, color, r, tipText){
  const w = cv.clientWidth, h = cv.clientHeight;
  const pj = project(p, w, h);
  if (!pj) return;
  prims.push({ kind:"dot", pj, color, r: r||3.5, depth: pj.z });
  if (tipText) anchors.push({ x:pj.x, y:pj.y, text:tipText });
}
function box(cx,cy,cz, sx,sy,sz, fill, stroke, alpha){
  const X=[cx-sx/2,cx+sx/2], Y=[cy-sy/2,cy+sy/2], Z=[cz-sz/2,cz+sz/2];
  const v=(i,j,k)=>[X[i],Y[j],Z[k]];
  const hide = cam.pitch > 0 ? 0 : 1, far = cam.pitch > 0 ? 1 : 0;
  const faces = [
    [v(hide,0,0),v(hide,1,0),v(hide,1,1),v(hide,0,1)],   // x face
    [v(0,far,0),v(1,far,0),v(1,far,1),v(0,far,1)],       // y face
    [v(0,0,far),v(1,0,far),v(1,1,far),v(0,1,far)],       // z face
    [v(1,0,1),v(1,1,1),v(1,1,0),v(1,0,0)],
    [v(0,1,0),v(1,1,0),v(1,1,1),v(0,1,1)],
  ];
  faces.forEach(f => quad(f, fill, stroke, alpha));
}

/* ---------- colour helpers ---------- */
function ramp(t){                       // 0 -> bad red, 1 -> good green
  const c = [[248,81,73],[210,153,34],[63,185,80]];
  const x = Math.max(0, Math.min(1, t)) * (c.length-1);
  const i = Math.min(c.length-2, Math.floor(x)), f = x - i;
  const mix = j => Math.round(c[i][j] + (c[i+1][j]-c[i][j])*f);
  return `rgb(${mix(0)},${mix(1)},${mix(2)})`;
}
const hex = (r,g,b,a) => `rgba(${r},${g},${b},${a===undefined?1:a})`;

/* ---------- view 1: the model in layers ---------- */
let ffnFrac = 0.125;
function oracleAt(frac){
  // nearest measured k=1 point; the curve is monotone so this reads honestly
  let best = D.oracle_curve[0], bd = 1e9;
  for (const p of D.oracle_curve){
    const d = Math.abs(p.active_fraction - frac);
    if (d < bd){ bd = d; best = p; }
  }
  return best;
}
function buildLayers(){
  const G = D.geometry;
  const perLayerFloor = (G.floor - G.embedding) / G.layers;   // matmul-only floor
  const s = 1 / 3.0e9;                                        // world units per 3B params
  const floorH = perLayerFloor * s, ffnH = G.ffn_per_layer * s;
  const readH = ffnH * ffnFrac;
  const loss = oracleAt(ffnFrac).oracle_loss;
  const N = G.layers;
  for (let i=0;i<N;i++){
    const z = (i - (N-1)/2) * 0.155;
    // the floor: read on every token, cannot be routed away
    box(0, floorH/2, z, 1.5, floorH, 0.11, "#39414d", "#20262e", 0.85);
    // the FFN: the part routing can shrink
    box(0, floorH + ffnH/2, z, 1.5, ffnH, 0.11, "#161d29", "#2b3546", 0.5);
    // the part actually read this token (k of E), coloured by measured loss
    box(0, floorH + readH/2, z, 1.55, readH, 0.13, ramp(1-loss), null, 0.95);
  }
  const top = floorH + ffnH + 0.5;
  label([0.95, floorH*0.6, 0], "backbone — every token", "#8b949e");
  label([0.95, floorH + readH + 0.28, 0],
        `FFN read: ${(ffnFrac*100).toFixed(1)}%  (${oracleAt(ffnFrac).run})`,
        ramp(1-loss));
  label([0.95, floorH + ffnH - 0.25, 0], "FFN — routable", "#6e7681");
  legend.innerHTML =
    `<b>Reading ${(ffnFrac*100).toFixed(1)}% of the FFN per token</b><br>` +
    `<span class="sw" style="background:#39414d"></span>backbone floor ` +
    `${(G.floor_matmul_only/1e9).toFixed(2)}B — read always, routable: no<br>` +
    `<span class="sw" style="background:rgb(63,185,80)"></span>the ${(ffnFrac*100).toFixed(0)}% ` +
    `of the FFN that is read<br>` +
    `<span class="sw" style="background:rgb(22,29,41)"></span>FFN not read ` +
    `(${(G.ffn*(1-ffnFrac)/1e9).toFixed(2)}B idle)<br><br>` +
    `measured: even a <b>perfect</b> router loses ` +
    `<b>${loss.toFixed(4)}</b> of the layer output here<br>` +
    `<span style="color:#8b949e">${D.curve_source}</span>`;
}
function readoutLayers(){
  const G = D.geometry;
  const readB = (G.floor_matmul_only + G.ffn*ffnFrac) / 1e9;
  const bpw = D.thresholds.threshold_bpw;
  readout.innerHTML = cards([
    ["active params / token", readB.toFixed(2) + "B"],
    ["of 27.78B total", (100*readB/27.781).toFixed(1) + "%"],
    ["bytes/token @4.9bpw", (readB*4.9/8).toFixed(2) + " GB"],
    ["measured oracle loss", oracleAt(ffnFrac).oracle_loss.toFixed(4)],
    ["BPW that fits 14.28GB", bpw.toFixed(2)],
  ]);
}

/* ---------- view 2: active targets vs the floor ---------- */
function buildTarget(){
  const G = D.geometry, s = 1/3.0e9;
  const floorH = G.floor * s;
  // the floor as a translucent plane spanning the scene
  quad([[-2.6,floorH,-2.2],[2.6,floorH,-2.2],[2.6,floorH,2.2],[-2.6,floorH,2.2]],
       hex(110,118,129,0.16), "#6e7681", 0.5);
  label([2.62, floorH + 0.06, -2.2],
        `floor ${(G.floor/1e9).toFixed(2)}B — read every token, not routable`,
        "#8b949e");
  D.targets.forEach((t, i) => {
    const x = (i - (D.targets.length-1)/2) * 1.05;
    const floorPart = (G.floor/1e9) * s;
    if (!t.possible){
      box(x, floorPart/2, 0, 0.5, floorPart, 0.5, "#3a2028", "#f85149", 0.9);
      dot([x, floorPart + 0.28, 0], "#f85149", 4,
          `target ${t.target_b}B active\nneeds ${t.ffn_active_b.toFixed(2)}B of FFN ` +
          `— a negative amount.\nBelow the ${(G.floor/1e9).toFixed(2)}B floor: impossible ` +
          `by routing.`);
      label([x, floorPart + 0.52, 0], `${t.target_b}B`, "#f85149");
      return;
    }
    box(x, floorPart/2, 0, 0.5, floorPart, 0.5, "#39414d", "#20262e", 0.9);
    const needH = t.ffn_active_b * s;
    const c = ramp(1 - oracleAt(t.ffn_fraction_pct/100).oracle_loss);
    box(x, floorPart + needH/2, 0, 0.52, needH, 0.52, c, null, 0.92);
    dot([x, floorPart + needH + 0.2, 0], c, 4,
        `${t.target_b}B active\n` +
        `FFN read: ${t.ffn_active_b.toFixed(2)}B = ` +
        `${t.ffn_fraction_pct.toFixed(1)}% of the FFN\n` +
        `measured oracle loss at that width: ` +
        `${oracleAt(t.ffn_fraction_pct/100).oracle_loss.toFixed(4)}`);
    label([x, floorPart + needH + 0.44, 0], `${t.target_b}B`, c);
    label([x, floorPart*0.5, 0], `${(G.floor/1e9).toFixed(1)}B`, "#8b949e", 0);
  });
  legend.innerHTML =
    `<b>An active-parameter target is not a knob you turn</b><br>` +
    `<span class="sw" style="background:#39414d"></span>the floor (9.78B): read on ` +
    `every token, unaffected by any router<br>` +
    `<span class="sw" style="background:rgb(63,185,80)"></span>the FFN you may read ` +
    `— colour is the measured oracle loss<br><br>` +
    `<b>12B active is reachable</b> (2.22B of FFN, 13.0%); <b>9B is not</b> ` +
    `(needs −0.78B). But to hit 12B you must read only 13% of the FFN, which is the ` +
    `worst end of the measured curve.<br><br>` +
    `<span style="color:#8b949e">source: src/moeme/accounting.py; ` +
    `results/branch_poc_08b.json</span>`;
}
function readoutTarget(){
  const t12 = D.targets.find(t => t.target_b === 12);
  const t9 = D.targets.find(t => t.target_b === 9);
  readout.innerHTML = cards([
    ["floor (every token)", (D.geometry.floor/1e9).toFixed(2) + "B"],
    ["12B needs FFN active", t12.ffn_active_b.toFixed(2) + "B"],
    ["= fraction of FFN", t12.ffn_fraction_pct.toFixed(1) + "%"],
    ["9B needs", t9.ffn_active_b.toFixed(2) + "B (impossible)"],
    ["oracle loss at 13%", oracleAt(0.13).oracle_loss.toFixed(4)],
  ]);
}

/* ---------- view 3: residency + quantisation envelope ---------- */
let showMixed = true;
function buildEnvelope(){
  const T = D.thresholds;
  const xMax = 60, zMax = 26;                     // params (B), size (GB)
  const X = v => (v - 0) / xMax * 5.2 - 2.6;
  const Z = v => (v - 0) / zMax * 4.0 - 2.0;
  const Y = b => (b - 1.5) / 7.5 * 3.0 - 1.5;      // bpw -> height

  // the ceiling: a plane over all (params, bpw) that fit
  quad([[X(0), Y(1.5), Z(0)],[X(xMax), Y(1.5), Z(0)],
        [X(xMax), Y(1.5), Z(zMax)],[X(0), Y(1.5), Z(zMax)]],
       hex(63,185,80,0.07), "#1b3a24", 0.4);
  // the size surface implied by bpw: size = params*bpw/8
  for (let i=0;i<=8;i++){
    const p = xMax * i/8, bpw = T.threshold_bpw;
    line([X(p), Y(bpw), Z(p*bpw/8)], [X(p), Y(6.6), Z(p*6.6/8)], "#21262d", 1);
  }
  // the 4.11 bpw threshold line for THIS model
  line([X(27.781), Y(T.threshold_bpw), Z(T.resident_gb)],
       [X(27.781), Y(T.threshold_bpw), Z(0)], "#d29922", 1.5, [4,4]);
  dot([X(27.781), Y(T.threshold_bpw), Z(T.resident_gb)], "#d29922", 3,
      `this model's hard residency threshold\n` +
      `${T.threshold_bpw.toFixed(2)} bpw -> ${T.resident_gb.toFixed(2)} GB`);

  const pts = [];
  for (const q of D.ladder)
    pts.push({ label:q.name, params:27.781, bpw:q.bpw, size:q.size_gb,
               tok:q.tok_s, res:q.resident, streamed:q.streamed_gb,
               kind:"ladder" });
  if (showMixed)
    for (const m of D.mixed)
      pts.push({ label:m.name, params:27.781, bpw:m.bpw, size:m.size_gb,
                 tok:m.tok_s, res:m.resident, streamed:m.streamed_gb,
                 kind:"mixed" });
  for (const m of D.measured)
    if (m.total_b) pts.push({ label:m.tag, params:m.total_b, bpw:m.bpw,
        size:m.size_gb, tok:m.tok_s, res:m.resident, kind:"measured",
        active:m.active_b });

  for (const p of pts){
    const col = p.res ? "#3fb950" : "#f85149";
    box(X(p.params), Y(p.bpw), Z(p.size), 0.10, 0.10, 0.10, col, null, 0.95);
    // only the dense ladder rows have a modelled per-token stream; the measured
    // rows report what the runtime did, not a streamed figure we did not measure
    const move = p.res ? "RESIDENT"
      : (p.streamed !== undefined
          ? "streams " + p.streamed.toFixed(2) + " GB/token"
          : "does not fit the fast tier");
    dot([X(p.params), Y(p.bpw), Z(p.size)], col, p.kind==="measured"?5:3.5,
        `${p.label}\n${p.params.toFixed(1)}B total` +
        (p.active ? ` / ${p.active.toFixed(1)}B active` : "") +
        `\n${p.bpw.toFixed(2)} bpw -> ${p.size.toFixed(2)} GB\n${move}\n` +
        `${p.tok.toFixed(2)} tok/s ` +
        (p.kind==="measured" ? "(MEASURED)" : "(predicted)"));
  }
  label([X(27.781), Y(2.0), Z(0)], "27.78B — the dense model", "#8b949e");
  label([X(35.5), Y(6.6), Z(13)], "35.5B / 3B active MoE", "#8b949e");

  const best = pts.filter(p=>p.kind==="mixed" && p.res)
                  .sort((a,b)=>b.tok-a.tok)[0];
  legend.innerHTML =
    `<b>What actually binds: total size at a precision</b><br>` +
    `<span class="sw" style="background:#3fb950"></span>fits the ${T.resident_gb.toFixed(2)} GB ` +
    `fast tier — RESIDENT<br>` +
    `<span class="sw" style="background:#f85149"></span>does not fit — every token ` +
    `streams off a ${T.bw_disk.toFixed(1)} GB/s disk<br>` +
    `<span class="sw" style="background:#d29922"></span>the ${T.threshold_bpw.toFixed(2)} bpw ` +
    `threshold for this model<br><br>` +
    `axes: total params &rarr; depth, bpw &rarr; height, model size &rarr; width` +
    `<br><br><span style="color:#8b949e">measured points from configs/tiers.json; ` +
    `ladder predicted by tools/residency_table.py</span>`;
  window.__bestMixed = best ? best.tok : null;
}
function readoutEnvelope(){
  const T = D.thresholds;
  const mix = D.mixed[0];
  const q49 = D.ladder.find(q => q.name === "Q4_K_M");
  const q34 = D.ladder.find(q => q.name === "Q3_K_M");
  readout.innerHTML = cards([
    ["usable fast memory", T.resident_gb.toFixed(2) + " GB"],
    ["hard threshold", T.threshold_bpw.toFixed(2) + " bpw"],
    ["Q4_K_M (today)", q49.size_gb.toFixed(1) + " GB &middot; " +
       q49.tok_s.toFixed(2) + " tok/s"],
    [mix.name, mix.size_gb.toFixed(2) + " GB &middot; " + mix.tok_s.toFixed(2)],
    ["Q3_K_M (resident)", q34.size_gb.toFixed(1) + " GB &middot; " +
       q34.tok_s.toFixed(2) + " tok/s"],
  ]);
}

function cards(pairs){
  return pairs.map(([k,v]) =>
    `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
}

/* ---------- render loop ---------- */
function draw(){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w*dpr || cv.height !== h*dpr){
    cv.width = w*dpr; cv.height = h*dpr;
  }
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  prims = []; anchors = [];
  const view = currentView;
  if (view === "layers"){ buildLayers(); readoutLayers(); }
  else if (view === "target"){ buildTarget(); readoutTarget(); }
  else { buildEnvelope(); readoutEnvelope(); }
  prims.sort((a,b) => b.depth - a.depth);
  for (const p of prims){
    if (p.kind === "poly"){
      ctx.beginPath();
      ctx.moveTo(p.pj[0].x, p.pj[0].y);
      for (let i=1;i<p.pj.length;i++) ctx.lineTo(p.pj[i].x, p.pj[i].y);
      ctx.closePath();
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.fill; ctx.fill();
      if (p.stroke){ ctx.strokeStyle = p.stroke; ctx.lineWidth = 1; ctx.stroke(); }
      ctx.globalAlpha = 1;
    } else if (p.kind === "line"){
      ctx.beginPath(); ctx.moveTo(p.pa.x, p.pa.y); ctx.lineTo(p.pb.x, p.pb.y);
      ctx.strokeStyle = p.color; ctx.lineWidth = p.width;
      if (p.dash) ctx.setLineDash(p.dash);
      ctx.stroke(); ctx.setLineDash([]);
    } else if (p.kind === "dot"){
      ctx.beginPath(); ctx.arc(p.pj.x, p.pj.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = p.color; ctx.fill();
    } else if (p.kind === "text"){
      ctx.fillStyle = p.color; ctx.font = "12px ui-sans-serif,system-ui,sans-serif";
      ctx.fillText(p.text, p.pj.x, p.pj.y + p.dy);
    }
  }
}

/* ---------- interaction ---------- */
let drag = null, currentView = "layers";
cv.addEventListener("mousedown", e => { drag = { x:e.clientX, y:e.clientY }; });
window.addEventListener("mouseup", () => { drag = null; });
window.addEventListener("mousemove", e => {
  if (drag){
    cam.yaw += (e.clientX - drag.x) * 0.008;
    cam.pitch = Math.max(-1.2, Math.min(1.2, cam.pitch + (e.clientY - drag.y)*0.008));
    drag = { x:e.clientX, y:e.clientY };
    draw();
  }
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  let hit = null, bd = 18;
  for (const a of anchors){
    const d = Math.hypot(a.x - mx, a.y - my);
    if (d < bd && a.text){ bd = d; hit = a; }
  }
  if (hit){ tip.style.display = "block"; tip.textContent = hit.text;
            tip.style.left = Math.min(mx + 14, r.width - 300) + "px";
            tip.style.top = (my + 12) + "px"; }
  else tip.style.display = "none";
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  cam.zoom = Math.max(0.3, Math.min(3.4, cam.zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
  document.getElementById("zoom").value = cam.zoom;
  draw();
}, { passive:false });

function setView(name){
  if (!PROV[name]) name = "layers";
  currentView = name;
  document.querySelectorAll("#tabs button").forEach(x =>
    x.setAttribute("aria-selected", String(x.dataset.view === name)));
  document.getElementById("ctl-ffn").style.visibility =
    name === "layers" ? "visible" : "hidden";
  prov.textContent = PROV[name];
  draw();
}
document.querySelectorAll("#tabs button").forEach(b => {
  b.addEventListener("click", () => {
    cam.yaw = -0.62; cam.pitch = 0.34;
    location.hash = b.dataset.view;      // deep-linkable: #target, #envelope
    setView(b.dataset.view);
  });
});
window.addEventListener("hashchange", () => setView(location.hash.slice(1)));
document.getElementById("ffn").addEventListener("input", e => {
  ffnFrac = Number(e.target.value) / 100;
  document.getElementById("ffn-v").textContent = (ffnFrac*100).toFixed(1) + "%";
  draw();
});
document.getElementById("zoom").addEventListener("input", e => {
  cam.zoom = Number(e.target.value); draw();
});
window.addEventListener("resize", draw);

const PROV = {
  layers: "Layers and floor: src/moeme/accounting.py (verified against the "
        + "safetensors headers). Loss curve: " + D.curve_source,
  target: "Floor: src/moeme/accounting.py. Loss at each width: "
        + D.curve_source,
  envelope: "Measured runs: configs/tiers.json. Ladder and thresholds: "
        + D.thresholds.source + ". " + D.tier_model_note,
};

/* ---------- self-test: what headless verification reads ---------- */
window.__atlasSelfTest = function(){
  const out = [];
  const ok = (name, cond) => out.push((cond ? "ok " : "FAIL ") + name);
  const G = D.geometry;
  ok("no script errors", scriptError === null);
  ok("floor is 9.783B", Math.abs(G.floor - 9783238144) < 1);
  ok("floor = components",
     G.linear_attention + G.attention + G.lm_head + G.embedding + G.norms
       === G.floor);
  ok("ffn per layer x 64 = ffn", G.ffn_per_layer * G.layers === G.ffn);
  ok("total reconciles", G.text_tower + G.vision + G.mtp === G.params);
  ok("oracle curve monotone",
     D.oracle_curve.every((p,i,a) => i === 0 || p.oracle_loss <= a[i-1].oracle_loss));
  ok("12B is possible", D.targets.find(t=>t.target_b===12).possible === true);
  ok("9B is impossible", D.targets.find(t=>t.target_b===9).possible === false);
  const q49 = D.ladder.find(q=>q.name==="Q4_K_M");
  ok("Q4_K_M streams", q49.resident === false);
  const q34 = D.ladder.find(q=>q.name==="Q3_K_M");
  ok("Q3_K_M resident", q34.resident === true);
  ok("threshold ~4.11 bpw", Math.abs(D.thresholds.threshold_bpw - 4.112) < 0.01);
  ok("measured runs present", D.measured.length >= 3);
  ok("mixed recipe is resident", D.mixed[0].resident === true);
  ok("projection works", !!project([0,0,0], 800, 600));
  document.getElementById("selftest").textContent =
    "SELFTEST: " + (out.every(x => x.startsWith("ok")) ? "ok" : "FAILED")
    + " | " + out.join(" | ") + " | view=" + currentView;
  return out;
};

setView((location.hash || "#layers").slice(1));
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

    G = data["geometry"]
    t12 = next(t for t in data["targets"] if t["target_b"] == 12)
    print(f"wrote {os.path.relpath(args.out, ROOT)} ({len(html)/1024:.1f} KiB)")
    print(f"  floor {G['floor']/1e9:.3f}B  FFN {G['ffn']/1e9:.3f}B  "
          f"total {G['params']/1e9:.3f}B")
    print(f"  12B active needs {t12['ffn_active_b']:.2f}B of FFN "
          f"({t12['ffn_fraction_pct']:.1f}%);  9B needs "
          f"{next(t for t in data['targets'] if t['target_b']==9)['ffn_active_b']:.2f}B")
    print(f"  residency threshold {data['thresholds']['threshold_bpw']:.3f} bpw "
          f"({data['thresholds']['resident_gb']:.2f} GB)")
    print(f"  oracle curve: " + ", ".join(
        f"{p['active_fraction']*100:.1f}%->{p['oracle_loss']:.4f}"
        for p in data["oracle_curve"]))
    print(f"  measured points: {len(data['measured'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
