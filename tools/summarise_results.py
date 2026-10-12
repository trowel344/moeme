#!/usr/bin/env python3
"""
summarise_results.py -- re-print every headline number from results/*.json.

Keeps the reports and the raw evidence in sync: if a number is quoted in
docs/, it came out of this script.

Usage:
    python3 tools/summarise_results.py
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

R = "results"


def load(name: str):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def section_hardware() -> None:
    d = load("hardware.json")
    if not d:
        print("hardware.json missing -- run `make doctor`")
        return
    hr("HARDWARE (measured, results/hardware.json)")
    inv, gpu = d["inventory"], d.get("gpu_bench", {})
    print(f"  cpu            : {inv['cpu_model']} ({inv['cpu_logical']} threads)")
    print(f"  ram total      : {inv['ram_total_bytes']/1e9:.2f} GB")
    print(f"  ram available  : {inv['ram_available_bytes']/1e9:.2f} GB")
    print(f"  swap free      : {inv['swap_free_bytes']/1e9:.2f} GB of "
          f"{inv['swap_total_bytes']/1e9:.2f} GB")
    g = inv.get("gpu", {})
    if g.get("available"):
        print(f"  gpu            : {g['name']}")
        print(f"  vram           : {g['vram_total_mib']} MiB total, "
              f"{g['vram_used_mib']} MiB already used by the desktop, "
              f"{g['vram_free_mib']} MiB free")
    if gpu:
        print(f"  gpu bw         : {gpu.get('copy_bw_gbps', 0):.1f} GB/s copy, "
              f"{gpu.get('read_bw_gbps', 0):.1f} GB/s read")
        if "gemv_ms" in gpu:
            print(f"  decode GEMV    : {gpu['gemv_ms']:.3f} ms for one "
                  f"17408x5120 block ({gpu['gemv_effective_bw_gbps']:.1f} GB/s)")
    rb = d.get("ram_bandwidth")
    if rb:
        print(f"  ram bw         : {rb['copy_bw_gbps']:.1f} GB/s copy, "
              f"{rb['read_bw_gbps']:.1f} GB/s single-thread read")
    disk = d.get("disk")
    if disk:
        print(f"  nvme seq read  : {disk['sequential_read_gbps_pass2']:.2f} GB/s "
              f"(file larger than RAM, writeback drained)")
        for k, v in disk.get("random", {}).items():
            print(f"  nvme {k:14s}: {v['throughput_gbps']:.2f} GB/s  "
                  f"({v['median_latency_us']:.0f} us, {v['iops']:.0f} IOPS)")
    mm = d.get("mmap_cold")
    if mm:
        print(f"  mmap cold 1M   : {mm['throughput_gbps']:.2f} GB/s "
              f"({mm['median_latency_ms']:.2f} ms per 1 MiB touch)")


def section_inference() -> None:
    hr("INFERENCE BASELINES (measured, ollama on this machine)")
    # The two MoE files are the sparse-vs-dense evidence: same machine, same
    # runtime, same 14.3 GB of usable fast memory, so the only variable is how
    # many bytes a token actually needs.
    rows = []
    for f in ("inference_safe.json", "inference_27b.json",
              "inference_moe.json", "inference_moe_q4.json"):
        d = load(f)
        if d:
            rows += d["results"]
    if not rows:
        print("  no inference results -- run `make bench-inference`")
        return
    print(f"  {'model':>26} {'decode tok/s':>13} {'prefill tok/s':>14} "
          f"{'peak VRAM':>10} {'load s':>7}")
    for r in rows:
        if r.get("error"):
            print(f"  {r['model']:>26}   ERROR: {r['error'][:60]}")
            continue
        d = r.get("decode_tok_s") or 0.0
        p = r.get("prefill_tok_s") or 0.0
        print(f"  {r['model']:>26} {d:>13.3f} {p:>14.2f} "
              f"{r['peak_vram_mib']:>8} MiB {r.get('load_s', 0):>7.1f}")
    print("\n  Note the two 35.5B `qwen35moe` rows: 256 experts, top-8, so a token")
    print("  reads 8 full experts rather than the whole model. At 13.0 GB (Q2_K)")
    print("  it fits the 14.3 GB of usable fast memory and decodes at 19.4 tok/s;")
    print("  the same model at 22.0 GB (Q4_K_M) does not fit, and drops to 6.45.")
    print("  The dense 27.8B is smaller than both and still slower than either,")
    print("  because a dense layer has no cold subset: every weight is hot.")


def section_eval() -> None:
    """E18/E19: the quality gate -- fixed tasks, fixed decode settings.

    Printed per model AND on the intersection of task ids, because a slow model
    can be evaluated on a subset while a fast one runs the full set; the
    intersection is the only place the two rates are comparable.
    """
    files = sorted(glob.glob(os.path.join(R, "eval_*.json")))
    if not files:
        print("\nno eval_*.json -- run `make eval MODEL=...`")
        return
    hr("QUALITY GATE (results/eval_*.json)")
    print("  Fixed task ids, temperature 0, fixed seed. Coding tasks are EXECUTED")
    print("  against fixed cases, not string-matched. `rate` = passed / run.\n")
    runs = {os.path.basename(f): json.load(open(f)) for f in files}
    entries: list[tuple[str, tuple, dict[str, bool]]] = []
    for fname, d in runs.items():
        ids = d.get("task_ids")
        if ids is None:
            print(f"  {fname}: recorded before --tasks existed, so no task_ids "
                  f"(think={d.get('think')}, num_predict={d.get('num_predict')})")
        else:
            # The engine is part of the run's identity now: the same weights
            # through llama.cpp at an explicit placement is a different
            # experiment from the ollama tag, and the report has to say which.
            backend = d.get("backend", "ollama")
            place = (d.get("engine") or {}).get("placement_flags")
            print(f"  {fname}: {len(ids)} tasks selected, think={d.get('think')}, "
                  f"num_predict={d.get('num_predict')}, engine={backend}"
                  + (f" [{place}]" if place else ""))
        for model, m in sorted(d.get("models", {}).items()):
            rows = m.get("tasks") or []
            if not rows:
                continue
            dec = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
            # A run still in progress is checkpointed with `overall` only, so
            # neither `rate` nor `wall_s` is guaranteed to be present. Never let
            # a partial file crash the summary, and never print a partial run as
            # if it were the finished score.
            total_sel = m.get("tasks_selected",
                               len(ids) if ids is not None else len(rows))
            partial = "  [PARTIAL]" if len(rows) < total_sel else ""
            passed = m["overall"].get("passed", sum(1 for r in rows if r["passed"]))
            rate = m["overall"].get("rate")
            rate = rate if rate is not None else (passed / len(rows))
            wall = m.get("wall_s")
            wall_s = f"{wall:>6.0f}s" if wall else "     -"
            print(f"    {model:<28} {passed:>2}/{len(rows):<3} "
                  f"= {100*rate:>5.1f}%  "
                  f"wall {wall_s}  "
                  f"decode {statistics.median(dec) if dec else float('nan'):>5.1f} tok/s "
                  f"(median){partial}")
            trunc = m.get("truncated_tasks") or 0
            un = m.get("overall_untruncated")
            if trunc:
                # A row the token cap cut off holds no usable answer, so it stays
                # a failure in the score above. This is the score with those rows
                # removed -- the only one that can be compared across different
                # caps, because it stops a harness budget from reading as a
                # capability gap.
                tail = (f"; excluding them {un['passed']}/{un['total']} = "
                        f"{100*un['rate']:.1f}%" if un else "")
                # Reports written before num_predict was recorded have no cap to
                # name; saying "num_predict=None" would invent one.
                cap = (f"num_predict={d['num_predict']}"
                       if d.get("num_predict") else "the token cap")
                print(f"        ({trunc} row(s) hit {cap}{tail})")
            for cat, s in sorted((m.get("summary") or {}).items()):
                capped = f"  [{s['truncated']} capped]" if s.get("truncated") else ""
                print(f"        {cat:<12} {s['passed']}/{s['total']} = "
                      f"{100*s['rate']:>3.0f}%{capped}")
            print()
            if ids:
                # Group by settings as well as ids: two runs can share a task
                # list and still not be comparable (a 192-token cap against a
                # 768-token cap decides the reasoning column on its own).
                entries.append((model,
                                (tuple(ids), d.get("num_predict"), d.get("think")),
                                {r["id"]: bool(r["passed"]) for r in rows},
                                len(rows) >= len(ids),
                                d.get("backend", "ollama")))
    # Only models that ran the SAME task ids are comparable, and a model can
    # legitimately appear twice (a full run and a short subset). Group by the
    # exact id tuple instead of intersecting everything, which would silently
    # produce a non-empty table out of two different task sets.
    by_set: dict[tuple, list] = {}
    for label, key, res, complete, backend in entries:
        by_set.setdefault(key, []).append((label, res, complete, backend))
    for key, group in by_set.items():
        if len(group) < 2:
            continue
        tids, npred, think = key
        engines = ", ".join(sorted({b for _, _, _, b in group}))
        print(f"  Head-to-head on the {len(tids)} task(s) these runs share "
              f"(same ids, num_predict={npred}, think={think}; engines: {engines}):")
        for label, res, complete, backend in sorted(group, key=lambda kv: kv[0]):
            p = sum(1 for t in tids if res.get(t))
            # A run that has only reached the first few tasks must not read as a
            # finished score next to a complete one.
            note = "" if complete else f"  (only {len(res)} of {len(tids)} run so far)"
            tag = "" if backend == "ollama" else f"  [{backend}]"
            print(f"    {label:<28} {p:>2}/{len(tids)} = {100*p/len(tids):>5.1f}%{note}{tag}")
        print()


def section_placement() -> None:
    """E20: explicit --n-cpu-moe placement vs the runtime's own offload policy."""
    d = load("moe_placement.json")
    if not d:
        print("\nmoe_placement.json missing -- see docs/quality_vs_speed.md")
        return
    hr("EXPERT PLACEMENT (results/moe_placement.json)")
    a = d["artifact"]
    print(f"  {a['ollama_tag']}  {a['arch']} {a['quant']}  "
          f"{a['total_params']/1e9:.1f}B total / {a['active_params']/1e9:.1f}B active")
    print(f"  engine: {d['engine']['build']}\n")
    print(f"  {'placement':<16} {'prompt tok/s':>12} {'decode tok/s':>12}  source")
    for p in d["placements"]:
        print(f"  {p['name']:<16} {p['prompt_tok_s']:>12.1f} {p['decode_tok_s']:>12.1f}  "
              f"{p['source'][:44]}")
    b = d["ollama_baseline"]
    print(f"  {'ollama default':<16} {b['prefill_tok_s_end_to_end']:>12.1f} "
          f"{b['decode_tok_s_end_to_end']:>12.1f}  {b['policy'][:44]}")
    best = max(d["placements"], key=lambda p: p["decode_tok_s"])
    print(f"\n  Best explicit placement ({best['name']}): {best['decode_tok_s']:.1f} tok/s")
    print(f"    x{best['decode_tok_s']/b['decode_tok_s_end_to_end']:.2f} the ollama "
          f"end-to-end decode ({b['decode_tok_s_end_to_end']:.2f})")
    print(f"    x{best['decode_tok_s']/b['decode_tok_s_mean_over_30_eval_tasks']:.2f} "
          f"ollama's OWN mean over 30 eval tasks "
          f"({b['decode_tok_s_mean_over_30_eval_tasks']:.1f}, median "
          f"{b['decode_tok_s_median_over_30_eval_tasks']:.1f})")
    print(f"    prompt processing x{best['prompt_tok_s']/b['prefill_tok_s_end_to_end']:.1f}"
          f" ({best['prompt_tok_s']:.1f} vs {b['prefill_tok_s_end_to_end']:.1f} tok/s)")
    s = d["surplus"]
    print(f"\n  surplus: predicted {s['planner_prediction_decode_tok_s']:.1f} tok/s for "
          f"gpu_layers={s['planner_plan']['gpu_layers']}, "
          f"n_cpu_moe={s['planner_plan']['n_cpu_moe_layers']} -- unverified:")
    print(f"    `surplus bench` {s['bench_outcome']} ({s['bench_error']})")
    print(f"    {s['isolated_cause']}")


def section_real_ffn() -> None:
    d = load("real_ffn.json")
    if not d:
        print("\nreal_ffn.json missing -- run `make verify-real-ffn`")
        return
    hr("REAL Qwen3.8-27B WEIGHTS: partitioning exactness (results/real_ffn.json)")
    g = d["geometry"]
    print(f"  layer {d['layer']}: hidden={g['hidden']} intermediate={g['intermediate']} "
          f"params={g['params']:,}")
    worst = max(v for row in d["exactness_worst_rel_error"].values()
                for v in row.values())
    print(f"  worst relative L2 error, all experts active vs dense: {worst:.3e}")
    print("  -> partitioning is exactly dense-equivalent on the real artefact")
    tp = d.get("throughput", {})
    if "dense_ms" in tp:
        print(f"\n  throughput at batch 1 (GPU):")
        print(f"    dense           {tp['dense_ms']:.3f} ms")
        for k, v in tp.items():
            if k.endswith("_all_active_ms"):
                e = k.split("_")[0]
                print(f"    {e:4s} all-active  {v:.3f} ms  "
                      f"({tp[e+'_all_active_overhead_vs_dense']:.2f}x dense)")


def section_pilot() -> None:
    d = load("pilot_sparsity.json")
    if not d:
        print("\npilot_sparsity.json missing -- run `make pilot`")
        return
    hr("PILOT SPARSITY STUDY (results/pilot_sparsity.json)")
    print(f"  model: {d['model']}  calibration: {d['calibration_chars']:,} chars")
    print("  metric: relative L2 residual of the k-expert reconstruction\n")
    print(f"  {'layer':>5} {'plan':>12} {'E':>4} {'k':>3} {'active':>7} "
          f"{'largest':>8} {'oracle':>7} {'learned':>8}")
    for L in sorted(d["layers"], key=int):
        for key, v in d["layers"][L].items():
            if not isinstance(v, dict) or "oracle" not in v:
                continue
            E = int(key.split("_E")[1])
            plan = key.split("_E")[0]
            for k, o in sorted(v["oracle"].items(), key=lambda kv: int(kv[0])):
                if k == "1" or int(k) == E // 8:
                    print(f"  {L:>5} {plan:>12} {E:>4} {k:>3} "
                          f"{100*int(k)/E:>6.1f}% "
                          f"{v['expert_sizes']['largest_share']:>8.3f} "
                          f"{o['mean_rel_residual']:>7.4f} "
                          f"{v['learned_router'][k]['mean_rel_residual']:>8.4f}")


def section_upcycle() -> None:
    """E14 (local, 4 layers), its re-run, and E15 (Kaggle, 24 layers).

    Each line prints the **eval window** it was averaged over, because perplexity
    on this model and this text is 19.57 over 4 val blocks and 23.42 over 8: a 20%
    swing from one flag that changes no weights and no data. E14's report predates
    that field, which is precisely why its absolute numbers are not quotable and
    why its re-run is listed beside it.
    """
    sources = (("E14 local  ", "upcycle_poc.json"),
               ("E14 re-run ", "upcycle_poc_rerun.json"),
               ("E15 Kaggle ", "kaggle_upcycle_poc.json"),
               ("E15 seed 1 ", "kaggle_upcycle_poc_seed1.json"))
    if not any(load(f) for _, f in sources):
        print("\nupcycle_poc.json missing -- run `make upcycle` or the Kaggle notebook")
        return
    hr("UPCYCLING POC: trained top-k vs the frozen geometric partition")
    print("  residual = relative L2 error of the k-expert reconstruction vs the")
    print("  true dense FFN output. `frozen` is the pilot's untrained ceiling;")
    print("  `trained oracle` is the best-possible k-expert pick after training.\n")
    for label, fname in sources:
        d = load(fname)
        if not d:
            continue
        n_layers = len(d["runs"][0]["layers"]) if d.get("runs") else 0
        blocks = d.get("eval_blocks")
        window = (f"{blocks} val blocks" if blocks
                  else "eval window NOT recorded -- not comparable")
        print(f"  {label} {d['model']}  device={d.get('device')}  "
              f"{n_layers} layers  dense ppl {d.get('dense_val_ppl', float('nan')):.3f}")
        print(f"    [{window}]  [{d.get('corpus', 'corpus NOT recorded')}]")
        print(f"    {'E':>4} {'k':>2} {'frozen':>8} {'oracle':>8} {'router':>8} "
              f"{'beats?':>7} {'ppl untr':>10} {'ppl tr':>9}")
        for r in d.get("runs", []):
            if "frozen_oracle_mean_on_common_layers" not in r:
                continue
            print(f"    {r['config']['n_experts']:>4} {r['config']['top_k']:>2} "
                  f"{r['frozen_oracle_mean_on_common_layers']:>8.4f} "
                  f"{r['trained_oracle_mean_on_common_layers']:>8.4f} "
                  f"{r['trained_router_mean_on_common_layers']:>8.4f} "
                  f"{str(r['beats_frozen_oracle']):>7} "
                  f"{r.get('untrained_val_ppl', float('nan')):>10.1f} "
                  f"{r.get('trained_val_ppl', float('nan')):>9.1f}")
        print()
    print("  `beats?` compares only the layers the pilot measured (E15: 0/3/8/23).")
    print("  At E=8 the two agree; at E=32 and E=64 training makes the residual")
    print("  HIGHER than the frozen partition, and the gap grows with E.")
    e14, rerun = load("upcycle_poc.json"), load("upcycle_poc_rerun.json")
    if e14 and rerun:
        print("\n  The rows above that say 'eval window NOT recorded' were all written"
              " before")
        print("  that field existed, so every absolute perplexity in them is"
              " unquotable:")
        print("  on this model and this text perplexity is 19.57 over 4 val blocks and")
        print("  23.42 over 8. What is comparable across them is the x-dense ratio,")
        print("  which reproduces (1.55/1.41/1.31 in E14 vs 1.52/1.40/1.29 in its")
        print(f"  re-run, dense ppl {e14.get('dense_val_ppl', float('nan')):.3f} vs "
              f"{rerun.get('dense_val_ppl', float('nan')):.3f} on the same data).")
        print("  A missing window is recoverable, not lost: perplexity is a running")
        print("  mean of per-block losses, so it identifies the k that produced it.")
        print("  E14's is recovered as 6 val blocks -- see")
        print("  `make recover-eval-window-selftest` or results/eval_window_recovery.json.")
        print("  The re-run's window is in logs/upcycle_rerun.cmd.sh, and the two")
        print("  probes that bracket it are logs/blocks8.log and logs/hdr.log.")

    # E20 -- the same question one level down: not "how much of the layer output
    # is lost" but "does it still write the same text". A perplexity ratio cannot
    # answer that. Printing the two all-experts rows is the point: only the
    # *fresh* one is a control. Both are labelled here so neither can be read as
    # the other, which is the one mistake this table invites.
    d = load("upcycled_08b_vs_dense.json")
    if not d:
        return
    hr("UPCYCLED 0.8B vs DENSE: does it still write the same text? (E20)")
    print(f"  {d['model']}, {len(d['layers_upcycled'])} of {d['n_layers_total']} "
          f"layers upcycled {d['layers_upcycled']}, E={d['config']['n_experts']} "
          f"top-{d['config']['top_k']}, checkpoint step {d.get('checkpoint_step')}")
    print(f"  [{d.get('eval_blocks')} val blocks]  [{d.get('corpus')}]\n")
    k = d["config"]["top_k"]
    rows = (("dense baseline", d["dense_val_ppl"]),
            (f"fresh partition, top-{k} (the post-hoc path)", d["fresh_val_ppl"]),
            (f"trained, top-{k}", d["moe_val_ppl"]),
            ("fresh partition, all experts  <- CONTROL", d["dense_limit_val_ppl"]),
            ("trained partition, all experts (not a control)",
             d.get("trained_all_experts_val_ppl", float("nan"))))
    print(f"  {'configuration':<48} {'ppl':>9}")
    for label, v in rows:
        print(f"  {label:<48} {v:>9.4f}")
    ratio = d["trained_all_experts_val_ppl"] / d["dense_val_ppl"]
    print(f"\n  trained/dense {d['ppl_ratio_moe_over_dense']:.2f}x | control relative "
          f"error {d['dense_limit_rel_error']:.2e} | trained all-experts {ratio:.0f}x")
    print(f"  continuations: {d['prompts_identical']}/{d['prompts_total']} identical to "
          f"dense, mean token agreement {d['mean_token_match_fraction']:.3f}, first")
    print(f"  divergence at "
          f"{[r['first_divergence_index'] for r in d['prompts']]}")
    print("  A fresh partition with every expert active IS the dense layer, so that")
    print(f"  row is the control and it passes; the trained one is {ratio:.0f}x dense because")
    print("  training moved the experts off the slices they were cut from. Re-run:")
    print("  `make compare-upcycled-08b`.")


def section_branch() -> None:
    """E16: complete modules (replica) vs disjoint slices (tile).

    Two residuals are printed because they answer different questions. The
    `best1`/`oracle` columns are per-token *oracle* errors and are the only fair
    way to compare the two constructions at equal active width; `router` is what
    the untrained router actually achieves and is an artefact of initialisation.
    A replica has one distinct block, so for every replica row
    best1 == oracle == router.
    """
    d = load("branch_poc_08b.json")
    if not d:
        print("\nbranch_poc_08b.json missing -- run `make branch`")
        return
    hr("BRANCHING vs SLICING (results/branch_poc_08b.json)")
    print(f"  model: {d['model']}  dtype: {d.get('weight_dtype')}  "
          f"layers: {d.get('layers')}  corpus: {d.get('corpus')}")
    print(f"  dense val ppl {d.get('dense_val_ppl', float('nan')):.4f}  "
          f"(residuals are vs the ORIGINAL dense layer; bf16 floor ~4e-3)\n")
    print(f"  {'config':<34} {'active':>6} {'stored':>7} {'best1':>7} "
          f"{'oracle':>7} {'router':>7} {'ppl':>7}")
    for r in d.get("runs", []):
        b = r.get("untrained")
        if not b:
            continue
        t = r.get("trained") or {}
        router = (f"{b['residual_router']:.4f}"
                  if not t else f"{b['residual_router']:.4f}->{t['residual_router']:.4f}")
        print(f"  {r['name']:<34} {b['active_width_fraction']:>6.3f} "
              f"{b['stored_width_fraction']:>6.2f}x "
              f"{b['residual_best_single_expert']:>7.4f} "
              f"{b['residual_oracle_topk']:>7.4f} {router:>13} "
              f"{r.get('val_ppl', float('nan')):>7.2f}")
    print("\n  A full-width replica at k=1 reproduces the dense model exactly")
    print("  (ppl 15.49 vs 15.48) at 4.00x the FFN storage; no slicing")
    print("  configuration does unless k == E. Best-tile and best-replica agree")
    print("  to ~0.02 at equal active width, so width -- not tiling -- sets the")
    print("  deficit. In every trained row the dense residual RISES while ppl")
    print("  falls, so that residual cannot arbitrate a trained model.")
    btx = [r for r in d.get("runs", []) if r.get("btx")]
    if btx:
        print("\n  branch-train-miX (experts pre-trained apart on code/math/chat/wiki):")
        for r in btx:
            u = r.get("expert_usage_on_val") or {}
            spread = ", ".join(f"l{i} " + str([round(x, 2) for x in v])
                              for i, v in list(u.items())[:2])
            print(f"    {r['name']:<26} ppl {r.get('val_ppl', float('nan')):>6.2f}  "
                  f"usage {spread}")
        print("    -> specialisation helps at 25% active width (21.64 vs 23.19 for")
        print("       the same architecture trained on mixed data for the same steps)")
        print("       and vanishes at full width (19.14 vs 19.11).")


def section_quant() -> None:
    """E19: what a quantisation costs, on identical tokens.

    The gate in section_eval is pass/fail on exact-match checks; this is the
    instrument for fidelity. Three things are printed together on purpose: the
    estimate, the *scored window* (llama-perplexity scores chunks*n_ctx tokens,
    not the whole corpus) and the corpus sha256. A PPL number quoted without both
    is not comparable to anything. A run the memory guard had to end has no
    estimate at all, and that is printed as `--` with its exit code rather than
    as a zero -- on this host it is the correct answer for the 21 GB artifact.
    """
    d = load("quant_perplexity_35b.json")
    if not d:
        print("\nquant_perplexity_35b.json missing -- run `make quant-ppl`")
        return
    hr("QUANTISATION COST (results/quant_perplexity_35b.json)")
    c = d["corpus"]
    print(f"  corpus : {c['bytes']} bytes from {len(c['files'])} files, "
          f"sha256 {c['sha256'][:16]}...")
    print(f"  flags  : -c {d.get('ctx')} --chunks {d.get('chunks')} "
          f"-t {d.get('threads')} -ngl {d.get('gpu_layers')}")
    print(f"\n  {'label':<10} {'quant':<9} {'PPL':>9} {'stderr':>7} "
          f"{'window':>7} {'wall':>7}  verdict")
    for label, run in d["runs"].items():
        ppl = run.get("ppl")
        window = run.get("tokens_scored")
        if run.get("guard_ended_it"):
            verdict = "GUARD TERMINATED IT -- no estimate, not a score"
        elif run.get("exit_code") == 0:
            verdict = "ok"
        else:
            verdict = f"exit {run.get('exit_code')} -- no estimate, not a score"
        ppl_s = f"{ppl:.4f}" if ppl else "--"
        err_s = f"{run['stderr']:.3f}" if run.get("stderr") else "-"
        win_s = str(window) if window else "-"
        print(f"  {label:<10} {str(run.get('quant')):<9} {ppl_s:>9} {err_s:>7} "
              f"{win_s:>7} {run.get('wall_s', 0):>6.0f}s  {verdict}")
        if run.get("scored_window_note"):
            print(f"             {run['scored_window_note']}")
    scored = {k: v["ppl"] for k, v in d["runs"].items() if v.get("ppl")}
    if len(scored) == 2:
        (lo, a), (hi, b) = sorted(scored.items(), key=lambda kv: kv[1])
        print(f"\n  {hi} is {b - a:+.4f} PPL ({100 * (b - a) / a:+.1f}%) against {lo}")
    elif not scored:
        print("\n  no run produced an estimate")
    if any(v.get("ppl") and not v.get("tokens_scored") for v in d["runs"].values()):
        print("\n  A '-' window means the record predates the scored-window field. The run "
              "still\n  printed it -- it is in its log (logs/ppl_<label>.log) -- and a "
              "re-run of\n  `make quant-ppl` fills the field in; do not read a corpus "
              "byte count as the\n  window that was measured.")
    print("\n  The gate above reports pass/fail on easy exact-match checks and cannot")
    print("  see quantisation damage: the same 35B MoE at Q2_K and Q4_K_M scores")
    print("  17/19 vs 16/19 while 13 of the 19 responses are different text. Use the")
    print("  gate for capability and this for fidelity -- and quote a PPL number only")
    print("  with its corpus sha256 and scored window.")


def main() -> int:
    print("moeme result summary")
    section_hardware()
    section_inference()
    section_placement()
    section_eval()
    section_quant()
    section_real_ffn()
    section_pilot()
    section_upcycle()
    section_branch()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
