# Branching vs slicing: are MoE experts *complete modules* or *slices*?

E14/E15 upcycled a dense FFN by **slicing** its intermediate axis into E disjoint
experts and keeping k. The standing objection to that whole line of work is:

> Real MoE models do not slice anything. Mixtral, Switch and BTX build their
> experts from **complete** feed-forward networks. Activating one performs a
> whole layer's maths, so there is no partial-sum deficit to begin with. The
> negative result may be an artefact of the construction, not a fact about MoE.

That objection is correct, and this document reports the experiment that tests
it. Two constructions are built from the *same* dense layer, evaluated on the
*same* activations with the *same* metric, and compared at *equal active width*
— which is what separates two claims that are easy to conflate:

| claim | what it is about |
|---|---|
| "slicing hurts" | tiles vs replicas |
| "reading less hurts" | width |

## What was built

`src/moeme/branch.py` — `BranchMoEFFN`, a top-k FFN whose experts can be
initialised two ways, from the same dense weights, with the same router, losses
and diagnostics as `moeme.upcycle.MoEFFN`:

* `init="tile"` — expert *j* is the *j*-th disjoint block of intermediate units.
  At `expert_units = I/E` this **is** the E14/E15 construction, and summing all
  experts reproduces the dense layer. Storage stays at 1x the dense FFN.
* `init="replica"` — every expert is initialised from the *same* block. At
  `expert_units = I` every expert is a complete copy, so `top_k = 1` reproduces
  the dense layer **exactly** — at E x the storage.

Three mechanisms were added to make the "cut it into sub-models and give each one
its own data" idea testable rather than merely arguable:

1. **`n_shared` / `shared_units`** — always-on experts added to every token, so
   the units that are active for most inputs can be moved out of the router's
   remit (the DeepSeek-MoE / CMoE fix for router collapse).
2. **`set_force_expert(e)`** — bypasses the router and sends every token to one
   expert at weight 1.0, with no balancing loss. This is what lets an expert be
   trained **alone, on its own data, before any router exists**.
3. **`combine`** — `sum` (E14/E15's default), `mean`, or `softmax`.

`experiments/branch_poc.py` — the driver. `--btx` adds the branch-train-miX
configurations: stage 2 trains each expert independently on a *different domain*
(code, math, chat, wiki), stage 3 trains the router and experts jointly on the
mixed stream. `--self-test` checks the whole pipeline offline in ~2 s.

## Method

Qwen3.5-0.8B, `bf16`, layers `[0, 3, 8, 23]` of 24 replaced, 3M characters of
streamed `fineweb-edu` (661,966 tokens; 4 held-out val blocks), seq 512, effective
batch 2048, 80 cooperation steps, 40 steps per expert in stage 2, lr 1e-4,
`aux_loss_coef` 1e-2. TF32 is **disabled**, so an identity that should be exact
reads as exact and a small difference is architectural rather than arithmetic.
BF16 still leaves a ~4e-3 relative floor on residuals; the exactness identities
are verified to 1e-5 in fp32 by `--self-test` and by `tests/test_branch.py`.

All residuals are against the **original dense** layer, which is the only
reference that means the same thing for a tile and a replica. (`oracle_topk` in
`moeme.ffn` compares against the sum of all experts — the dense layer for a tile,
E x the dense layer for a replica — so it is not usable here.)

## Results

dense val ppl **15.4803**. `residual_*` are mean relative L2 errors over the four
upcycled layers; `ppl` is the whole model on the held-out blocks.

| config | active width | stored | best-1 | oracle top-k | router | all-sum | ppl |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tile_E8_k1_12pct` | 0.125 | 1.00x | 0.7104 | 0.7104 | 0.9231 | 0.005 | 54.00 |
| `tile_E4_k1_25pct` | 0.250 | 1.00x | 0.6339 | 0.6339 | 0.8878 | 0.005 | 45.63 |
| `tile_E2_k1_50pct` | 0.500 | 1.00x | 0.4975 | 0.4975 | 0.6656 | 0.004 | 26.02 |
| `tile_E1_k1_100pct` | 1.000 | 1.00x | 0.0000 | 0.0000 | 0.0000 | 0.000 | 15.48 |
| `tile_E4_k2_50pct` | 0.500 | 1.00x | 0.6339 | 0.4820 | 0.7392 | 0.005 | 34.34 |
| `tile_E4_k4_100pct` | 1.000 | 1.00x | 0.6339 | 0.0051 | 0.0058 | 0.005 | 15.47 |
| `replica_E4_k1_12pct` | 0.125 | 0.50x | 0.7307 | 0.7307 | 0.7307 | 1.831 | 33.88 |
| `replica_E4_k1_25pct` | 0.250 | 1.00x | 0.6530 | 0.6530 | 0.6530 | 2.117 | 24.93 |
| `replica_E4_k1_50pct` | 0.500 | 2.00x | 0.5134 | 0.5134 | 0.5134 | 2.493 | 20.11 |
| **`replica_E4_k1_full`** | **1.000** | **4.00x** | **0.0000** | **0.0000** | **0.0038** | 3.000 | **15.49** |
| `trained_tile_E4_k1_25pct` | 0.250 | 1.00x | 0.6339 | 0.6339 | 0.8852→0.9303 | 0.005 | 23.19 |
| `trained_replica_E4_k1_full` | 1.000 | 4.00x | 0.0000 | 0.0000 | 0.0044→0.5491 | 3.000 | 19.11 |
| `trained_replica_E4_k1_full_shared` | 1.000 | 4.00x | 0.0000 | 0.0000 | 0.0035→0.5142 | 3.000 | 22.78 |
| `btx_replica_k1_full` | 1.000 | 4.00x | 0.0000 | 0.0000 | 0.0044→0.6296 | 3.000 | 19.14 |
| `btx_replica_k1_half` | 0.500 | 2.00x | 0.5134 | 0.5134 | 0.5134→0.7336 | 2.493 | 19.80 |
| `btx_tile_k1_25pct` | 0.250 | 1.00x | 0.6339 | 0.6339 | 0.8477→0.9237 | 0.005 | **21.64** |

## Findings

**1. The objection is right: complete modules are lossless, slices are not.**
`replica_E4_k1_full` reproduces the dense model at k=1 — ppl 15.49 against the
dense 15.48, residual 0.0038 (the bf16 floor; exactly 0 at full width with one
expert, and 1e-6 in fp32). No slicing configuration can do this unless k = E
(`tile_E4_k4_100pct`, which is just the dense layer with extra bookkeeping). So
"experts must be complete modules" is confirmed, and the reason partition-based
upcycling degrades is structural, not a training failure.

**2. Width, not tiling, is what the deficit tracks.** At equal active width the
best-single-expert residual of a tile and a replica agree to ~0.02:

| active width | best tile | best replica |
|---|---:|---:|
| 12.5% | 0.7104 | 0.7307 |
| 25% | 0.6339 | 0.6530 |
| 50% | 0.4975 | 0.5134 |

Reading 25% of a layer costs you ~0.64 of that layer's output whether the 25% is
contiguous (a tile) or repeated (a replica). The oracle comparison at 50%
(`tile_E4_k2_50pct` 0.4820 vs `replica_E4_k1_50pct` 0.5134) even favours tiling,
which is expected: a tile can choose the *best* of E blocks, a replica has only
one block and the router has nothing to choose — visible in the table as
`best1 == oracle == router` for every replica row.

**3. The huge ppl gap between tiles and replicas at equal width is the router,
not the architecture.** At 25% active: `tile_E4_k1_25pct` ppl 45.63 versus
`replica_E4_k1_25pct` 24.93, with nearly identical single-expert residuals
(0.6339 vs 0.6530). The difference is that an *untrained* tile router picks a
random block (`router` 0.8878 against `best1` 0.6339), while a replica's "choice"
is vacuous. Training the tile router recovers most of it (45.63 → 23.19). Any
comparison of these two constructions that reports only end-to-end ppl is
measuring router initialisation.

**4. Residual against the dense function is the wrong objective once anything is
trained.** The router's residual *rises* in every trained row, whichever way ppl
moves: `trained_tile_E4_k1_25pct` 0.8852 → 0.9303 with ppl improving 45.63 →
23.19, and `trained_replica_E4_k1_full` 0.0044 → 0.5491 with ppl *worsening*
15.49 → 19.11. A MoE that is being
finetuned on a corpus is no longer trying to reproduce the dense function, and
drift away from it is what training *does*. This is the metric E14/E15 used as
the decisive gate, and it cannot arbitrate a trained model.

**5. Training a *lossless* configuration can only make it worse, and 80 steps
shows exactly that.** `replica_E4_k1_full` starts at dense quality by
construction. After 80 steps: 19.11 (joint) / 19.14 (domain-specialised) /
22.78 (with a shared expert). Specialisation did not recover the loss — it
matched joint training to two decimal places. At this budget the honest statement
is: *the untrained exact replica is the best sparse configuration measured*, and
everything that trains moves away from it.

**6. Giving each module its own domain does help — exactly where a division of
labour exists.** `btx_tile_k1_25pct` (stage 2 on four domains, then stage 3)
reaches ppl **21.64** at 1.00x storage and 25% active width, against **23.19**
for the identical architecture trained on mixed data for the same 80 stage-3
steps — 1.55 ppl better. At full width the effect vanishes (19.14 vs 19.11),
which is coherent: a full-width replica starts as an exact copy of the dense
function, so there is nothing for specialisation to add, while a 25%-wide tile
has to *earn* the missing 75% and pre-specialised experts give the router
something to route between.

**7. The routed experts do not collapse.** Post-training usage on the validation
batch is spread — layer 0 `[0.26, 0.25, 0.22, 0.27]`, layer 3
`[0.20, 0.18, 0.39, 0.23]`, layer 23 `[0.28, 0.17, 0.07, 0.48]` — so the
`eff_experts = 33.2/64` failure of E15 does not reproduce here. That is a
consequence of construction, not of tuning: E15's collapse came from
always-active neurons scattered across disjoint slices, which is precisely what
the tile-with-shared-expert and replica constructions avoid.

**8. The wall is unchanged, and it is made of storage.** The lossless case costs
**4.00x** the dense FFN. Nothing here makes a sparse model smaller than the dense
model it came from; it makes the *per-token read* smaller. On an 8 GB machine
that is the constraint that binds, which is why the residency table and the
quantisation ladder remain the route to a 27B on this laptop.

## What would make it better (next experiments, in order)

1. **A shared expert holding the *dense* FFN, with routed specialists added on
   top, and the router output zero-initialised.** This is the one configuration
   that is *guaranteed* to start at dense quality and can only improve: at
   initialisation the routed contribution is zero, so the layer is exactly dense,
   and specialists add capacity without ever being able to subtract it. The
   `_shared` run here did not test this (it gave the shared expert a *full* extra
   FFN, 2x width, and took the result to 22.78) — the correct version is one
   dense-width shared expert plus narrow routed experts with a zeroed router.
   This is the concrete way to "keep the 9B active slice at full quality".
2. **Train long enough to beat dense, or do not train.** Every trained
   configuration here is worse than the untrained exact replica. 80 steps on
   662k tokens is 3-4 orders of magnitude below what BTX used. Either scale the
   corpus by ~1000x, or restrict training to the router.
3. **Judge trained MoE by task metrics, not by dense residual.** Finding 4 is the
   methodological lesson; the gate set in `experiments/eval_gate.py` exists and
   should be the arbiter.
4. **Re-test `combine="softmax"` against `sum` with the router actually in the
   output path.** From the language-model objective alone `sum` gives the router
   **no gradient at all** (`grad is None`); with the balancing loss included it
   reaches ~1e-2 against ~2.6e1 for `softmax` at E=8, k=2 — ~1.6e3 less, i.e. ~3
   orders, not the 5 an earlier probe recorded. At `k=1` the two modes coincide
   exactly. Pinned by
   `tests/test_branch.py::test_softmax_carries_far_more_router_gradient_than_sum`;
   the tile-router recovery in finding 3 was obtained
   with `softmax` and would not have been possible with `sum`.
5. **Only then re-run at 27B.** Nothing in this experiment changes the arithmetic
   in `docs/upcycling_design.md`: at 9B active of 27B total the selectable part
   is 2.9% of the FFN, so the wall is reached before the architecture is.

## Limitations

* One model (Qwen3.5-0.8B), 4 of 24 layers upcycled, one seed. Whole-model ppl
  moves only through those four layers, so ppl differences are *understated*
  relative to a full conversion.
* 80 cooperation steps and 40 expert steps is a pipeline check, not a training
  budget; findings 5 and 6 are statements about *this* budget.
* The stage-2 domains are 800k characters each with 2-block held-out sets, so
  `domain_ppl` is noisy and is reported as directional only.
* bf16 leaves a ~4e-3 residual floor. Identities that should be exact are
  verified in fp32 by `experiments/branch_poc.py --self-test` (18 checks) and
  `tests/test_branch.py` (38 tests).
* `stored_width_fraction` counts FFN width only. It excludes the router and does
  not include the embedding/attention backbone, which every token reads
  regardless of routing — see `docs/feasibility_report.md` §3 for why that
  backbone is what makes "9B active of 27B" unreachable by routing alone.

Raw evidence: `results/branch_poc_08b.json`, `results/branch_poc_08b_table.txt`.
