# Running the upcycling PoC on Kaggle

`upcycle_poc.ipynb` is the decisive experiment for the MoE route: **at the ratio a
27B conversion needs (~1.6% active), does *training* beat the frozen ceiling?**

The local pilot measured that ceiling with no weight updates. For `plan=random`,
k=1, on layers 0, 3, 8 and 23 it is:

| E | active | frozen oracle residual |
|---|---|---|
| 8 | 12.5% | 0.8624 |
| 32 | 3.1% | 0.9188 |
| 64 | 1.6% | 0.9277 |

Locally that could only be tested at E=8 on 4 layers, because cost scales with
(upcycled layers x experts): E=64 over all 24 layers is ~48x the 0.6 s/step the
4-layer runs cost, which is the difference between minutes and a lost evening.
Kaggle gives a CUDA device and a 12 h session.

**This has been run.** Kernel `trowel3444/moeme-upcycling-poc`, one 3.25 h session,
24 layers at E=8/32/64 plus a seed-1 replicate. The answer is below.

## Running it

The repo is attached as a Kaggle **Dataset** and the notebook copies it into
`/kaggle/working/moeme`.

```bash
# 1. package the repo (kaggle datasets create SKIPS directories by default, so the
#    tarball is what makes the contents deterministic)
tar -czf moeme-repo.tar.gz --exclude='./.git' --exclude='./.venv' \
    --exclude='./.cache' --exclude='__pycache__' -C . .
# ...upload it as a Dataset version, then:

# 2. push and run
kaggle kernels push -p kaggle/
kaggle kernels status trowel3444/moeme-upcycling-poc
kaggle kernels logs   trowel3444/moeme-upcycling-poc        # or view on the web

# 3. collect
kaggle kernels output trowel3444/moeme-upcycling-poc -p out/
```

`kernel-metadata.json` in this directory carries the settings that matter:
`enable_gpu: true`, `enable_internet: true` (needed to `pip install`, to download
the model, and to stream the corpus), and `dataset_sources` naming the repo
dataset. The title must slugify to the `id`, or the CLI warns.

## Result of the full run

`results/kaggle_upcycle_poc.json` and `results/kaggle_upcycle_poc_seed1.json`;
raw session log `kaggle/kaggle_run.log`. 24 layers upcycled, k=1,
`combine=sum`, `plan=random`, 300 steps, 265.8M trainable params, real
`fineweb-edu` (883,186 tokens; 1638 train / 86 val blocks of 512), bf16 autocast,
dense val ppl **19.562**.

| E | active | frozen pilot oracle | oracle trained | router trained | beats frozen | ppl untrained | ppl trained | s/step |
|---|---:|---:|---:|---:|:--:|---:|---:|---:|
| 8 | 12.5% | 0.8624 | **0.8560** | 0.8951 | yes, by 0.006 | 653,536 | 109.5 | 5.05 |
| 32 | 3.13% | 0.9188 | **0.9355** | 0.9653 | **no** | 1,464,554 | 159.3 | 8.12 |
| 64 | 1.56% | 0.9277 | **0.9549** | 0.9806 | **no** | 624,690 | 202.5 | 11.89 |

**Training loses at the ratio that matters, and loses more the sparser it gets.**
At E=8 the mild training win E14 found locally is reproduced (0.8560 vs 0.8624).
At E=32 and E=64 training does not merely fail to beat the frozen geometric
partition — it makes the reconstruction *worse* than leaving the split alone, and
the deficit grows with E (+0.017, then +0.027). The learned router is worse than
the oracle at every ratio and far worse than the frozen split. A second seed at
E=64 reproduces the inversion to three decimals (oracle 0.9551 vs 0.9549, router
0.9802 vs 0.9806), so it is not seed noise.

The mechanism: `vs_original_dense` — the residual against the dense FFN the
experts were partitioned from — **rises** under training at every E (0.9324 →
0.9480, 0.9839 → 0.9902, 0.9919 → 0.9974) while perplexity drops three orders of
magnitude. Training re-purposes experts away from the split they were initialised
to be; that is what buys the perplexity and what costs the oracle ceiling.

End to end it is still far off: 109.5 ppl at E=8 is 5.6x dense, and 202.5 at E=64
(the target ratio) is 10.4x dense. `docs/upcycling_design.md` §9 said to abandon
the retrofit if trained `oracle_topk` did not fall below the frozen oracle at the
same (E, k). At E=32 and E=64 it does not. **The route is dead on its own gate.**

### Cost, measured rather than estimated

11.89 s/step at E=64, 8.12 at E=32, 5.05 at E=8 — the MoE forward loops over all
E experts, so cost tracks layers x E. Whole session 11,695 s (3.25 h) against the
12 h cap, including the self-test and the seed-1 replicate; the four training runs
themselves are 11,333 s. `--max-seconds 5400` per config never fired
(`stopped_early: false` everywhere).

## What an 8-step probe established

The cells were first run as a short probe (`--steps 8`) before committing the
session, and it was worth the 8 minutes — it failed on an artifact bug the first
time. Worth repeating whenever the environment changes:

- **The install works**: `transformers==5.6.2`, real `fineweb-edu` streaming
  (4,004,296 chars → 883,186 tokens), cuda with **bf16** autocast, and
  `dense val ppl 19.6`, i.e. sane for this model on natural text.
- **`causal-conv1d` does not build** on the free image (no matching nvcc/CUDA);
  `flash-linear-attention` does install. With only one of the two present the
  fused Gated DeltaNet path is still unavailable, so the model runs its
  **pure-torch fallback**. That is correct -- verified in this repo by making both
  packages invisible to `find_spec` and running a real forward pass -- and it is
  **11.9 s per training step** at E=64 across 24 layers at seq_len 512 with
  grad_accum 4, after a one-off ~206 s on the first step. Expect the build to fail
  and do not spend the session chasing it.
- **The plumbing is right**: the untrained oracle reproduces the pilot's frozen
  number on the *same four layers* to within 0.0015 (E=64 0.9292 vs 0.9277, E=32
  0.9194 vs 0.9188, E=8 0.8583 vs 0.8624) -- two independent pipelines on
  different devices agreeing. The full run's dense-limit check (`top_k = E`)
  landed at 19.573/19.592/19.587 against a dense 19.562, so the upcycle is
  exactly dense-equivalent on the real model on real text too.
- **Load balancing at E=64 is poor and the default aux loss does not fix it**:
  `eff_experts` ran **33.2 of 64** with `load_max_over_mean` up to **9.25**. A
  probe that stops after 8 steps sees a much lower figure (21.6), so measure this
  only after a real run.

## Constraints the driver is built around

| Constraint | How it is handled |
|---|---|
| unknown device speed | `--max-seconds` caps each configuration's training loop and the report is still written; `stopped_early` and `steps_completed` record any truncation |
| a configuration that raises | recorded as an `error` entry in the report, remaining configurations still run, process exits non-zero |
| ~12 h session cap | configurations run in the order given, and the JSON is rewritten after every one |
| ~20 GB output cap | `--save-every 0` in the run cells: a checkpoint at E=64 over 24 layers is ~1.6 GB, and several of them would consume the cap. Re-enable only if `--resume` is needed |
| T4 bf16 is emulated, not tensor-core | measured `bf16 True` on this image and the run used it; `pick_amp_dtype` falls back to fp16 + `GradScaler` if a device reports no bf16 |
| 2x T4 has no NVLink | the PoC is a single-GPU job; multi-GPU buys nothing at this scale |
| idle timeout | the training loop prints a loss and elapsed-time line at least every 25 steps |
| a silent corpus fallback | the corpus source is recorded in the report, and a built-in-corpus run prints an explicit warning, because perplexities from a repeated 1 kB text look finite but mean nothing |

> Kaggle free-tier quotas (hours per week, session length) are not verified here.

## What would make a good next session

Not more of this grid. E15 answered the question it was built to answer, and the
remaining MoE cells are refinements of a 0.027 deficit:

- **The resident-quantisation check on the real 27B** (2/3/4 bpw across 2x T4),
  which validates the free ~8x path end to end rather than predicting it. This is
  the highest-value use of a Kaggle session now — `docs/decision_memo.md` Path A.
- **If the MoE route is revisited, sweep the aux loss first.** 33.2 of 64
  effective experts and `load_max_over_mean` 9.25 at E=64 means the default
  `--aux-loss-coef 1e-2`, carried over from the E=8 runs, is leaving routing on
  the table. Cheap to test (E=8 at 5.05 s/step), but temper expectations: the
  oracle route, which needs no router at all, also loses to the frozen split.
- **A `combine="softmax"` run** to test the partial-sum hypothesis directly. If
  the deficit is output-scale rather than routing, softmax gating should shrink
  it — at the cost of the exact dense limit, which `tests/test_upcycle.py` pins
  deliberately.
