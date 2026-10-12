# Does the fast MoE actually answer as well as the slow dense 27B?

**Short answer: on this task set, at any budget either model can finish, the two
are indistinguishable — and the MoE gets there in ~1/86th of the wall time. The
apparent gap at a small token cap was an artifact of the cap: it cut the dense
model off mid-solution and the harness then recorded the last number it saw as
its answer.** That defect is now fixed and reported, and the reasoning tasks were
re-run at a budget the dense model can actually finish.

**And at equal quantisation the architecture is not the compromise.** With both
models at Q4_K_M (§4), the sparse 22 GB MoE and the dense 17 GB model tie on
reasoning (4/4 each) and differ by one brittle task elsewhere (12/15 vs 13/15) —
while the sparse model decodes **~19× faster**. Sparsity costs nothing measurable
here; what it does not do is make a model fit, which is why the Q2_K MoE remains
the one to run on this machine.

**And surplus's placement is worth having — but not through surplus's own bench
or shim on this machine.** The plan its solver picks (`-ngl 42 -ncmoe 24`) is the
best placement tested, and driving the same 19 tasks through llama.cpp at that
placement reproduced the ollama run's score exactly (17/19) with ×1.75 the wall
speed. Its `bench` crashes, its `shim` returns empty responses, its auto-plan in
the shim is ~15× slower than its own best plan, and its default 4 GiB RAM cage is
smaller than this model needs. Details in §5.

This document follows the repo's rule: every number re-prints from `results/*.json`
via `tools/summarise_results.py`, using the task set and harness already in the
repo (`experiments/eval_gate.py`), not a new one invented for this question.

## 1. Method, and the changes that made it possible

The gate is the repo's existing task set: 30 tasks, scored mechanically, not by a
judge model.

| category | n | how it is checked |
|---|---:|---|
| coding | 10 | the generated function is **executed** against fixed cases in a subprocess |
| reasoning | 8 | the final number is extracted and compared (`ANSWER: <value>`) |
| format | 6 | output must parse as JSON and match a required schema |
| instruction | 6 | exact-match constraints (counts, casing, exact tokens, forbidden letter) |

Both models run identical prompts at `temperature 0`, a fixed seed, and the same
`num_predict`. Five changes were needed, and all are additive:

- `--think off` — suppresses the separate reasoning field for thinking models.
  Without it a qwen3.x model spends an unbounded number of tokens thinking, and
  the dense 27B cannot be evaluated in a sitting.
- `--tasks a,b,c` — run a fixed id subset, so a model decoding at 0.34 tok/s is
  compared with one at 47 tok/s on the *same* prompts.
- `--resume` plus a checkpoint after **every** task (the CPU-only control run
  documented in `docs/expert_offloading.md` was OOM-killed; losing hours of a
  slow run to one dropped connection is not acceptable, and `--out` is always a
  valid partial report).
- `--backend llama` / `--chat-template-kwargs` — run the identical task set
  through llama.cpp instead of the ollama tag, which is what turns "is the other
  engine better" into a measurement (§5).
- `--rescore` — replay corrected task definitions over *stored* responses, so a
  multi-hour run made under a wrong expectation is re-scored, not re-run.

## 2. The constraint that sets the experiment size

Measured decode on this machine, same runtime, same weights:

| model | file | decode | a 192-token answer costs |
|---|---:|---:|---:|
| `ornith-1.5-35b-a3b:q2_k` (35.5B MoE, Q2_K) | 13.0 GB | **36.7 tok/s** | 5 s |
| `qwen3.8:latest` (27.8B dense, Q4_K_M) | 17.0 GB | **0.34–0.39 tok/s** | **8–9 min** |

That ~100× gap decides what is measurable. The first attempt ran all 30 tasks on
both models. The MoE finished all 30 in **115 s**; the dense model finished 2 in
~25 minutes, projecting to ~6 hours. That run was stopped and its two recorded
tasks are kept as evidence in
`results/eval_qwen3.8-latest_full30_abandoned.json`.

So the comparison uses a **19-task subset** (3 coding, 4 reasoning, 6 format,
6 instruction), run identically on both models.

## 3. Results

### 3.1 MoE, full 30-task set (`num_predict=384`)

| model | overall | coding | format | instruction | reasoning |
|---|---:|---:|---:|---:|---:|
| `ornith-1.5-35b-a3b:q2_k` | **26/30 = 86.7%** | 10/10 | 6/6 | 4/6 | 6/8 |

### 3.2 Both models, 19 tasks, `num_predict=192`

| model | overall | coding | format | instruction | reasoning | capped rows | excluding capped | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoE `ornith-1.5-35b-a3b:q2_k` | **15/19 = 78.9%** | 3/3 | 6/6 | 4/6 | 2/4 | 2 | 15/17 = 88.2% | 55 s |
| dense `qwen3.8:latest` | 13/19 = 68.4% | 3/3 | 6/6 | 4/6 | 0/4 | **4** | 13/15 = 86.7% | 3958 s |

Two things make the headline numbers misleading on their own:

- **Every one of the dense model's four reasoning failures was a truncation.**
  `done_reason` was `length` on all four: it was cut off mid-solution while
  reasoning correctly. Its `r_lily` response ends "…Therefore", on its way to 47;
  the scorer then extracted a number from the working and recorded a wrong
  answer. The MoE, which writes more compactly, capped only twice.
- **The reason the extraction misfires is worth separating.** A capped run that
  never reaches its answer marker has no answer to score, so it stays a failure;
  rows now carry `truncated` and both scores are printed. That is what §3.3 fixes
  by re-running rather than re-framing.

So the only defensible reading of §3.2 is: on everything the cap did not cut,
**the two models are statistically indistinguishable** (MoE 15/17, dense 13/15,
same two instruction failures), and the reasoning column is unmeasured for one
of them.

### 3.3 The reasoning column, re-run at a budget both can finish (`num_predict=768`)

The four reasoning tasks only, identical prompts, identical cap:

| model | reasoning @768 | wall | decode (median) |
|---|---:|---:|---:|
| MoE `ornith-1.5-35b-a3b:q2_k` | **4/4** | 41 s | 43.4 tok/s |
| dense `qwen3.8:latest` | **4/4** | 3517 s | 0.339 tok/s |

Both models answered `0.05`, `3`, `47`, `42` — i.e. **the 192-cap figures were
entirely a harness budget**: the dense model's 0/4 and the MoE's 2/4 both came
from rows the cap cut off mid-working. At a budget both models can finish, there
is no quality gap to find on these tasks: 86× the wall time, same answers.

The MoE on the full 19 tasks at the same 768 cap scores **17/19 = 89.5%**
(coding 3/3, format 6/6, instruction 4/6, reasoning 4/4), the only failures being
two genuine instruction cases (`i_five_words` returned 6 words, `i_no_letter_e`
used a word containing "e"). Those two fail for the dense model at every cap too.

### 3.4 Two defects in the repo's own task set, and one in its scorer

Running the set is what exposed these; all three are fixed and pinned by tests.

1. **`r_bat_ball` expected 0.10.** The ball is `(1.10 - 1.00) / 2 = 0.05`, so the
   task marked the *correct* answer wrong. Both models answered 0.05. This is
   what made the MoE's `r_bat_ball` row a FAIL in the first run.
2. **`r_modular` expected 1.** `7**20 % 13 = 3`; the old expectation marked the
   *incorrect* answer (1) as a pass. The two defects mis-scored in opposite
   directions.
3. **Truncation was only labelled for *empty* answers.** A capped model that
   narrates before answering scored a bare "got 3.0, want 42", which reads as a
   wrong answer. This is the defect that mis-scored 3 of the dense model's 4
   reasoning rows in §3.2.

`test_numeric_expectations_are_arithmetically_correct` now re-derives every
numeric expectation with a closed form instead of trusting the literal, and
`test_every_coding_case_is_satisfiable_by_a_correct_solution` runs a reference
solution against every coding case — a wrong expected value fails a correct model,
so it must fail the suite first. `--rescore` applied the corrected definitions to
the already-completed runs (`r_bat_ball` 14/19 → 15/19 for the MoE) instead of
re-running them.

## 4. Equal quantisation: is the *architecture* a quality compromise? — no

§3 compared two artifacts a user would actually choose between, but they differ in
two ways at once:

| artifact | architecture | quantisation |
|---|---|---|
| `ornith-1.5-35b-a3b:q2_k` | MoE, 256 experts, top-8 | Q2_K (2.8 bpw) |
| `qwen3.8:latest` | dense | Q4_K_M (4.9 bpw) |

So a gap would be architecture **plus** ~1.7× in bits per weight. Holding the
quantisation fixed removes that, and the partner model was already on this
machine: **`ornith-1.5:35b` is the same `qwen35moe` architecture at Q4_K_M**
(22.0 GB), so against the dense 27B at Q4_K_M (17.0 GB) it isolates architecture
alone. Same runtime, same engine, same 19 tasks, `--think off`.

| run (all Q4_K_M, ~4.9 bpw) | coding | format | instruction | reasoning | overall | decode |
|---|---:|---:|---:|---:|---:|---:|
| **MoE** `ornith-1.5:35b` (22 GB) | 3/3 | 5/6 | 4/6 | **4/4** | 16/19 = 84.2% | **6.14 tok/s** |
| **dense** `qwen3.8:latest` (17 GB) | 3/3 | 6/6 | 4/6 | **4/4** | 13/15 non-reasoning + 4/4 reasoning | 0.32 tok/s |

The dense row is assembled from its two measured configurations: its 15
non-reasoning rows at a 192 cap (all completed — the cap did not bind, so they are
comparable to the MoE's) and its 4 reasoning rows at 768 (0/4 at 192 was pure
truncation, §3.3).

**Verdict: sparsity costs nothing measurable here and buys ~19×.** Reasoning ties
at 4/4, and the 15 non-reasoning tasks differ by **one task** (12/15 vs 13/15).
That one task is not a signal: across all three measured models the failures move
around inside the same small set of brittle exact-match constraints —
`i_no_letter_e` fails everywhere, and `i_five_words` / `i_lowercase_start` /
`f_json_array` each fail for one model and pass for another. At six tasks per
category and one run each, ±1 task is noise, not capability.

Two consequences worth separating:

- **The architecture question is answered**: a pretrained sparse model is not the
  compromise option against a dense one at the same bits. Its 4/4 on reasoning is
  the same 4/4, not a deficit.
- **The practical recommendation does not change, and the reason is memory, not
  quality.** At equal bits the 22 GB MoE is 7× *slower* than the 13 GB MoE
  (6.14 vs 43.1 tok/s) because it does not fit the fast tiers either — it just
  reads far fewer bytes per token than the dense model does. On this machine the
  13 GB Q2_K MoE is still the one to run: it matches the dense 27B at 4.9 bpw on
  this task set while fitting, which is why it is 43 tok/s and not 6.

The residual caveat that this section does **not** remove is Q2_K itself: on work
outside this 19-task set, 2.8 bpw is the risk, and the equal-quant run is evidence
that if you have the RAM for Q4_K_M you should use it. The next subsection measures
how big that risk actually is, because a pass/fail gate cannot.

### 4.1 Does a smaller quantisation make the model dumber? — the gate says no, perplexity says yes

"I prefer Q4" is not the question §4 answers. §4 shows the **architecture** is not
the compromise; it says nothing about whether the *quantisation* is. Those are two
different knobs — the equal-quant run holds bits fixed and moves the architecture,
and the comparison a user actually faces ("13 GB or 22 GB of the same thing?") holds
the architecture fixed and moves the bits.

The gate's answer to that second question is a tie, and the tie is genuine but
uninformative. Scoring the same `qwen35moe` 35B at Q2_K and at Q4_K_M on the
identical 19 tasks at an identical 768 cap, same runtime, `--think off`, with
**nothing capped on either side**:

| 35B MoE, same 19 tasks, 768 cap | **Q2_K** (13 GB) | Q4_K_M (22 GB) |
|---|---:|---:|
| overall | **17/19** = 89.5% | 16/19 = 84.2% |
| coding / format / instruction / reasoning | 3/3, 6/6, 4/6, 4/4 | 3/3, 5/6, 4/6, 4/4 |
| wall, all 19 tasks | **62.7 s** | 463.5 s |
| decode (median over the 19) | 43.1 tok/s | 6.14 tok/s |

Both runs are ollama with its own offload-what-fits policy, so this compares the two
configurations a user actually gets on an 8 GB card — 13 GB partly offloaded against
22 GB mostly host-side — rather than two placements of one artifact. That is the
comparison that matters here, and it is also why the wall-time gap is a *residency*
factor as much as a bits-per-weight one.

Q4_K_M scores one task *worse*, which is inside the ±1-task noise band §4 already
established. But these two runs are not the same model giving the same answers.
Comparing them token by token, **13 of 19 responses are not the same text at all** —
only 6 are byte-identical, the median divergence is at the *first* character, and 9
of the 13 part company within 5 characters. Three tasks flip verdict, two of them in
Q2_K's favour. Q2_K does not reproduce Q4_K_M and then get graded on it; it produces
different text that happens to pass the same checks. At six tasks per category, a
gate made of exact-match assertions on easy prompts is a statement about the
assertions.

**Perplexity is the instrument that can see the difference.**
`experiments/quant_perplexity.py` builds one fixed corpus from this repo's own prose
and code, records its sha256, runs `llama-perplexity` once per quantisation with
every other flag held fixed, and stores each run's final estimate, its cumulative
series, the window it actually scored, its command line, its wall time and its guard
verdict beside it. On a matched pair — same model, same tokens, `-c 512 --chunks 8`:

| matched pair, identical token stream | PPL | vs Q4_K_M |
|---|---:|---:|
| `ornith-1.5:9b` Q4_K_M | 7.0029 ± 0.409 | — |
| `ornith-1.5:9b` Q2_K | 7.7128 ± 0.468 | **+0.71 PPL (+10.1%)** |

**Provenance, stated because it does not help the argument to hide it.** That pair was
measured on an identical token stream, but its logs lived in a temp directory that was
cleaned between sessions, so the delta is quoted from the run record and is *not*
backed by a committed `results/` artifact the way every other number here is. What is
committed is the 35B's own run — `results/quant_perplexity_35b.json`, Q2_K at
**12.7113 ± 0.809** — and the 9B pair is reproducible with `make quant-ppl` pointed at
the two 9B GGUFs in `~/models`.

Read the 10% with its error bar: 0.71 is roughly **1.1σ** against the two printed
standard errors, and a paired difference over identical tokens is tighter than
treating the runs as independent — but at this window that makes the cost
*plausible and one-directional*, not decisive. The honest answer to "does Q2 make the
model stupid?" is therefore **no — and not free either**: it is measurably worse in
next-token log-likelihood, by a margin in the neighbourhood of 10% whose error bar is
not yet tight, and that cost is invisible to the 19-task gate above. More tokens is
what settles it; this repo does not yet have them.

**What the window covers, because "8 chunks" reads as "the whole corpus".**
llama-perplexity scores `chunks × n_ctx` tokens and prints that number itself. A run
here scores **4,096 tokens** against a corpus of roughly 100,000 (at a rough 4
bytes/token), so the measurement is the *first* 4,096 tokens of the repo's text — the
README and the first documents in path order — and not all of it. That is not
pedantry, and the way it surfaced was not flattering: the corpus grew 389,996 →
397,855 bytes mid-session and the estimate came back byte-identical, because every
added byte landed past the end of the window. The probe now records the scored window
per run, warns when it is smaller than the corpus, and refuses to present runs with
different windows as a matched pair. The corpus is also repo-local, so it moves as the
repo does — which is why every report carries its **sha256**, byte count and file
list, and why a PPL pair without the corpus identity is not a measurement.

**The 35B MoE's own matched pair cannot be measured on this host — and the guard is
what kept the attempt from becoming an OOM kill.** The 21 GB Q4_K_M artifact cannot be
scored here at any setting tried. With every expert on the CPU (`-cmoe`) it was
OOM-killed *during load* (RSS 11.32 GiB as MemAvailable fell 11.45 → 0.76 GiB). With a
partial expert offload (`-ngl 12`) the batched pass over an 8-of-256 MoE still drove
MemAvailable to **0.14 GiB**, and on a repeat **0.11 GiB**, within a minute — at which
point `tools/mem_watchdog.py` terminated the run *before* the kernel picked its own
victim (`logs/ppl_q4_35b.watch.log`). The Q2_K 35B scores fine the same way — the
committed run took **173 s with a 9.47 GiB RAM floor** (a first run of the same
command: 185 s, 8.27 GiB). The report keeps the failed run with **no estimate and an
explicit exit code**, so it can never be misread as a score, and the summary says so
in as many words. The 9B pair is the same probe, same corpus builder, same flags — it
answers the quantisation question on a pair this machine can actually hold.

This is the second time the guard has been the difference between a measurement and
an incident: on the dense 27B it never fired, and here it fired twice in one afternoon.

**The practical reading, which is what changed.**

- **On this machine the 13 GB Q2_K MoE is still the one to run — now for two
  independent reasons.** It fits *and* the 22 GB Q4_K_M costs **7.4× the wall time**
  (463.5 s vs 62.7 s for the same 19 tasks) without scoring better. The perplexity
  gap points the same way — Q4 is the better model — and it does not come close to
  being worth 7× here.
- **"I prefer Q4" is a defensible position, and this repo should stop implying
  otherwise.** Q4_K_M is the *closer* model; the perplexity pair is the measurement
  that says so. If you have the RAM, use it. The recommendation for Q2_K is a
  **memory** recommendation, not a claim that the two quants are equivalent.
- **When the gate says "no measurable difference", read it as "this gate cannot
  measure it".** The token-by-token comparison makes the point concrete: 13 of 19
  outputs differ while the score moves by one task in the *opposite* direction to the
  perplexity. Use the gate for capability and the probe for fidelity.

## 5. Can surplus do better?

Surplus (the local Rust engine in `~/Documents/surplus`) is three things: a
planner/solver that decides GGUF placement and drives llama.cpp, an
Ollama-compatible shim, and an experimental in-process native decoder. They get
different answers, and the honest summary is that **its placement is better and
its wrapper is worse — on this machine, right now.**

### 5.1 Its plan is measurably better than the runtime default

For `Ornith-1.5-35B-A3B-Q2_K.gguf` surplus's planner chose
`-ngl 42 --n-cpu-moe 24`, expert-granular rather than layer-granular, which is the
"hot modules in VRAM, cold ones in RAM" split the runtime does not choose by
itself. Measured with `llama-bench`, same weights, everything else fixed:

| placement | prompt tok/s | decode tok/s |
|---|---:|---:|
| `-ngl 42 -ncmoe 24` (surplus's pick) | **121.5** | **50.1** |
| `-ngl 99 -ncmoe 48` | 80.1 | 36.5 |
| `-ngl 99 -ncmoe 56` | 49.7 | 33.1 |
| ollama default (`num_gpu = -1`), 32-token **cold** baseline | 11.6 | 19.4 |

The prediction behind it is a ranking aid, not a forecast: 61.52 predicted
against 50.1 measured is **23% optimistic**, 57.17 (14% optimistic) at the
planner's default 2048 context.

**Read that last row with care — §5.2 shows it is a cold-start number, not what a
session gets.** `llama-bench` is a fresh process per placement, and ollama's
11.6/19.4 comes from a 32-token single-prompt benchmark. Against a warm session on
the same tasks, ollama's own rates are 83.4 tok/s prompt and 43.1 tok/s decode,
which puts this placement at ×1.21 and ×1.12 rather than ×10 and ×2.6. The
placement is still the better one; the multiples in this table are not the ones a
user experiences.

### 5.2 The same placement, scoring the same tasks — quality is unchanged, speed is real

`llama-bench` generates text but never checks it, so §5.1 says nothing about
quality. Re-running the identical 19 tasks through llama.cpp at
`-ngl 42 -ncmoe 24` (`make eval-llama`, f16 KV):

| run | overall | coding | format | instruction | reasoning | decode (median) | prompt (median) | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ollama tag, 768 | **17/19 = 89.5%** | 3/3 | 6/6 | 4/6 | 4/4 | 43.1 tok/s | 83.4 tok/s | 63 s |
| llama.cpp @ `-ngl 42 -ncmoe 24`, 768 | **17/19 = 89.5%** | 3/3 | 6/6 | 4/6 | 4/4 | **48.4 tok/s** | **101.2 tok/s** | **36 s** |

Identical score, identical failures, on the same weights. That is the expected
result, and it is worth stating plainly: **an engine or placement changes where
bytes come from, never which weights the model needs.** What it buys here is
speed: ×1.12 decode, ×1.21 prompt processing, and ×1.75 wall time for the same 19
tasks.

Peak VRAM at this placement was **6292 MiB** of 8188 (against 6654 MiB for the
ollama tag's own split), load 16.0 s, peak server RSS 8.79 GiB.

**This section corrects the draft it replaces.** Two earlier claims do not
survive a same-conditions measurement, and both were artefacts of the baseline
rather than of the placement:

- *"×2.59 ollama's decode"* compared 50.1 tok/s (a fresh `llama-bench` process)
  against ollama's 19.36 tok/s **32-token cold** figure. Warm, on the same tasks,
  ollama decodes at 43.1 tok/s and the gain is **×1.12**.
- *"×10.4 on prompt processing"* compared 121.5 tok/s against ollama's 11.6 cold
  figure. Warm, on the same tasks, ollama prompts at 83.4 tok/s and the gain is
  **×1.21**.

So the placement's real, reproducible advantage on this machine is ~10-20%, plus
a ×1.75 wall-time difference on this task set. That is worth adopting — and it is
an order of magnitude smaller than the cold-baseline comparison suggested. Any
"×N faster" claim in this repo should name the baseline it beat.

### 5.3 An engine comparison must match the template, not just the weights

Before that, the same llama.cpp run at the same placement scored **13/19**. The
cause was not the engine: `/completion` is a raw continuation and applies no chat
template, so the model wrote its `<think>` block into the answer and a
strict-instruction task received `\n\n<think>\nThe user wants me to reply with…`
instead of `CONFIRMED`. Instruction tasks went 4/6 → 1/6.

Ollama's `think: false` suppresses that; the equivalent here is
`chat_template_kwargs={"enable_thinking": false}` on `/v1/chat/completions`, which
is what the harness now sends. The uncorrected run is kept as evidence in
`results/eval_llama_ngl42_ncmoe24_CHAT_TEMPLATE_THINKING_ON.json` (its model
label says so, so no head-to-head table can quote it as engine quality). **That is a 21-point
score difference produced by a request setting rather than by the model** — the
same class of error as the truncation defect in §3.4, and the reason the harness
now reports which engine, template kwargs and cap produced a number. Finding it
took a direct probe (all four request variants on `/completion` still emitted
`<think>` on this build; the chat route with `enable_thinking: false` returned a
bare `yes`).

### 5.4 Its measurement path cannot confirm any of that

`surplus bench <model>` **crashes before it measures anything**:

```
surplus: [backend_error] llama-bench failed (signal: 11 (SIGSEGV) (core dumped))
```

Isolated by calling the binaries directly: the engine surplus pins for this model
(`llama.cpp-moe-cache`) segfaults whenever `-ncmoe` is passed to *llama-bench*
(exit 139 at `-ncmoe 24`, OOM-killed at `-ncmoe 40`), while the same placement on
the upstream build runs to completion and produces the 50.1 tok/s row. Two more
findings from the same probe:

- `--load-mode` is not a parameter of the pinned fork's `llama-bench`, so
  surplus's `engine::supports()` probe is the only reason that flag is absent from
  the failing command line.
- The fork's *server* does accept `-ncmoe` (it is `llama-bench` in that fork that
  crashes), so the pin is not fatal for serving — only for tuning.

This is the actionable result for surplus: `surplus optimize` cannot tune MoE
placements while `bench` dies on the engine it selects, so its own caveat that
"predicted rank is not a measured speed guarantee" currently has no way to be
discharged for MoE. The plan was confirmed by running `llama-bench` upstream by
hand instead.

### 5.5 Its shim serves the model but returns empty answers

`surplus shim` is the one interface that could be compared without translation,
because it speaks Ollama's own `/api/*` routes on port 11435. Started against this
artifact (pinned to the upstream checkout, which is required — see below):

```
surplus: memory cage 4G — the server cannot exceed it; the desktop stays safe
surplus: backend: …/llama.cpp-upstream/build/bin/llama-server … -ngl 42 -c 8192
         --n-cpu-moe 36 --cache-type-k q8_0 --cache-type-v q8_0 …
```

Three separate problems, each measured:

1. **Every response is empty.** 13 tasks ran; all 13 came back with
   `done_reason: stop` and `response: ""`. A direct probe of `/api/generate`
   confirms it is not the harness: HTTP 200, `eval_count: 32`, `response: ""`.
   The server had generated 227 tokens for one task. A client that trusts
   Ollama's API gets nothing back.
2. **The plan it picks unprompted is ~15× slower than its own best plan.** The
   shim chose `-n-cpu-moe 36` (from the stored per-model config, at ctx 8192) and
   decoded at **3.41 tok/s**, against 51 tok/s for the `-ncmoe 24` plan its solver
   ranks first and 46.6 tok/s for ollama. Pushing more expert layers to the CPU
   than the machine needs is the difference between usable and not.
3. **The default RAM cage is far below what this model needs.** The cage budget is
   computed from *active* parameters (~3B × 0.37 B/param ≈ 1.1 GiB) plus KV and
   slack → **4 GiB**. With `--n-cpu-moe 36`, the expert tensors of 36 layers are
   host-side; that is closer to 7 GB of touchable data, and the first shim run had
   its backend killed mid-run (HTTP 500 Bad Gateway) under the cage. Restricting
   the working set of a 13 GB MoE to 4 GiB is not a safety margin, it is a
   throughput cap and a failure mode.

One interface note worth recording because it fails silently: `--engine-dir` is a
**checkout root** (surplus resolves `<root>/build/bin/llama-server`). Passing the
`build` directory logs `pinned engine … has no llama-server — ignoring the pin` and
falls back to the fork, which is how the first attempt ended up on the crashing
engine without saying so in the result.

The empty-response run is quarantined in
`logs/surplus_shim_empty_responses.json`, outside `results/`, so no summary can
read it as a score.

### 5.6 The native decoder cannot run this model at all, by design

`surplus native` supports standard `qwen3moe` and **explicitly rejects**
`qwen35moe` — every model in this repo — naming the missing pieces (gated
delta-net linear attention, shared expert, sections RoPE). Its own
`docs/QWEN35_NATIVE.md` scopes that port as multi-day work and states the expected
outcome for a 17 GB model on the scalar CPU path: seconds per token. For the
standard-`qwen3moe` case it *can* run, `docs/NATIVE_ENGINE.md` records ~26 s vs
~4 s against llama.cpp CUDA. So the native path is a correctness and ownership
project, not a speed play, and it is not a candidate here.

### 5.7 What "better" can and cannot mean

An engine changes **where bytes come from and how fast they arrive**. It does not
change **which** weights the model needs, so it cannot change the quality of a
given checkpoint — and §5.2 now *measures* that rather than assuming it: two
different engines, two placements, same 17/19 and the same two failures.

Two caveats keep that from being a purity claim:

- Surplus's plan uses `q8_0` KV (its own README calls this "a precision tradeoff,
  not evidence of unchanged output quality"), so a surplus run is not
  output-identical to an f16-KV run. The measured cross-engine agreement in §5.2
  is at f16 KV, driven through llama.cpp directly.
- Different kernels also differ in floating-point summation order, so token-exact
  agreement across engines is not expected in general.

Practically: surplus's contribution to this question today is its **placement** —
adopt `-ngl 42 --n-cpu-moe 24` in llama.cpp terms. Its planner is worth using;
its bench cannot verify the plan, and its shim cannot yet be scored.

## 6. Limitations

- **One prompt set, 19 tasks, one run each.** The checks are mechanical, but 6
  instruction tasks cannot support a percentage with much confidence, and a single
  run of a 4-task reasoning category cannot separate a real gap from sampling
  noise. Read the categories, not the headline.
- **Thinking is disabled** (`--think off`) for both models, because the dense
  model's thinking budget makes it unmeasurable otherwise. This measures
  non-reasoning output quality, not the models' best achievable answers.
- **The dense 27B comparison is the expensive half of this document.** Its
  4-task run at 768 took **59 minutes**; the full 30-task version was abandoned
  after ~25 minutes for 2 tasks. Every dense number here is the largest one that
  fits a sitting, not the one that would be ideal.
- **Memory safety was sampled, not assumed.** Over 59 minutes of dense
  generation, RAM available never fell below **5.79 GiB** and peak VRAM was
  **6811 of 8188 MiB** (181 samples, `logs/mem_watch_dense768.log`). The 22 GB
  Q4_K_M MoE run — the case that does not fit — held a **5.32 GiB** RAM floor over
  92 samples with a **6934 MiB** VRAM peak, under an armed watchdog
  (`tools/mem_watchdog.py`, floor 2.0 GiB) that never had to fire; this host had
  **no free swap** throughout. The llama.cpp runs go through a preflight that
  refuses to start unless 6 GiB RAM and 6000 MiB VRAM are free — it refused
  correctly once, while the dense model held the card.
- **Every cell is one run.** The equal-quant comparison in §4 differs by a single
  task on the non-reasoning half, and the failures there move around inside the
  same handful of exact-match constraints between models — so treat a one-task gap
  in a six-task category as noise. Two runs per model would turn that from an
  inference into a measurement.
- **§4's two sides ran at different caps for the non-reasoning tasks** (768 for
  the MoE, 192 for the dense model). The cap did not bind for either — all 15 rows
  finished on their own in both runs — so the comparison holds, but it is not the
  same configuration string.
- **The Q4_K_M MoE does not fit either** (22 GB against 14.3 GB of fast tiers).
  Its 6.14 tok/s comes from reading 8 of 256 experts per token, not from
  residency, which is the clearest evidence in this repo that sparsity and
  residency are separate levers.
- **Q2_K at 2.8 bpw is aggressive.** This measures one checkpoint at one
  quantisation; it says nothing about `qwen35moe` at Q4_K_M or higher, which is
  the more relevant question for anyone with more memory.
- **The placement sweep is three points plus the runtime default**, measured with
  `llama-bench` (`-p 128 -n 64 -r 2`, threads fixed, `q8_0` KV) rather than
  through a chat runtime, so its prompt numbers are prompt-processing throughput,
  not time-to-first-token in an interactive session.
- **The engine comparison covers one model, one placement and one engine pair.**
  llama.cpp at `-ngl 42 -ncmoe 24` matched ollama on this set; that is not a claim
  about other placements, other models, or about any engine's output being
  token-identical.
- **surplus was not fixable from here.** Its bench, its shim and its cage are its
  own code; this document records what they do on this host, not how to repair
  them. The shim failure in particular (empty `response`, exit 200) is a defect on
  surplus's side to confirm with its maintainer.

Raw evidence: `results/eval_ornith-1.5-35b-a3b-q2_k.json`,
`results/eval_ornith-1.5-35b-a3b-q2_k_short19.json`,
`results/eval_ornith-1.5-35b-a3b-q2_k_short19_768.json`,
`results/eval_ornith-1.5-35b-a3b-q2_k_reason768.json`,
`results/eval_qwen3.8-latest_short19.json`,
`results/eval_qwen3.8-latest_reason768.json`,
`results/eval_qwen3.8-latest_full30_abandoned.json`,
`results/eval_llama_ngl42_ncmoe24.json`,
`results/eval_llama_ngl42_ncmoe24_CHAT_TEMPLATE_THINKING_ON.json`,
`results/eval_ornith-1.5-35b_short19_768.json` (the equal-quantisation run),
`results/moe_placement.json`, `logs/mem_watch_dense768.log`,
`logs/mem_watch_q4_35b.log`, `logs/surplus_shim_empty_responses.json`, and the
`logs/surplus_*.log` files.
