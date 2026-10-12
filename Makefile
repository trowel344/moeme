# moeme -- edge MoE feasibility study. One-command entry points.
#
# Everything here is non-destructive: temp files are removed by the scripts,
# and the only network traffic is read-only Hugging Face range requests.
#
# NOTE: recipe lines must be indented with a TAB, not spaces.

PY          ?= .venv/bin/python
PY_SYS      ?= python3
MODEL       ?= qwen3:4b
NUM_PREDICT ?= 1536
UP_STEPS    ?= 400
# Extra flags for `make eval`, e.g. EVAL_EXTRA="--think off --tasks i_only_word"
EVAL_EXTRA  ?=

# Engine comparison (`make eval-llama`): the SAME task set through llama.cpp at
# an explicit expert placement, so an engine/placement change is measured for
# correctness and not only for speed. Needs a llama-server build and the GGUF.
LLAMA_SERVER ?= $(HOME)/llama.cpp-upstream/build/bin/llama-server
LLAMA_GGUF   ?= $(HOME)/models/Ornith-1.5-35B-A3B-Q2_K.gguf
LLAMA_PORT   ?= 8080
# surplus's planner pick for this artifact (see results/moe_placement.json)
LLAMA_PLACE  ?= -ngl 42 -ncmoe 24
LLAMA_OUT    ?= results/eval_llama_ngl42_ncmoe24.json

# surplus's own engine (see tools/eval_surplus_shim.py). The shim is
# Ollama-shaped on port 11435, so the same gate scores it unchanged. Surplus's
# pinned llama.cpp engine segfaults on the -ncmoe placement its planner picks
# for this artifact, so point ENGINE_DIR at an upstream checkout that does not.
SURPLUS      ?= surplus
SHIM_PORT    ?= 11435
# A checkout ROOT: surplus looks for <root>/build/bin/llama-server, and silently
# falls back to its own pin if the path does not resolve.
SURPLUS_ENGINE_DIR ?= $(HOME)/llama.cpp-upstream
# The same 19-task subset the ollama comparison used.
SHORT19      ?= c_sum_evens,c_reverse_words,c_gcd,r_bat_ball,r_work_rate,r_lily,r_sequence,f_json_three_actions,f_json_tool_call,f_json_no_prose,f_json_nested,f_json_array,f_json_nulls,i_exact_three_bullets,i_only_word,i_uppercase,i_five_words,i_lowercase_start,i_no_letter_e
LLAMA_EXTRA  ?= --tasks $(SHORT19) --num-predict 768

.PHONY: help venv test doctor bench-inference verify-sources \
        verify-real-ffn pilot upcycle upcycle-selftest branch branch-selftest \
        compare-upcycled-08b self-teach \
        eval eval-llama eval-surplus quant-ppl quant-ppl-selftest \
        recover-eval-window recover-eval-window-selftest report \
        residency clean clean-results all

help:
	@echo "moeme targets:"
	@echo "  make venv             create .venv (reuses system torch) and install pytest"
	@echo "  make test             run the offline unit test suite"
	@echo "  make doctor           one-command local hardware diagnostics (~90s)"
	@echo "  make bench-inference  benchmark quantised models on this machine"
	@echo "  make verify-sources   re-verify published model geometry from real artefacts"
	@echo "  make verify-real-ffn  verify FFN exactness on real 27B weights (535 MB download)"
	@echo "  make pilot            pilot sparsity study, real activations (needs GPU, 1.8 GB)"
	@echo "  make upcycle-selftest upcycling PoC pipeline check, offline, tiny model (~2 s)"
	@echo "  make upcycle          upcycling PoC on a real model (needs GPU; see kaggle/)"
	@echo "  make branch-selftest  branching-vs-slicing pipeline check, offline, CPU (~2 s)"
	@echo "  make branch           branching (complete modules) vs slicing (needs GPU)"
	@echo "  make compare-upcycled-08b  does the upcycled 0.8B still answer like the"
	@echo "                        dense one? perplexity + greedy text, same tokens"
	@echo "  make self-teach       teach the upcycled layers from the model's own dense"
	@echo "                        FFN, with no corpus (needs GPU, ~4 min)"
	@echo "  make quant-ppl        perplexity of one model at two quantisations, on"
	@echo "                        identical tokens (QUANT_PPL_MODELS overrides)"
	@echo "  make recover-eval-window  recover the --eval-blocks window a perplexity"
	@echo "                        report never recorded, by measuring the val loss"
	@echo "                        curve and inverting it (needs the model, ~3 min)"
	@echo "  make recover-eval-window-selftest  check that inversion offline (~1 s)"
	@echo "  make eval             evaluation gate set (MODEL=..., NUM_PREDICT=...)"
	@echo "                        slow models: EVAL_EXTRA=\"--think off --num-predict 384\""
	@echo "                        subsets:     EVAL_EXTRA=\"--tasks r_lily,f_json_nulls\""
	@echo "                        interrupt -> rerun after adding EVAL_EXTRA=\"--resume\""
	@echo "  make eval-llama       the same tasks through llama.cpp at an explicit"
	@echo "                        expert placement (LLAMA_PLACE, default -ngl 42"
	@echo "                        -ncmoe 24 = surplus's planner pick). Checks free"
	@echo "                        VRAM/RAM first; use LLAMA_EXTRA to change the tasks"
	@echo "  make eval-surplus     the same tasks through surplus's own shim"
	@echo "                        (SURPLUS_ENGINE_DIR overrides its pinned llama.cpp)"
	@echo "  make report           re-print the summary tables from results/*.json"
	@echo "  make all              venv + test + doctor + bench-inference"
	@echo "  make clean-results    delete results/ (regenerable)"
	@echo ""
	@echo "  NOTE: thinking models (qwen3 / qwen3.5 / qwen3.8) need a large NUM_PREDICT"
	@echo "  or they truncate mid-reasoning and return empty answers. eval_gate.py"
	@echo "  reports those as TRUNCATED rather than as capability failures."

venv:
	$(PY_SYS) -m venv --system-site-packages .venv
	.venv/bin/pip install -q pytest

test:
	$(PY) -m pytest tests -q

doctor:
	$(PY_SYS) tools/bench_hardware.py --out results/hardware.json

bench-inference:
	$(PY_SYS) tools/bench_inference.py \
		--spec qwen3:4b:-1:64 \
		--spec ornith-1.5:9b:-1:32 \
		--out results/inference_safe.json

verify-sources:
	$(PY_SYS) tools/verify_model_geometry.py Qwen/Qwen3.8-27B \
		--json results/qwen3.8-27b-geometry.json
	$(PY_SYS) tools/verify_model_geometry.py Qwen/Qwen3.8-Flash-Next \
		--json results/qwen3.8-flash-next-geometry.json

verify-real-ffn:
	$(PY_SYS) experiments/verify_real_ffn.py --out results/real_ffn.json

pilot:
	$(PY_SYS) experiments/pilot_sparsity.py --out results/pilot_sparsity.json
	$(PY_SYS) tools/summarise_results.py

upcycle-selftest:
	$(PY_SYS) experiments/upcycle_poc.py --self-test

upcycle:
	$(PY_SYS) experiments/upcycle_poc.py \
		--experts 8 32 --topks 1 2 4 --layers all \
		--steps $(UP_STEPS) --resume \
		--out results/upcycle_poc.json

branch-selftest:
	$(PY_SYS) experiments/branch_poc.py --self-test

branch:
	$(PY_SYS) experiments/branch_poc.py \
		--layers 0 3 8 23 --steps 80 --branch-steps 40 --btx \
		--resume --out results/branch_poc_08b.json

# The upcycling PoC leaves a perplexity ratio, which does not say whether the
# converted model still *answers* like the stock one. This loads the trained
# checkpoint over a fresh partition of the same four layers and compares both
# models on identical tokens: perplexity, and 48 greedy tokens on six fixed
# prompts. It also measures the dense-limit control on the fresh partition,
# where all-experts-active must reproduce dense perplexity -- that is the check
# that separates "the sparse model is different" from "the plumbing is broken".
compare-upcycled-08b:
	$(PY_SYS) experiments/compare_upcycled_08b.py \
		--out results/upcycled_08b_vs_dense.json

# The corpus-free version of the upcycling question: train the converted layers
# against *their own dense FFN* -- no labels, no pretraining corpus, just enough
# text to generate activations -- and measure whether that transfers to held-out
# activations or only memorises the calibration set. It also measures the
# least-squares optimal output scale, which decides whether the E/k rescaling
# the literature calls essential applies to a *partition* (measured: it does
# not, s* sits near 1 wherever k is, so ~96% of the deficit is directional).
self-teach:
	$(PY_SYS) experiments/self_teach_poc.py \
		--out results/self_teach_08b.json

eval:
	$(PY_SYS) experiments/eval_gate.py --model $(MODEL) \
		--num-predict $(NUM_PREDICT) $(EVAL_EXTRA) \
		--out results/eval_$(subst :,-,$(MODEL)).json

# How much a quantisation costs the model, on identical tokens. The gate is
# pass/fail and cannot see this; perplexity can. Locally this runs on the
# Ornith-1.5-35B-A3B pair, whose GGUFs are already in ~/models. Note that the
# Q4_K_M file needs a partial offload here: keeping every expert on the CPU size
# (-cmoe) makes a 21 GB model on a 15 GiB host get OOM-killed during load.
QUANT_PPL_MODELS ?= q2_k=$$HOME/models/Ornith-1.5-35B-A3B-Q2_K.gguf \
	q4_k_m=$$HOME/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf
quant-ppl:
	$(PY_SYS) experiments/quant_perplexity.py \
		$(foreach m,$(QUANT_PPL_MODELS),--model $(m)) \
		--out results/quant_perplexity_35b.json

quant-ppl-selftest:
	$(PY_SYS) experiments/quant_perplexity.py --self-test

# A perplexity report that does not record its eval window cannot be reproduced.
# It can still be *recovered*: ppl(k) is a running mean, so the reported number is a
# fingerprint of the k that produced it. E14's was recovered as 6 val blocks; add
# more --target values to identify other reports. Re-decide without a GPU by adding
# --curve-from results/eval_window_recovery.json.
recover-eval-window:
	$(PY_SYS) experiments/recover_eval_window.py \
		--target 21.7359 --target 21.7396 \
		--out results/eval_window_recovery.json

recover-eval-window-selftest:
	$(PY_SYS) experiments/recover_eval_window.py --self-test

eval-llama:
	$(PY_SYS) tools/eval_llama_engine.py \
		--server "$(LLAMA_SERVER)" --model "$(LLAMA_GGUF)" \
		--port $(LLAMA_PORT) --placement "$(LLAMA_PLACE)" $(LLAMA_EXTRA) \
		--out $(LLAMA_OUT)

eval-surplus:
	$(PY_SYS) tools/eval_surplus_shim.py \
		--shim $(SURPLUS) --model "$(LLAMA_GGUF)" --port $(SHIM_PORT) \
		--engine-dir "$(SURPLUS_ENGINE_DIR)" \
		--tasks $(SHORT19) --num-predict 768 \
		--out results/eval_surplus_shim_short19.json

report:
	$(PY_SYS) tools/summarise_results.py

residency:
	$(PY_SYS) tools/residency_table.py

clean-results:
	rm -rf results

clean: clean-results
	rm -rf .venv .cache/moeme

all: test doctor bench-inference
