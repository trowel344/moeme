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

.PHONY: help venv test doctor bench-inference verify-sources \
        verify-real-ffn pilot eval report clean clean-results all

help:
	@echo "moeme targets:"
	@echo "  make venv             create .venv (reuses system torch) and install pytest"
	@echo "  make test             run the offline unit test suite (124 tests)"
	@echo "  make doctor           one-command local hardware diagnostics (~90s)"
	@echo "  make bench-inference  benchmark quantised models on this machine"
	@echo "  make verify-sources   re-verify published model geometry from real artefacts"
	@echo "  make verify-real-ffn  verify FFN exactness on real 27B weights (535 MB download)"
	@echo "  make pilot            pilot sparsity study, real activations (needs GPU, 1.8 GB)"
	@echo "  make eval             evaluation gate set (MODEL=..., NUM_PREDICT=...)"
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

eval:
	$(PY_SYS) experiments/eval_gate.py --model $(MODEL) \
		--num-predict $(NUM_PREDICT) \
		--out results/eval_$(subst :,-,$(MODEL)).json

report:
	$(PY_SYS) tools/summarise_results.py

residency:
	$(PY_SYS) tools/residency_table.py

clean-results:
	rm -rf results

clean: clean-results
	rm -rf .venv .cache/moeme

all: test doctor bench-inference
