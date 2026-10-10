# moeme -- edge MoE feasibility study. One-command entry points.
#
# Everything here is non-destructive: temp files are removed by the scripts,
# and the only network traffic is read-only Hugging Face range requests.

PY      ?= .venv/bin/python
PY_SYS  ?= python3

.PHONY: help venv test test-fast doctor bench-inference verify-sources \
        verify-real-ffn pilot report clean clean-results all

help:
	@echo "moeme targets:"
	@echo "  make venv             create .venv (reuses system torch) and install pytest"
	@echo "  make test             run the offline unit test suite"
	@echo "  make doctor           one-command local hardware diagnostics (~90s, writes results/hardware.json)"
	@echo "  make bench-inference  benchmark quantised models on this machine"
	@echo "  make verify-sources   re-verify published model geometry from the real artefacts"
	@echo "  make verify-real-ffn  verify FFN partitioning exactness on real 27B weights (535 MB download)"
	@echo "  make pilot            pilot sparsity study with real activations (needs GPU, ~1.8 GB download)"
	@echo "  make report           re-print the summary tables from results/*.json"
	@echo "  make all              venv + test + doctor + bench-inference"
	@echo "  make clean-results    delete results/ (regenerable)"

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

report:
	$(PY_SYS) tools/summarise_results.py

clean-results:
	rm -rf results

clean: clean-results
	rm -rf .venv .cache/moeme

all: test doctor bench-inference
