"""
Tests for tools/eval_llama_engine.py -- the llama.cpp placement driver.

This is the code that decides whether a 13 GB model is allowed to start on an
8 GB card. Getting it wrong does not produce a wrong number, it takes the desktop
down (the CPU-only control run in docs/expert_offloading.md was OOM-killed), so
the refusal path is pinned here, offline, without a GPU.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import eval_llama_engine as engine  # noqa: E402

GIB = 2 ** 30


def _meminfo(available_gib: float) -> dict:
    return {"MemAvailable": int(available_gib * GIB), "MemTotal": 15 * GIB}


@pytest.fixture
def plenty(monkeypatch):
    monkeypatch.setattr(engine, "_read_meminfo", lambda: _meminfo(10.0))
    monkeypatch.setattr(engine, "vram_used_mib", lambda: 300)
    monkeypatch.setattr(engine, "vram_total_mib", lambda: 8188)


def test_preflight_allows_a_run_that_fits(plenty):
    facts = engine.preflight(6000, 6.0, force=False)
    assert facts["vram_used_mib_before"] == 300
    assert facts["ram_available_bytes"] == int(10.0 * GIB)


def test_preflight_refuses_when_ram_is_short(monkeypatch):
    monkeypatch.setattr(engine, "_read_meminfo", lambda: _meminfo(4.0))
    monkeypatch.setattr(engine, "vram_used_mib", lambda: 300)
    monkeypatch.setattr(engine, "vram_total_mib", lambda: 8188)
    with pytest.raises(SystemExit) as e:
        engine.preflight(6000, 6.0, force=False)
    assert "RAM available" in str(e.value)
    # The message has to say what to do about it, not just refuse.
    assert "ollama stop" in str(e.value)


def test_preflight_refuses_when_vram_is_already_occupied(monkeypatch):
    """A resident model is the usual cause, and the usual fix. Refusing is the
    point: llama.cpp would otherwise either fail mid-load or evict something the
    user is using."""
    monkeypatch.setattr(engine, "_read_meminfo", lambda: _meminfo(10.0))
    monkeypatch.setattr(engine, "vram_used_mib", lambda: 6600)
    monkeypatch.setattr(engine, "vram_total_mib", lambda: 8188)
    with pytest.raises(SystemExit) as e:
        engine.preflight(6000, 6.0, force=False)
    assert "VRAM free" in str(e.value)


def test_preflight_force_proceeds_and_says_so(monkeypatch, capsys):
    monkeypatch.setattr(engine, "_read_meminfo", lambda: _meminfo(2.0))
    monkeypatch.setattr(engine, "vram_used_mib", lambda: 6600)
    monkeypatch.setattr(engine, "vram_total_mib", lambda: 8188)
    facts = engine.preflight(6000, 6.0, force=True)
    assert facts["ram_available_bytes"] == int(2.0 * GIB)
    assert "WARNING" in capsys.readouterr().out


def test_preflight_tolerates_a_machine_without_nvidia_smi(monkeypatch):
    """The VRAM check is skipped rather than faked when nvidia-smi is absent."""
    monkeypatch.setattr(engine, "_read_meminfo", lambda: _meminfo(10.0))
    monkeypatch.setattr(engine, "vram_used_mib", lambda: None)
    monkeypatch.setattr(engine, "vram_total_mib", lambda: None)
    facts = engine.preflight(6000, 6.0, force=False)
    assert facts["vram_used_mib_before"] is None


def test_sampler_reads_its_own_peak_rss():
    """Peak, not current: an average hides the moment a run got close to the edge."""
    assert engine.Sampler._rss(os.getpid()) > 0


class _Proc:
    def __init__(self, code=None):
        self._code = code

    def poll(self):
        return self._code


class _Health:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_wait_ready_reports_a_load_that_died(monkeypatch):
    """A failed allocation during load must not be reported as a slow start."""
    with pytest.raises(SystemExit) as e:
        engine.wait_ready("http://127.0.0.1:8080", 5.0, _Proc(code=1))
    assert "exited with code 1" in str(e.value)
    assert "allocation failure" in str(e.value)


def test_wait_ready_returns_when_the_server_is_up(monkeypatch):
    monkeypatch.setattr(engine.urllib.request, "urlopen",
                        lambda url, timeout=None: _Health())
    engine.wait_ready("http://127.0.0.1:8080", 5.0, _Proc())


def test_surplus_plan_parses_the_json_after_the_human_lines(monkeypatch):
    """`surplus plan` prints two prose lines before the object, and the plan is
    recorded next to the measured result so its prediction can be checked."""
    import eval_surplus_shim as shim

    out = ("tuning: none (conservative planner defaults)\n"
           "model_group: moe-split (experts split across GPU/CPU: --n-cpu-moe 24)\n"
           '{\n  "gpu_layers": "42",\n  "n_cpu_moe_layers": 24,\n'
           '  "cache_type_k": "q8_0",\n  "predicted_decode_tps": 57.17\n}\n')

    class P:
        returncode, stdout, stderr = 0, out, ""

    monkeypatch.setattr(shim.subprocess, "run", lambda *a, **k: P())
    plan = shim.surplus_plan("surplus", "m.gguf")
    assert plan["n_cpu_moe_layers"] == 24
    assert plan["predicted_decode_tps"] == 57.17


def test_surplus_plan_failure_does_not_abort_the_run(monkeypatch):
    """A planner that cannot answer is a missing annotation, not a reason to
    skip a measurement that is otherwise fine."""
    import eval_surplus_shim as shim

    class P:
        returncode, stdout, stderr = 1, "", "no such file"

    monkeypatch.setattr(shim.subprocess, "run", lambda *a, **k: P())
    assert "error" in shim.surplus_plan("surplus", "m.gguf")


def test_wait_ready_gives_up_with_the_last_status(monkeypatch):
    class Loading:
        status = 503

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(engine.urllib.request, "urlopen",
                        lambda url, timeout=None: Loading())
    monkeypatch.setattr(engine.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit) as e:
        engine.wait_ready("http://127.0.0.1:8080", 0.01, _Proc())
    assert "not ready" in str(e.value)
