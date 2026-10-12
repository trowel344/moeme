"""The PoC driver's cloud-safety behaviour.

Why this file exists: `experiments/upcycle_poc.py` is meant to be run
unattended on a rented device whose speed is unknown in advance (a free Kaggle
T4). The failure mode that wastes a whole session is a run that is killed by
the session cap having written nothing -- so the driver has two guarantees:

  1. `--max-seconds` stops the training loop and still reports what it measured.
  2. A configuration that raises is recorded as an `error` entry, the remaining
     configurations still run, and the process exits non-zero so the failure is
     not swallowed.

Neither is observable from the library tests in `test_upcycle.py`; they both
live in the driver. These tests use the driver's own tiny random model, so they
need no download, no GPU, and no network.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

import upcycle_poc as poc                    # noqa: E402
from moeme.upcycle import MoEConfig          # noqa: E402

CPU = torch.device("cpu")


def _run_config(*, steps, max_seconds, tmp_path, seed=0, resume=False,
                save_every=0):
    """One `train_one_config` call against the tiny model, on the CPU."""
    torch.manual_seed(seed)
    model = poc._TinyLM()
    blocks_mod = model.model.layers
    originals = [(i, blocks_mod[i].mlp) for i in range(len(blocks_mod))]

    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (2048,), generator=g)
    blocks = poc.make_blocks(ids, seq_len=128)
    train_blocks, val_blocks = blocks[:-2], blocks[-2:]

    model.eval()
    dense_ppl = poc.eval_perplexity(model, val_blocks, CPU, None, max_blocks=2)
    reference = poc.capture_reference(model, val_blocks, [0, 1, 2], CPU)

    config = MoEConfig(n_experts=2, top_k=1, plan="random", aux_loss_coef=0.01)
    args = SimpleNamespace(layers=["all"], lr=1e-3, weight_decay=0.0,
                           grad_accum=1, steps=steps, clip=1.0, seed=seed,
                           resume=resume, max_seconds=max_seconds,
                           ckpt_dir=str(tmp_path), eval_blocks=2,
                           save_every=save_every)
    return poc.train_one_config(
        model=model, originals=originals, blocks_mod=blocks_mod, config=config,
        args=args, train_blocks=train_blocks, val_blocks=val_blocks,
        reference=reference, device=CPU, amp_dtype=None,
        scaler=poc.make_grad_scaler(None), dense_ppl=dense_ppl,
    )


# ---------------------------------------------------------------------------
# the report header: what reproducing `dense_val_ppl` requires
# ---------------------------------------------------------------------------

def test_report_header_records_what_reproducing_the_baseline_needs():
    """The dense baseline is a function of the corpus AND the eval window.

    Both were absent from a header once, and each one cost a re-run instead of a
    re-read. The eval window is the sharper case: on this model and this text the
    perplexity is 19.57 over 4 val blocks and 23.42 over 8 -- same weights, same
    words -- so a header that omits it cannot reproduce the number printed beside
    it. That is exactly how E14's 21.74 became unreproducible against E15's
    19.56, and why this test exists rather than a comment.
    """
    args = SimpleNamespace(model="tiny", eval_blocks=4, seq_len=128)
    header = poc.make_report(args, device=CPU, amp_dtype=None,
                             corpus_source="dataset:HuggingFaceFW/fineweb-edu",
                             corpus_tokens=883_186, dense_ppl=19.5707,
                             n_train_blocks=1638, n_val_blocks=86)
    assert header["eval_blocks"] == 4
    assert header["corpus"] == "dataset:HuggingFaceFW/fineweb-edu"
    assert header["corpus_tokens"] == 883_186
    assert header["dense_val_ppl"] == 19.5707
    assert header["train_blocks"] == 1638 and header["val_blocks"] == 86
    assert header["runs"] == [], "the header must not pretend to hold results"


def test_report_header_follows_the_eval_window_it_was_given():
    # The window is recorded, not defaulted: a header that hard-coded one would
    # read as evidence while describing a run that did something else.
    for k in (2, 4, 8, 86):
        args = SimpleNamespace(model="tiny", eval_blocks=k, seq_len=512)
        header = poc.make_report(args, device=CPU, amp_dtype=None,
                                 corpus_source="built-in", corpus_tokens=1,
                                 dense_ppl=1.0, n_train_blocks=0, n_val_blocks=0)
        assert header["eval_blocks"] == k


# ---------------------------------------------------------------------------
# --max-seconds
# ---------------------------------------------------------------------------

def test_no_budget_runs_every_step(tmp_path):
    res = _run_config(steps=5, max_seconds=0.0, tmp_path=tmp_path)
    assert res["stopped_early"] is False
    assert res["steps_completed"] == 5
    assert res["steps"] == 5
    assert res["max_seconds"] == 0.0


def test_loss_curve_carries_elapsed_time(tmp_path):
    """The report must record step cost, not just print it.

    Sizing a later run on a different device depends on the measured cost, and
    the printed log is not the artefact anyone reads back.
    """
    res = _run_config(steps=6, max_seconds=0.0, tmp_path=tmp_path)
    curve = res["loss_curve"]
    assert len(curve) >= 2
    for point in curve:
        assert {"step", "loss", "elapsed"} <= set(point), point
    elapsed = [p["elapsed"] for p in curve]
    assert all(e >= 0 for e in elapsed)
    assert elapsed == sorted(elapsed), "elapsed time must not go backwards"


def test_budget_stops_early_but_still_reports_everything(tmp_path):
    """The whole point: a slow device returns fewer steps, not nothing."""
    res = _run_config(steps=1_000_000, max_seconds=0.001, tmp_path=tmp_path)
    assert res["stopped_early"] is True
    assert 1 <= res["steps_completed"] < 1_000_000

    # the report must be as complete as a full run's, or the evidence is lost
    for key in ("dense_val_ppl", "dense_limit_val_ppl", "untrained_val_ppl",
                "trained_val_ppl", "metrics_untrained", "metrics_trained",
                "routing", "loss_curve", "trainable_params"):
        assert key in res, f"early stop dropped {key!r} from the report"
    assert res["metrics_trained"]["oracle_topk_mean"] >= 0
    assert res["routing"]["effective_experts_mean"] > 0


def test_resumed_run_is_budgeted_against_its_own_clock(tmp_path):
    """Resuming must not hand the new session an unlimited budget.

    A resumed run starts at a saved step, so its loop is short; the budget has
    to be measured from the start of *this* session, not from the checkpoint.
    """
    ckpt = tmp_path / "ck"
    first = _run_config(steps=2, max_seconds=0.0, tmp_path=ckpt, save_every=1)
    assert first["steps_completed"] == 2
    assert list(ckpt.glob("*.pt")), "no checkpoint written, so resume is untested"

    resumed = _run_config(steps=1_000_000, max_seconds=0.001, tmp_path=ckpt,
                          resume=True, save_every=0)
    assert resumed["stopped_early"] is True
    assert resumed["steps_completed"] < 1_000_000


# ---------------------------------------------------------------------------
# a failing configuration must not destroy the whole run
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        ids = torch.arange(max(len(text), 1), dtype=torch.long) % 256
        return SimpleNamespace(input_ids=ids.unsqueeze(0))


def _run_main(monkeypatch, tmp_path, argv, use_text_file=True):
    """Run `main()` on the tiny model, with no network and a local corpus."""
    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeTokenizer()))
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", SimpleNamespace(
        from_pretrained=lambda *a, **k: poc._TinyLM()))

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 200)
    out = tmp_path / "report.json"
    corpus_arg = ["--text-file", str(corpus)] if use_text_file else []

    monkeypatch.setattr(sys, "argv", [
        "upcycle_poc.py", *corpus_arg, "--out", str(out), "--offline", *argv,
    ])
    return poc.main(), json.loads(out.read_text())


def test_one_failing_config_is_recorded_and_the_others_still_run(monkeypatch, tmp_path):
    real = poc.train_one_config
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated CUDA out of memory")
        return real(**kwargs)

    monkeypatch.setattr(poc, "train_one_config", flaky)
    code, report = _run_main(monkeypatch, tmp_path,
                             ["--experts", "2", "4", "--topks", "1", "--steps", "3"])

    assert calls["n"] == 2, "the second configuration was not attempted"
    assert len(report["runs"]) == 2

    bad, good = report["runs"]
    assert "simulated CUDA out of memory" in bad["error"]
    assert bad["layers"], "the error entry must still say which layers were upcycled"
    assert "traceback" in bad

    # the surviving configuration must be a complete, usable result
    assert good["steps_completed"] == 3
    assert good["trained_val_ppl"] > 0

    # and the failure must be visible in the exit status, not swallowed
    assert code == 1


def test_a_clean_run_exits_zero(monkeypatch, tmp_path):
    code, report = _run_main(monkeypatch, tmp_path,
                             ["--experts", "2", "--topks", "1", "--steps", "3"])
    assert code == 0
    assert len(report["runs"]) == 1
    assert "error" not in report["runs"][0]
    assert report["corpus"].startswith("text-file:")
    assert report["corpus_tokens"] > 0


def test_the_built_in_fallback_corpus_is_named_in_the_report(monkeypatch, tmp_path):
    """A silent corpus fallback would make every perplexity meaningless.

    The built-in corpus still yields finite, plausible-looking perplexities, so
    the report has to say which corpus it actually used.
    """
    code, report = _run_main(monkeypatch, tmp_path,
                             ["--experts", "2", "--topks", "1", "--steps", "2"],
                             use_text_file=False)
    assert code == 0
    assert report["corpus"] == "built-in"
    assert report["corpus_tokens"] > 0
