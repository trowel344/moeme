"""
Tests for the evaluation harness's extraction and scoring logic.

This matters more than it looks: the harness is what turns model output into
pass/fail, so a buggy scorer silently invalidates every quality number quoted in
the reports. These tests run offline and in milliseconds, so `make test` covers
the scorer without needing a model or a GPU.

The end-to-end harness was separately exercised against a real local model
(results/eval_qwen3_4b_subset.json); these tests pin the logic it relies on.

The second half covers the runner rather than the scorer: which tasks a run
selects, what the request payload says, and whether an interrupted multi-hour run
keeps its results. Those failure modes are silent -- they produce a plausible
score for the wrong task set, or lose hours of work -- so they are worth pinning
without a model.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import eval_gate  # noqa: E402
from eval_gate import (  # noqa: E402
    TASKS,
    check_json,
    evaluate,
    extract_code,
    extract_json,
    extract_number,
    extract_time,
    mark_truncation,
    row_is_truncated,
    run_code,
)

# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def test_extract_code_from_fenced_block():
    text = "Here you go:\n```python\ndef solve(x):\n    return x\n```\nDone."
    assert "def solve" in extract_code(text)
    assert "Done." not in extract_code(text)


def test_extract_code_without_language_tag():
    assert "def solve" in extract_code("```\ndef solve(a):\n    return a\n```")


def test_extract_code_from_bare_function():
    assert extract_code("def solve(n):\n    return n + 1").startswith("def solve")


def test_extract_code_gives_empty_string_when_absent():
    assert extract_code("There is no code here.") == ""


def test_extract_json_from_fenced_block():
    obj, err = extract_json('```json\n{"a": 1}\n```')
    assert obj == {"a": 1} and err == ""


def test_extract_json_from_prose_wrapped_object():
    obj, err = extract_json('Sure! Here it is: {"a": [1, 2], "b": "x"} — hope that helps.')
    assert obj == {"a": [1, 2], "b": "x"} and err == ""


def test_extract_json_handles_braces_inside_strings():
    obj, err = extract_json('{"note": "a } brace", "n": 2}')
    assert obj == {"note": "a } brace", "n": 2}, err


def test_extract_json_from_array_root():
    obj, err = extract_json("Here: [1, 2, 3]")
    assert obj == [1, 2, 3] and err == ""


def test_extract_json_reports_failure():
    obj, err = extract_json("no json at all")
    assert obj is None and err


def test_extract_number_prefers_explicit_answer_marker():
    text = "We computed 3 and 4, then squared. ANSWER: 42"
    assert extract_number(text) == 42.0


def test_extract_number_handles_negative_and_decimal():
    assert extract_number("ANSWER: -2.5") == -2.5


def test_extract_number_returns_none_when_absent():
    assert extract_number("no digits here") is None


def test_extract_time_takes_clock_like_value():
    assert extract_time("It arrives at 16:50.") == "16:50"
    assert extract_time("no time") is None


# ---------------------------------------------------------------------------
# json schema checking
# ---------------------------------------------------------------------------

def test_check_json_accepts_valid_schema():
    ok, _ = check_json({"summary": "s", "actions": ["a"]},
                       {"schema": {"summary": str, "actions": list}})
    assert ok


def test_check_json_rejects_missing_key():
    ok, why = check_json({"summary": "s"},
                         {"schema": {"summary": str, "actions": list}})
    assert not ok and "actions" in why


def test_check_json_rejects_wrong_type():
    ok, why = check_json({"ok": "true", "count": 3},
                         {"schema": {"ok": bool, "count": int}})
    assert not ok and "ok" in why


def test_check_json_rejects_bool_masquerading_as_int():
    """Regression: bool subclasses int, so JSON `true` used to validate as an
    integer field. A config value must not slip through that way."""
    ok, why = check_json({"count": True}, {"schema": {"count": int}})
    assert not ok and "count" in why
    ok2, _ = check_json({"count": 3}, {"schema": {"count": int}})
    assert ok2


def test_check_json_rejects_bool_in_nested_position():
    ok, _ = check_json({"server": {"host": "h", "port": True}},
                       {"schema": {"server": dict},
                        "nested": {"server": {"host": str, "port": int}}})
    assert not ok


def test_check_json_enforces_list_length():
    ok, why = check_json({"summary": "s", "actions": ["a", "b"]},
                         {"schema": {"summary": str, "actions": list},
                          "list_len": {"actions": 3}})
    assert not ok and "3" in why


def test_check_json_checks_nested_types():
    ok, _ = check_json({"tool": "read_file", "arguments": {"path": "/tmp/x"}},
                       {"schema": {"tool": str, "arguments": dict},
                        "nested": {"arguments": {"path": str}}})
    assert ok
    bad, why = check_json({"tool": "read_file", "arguments": {"path": 5}},
                          {"schema": {"tool": str, "arguments": dict},
                           "nested": {"arguments": {"path": str}}})
    assert not bad and "path" in why


def test_check_json_array_root_with_item_keys():
    ok, _ = check_json([{"name": "a", "score": 1}, {"name": "b", "score": 2}],
                       {"json_root": list, "array_len": 2,
                        "item_keys": ["name", "score"]})
    assert ok
    bad, why = check_json([{"name": "a"}],
                          {"json_root": list, "array_len": 1,
                           "item_keys": ["name", "score"]})
    assert not bad and "score" in why


def test_check_json_rejects_null_input():
    ok, why = check_json(None, {"schema": {"a": str}})
    assert not ok and "no JSON" in why


# ---------------------------------------------------------------------------
# evaluate() dispatch
# ---------------------------------------------------------------------------

def test_evaluate_exact_requires_whole_response_match():
    assert evaluate({"check": "exact", "expect": "yes"}, "yes", True)[0]
    assert not evaluate({"check": "exact", "expect": "yes"}, "yes, it is", True)[0]


def test_evaluate_bullets_counts_and_requires_exact_count():
    task = {"check": "bullets", "n": 3}
    assert evaluate(task, "- a\n- b\n- c", True)[0]
    assert not evaluate(task, "- a\n- b", True)[0]
    assert not evaluate(task, "- a\n- b\n- c\n- d", True)[0]


def test_evaluate_word_count():
    task = {"check": "word_count", "n": 5}
    assert evaluate(task, "one two three four five", True)[0]
    assert not evaluate(task, "one two three four", True)[0]


def test_evaluate_starts_with():
    assert evaluate({"check": "starts_with", "ch": "c"}, "compilers are neat", True)[0]
    assert not evaluate({"check": "starts_with", "ch": "c"}, "Compilers are neat", True)[0]


def test_evaluate_forbid_char_reports_accurately_on_pass_and_fail():
    """Regression: the detail string used to claim 'contains e' even on a pass."""
    task = {"check": "forbid_char", "ch": "e"}
    ok, why = evaluate(task, "a binary shrub", True)
    assert ok and "no 'e'" in why
    bad, why2 = evaluate(task, "a binary tree", True)
    assert not bad and "contains 'e'" in why2


def test_evaluate_number_respects_tolerance():
    task = {"check": "number", "expect": 0.10, "tol": 1e-6}
    assert evaluate(task, "ANSWER: 0.1", True)[0]
    assert not evaluate(task, "ANSWER: 0.2", True)[0]


def test_evaluate_numeric_time():
    task = {"check": "numeric_time", "expect": "16:50"}
    assert evaluate(task, "Arrives at 16:50.", True)[0]
    assert not evaluate(task, "Arrives at 17:50.", True)[0]


def test_evaluate_json_task_end_to_end():
    task = {"check": "json", "schema": {"summary": str, "actions": list},
            "list_len": {"actions": 3}}
    good = '{"summary": "build broke", "actions": ["a", "b", "c"]}'
    assert evaluate(task, good, True)[0]
    bad = '{"summary": "build broke", "actions": ["a"]}'
    assert not evaluate(task, bad, True)[0]


def test_evaluate_code_requires_execution_enabled():
    task = {"check": "code", "cases": [([[1]], 1)]}
    ok, why = evaluate(task, "```python\ndef solve(x):\n    return x\n```", False)
    assert not ok and "execution disabled" in why


def test_evaluate_rejects_unknown_check():
    with pytest.raises(ValueError):
        evaluate({"check": "nonsense"}, "x", True)


# ---------------------------------------------------------------------------
# truncation accounting
# ---------------------------------------------------------------------------

def test_a_stopped_run_is_not_labelled_truncated():
    detail, trunc = mark_truncation("got 3.0, want 42", "3", "stop", 40)
    assert (detail, trunc) == ("got 3.0, want 42", False)


def test_truncation_is_labelled_on_a_NON_empty_answer():
    """Regression: only an EMPTY answer was labelled, so a capped run that
    narrated on its way to the answer recorded a bare wrong-answer verdict.
    Three of the dense 27B's four reasoning rows on the 19-task subset were
    exactly this -- truncated mid-solution on the way to the right number -- and
    were reported as reasoning failures."""
    detail, trunc = mark_truncation(
        "got 3.0, want 42",
        "Step 3: add the next difference to the last term",
        "length", 192)
    assert trunc and detail.startswith("got 3.0, want 42")
    assert "[truncated at 192 tokens]" in detail


def test_truncated_empty_answer_names_the_harness_setting():
    detail, trunc = mark_truncation("no number found", "", "length", 1536)
    assert trunc and detail.startswith("TRUNCATED")
    assert "raise --num-predict" in detail


def test_row_is_truncated_reads_the_flag_and_pre_flag_reports():
    assert row_is_truncated({"truncated": True, "detail": "ok"})
    # Reports written before the field existed carry the marker in `detail`.
    assert row_is_truncated({"detail": "TRUNCATED during reasoning (192 tokens"})
    assert not row_is_truncated({"truncated": False, "detail": "got 1, want 2"})
    assert not row_is_truncated({})


# ---------------------------------------------------------------------------
# code execution
# ---------------------------------------------------------------------------

def test_run_code_passes_correct_solution():
    ok, why = run_code("def solve(nums):\n    return sum(n for n in nums if n % 2 == 0)",
                       [([[1, 2, 3, 4]], 6), ([[]], 0)])
    assert ok and "2/2" in why


def test_run_code_fails_incorrect_solution():
    ok, why = run_code("def solve(nums):\n    return 0", [([[1, 2]], 2)])
    assert not ok and "1/1 cases failed" in why


def test_run_code_reports_runtime_error():
    ok, why = run_code("def solve(x):\n    return 1 / 0", [([[1]], 1)])
    assert not ok and "ZeroDivisionError" in why


def test_run_code_reports_syntax_error():
    ok, why = run_code("def solve(x)\n    return x", [([[1]], 1)])
    assert not ok and "runtime error" in why


def test_run_code_enforces_timeout():
    ok, why = run_code("def solve(x):\n    while True:\n        pass",
                       [([[1]], 1)], timeout=2.0)
    assert not ok and why == "timeout"


# ---------------------------------------------------------------------------
# runner plumbing: request payload, task selection, checkpointing, resume
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Minimal stand-in for the object urlopen yields in a `with` block."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_request(monkeypatch, response: dict) -> dict:
    seen: dict = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode())
        seen["timeout"] = timeout
        return _FakeResponse(response)

    monkeypatch.setattr(eval_gate.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_ask_omits_think_by_default(monkeypatch):
    """--think auto must send nothing, or builds without the parameter 400."""
    seen = _capture_request(monkeypatch, {"response": "yes", "eval_count": 1})
    eval_gate.ask("m", "p", 8, think="auto")
    assert "think" not in seen["payload"]


def test_ask_sends_think_false_when_disabled(monkeypatch):
    seen = _capture_request(monkeypatch, {"response": "yes", "eval_count": 1})
    eval_gate.ask("m", "p", 8, think="off")
    assert seen["payload"]["think"] is False
    assert seen["payload"]["options"]["num_predict"] == 8
    assert seen["payload"]["options"]["temperature"] == 0.0


def test_ask_records_prompt_processing_rate(monkeypatch):
    """Prefill is the number that decides time-to-first-token, so it has to be
    recorded per task rather than borrowed from a separate short benchmark."""
    _capture_request(monkeypatch, {"response": "x", "eval_count": 4,
                                   "eval_duration": 100_000_000,
                                   "prompt_eval_count": 200,
                                   "prompt_eval_duration": 2_000_000_000})
    _, meta = eval_gate.ask("m", "p", 8)
    assert meta["decode_tok_s"] == 40.0
    assert meta["prompt_tok_s"] == 100.0


def test_ask_sends_keep_alive_and_timeout(monkeypatch):
    seen = _capture_request(monkeypatch, {"response": "x", "eval_count": 1})
    eval_gate.ask("m", "p", 8, keep_alive="30m", timeout=123.0)
    assert seen["payload"]["keep_alive"] == "30m"
    assert seen["timeout"] == 123.0


def _chat_response(content: str, *, finish="stop", reasoning="", pred=50.0):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": content,
                                     "reasoning_content": reasoning}}],
            "timings": {"predicted_per_second": pred, "prompt_per_second": 121.5,
                        "predicted_n": 7}}


def test_llama_backend_uses_the_chat_route_and_parses_its_timings(monkeypatch):
    """The engine comparison is only worth anything if the other engine gets the
    identical prompt, greedy decode and budget, and if its own truncation signal
    reaches the scorer. The chat route (not /completion) is required: /completion
    applies no template, so a thinking model's <think> block lands in the answer
    and re-scores 13/19 against 17/19 for the same weights."""
    seen = _capture_request(monkeypatch, _chat_response("ANSWER: 42"))
    text, meta = eval_gate.ask_llama("p", 768, api="http://127.0.0.1:8080")
    assert seen["url"].endswith("/v1/chat/completions")
    p = seen["payload"]
    assert p["messages"] == [{"role": "user", "content": "p"}]
    assert p["n_predict"] == 768
    assert p["temperature"] == 0.0 and p["seed"] == eval_gate.SEED
    assert p["stream"] is False
    assert text == "ANSWER: 42"
    assert meta["done_reason"] == "stop"
    assert meta["decode_tok_s"] == 50.0 and meta["prompt_tok_s"] == 121.5
    assert meta["reasoning_only"] is False


def test_llama_finish_reason_length_maps_onto_truncation(monkeypatch):
    _capture_request(monkeypatch, _chat_response("step 3: add the next",
                                                 finish="length"))
    text, meta = eval_gate.ask_llama("p", 768)
    assert meta["done_reason"] == "length"
    detail, trunc = mark_truncation("no number found", text, meta["done_reason"],
                                    meta["eval_count"])
    assert trunc and "[truncated at 7 tokens]" in detail


def test_llama_reasoning_only_answer_is_flagged_not_scored_as_wrong(monkeypatch, tmp_path):
    """A build that ignores the kwarg puts the whole answer in reasoning_content
    and leaves `content` empty. Scoring that as a wrong answer would blame the
    model for a server setting."""
    seen = _capture_request(monkeypatch, _chat_response(
        "", reasoning="The user is asking whether 17 is prime. 17 is prime."))
    text, meta = eval_gate.ask_llama("p", 64)
    assert text == "" and meta["reasoning_only"] is True
    assert meta["thinking_chars"] > 0
    assert "content" in seen["payload"]["messages"][0]


def test_llama_can_suppress_thinking_through_the_template(monkeypatch):
    """A server applies the model's chat template, and a thinking model's template
    defaults to thinking ON. Ollama's `think: false` hides the <think> block; the
    llama.cpp side needs the equivalent kwarg or its answer is wrapped in one and
    every strict-instruction task fails for a reason that is not the engine."""
    seen = _capture_request(monkeypatch, {"content": "yes", "tokens_predicted": 1,
                                          "stopped_limit": False, "stop": True})
    eval_gate.ask_llama("p", 8, template_kwargs={"enable_thinking": False})
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    # and it is absent unless asked for
    seen2 = _capture_request(monkeypatch, {"content": "yes", "tokens_predicted": 1})
    eval_gate.ask_llama("p", 8)
    assert "chat_template_kwargs" not in seen2["payload"]


def test_chat_template_kwargs_must_be_a_json_object(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_gate, "ask_llama_ready", lambda api: True)
    for bad in ("not json", '[1, 2]'):
        monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--model", "m",
                                           "--backend", "llama", "--tasks",
                                           "i_only_word", "--chat-template-kwargs",
                                           bad, "--out", str(tmp_path / "e.json")])
        with pytest.raises(SystemExit):
            eval_gate.main()


def test_chat_template_kwargs_is_rejected_on_the_ollama_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--model", "m", "--tasks",
                                       "i_only_word", "--chat-template-kwargs",
                                       '{"enable_thinking": false}', "--out",
                                       str(tmp_path / "e.json")])
    with pytest.raises(SystemExit):
        eval_gate.main()


def test_llama_stopped_limit_also_maps_onto_truncation(monkeypatch):
    """Some builds report the cap with `stopped_limit` instead of finish_reason."""
    _capture_request(monkeypatch, {"stopped_limit": True,
                                   "choices": [{"message": {"content": "x"}}],
                                   "timings": {"predicted_per_second": 0.4}})
    _, meta = eval_gate.ask_llama("p", 768)
    assert meta["done_reason"] == "length"


def test_backend_selects_the_transport(monkeypatch):
    calls: list = []

    def fake_ask(model, prompt, num_predict=0, **kw):
        calls.append(("ollama", prompt))
        return "x", {}

    def fake_llama(prompt, num_predict=0, **kw):
        calls.append(("llama", prompt))
        return "x", {}

    monkeypatch.setattr(eval_gate, "ask", fake_ask)
    monkeypatch.setattr(eval_gate, "ask_llama", fake_llama)
    eval_gate.ask_backend("llama", "m", "p", 8, think="off", api="u",
                          keep_alive="30s", timeout=1.0)
    eval_gate.ask_backend("ollama", "m", "p", 8, think="off", api="u",
                          keep_alive="30s", timeout=1.0)
    assert calls == [("llama", "p"), ("ollama", "p")]


def test_llama_backend_refuses_to_start_a_server_on_its_own(monkeypatch, tmp_path):
    """The placement under test is the server's launch flags. Spawning a default
    server would silently measure something other than what was asked for."""
    monkeypatch.setattr(eval_gate, "ask_llama_ready", lambda api: False)
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--model", "m", "--backend",
                                      "llama", "--tasks", "i_only_word",
                                      "--out", str(tmp_path / "e.json")])
    with pytest.raises(SystemExit) as e:
        eval_gate.main()
    assert "llama-server" in str(e.value)


def test_backend_is_recorded_and_a_backend_switch_is_not_resumed(monkeypatch, tmp_path):
    out = tmp_path / "e.json"
    _run_main(monkeypatch, ["--model", "fake", "--tasks", "i_only_word",
                            "--out", str(out)])
    assert json.loads(out.read_text())["backend"] == "ollama"
    # A llama.cpp run reusing ollama rows would mix two engines in one score.
    monkeypatch.setattr(eval_gate, "ask_llama_ready", lambda api: True)
    llama_calls: list = []

    def fake_llama(prompt, num_predict=0, **kw):
        llama_calls.append(prompt)
        return "yes", {"eval_count": 1, "decode_tok_s": 50.0,
                       "done_reason": "stop"}

    monkeypatch.setattr(eval_gate, "ask_llama", fake_llama)
    _, ollama_calls = _run_main(monkeypatch,
                                ["--model", "fake", "--tasks", "i_only_word",
                                 "--backend", "llama", "--out", str(out),
                                 "--resume"])
    assert len(llama_calls) == 1 and ollama_calls == []
    assert json.loads(out.read_text())["backend"] == "llama"


def _run_main(monkeypatch, argv, answer=None, fail_from=None):
    """Run main() with the network stubbed out. Returns (rc, call count)."""
    calls: list = []

    def fake_ask(model, prompt, num_predict=0, *, think="auto", api="",
                 keep_alive="", timeout=0):
        calls.append(prompt)
        if fail_from is not None and len(calls) >= fail_from:
            raise RuntimeError("backend gone")
        return answer or "ANSWER: 42", {
            "eval_count": 1, "decode_tok_s": 9.0, "done_reason": "stop",
            "thinking_chars": 0, "num_predict": num_predict}

    monkeypatch.setattr(eval_gate, "ask", fake_ask)
    monkeypatch.setattr(eval_gate, "ensure_server", lambda: True)
    monkeypatch.setattr(sys, "argv", ["eval_gate.py"] + argv)
    return eval_gate.main(), calls


SUBSET = ["i_only_word", "i_uppercase", "r_sequence"]


def test_numeric_expectations_are_arithmetically_correct():
    """Regression: two expected values were simply wrong, and they mis-scored in
    opposite directions, so a correct answer was failed and a wrong answer
    passed. Re-derive the ones that have a closed form rather than trusting the
    literal in the task table."""
    by_id = {t["id"]: t for t in TASKS}
    assert by_id["r_bat_ball"]["expect"] == pytest.approx((1.10 - 1.00) / 2)
    assert by_id["r_modular"]["expect"] == 7 ** 20 % 13 == 3
    assert by_id["r_combinatorics"]["expect"] == 30  # 5! / (2! 2!) for LEVEL
    assert by_id["r_work_rate"]["expect"] == 3
    assert by_id["r_lily"]["expect"] == 47
    assert by_id["r_sequence"]["expect"] == 42
    assert by_id["r_percent"]["expect"] == pytest.approx(1.25 * 0.80 - 1.0)


REFERENCE_SOLUTIONS = {
    "c_sum_evens": "def solve(nums):\n    return sum(n for n in nums if n % 2 == 0)",
    "c_reverse_words": "def solve(s):\n    return ' '.join(s.split()[::-1])",
    "c_fizzbuzz": ("def solve(n):\n    return ['FizzBuzz' if i % 15 == 0 else 'Fizz' "
                   "if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i) "
                   "for i in range(1, n + 1)]"),
    "c_is_palindrome": ("def solve(s):\n    t = [c.lower() for c in s if c.isalnum()]\n"
                        "    return t == t[::-1]"),
    "c_binary_search": ("def solve(nums, t):\n    lo, hi = 0, len(nums) - 1\n"
                        "    while lo <= hi:\n        m = (lo + hi) // 2\n"
                        "        if nums[m] == t:\n            return m\n"
                        "        if nums[m] < t:\n            lo = m + 1\n"
                        "        else:\n            hi = m - 1\n    return -1"),
    "c_unique_ordered": ("def solve(items):\n    out = []\n    for x in items:\n"
                         "        if x not in out:\n            out.append(x)\n"
                         "    return out"),
    "c_gcd": "def solve(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "c_flatten": ("def solve(n):\n    out = []\n    for x in n:\n"
                  "        out.extend(solve(x) if isinstance(x, list) else [x])\n"
                  "    return out"),
    "c_two_sum": ("def solve(nums, t):\n    for i in range(len(nums)):\n"
                  "        for j in range(i + 1, len(nums)):\n"
                  "            if nums[i] + nums[j] == t:\n                return [i, j]\n"
                  "    return []"),
    "c_word_count": ("def solve(text):\n    d = {}\n    for w in text.split():\n"
                     "        k = w.lower()\n        d[k] = d.get(k, 0) + 1\n"
                     "    return d"),
}


@pytest.mark.parametrize("task", [t for t in TASKS if t["check"] == "code"],
                         ids=lambda t: t["id"])
def test_every_coding_case_is_satisfiable_by_a_correct_solution(task):
    """A case table with a wrong expected value fails EVERY model, including a
    correct one, and looks like a capability result. The numeric half of the set
    shipped two such defects (see test_numeric_expectations_are_arithmetically
    _correct); this is the same check for the executable half."""
    ok, why = run_code(REFERENCE_SOLUTIONS[task["id"]], task["cases"])
    assert ok, why


def test_rescore_reapplies_a_corrected_expectation(monkeypatch, tmp_path):
    """A corrected task definition must be replayable over stored responses."""
    out = tmp_path / "e.json"
    _run_main(monkeypatch, ["--model", "fake", "--tasks", "r_bat_ball",
                            "--out", str(out)], answer="ANSWER: 0.05")
    assert json.loads(out.read_text())["models"]["fake"]["overall"]["passed"] == 1
    # Simulate the old, wrong expectation and re-score the recorded text.
    original = next(t for t in TASKS if t["id"] == "r_bat_ball")
    monkeypatch.setitem(original, "expect", 0.10)
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--rescore", "--out", str(out)])
    assert eval_gate.main() == 0
    row = json.loads(out.read_text())["models"]["fake"]["tasks"][0]
    assert row["passed"] is False and "0.1" in row["detail"]


def test_rescore_leaves_rows_without_a_stored_response_alone(monkeypatch, tmp_path):
    out = tmp_path / "e.json"
    report = {"tasks_total": 1, "task_ids": ["r_sequence"], "think": "off",
              "num_predict": 8, "models": {"fake": {
                  "tasks": [{"id": "r_sequence", "category": "reasoning",
                             "passed": True, "detail": "legacy row"}]}}}
    out.write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--rescore", "--out", str(out)])
    assert eval_gate.main() == 0
    row = json.loads(out.read_text())["models"]["fake"]["tasks"][0]
    assert row["passed"] is True and row["detail"] == "legacy row"


def test_tasks_subset_is_canonical_order_and_narrower_than_the_full_set(
        monkeypatch, tmp_path):
    out = tmp_path / "e.json"
    rc, _ = _run_main(monkeypatch, ["--model", "fake", "--tasks",
                                    ",".join(reversed(SUBSET)),
                                    "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    canonical = [t["id"] for t in TASKS if t["id"] in set(SUBSET)]
    assert report["task_ids"] == canonical
    assert set(report["task_ids"]) <= {t["id"] for t in TASKS}


def test_unknown_task_id_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--model", "fake", "--tasks", "i_only_word,typo_id",
                               "--out", str(tmp_path / "e.json")])


def test_think_and_num_predict_are_recorded_for_provenance(monkeypatch, tmp_path):
    out = tmp_path / "e.json"
    _run_main(monkeypatch, ["--model", "fake", "--tasks", "i_only_word",
                            "--think", "off", "--num-predict", "192",
                            "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["think"] == "off" and report["num_predict"] == 192
    assert report["models"]["fake"]["num_predict"] == 192


def test_every_task_is_checkpointed_before_the_next_one_starts(
        monkeypatch, tmp_path):
    """A dropped connection on task 3 must not cost the first two results."""
    out = tmp_path / "e.json"
    _, calls = _run_main(monkeypatch,
                         ["--model", "fake", "--tasks", ",".join(SUBSET),
                          "--out", str(out)],
                         fail_from=3)
    assert len(calls) == 3
    report = json.loads(out.read_text())
    rows = report["models"]["fake"]["tasks"]
    assert len(rows) == 3
    assert "harness error" in rows[2]["detail"]
    assert rows[0]["passed"] is False or rows[0]["passed"] is True  # scored, not lost


def test_three_consecutive_harness_errors_abort_the_model(monkeypatch, tmp_path):
    out = tmp_path / "e.json"
    _, calls = _run_main(monkeypatch,
                         ["--model", "fake", "--tasks", ",".join(SUBSET),
                          "--out", str(out)], fail_from=1)
    # All three tasks are attempted once each, then the run stops instead of
    # hammering a dead backend for the rest of the set.
    assert len(calls) == 3
    report = json.loads(out.read_text())
    assert report["models"]["fake"]["harness_errors"] == 3


def test_resume_reuses_recorded_tasks_and_makes_no_new_calls(monkeypatch, tmp_path):
    out = tmp_path / "e.json"
    args = ["--model", "fake", "--tasks", ",".join(SUBSET), "--out", str(out)]
    _run_main(monkeypatch, args)
    first = json.loads(out.read_text())
    _, calls = _run_main(monkeypatch, args + ["--resume"])
    assert calls == []
    # Compare the results, not the whole document: `wall_s` legitimately grows
    # with the second invocation's (near-zero) elapsed time.
    second = json.loads(out.read_text())["models"]["fake"]
    assert second["tasks"] == first["models"]["fake"]["tasks"]
    assert second["overall"] == first["models"]["fake"]["overall"]


def test_resume_does_not_reuse_a_different_task_set(monkeypatch, tmp_path):
    """Reusing rows from another configuration would mix two experiments."""
    out = tmp_path / "e.json"
    _run_main(monkeypatch, ["--model", "fake", "--tasks", "i_only_word",
                            "--out", str(out)])
    _, calls = _run_main(monkeypatch, ["--model", "fake", "--tasks", "i_uppercase",
                                       "--out", str(out), "--resume"])
    assert len(calls) == 1
    assert json.loads(out.read_text())["task_ids"] == ["i_uppercase"]


def test_resume_does_not_reuse_a_different_decode_budget(monkeypatch, tmp_path):
    out = tmp_path / "e.json"
    _run_main(monkeypatch, ["--model", "fake", "--tasks", "i_only_word",
                            "--num-predict", "64", "--out", str(out)])
    _, calls = _run_main(monkeypatch, ["--model", "fake", "--tasks", "i_only_word",
                                       "--num-predict", "192", "--out", str(out),
                                       "--resume"])
    assert len(calls) == 1


def test_truncated_empty_answer_is_reported_as_a_harness_limit(monkeypatch, tmp_path):
    """An empty answer at the token cap is the harness's fault, not the model's."""
    out = tmp_path / "e.json"

    def fake_ask(model, prompt, num_predict=0, **kw):
        return "", {"eval_count": 192, "decode_tok_s": 1.0,
                    "done_reason": "length", "thinking_chars": 0}

    monkeypatch.setattr(eval_gate, "ask", fake_ask)
    monkeypatch.setattr(eval_gate, "ensure_server", lambda: True)
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--model", "fake",
                                      "--tasks", "r_sequence",
                                      "--out", str(out)])
    eval_gate.main()
    row = json.loads(out.read_text())["models"]["fake"]["tasks"][0]
    assert not row["passed"] and "TRUNCATED" in row["detail"]


def test_response_text_is_kept_for_re_scoring(monkeypatch, tmp_path):
    """The raw answer is stored so a scorer change can be replayed offline."""
    out = tmp_path / "e.json"
    _run_main(monkeypatch, ["--model", "fake", "--tasks", "i_uppercase",
                            "--out", str(out)], answer="confirmed")
    row = json.loads(out.read_text())["models"]["fake"]["tasks"][0]
    assert row["response"] == "confirmed"


def test_a_capped_run_reports_both_scores(monkeypatch, tmp_path):
    """A capped row still fails the strict score -- it holds no usable answer --
    but the report has to say how many rows the cap removed, and what the score
    is without them, or a harness setting reads as a capability result."""
    out = tmp_path / "e.json"

    def fake_ask(model, prompt, num_predict=0, **kw):
        return ("Step 3: the next difference is 12, so add it to the last term",
                {"eval_count": 192, "decode_tok_s": 1.0,
                 "done_reason": "length", "thinking_chars": 0})

    monkeypatch.setattr(eval_gate, "ask", fake_ask)
    monkeypatch.setattr(eval_gate, "ensure_server", lambda: True)
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--model", "fake",
                                      "--tasks", "r_sequence",
                                      "--out", str(out)])
    assert eval_gate.main() == 0
    m = json.loads(out.read_text())["models"]["fake"]
    row = m["tasks"][0]
    assert row["passed"] is False and row["truncated"] is True
    assert m["truncated_tasks"] == 1
    assert m["summary"]["reasoning"]["truncated"] == 1
    assert m["overall"] == {"passed": 0, "total": 1, "rate": 0.0}
    assert m["overall_untruncated"] == {"passed": 0, "total": 0, "rate": 0.0}


def test_rescore_re_derives_truncation_from_the_stored_done_reason(monkeypatch, tmp_path):
    """Truncation is a property of the recorded run, so --rescore must pick it up
    for a report written before the flag existed."""
    out = tmp_path / "e.json"
    answer = "The differences rise by 2, so the next term comes from the last one"
    report = {"tasks_total": 1, "task_ids": ["r_sequence"], "think": "off",
              "num_predict": 192, "models": {"fake": {
                  "tasks": [{"id": "r_sequence", "category": "reasoning",
                             "passed": False, "detail": "legacy row",
                             "response": answer, "done_reason": "length",
                             "eval_count": 192}]}}}
    out.write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--rescore", "--out", str(out)])
    assert eval_gate.main() == 0
    m = json.loads(out.read_text())["models"]["fake"]
    assert m["tasks"][0]["truncated"] is True
    assert m["truncated_tasks"] == 1 and m["overall_untruncated"]["total"] == 0
