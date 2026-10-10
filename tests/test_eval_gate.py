"""
Tests for the evaluation harness's extraction and scoring logic.

This matters more than it looks: the harness is what turns model output into
pass/fail, so a buggy scorer silently invalidates every quality number quoted in
the reports. These tests run offline and in milliseconds, so `make test` covers
the scorer without needing a model or a GPU.

The end-to-end harness was separately exercised against a real local model
(results/eval_qwen3_4b_subset.json); these tests pin the logic it relies on.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from eval_gate import (  # noqa: E402
    check_json,
    evaluate,
    extract_code,
    extract_json,
    extract_number,
    extract_time,
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
