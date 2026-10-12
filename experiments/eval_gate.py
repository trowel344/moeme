#!/usr/bin/env python3
"""
eval_gate.py -- reproducible evaluation set with executable checks.

Purpose: give `docs/decision_memo.md` Path A a real gate. "Q3 still feels fine"
is not a gate; this prints pass/fail per task category so the quantisation
ladder decision can be made on evidence.

Categories
----------
  coding       generated function is EXECUTED against fixed cases
  reasoning    numeric answer extracted and compared
  format       output must parse as JSON with the required keys/types
  instruction  explicit constraint compliance (counts, casing, exact tokens)

Design notes
------------
* Deterministic: temperature 0, fixed seed, fixed num_predict cap.
* Executable correctness, not string matching, for coding tasks.
* Failures are reported individually. Aggregate scores never hide a brittle
  failure: every task keeps its id, category, extracted answer and reason.

SAFETY: coding tasks execute model-generated Python in a subprocess with a
timeout, in a temporary directory, with the parent process's environment. That
is the point of the harness, but it does mean you are running code a local model
wrote. Use --no-execute to skip execution if you do not want that.

Scope: this is a STARTER set (30 tasks). The handoff specifies 30 coding /
30 reasoning / 20 instruction+tool / 20 chat plus context tests, ideally with
respected public benchmarks and contamination checks. This harness is the
scaffold and the plumbing those need; extend TASKS and add a contamination
check before treating any score as publishable.

Usage:
    python3 experiments/eval_gate.py --model qwen3:4b
    python3 experiments/eval_gate.py --model A --model B --out results/eval.json
    python3 experiments/eval_gate.py --model X --no-execute
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

# Thinking models (qwen3, qwen3.5, qwen3.8) spend an unbounded number of tokens
# in a separate `thinking` field before emitting `response`. With too small a
# cap they truncate mid-reasoning and return an EMPTY answer, which must not be
# silently scored as a capability failure. 512 is not enough; 1536 is the
# default here and truncations are reported explicitly.
#
# The first version of this only labelled truncation when the answer text was
# EMPTY. A capped model that narrates before it answers -- every verbose model at
# a small cap -- then recorded a bare "got 3.0, want 42", which reads as a wrong
# answer rather than a run the harness cut off. On the 19-task subset that
# mis-scored 3 of 4 reasoning failures of the dense 27B, all of which were
# truncated mid-solution on the way to the right number. Rows now carry
# `truncated`, and both scores are printed: the strict one (every row counts)
# and the one with capped rows removed.
API = "http://127.0.0.1:11434"
# llama.cpp `llama-server` speaks a different request shape. Same task set, same
# scorer, same greedy decode -- only the engine differs -- is what makes an
# engine comparison (surplus's placement, or llama.cpp against the ollama tag)
# mean anything, and it doubles as an agreement check: identical prompts at
# temperature 0 should give identical answers if the weights and the graph are
# the same. Start it with `llama-server -m <gguf> --port 8080 ...`.
LLAMA_API = "http://127.0.0.1:8080"
NUM_PREDICT = 1536
SEED = 1234

# Two expected values in this set were arithmetically wrong, in OPPOSITE
# directions, so they were scoring correct answers as failures and failures as
# passes (both are pinned by tests/test_eval_gate.py):
#
#   r_bat_ball  was expect=0.10. ball = (1.10 - 1.00) / 2 = 0.05.
#   r_modular   was expect=1.    7**20 % 13 = 3  (7**12 % 13 == 1, 20 % 12 == 8,
#                                and 7**8 % 13 == 3).
#
# `--rescore` replays the recorded responses through the corrected definitions,
# so runs made before this fix are re-scored from their stored text instead of
# being re-generated.

# ---------------------------------------------------------------------------
# task set: every check is mechanical
# ---------------------------------------------------------------------------

def _t(tid, cat, prompt, check, **kw):
    d = {"id": tid, "category": cat, "prompt": prompt, "check": check}
    d.update(kw)
    return d


TASKS = [
    # ---------------- coding: executed against fixed cases ----------------
    _t("c_sum_evens", "coding",
       "Write a Python function `solve(nums)` that returns the sum of the even "
       "numbers in the list `nums`. Reply with a single ```python code block only.",
       "code", cases=[([[1, 2, 3, 4]], 6), ([[]], 0), ([[2, 2, 5]], 4), ([[-2, 3]], -2)]),
    _t("c_reverse_words", "coding",
       "Write a Python function `solve(s)` that reverses the order of words in the "
       "string `s` (words separated by single spaces). Reply with a single "
       "```python code block only.",
       "code", cases=[(["hello world"], "world hello"), (["a"], "a"),
                      (["one two three"], "three two one")]),
    _t("c_fizzbuzz", "coding",
       "Write a Python function `solve(n)` that returns a list of strings for 1..n "
       "where multiples of 3 are 'Fizz', of 5 are 'Buzz', of both are 'FizzBuzz', "
       "else the number as a string. Reply with a single ```python code block only.",
       "code", cases=[([5], ["1", "2", "Fizz", "4", "Buzz"]),
                      ([15], ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8",
                              "Fizz", "Buzz", "11", "Fizz", "13", "14",
                              "FizzBuzz"])]),
    _t("c_is_palindrome", "coding",
       "Write a Python function `solve(s)` returning True if `s` is a palindrome "
       "ignoring case and non-alphanumeric characters, else False. Reply with a "
       "single ```python code block only.",
       "code", cases=[(["A man, a plan, a canal: Panama"], True),
                      (["race a car"], False), ([""], True)]),
    _t("c_binary_search", "coding",
       "Write a Python function `solve(nums, target)` that binary-searches a "
       "sorted list and returns the index of `target`, or -1 if absent. Reply "
       "with a single ```python code block only.",
       "code", cases=[([[1, 3, 5, 7], 5], 2), ([[1, 3, 5, 7], 2], -1),
                      ([[], 1], -1), ([[2], 2], 0)]),
    _t("c_unique_ordered", "coding",
       "Write a Python function `solve(items)` that removes duplicates from "
       "`items` while preserving first-occurrence order. Reply with a single "
       "```python code block only.",
       "code", cases=[([[3, 1, 3, 2, 1]], [3, 1, 2]), ([[]], [])]),
    _t("c_gcd", "coding",
       "Write a Python function `solve(a, b)` returning the greatest common "
       "divisor using Euclid's algorithm. Reply with a single ```python code "
       "block only.",
       "code", cases=[([12, 18], 6), ([7, 13], 1), ([0, 5], 5)]),
    _t("c_flatten", "coding",
       "Write a Python function `solve(nested)` that flattens arbitrarily nested "
       "lists into a single flat list. Reply with a single ```python code block "
       "only.",
       "code", cases=[([[1, [2, [3, 4]], 5]], [1, 2, 3, 4, 5]), ([[]], [])]),
    _t("c_two_sum", "coding",
       "Write a Python function `solve(nums, target)` returning a list of the two "
       "indices whose values sum to target, or [] if none. Reply with a single "
       "```python code block only.",
       "code", cases=[([[2, 7, 11, 15], 9], [0, 1]), ([[3, 3], 6], [0, 1]),
                      ([[1, 2], 99], [])]),
    _t("c_word_count", "coding",
       "Write a Python function `solve(text)` returning a dict mapping each "
       "lowercase word to its count, with words split on whitespace. Reply with a "
       "single ```python code block only.",
       "code", cases=[(["a b a"], {"a": 2, "b": 1}), ([""], {})]),

    # ---------------- reasoning: numeric ----------------
    _t("r_train", "reasoning",
       "A train leaves at 14:20 and travels 210 km at 84 km/h. What time does it "
       "arrive? Reply with the time as HH:MM on the final line.",
       "numeric_time", expect="16:50"),
    _t("r_bat_ball", "reasoning",
       "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the "
       "ball. How much does the ball cost, in currency units? End with "
       "'ANSWER: <value>'.",
       # ball + (ball + 1.00) = 1.10 -> ball = 0.05. This previously read 0.10,
       # which marked the correct answer (0.05) as a failure.
       "number", expect=0.05, tol=1e-6),
    _t("r_work_rate", "reasoning",
       "If 3 machines make 3 widgets in 3 minutes, how many minutes do 100 "
       "machines need to make 100 widgets? End with 'ANSWER: <value>'.",
       "number", expect=3, tol=1e-6),
    _t("r_lily", "reasoning",
       "A patch of lily pads doubles in size every day and covers a lake in 48 "
       "days. On which day was the lake half covered? End with 'ANSWER: <day>'.",
       "number", expect=47, tol=1e-6),
    _t("r_combinatorics", "reasoning",
       "How many distinct ways can the letters of the word 'LEVEL' be arranged? "
       "End with 'ANSWER: <value>'.",
       "number", expect=30, tol=1e-6),
    _t("r_modular", "reasoning",
       "What is 7^20 mod 13? End with 'ANSWER: <value>'.",
       # 7**12 % 13 == 1, 20 % 12 == 8, 7**8 % 13 == 3. This previously read 1,
       # which marked an incorrect answer (1) as a pass.
       "number", expect=3, tol=1e-6),
    _t("r_percent", "reasoning",
       "A price rises 25% then falls 20%. What is the net percentage change? End "
       "with 'ANSWER: <value>' where a rise is positive and a fall negative.",
       "number", expect=0.0, tol=1e-6),
    _t("r_sequence", "reasoning",
       "Find the next term of the sequence 2, 6, 12, 20, 30, ... End with "
       "'ANSWER: <value>'.",
       "number", expect=42, tol=1e-6),

    # ---------------- format: JSON validity + schema ----------------
    _t("f_json_three_actions", "format",
       "Summarise 'The build failed because a dependency was missing.' in one "
       "sentence, then list three action items. Reply with ONLY valid JSON: "
       '{"summary": string, "actions": array of exactly 3 strings}.',
       "json", schema={"summary": str, "actions": list}, list_len={"actions": 3}),
    _t("f_json_tool_call", "format",
       "Emit a tool call as ONLY valid JSON with exactly the keys 'tool' (string) "
       "and 'arguments' (object having key 'path' with a string value). Choose "
       "tool='read_file' and a plausible path.",
       "json", schema={"tool": str, "arguments": dict},
       nested={"arguments": {"path": str}}),
    _t("f_json_no_prose", "format",
       'Reply with ONLY this JSON object and no surrounding text: {"ok": true, '
       '"count": 3}',
       "json", schema={"ok": bool, "count": int}),
    _t("f_json_nested", "format",
       "Reply with ONLY valid JSON describing a config: an object with key "
       "'server' mapping to an object with keys 'host' (string) and 'port' "
       "(integer).",
       "json", schema={"server": dict},
       nested={"server": {"host": str, "port": int}}),
    _t("f_json_array", "format",
       "Reply with ONLY a JSON array of exactly two objects, each having keys "
       "'name' (string) and 'score' (number).",
       "json", json_root=list, array_len=2, item_keys=["name", "score"]),
    _t("f_json_nulls", "format",
       'Reply with ONLY valid JSON: an object with key "items" whose value is an '
       'empty array, and key "total" whose value is 0.',
       "json", schema={"items": list, "total": int}),

    # ---------------- instruction following: explicit constraints ----------------
    _t("i_exact_three_bullets", "instruction",
       "List exactly 3 benefits of unit testing. Use one line per benefit, each "
       "starting with '- '. Output nothing else.",
       "bullets", n=3),
    _t("i_only_word", "instruction",
       "Is the number 17 prime? Answer with only the single word yes or no, "
       "lowercase, with no punctuation and nothing else.",
       "exact", expect="yes"),
    _t("i_uppercase", "instruction",
       "Reply with the word 'confirmed' in UPPERCASE and nothing else.",
       "exact", expect="CONFIRMED"),
    _t("i_five_words", "instruction",
       "Describe the colour blue using exactly 5 words, no punctuation, nothing "
       "else.",
       "word_count", n=5),
    _t("i_lowercase_start", "instruction",
       "Write one sentence about compilers that begins with the lowercase letter "
       "'c'. Output only that sentence.",
       "starts_with", ch="c"),
    _t("i_no_letter_e", "instruction",
       "Write a short sentence about binary trees that does NOT contain the "
       "letter 'e' anywhere. Output only the sentence.",
       "forbid_char", ch="e"),
]


# ---------------------------------------------------------------------------
# extraction helpers
# ---------------------------------------------------------------------------

def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(^|\n)(def solve\(.*)", text, re.S)
    return m.group(2).strip() if m else ""


def extract_json(text: str) -> tuple[object | None, str]:
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    body = m.group(1).strip() if m else text.strip()
    # try whole body first, then scan for the first balanced {...} or [...]
    for candidate in (body, _first_balanced(body, "{", "}"), _first_balanced(body, "[", "]")):
        if not candidate:
            continue
        try:
            return json.loads(candidate), ""
        except json.JSONDecodeError as e:
            last = str(e)
    return None, "no parseable JSON found"


def _first_balanced(s: str, open_ch: str, close_ch: str) -> str:
    start = s.find(open_ch)
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        start = s.find(open_ch, start + 1)
    return ""


def extract_number(text: str):
    m = re.findall(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text, re.I)
    if m:
        return float(m[-1])
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def extract_time(text: str):
    m = re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    return f"{int(m[-1][0]):02d}:{m[-1][1]}" if m else None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def run_code(code: str, cases: list, timeout: float = 10.0) -> tuple[bool, str]:
    """Execute model-generated code against cases in a subprocess.

    The candidate source is concatenated at column 0 rather than interpolated
    into an indented template: a template plus textwrap.dedent collapses the
    function body's indentation and yields IndentationError on every correct
    solution, which silently makes the whole coding category fail.
    """
    driver = (
        "import json, sys\n"
        f"{code}\n"
        "cases = json.loads(sys.argv[1])\n"
        "results = []\n"
        "for args, expected in cases:\n"
        "    try:\n"
        "        got = solve(*args)\n"
        "    except Exception as e:\n"
        "        results.append(['error', type(e).__name__ + ': ' + str(e)])\n"
        "        continue\n"
        "    results.append(['ok', got == expected])\n"
        "print(json.dumps(results))\n"
    )
    with tempfile.TemporaryDirectory(prefix="moeme_eval_") as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w") as fh:
            fh.write(driver)
        try:
            p = subprocess.run(
                [sys.executable, path, json.dumps(cases)],
                capture_output=True, text=True, timeout=timeout, cwd=d,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if p.returncode != 0:
        return False, f"runtime error: {(p.stderr or '').strip().splitlines()[-1][:120]}"
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return False, "could not parse driver output"
    bad = [r for r in res if r[0] != "ok" or r[1] is not True]
    if bad:
        # Surface the exception type: "1/1 cases failed" alone is not enough to
        # tell a wrong answer from a crash.
        errs = [r[1] for r in res if r[0] == "error"]
        if errs:
            return False, f"{len(bad)}/{len(res)} cases failed (e.g. {str(errs[0])[:100]})"
        return False, f"{len(bad)}/{len(res)} cases failed"
    return True, f"{len(res)}/{len(res)} cases passed"


def check_json(obj, task) -> tuple[bool, str]:
    if obj is None:
        return False, "no JSON"
    if task.get("json_root") is list and not isinstance(obj, list):
        return False, "root is not an array"
    if task.get("json_root") is list:
        if "array_len" in task and len(obj) != task["array_len"]:
            return False, f"array length {len(obj)} != {task['array_len']}"
        for item in obj:
            for k in task.get("item_keys", []):
                if not isinstance(item, dict) or k not in item:
                    return False, f"item missing key {k!r}"
        return True, "ok"
    def type_ok(value, typ) -> bool:
        # bool subclasses int in Python, so a JSON `true` would otherwise
        # validate against an integer field. Reject that explicitly.
        if typ in (int, float) and isinstance(value, bool):
            return False
        return isinstance(value, typ)

    for key, typ in task.get("schema", {}).items():
        if key not in obj:
            return False, f"missing key {key!r}"
        if not type_ok(obj[key], typ):
            return False, f"{key!r} is {type(obj[key]).__name__}, want {typ.__name__}"
    for parent, sub in task.get("nested", {}).items():
        for k, typ in sub.items():
            if k not in obj.get(parent, {}) or not type_ok(obj[parent][k], typ):
                return False, f"{parent}.{k} wrong type/missing"
    for key, n in task.get("list_len", {}).items():
        if len(obj.get(key, [])) != n:
            return False, f"{key!r} has {len(obj.get(key, []))} items, want {n}"
    return True, "ok"


def evaluate(task, text: str, execute: bool) -> tuple[bool, str]:
    c = task["check"]
    if c == "code":
        code = extract_code(text)
        if "def solve" not in code:
            return False, "no solve() function found"
        if not execute:
            return False, "execution disabled (--no-execute)"
        return run_code(code, task["cases"])
    if c == "number":
        v = extract_number(text)
        if v is None:
            return False, "no number found"
        return (abs(v - task["expect"]) <= task["tol"], f"got {v}, want {task['expect']}")
    if c == "numeric_time":
        v = extract_time(text)
        return (v == task["expect"], f"got {v}, want {task['expect']}")
    if c == "json":
        return check_json(extract_json(text)[0], task)
    if c == "exact":
        got = text.strip()
        return (got == task["expect"], f"got {got[:40]!r}")
    if c == "bullets":
        lines = [l for l in text.strip().splitlines() if l.strip().startswith("- ")]
        return (len(lines) == task["n"], f"{len(lines)} bullets, want {task['n']}")
    if c == "word_count":
        n = len(text.split())
        return (n == task["n"], f"{n} words, want {task['n']}")
    if c == "starts_with":
        s = text.strip()
        return (s[:1] == task["ch"], f"starts with {s[:1]!r}")
    if c == "forbid_char":
        present = task["ch"].lower() in text.lower()
        return (not present,
                f"contains {task['ch']!r}" if present else f"no {task['ch']!r}")
    raise ValueError(f"unknown check {c}")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{API}/api/version", timeout=5) as r:
            json.load(r)
        return True
    except Exception:
        return False


def ensure_server() -> bool:
    if server_up():
        return True
    print("starting ollama serve ...", flush=True)
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(40):
        time.sleep(0.5)
        if server_up():
            return True
    return False


def mark_truncation(detail: str, text: str, done_reason,
                    eval_count) -> tuple[str, bool]:
    """Label a row the token cap cut off, so it is not read as a wrong answer.

    `detail` is the scorer's verdict and is kept: a truncated row still holds no
    usable answer, so it stays a failure in the strict score. The marker and the
    returned flag are what let a reader (and the summary) separate "the model
    could not do this" from "the harness stopped it mid-sentence".
    """
    if done_reason != "length":
        return detail, False
    if not text.strip():
        return (f"TRUNCATED during reasoning ({eval_count} tokens, empty answer; "
                f"raise --num-predict or use --think off)"), True
    return f"{detail} [truncated at {eval_count} tokens]", True


def row_is_truncated(row: dict) -> bool:
    """True for rows that hit the cap, including reports written before the
    `truncated` field existed (they are marked only in `detail`)."""
    return bool(row.get("truncated")) or "TRUNCATED" in (row.get("detail") or "")


def ask_llama(prompt: str, num_predict: int = NUM_PREDICT, *,
              api: str = LLAMA_API, timeout: float = 3600.0,
              template_kwargs: dict | None = None) -> tuple[str, dict]:
    """One greedy completion from a llama.cpp `llama-server`.

    Via /v1/chat/completions, not /completion, and that is the whole point.
    /completion is a raw continuation: it applies no chat template, so a
    thinking model starts from a bare instruction and writes its `<think>` block
    into the answer. Measured here, that alone cost a placement's score 13/19
    against 17/19 on identical weights and prompts -- a template default
    reported as engine quality. The chat route applies the model's template, so
    `chat_template_kwargs={"enable_thinking": false}` reproduces what the ollama
    runs get from `think: false`, and a bare `yes` comes back instead of
    `<think>...why 17 is prime...</think>\nyes`.

    Same returned shape as `ask`, so extraction, scoring, truncation labelling
    and checkpointing are shared rather than reimplemented per engine. llama.cpp
    reports a cap hit as finish_reason "length" (or `stopped_limit`) where ollama
    says `done_reason == "length"`; both map onto the one condition
    `mark_truncation` understands.
    """
    payload = {"messages": [{"role": "user", "content": prompt}],
               "n_predict": num_predict, "temperature": 0.0, "seed": SEED,
               "stream": False}
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    req = urllib.request.Request(f"{api}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    # A build that ignores the kwarg still answers, just not in `content`.
    # Flagging it beats either scoring an empty answer as a wrong one or
    # quietly scoring the reasoning as the answer.
    reasoning = message.get("reasoning_content") or ""
    capped = (choice.get("finish_reason") == "length"
              or bool(resp.get("stopped_limit")))
    timings = resp.get("timings") or {}
    return message.get("content") or "", {
        "eval_count": timings.get("predicted_n"),
        "decode_tok_s": timings.get("predicted_per_second"),
        "prompt_tok_s": timings.get("prompt_per_second"),
        "done_reason": "length" if capped else "stop",
        "thinking_chars": len(reasoning),
        "reasoning_only": bool(reasoning) and not (message.get("content") or ""),
        "num_predict": num_predict,
    }


def ask_backend(backend: str, model: str, prompt: str, num_predict: int, *,
                think: str, api: str, keep_alive: str, timeout: float,
                template_kwargs: dict | None = None) -> tuple[str, dict]:
    if backend == "llama":
        return ask_llama(prompt, num_predict, api=api, timeout=timeout,
                         template_kwargs=template_kwargs)
    return ask(model, prompt, num_predict, think=think, api=api,
               keep_alive=keep_alive, timeout=timeout)


def ask_llama_ready(api: str) -> bool:
    """llama-server answers /health with 503 while it is still loading weights,
    which is not the same thing as a wrong URL, but either way the run cannot
    start yet."""
    try:
        with urllib.request.urlopen(f"{api}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ask(model: str, prompt: str, num_predict: int = NUM_PREDICT, *,
        think: str = "auto", api: str = API, keep_alive: str = "30s",
        timeout: float = 1800.0) -> tuple[str, dict]:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "keep_alive": keep_alive,
               "options": {"temperature": 0.0, "seed": SEED,
                           "num_predict": num_predict}}
    # Only send `think` when asked for a specific behaviour. Leaving the key out
    # keeps the model's own default, which is what --think auto means, and it
    # avoids a hard error on builds that predate the parameter.
    if think == "off":
        payload["think"] = False
    elif think == "on":
        payload["think"] = True
    req = urllib.request.Request(f"{api}/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    ec, ed = resp.get("eval_count") or 0, resp.get("eval_duration") or 0
    pc, pd = resp.get("prompt_eval_count") or 0, resp.get("prompt_eval_duration") or 0
    return resp.get("response", ""), {
        "eval_count": ec,
        "decode_tok_s": (ec / (ed / 1e9)) if ed else None,
        # Prompt-processing rate on the same terms as the llama.cpp backend, so
        # prefill can be compared between engines instead of only between a run
        # and a separate 32-token benchmark.
        "prompt_tok_s": (pc / (pd / 1e9)) if pd else None,
        "done_reason": resp.get("done_reason"),
        "thinking_chars": len(resp.get("thinking") or ""),
        "num_predict": num_predict,
    }


def rescore(args) -> int:
    """Re-run the current task definitions over the RECORDED responses.

    Model answers are stored verbatim, so a corrected expectation (or a scorer
    fix) can be applied to an existing multi-hour run without re-issuing a single
    prompt. Rows recorded before responses were stored are left exactly as they
    were and counted in the output, never silently turned into failures.
    """
    if not os.path.exists(args.out):
        raise SystemExit(f"--rescore needs an existing --out, and {args.out} does not exist")
    with open(args.out) as fh:
        report = json.load(fh)
    if "task_ids" not in report:
        raise SystemExit(f"{args.out} predates --tasks, so there is no task_ids "
                         f"list to re-score against")
    by_id = {t["id"]: t for t in TASKS}
    changed = unchanged = skipped = 0
    for model, m in sorted(report.get("models", {}).items()):
        by_cat: dict[str, list] = {}
        for r in m.get("tasks", []):
            response, task = r.get("response"), by_id.get(r["id"])
            if response is None or task is None:
                skipped += 1
            else:
                ok, detail = evaluate(task, response, execute=not args.no_execute)
                # Truncation is a property of the recorded run, not of the
                # definition, so it is re-derived from the stored done_reason.
                # Reviews a report the harness wrote before it tracked this.
                detail, r["truncated"] = mark_truncation(
                    detail, response, r.get("done_reason"),
                    r.get("eval_count") or r.get("response_chars"))
                if ok != r["passed"]:
                    changed += 1
                    print(f"  {model} {r['id']}: {r['passed']} -> {ok}  ({detail})")
                else:
                    unchanged += 1
                r["passed"], r["detail"] = ok, detail
            by_cat.setdefault(r["category"], []).append(r["passed"])
        rows = m.get("tasks", [])
        passed = sum(1 for r in rows if r["passed"])
        m["summary"] = {c: {"passed": sum(v), "total": len(v),
                            "rate": sum(v) / len(v),
                            "truncated": sum(1 for r in rows
                                             if r["category"] == c
                                             and row_is_truncated(r))}
                        for c, v in sorted(by_cat.items())}
        m["overall"] = {"passed": passed, "total": len(rows),
                        "rate": (passed / len(rows)) if rows else 0.0}
        scored_rows = [r for r in rows if not row_is_truncated(r)]
        scored_pass = sum(1 for r in scored_rows if r["passed"])
        m["overall_untruncated"] = {
            "passed": scored_pass, "total": len(scored_rows),
            "rate": (scored_pass / len(scored_rows)) if scored_rows else 0.0}
        m["truncated_tasks"] = sum(1 for r in rows if row_is_truncated(r))
        m["harness_errors"] = sum(1 for r in rows
                                  if "harness error" in (r.get("detail") or ""))
        tail = (f", {m['truncated_tasks']} truncated ("
                f"{scored_pass}/{len(scored_rows)} excluding capped rows)"
                if m["truncated_tasks"] else "")
        print(f"  {model}: now {passed}/{len(rows)}{tail}")
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, args.out)
    print(f"re-scored {changed + unchanged} task(s): {changed} changed, "
          f"{unchanged} unchanged, {skipped} with no stored response "
          f"(left as recorded) -> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--no-execute", action="store_true",
                    help="do not run model-generated code (coding tasks auto-fail)")
    ap.add_argument("--only", default=None, help="comma-separated categories")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task ids -- a fixed subset, for a model "
                         "too slow to run the full set in one sitting")
    ap.add_argument("--num-predict", type=int, default=NUM_PREDICT)
    ap.add_argument("--think", choices=("auto", "off", "on"), default="auto",
                    help="thinking models: 'off' suppresses the separate reasoning "
                         "field, which is what makes a small --num-predict usable. "
                         "'auto' sends nothing and keeps the model default")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help="JSON passed to a llama.cpp server as "
                         "chat_template_kwargs, e.g. '{\"enable_thinking\": false}'. "
                         "Use it to match --think off: the server's template "
                         "otherwise leaves a thinking model's <think> block in "
                         "the answer, which fails strict-instruction tasks")
    ap.add_argument("--backend", choices=("ollama", "llama"), default="ollama",
                    help="ollama /api/generate (default) or a llama.cpp "
                         "llama-server POST /completion, for comparing engines on "
                         "the identical task set")
    ap.add_argument("--api", default=None, help="base URL of the backend")
    ap.add_argument("--keep-alive", default="30s",
                    help="how long the model stays resident between tasks "
                         "(reloading the dense 27B costs ~51 s)")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-request HTTP timeout in seconds")
    ap.add_argument("--resume", action="store_true",
                    help="reuse tasks already recorded in --out and continue from there")
    ap.add_argument("--pidfile", default=None,
                    help="write this process's PID here at startup, so a memory "
                         "watchdog (tools/mem_watchdog.py) can guard this run. "
                         "Without it the guard has to be aimed by hand at a "
                         "wrapper process, which is how a guard ends up armed on "
                         "the wrong PID.")
    ap.add_argument("--rescore", action="store_true",
                    help="do not run a model: re-evaluate the responses already "
                         "stored in --out against the current task definitions")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args)
    if not args.model:
        raise SystemExit("give at least one --model")
    if args.pidfile:
        os.makedirs(os.path.dirname(args.pidfile) or ".", exist_ok=True)
        with open(args.pidfile, "w") as fh:
            fh.write(str(os.getpid()))
        print(f"NOTE: pid {os.getpid()} written to {args.pidfile}", flush=True)
    api = args.api or (LLAMA_API if args.backend == "llama" else API)
    template_kwargs = None
    if args.chat_template_kwargs:
        if args.backend != "llama":
            raise SystemExit("--chat-template-kwargs only applies to --backend llama")
        try:
            template_kwargs = json.loads(args.chat_template_kwargs)
        except ValueError as e:
            raise SystemExit(f"--chat-template-kwargs is not valid JSON: {e}")
        if not isinstance(template_kwargs, dict):
            raise SystemExit("--chat-template-kwargs must be a JSON object")
        print(f"NOTE: sending chat_template_kwargs={template_kwargs}\n")
    if args.backend == "llama":
        # Do not spawn it: the whole point is to control the placement the
        # server was launched with, and a silent default launch would hide it.
        if not ask_llama_ready(api):
            raise SystemExit(
                f"no llama.cpp server at {api} (or it is still loading). Start "
                f"one, then rerun --backend llama --api {api}:\n"
                f"  llama-server -m <gguf> --port 8080 -ngl 42 -ncmoe 24 -c 4096")
        print(f"NOTE: measuring llama.cpp at {api}; the placement is whatever "
              f"that server was launched with.\n")
    elif not ensure_server():
        raise SystemExit("could not reach or start the ollama server")

    tasks = TASKS
    if args.only:
        cats = {c.strip() for c in args.only.split(",")}
        tasks = [t for t in tasks if t["category"] in cats]
    if args.tasks:
        want = {t.strip() for t in args.tasks.split(",") if t.strip()}
        unknown = sorted(want - {t["id"] for t in TASKS})
        if unknown:
            raise SystemExit(f"unknown task id(s): {', '.join(unknown)}")
        # Filter in canonical TASKS order so a subset run stays comparable with
        # the full run task by task, and so --resume sees the same id list.
        tasks = [t for t in tasks if t["id"] in want]
    if not tasks:
        raise SystemExit("no tasks selected")

    if not args.no_execute:
        print("NOTE: coding tasks execute model-generated Python in a subprocess "
              "with a 10 s timeout.\n")

    report = {"tasks_total": len(tasks), "models": {},
              "task_ids": [t["id"] for t in tasks],
              "backend": args.backend, "api": api,
              "chat_template_kwargs": template_kwargs,
              "think": args.think, "num_predict": args.num_predict}
    if args.resume and os.path.exists(args.out):
        try:
            with open(args.out) as fh:
                prev = json.load(fh)
        except (OSError, ValueError) as e:
            print(f"NOTE: cannot read {args.out} ({e}); starting fresh", flush=True)
            prev = {}
        if (prev.get("task_ids") == report["task_ids"]
                and prev.get("num_predict") == args.num_predict
                and prev.get("think") == args.think
                # Reports written before --backend existed are ollama reports.
                and prev.get("backend", "ollama") == args.backend):
            report["models"] = prev.get("models", {})
            have = ", ".join(f"{m}: {len(v.get('tasks', []))}"
                             for m, v in report["models"].items()) or "nothing"
            print(f"resuming {args.out}: {have}", flush=True)
        elif prev:
            print(f"NOTE: {args.out} is a different run (task_ids / num_predict / "
                  f"think / backend differ); starting fresh", flush=True)

    def save() -> None:
        """Checkpoint after every task.

        A 30-task run on the dense 27B takes hours. Losing it to one dropped
        connection -- the exact failure the CPU-only control hit -- is not
        acceptable, so the file on disk is always a valid partial report.
        """
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(report, fh, indent=2)
        os.replace(tmp, args.out)

    for model in args.model:
        prev_run = report["models"].get(model) or {}
        results = {r["id"]: r for r in prev_run.get("tasks", [])}
        print(f"=== {model} ({len(tasks)} tasks, {len(results)} already "
              f"recorded) ===", flush=True)
        t0 = time.perf_counter()
        consecutive_errors = 0
        for i, task in enumerate(tasks, 1):
            if task["id"] in results:
                print(f"  [{i:>2}/{len(tasks)}] {task['id']:<22} SKIP  already recorded",
                      flush=True)
                continue
            try:
                text, meta = ask_backend(args.backend, model, task["prompt"],
                                         args.num_predict, think=args.think,
                                         api=api, keep_alive=args.keep_alive,
                                         timeout=args.timeout,
                                         template_kwargs=template_kwargs)
                ok, detail = evaluate(task, text, execute=not args.no_execute)
                if not ok and meta.get("reasoning_only"):
                    detail = (f"{detail} [the engine answered only in its reasoning "
                              f"field -- thinking was not suppressed for this "
                              f"request, so there is no answer to score]")
                detail, truncated = mark_truncation(
                    detail, text, meta.get("done_reason"), meta.get("eval_count"))
                consecutive_errors = 0
            except Exception as e:
                text, meta, ok, detail = "", {}, False, f"harness error: {e}"
                truncated = False
                consecutive_errors += 1
            results[task["id"]] = {
                "id": task["id"], "category": task["category"],
                "passed": ok, "detail": detail, "truncated": truncated,
                "response": text[:1200],
                "eval_count": meta.get("eval_count"),
                "decode_tok_s": meta.get("decode_tok_s"),
                "prompt_tok_s": meta.get("prompt_tok_s"),
                "done_reason": meta.get("done_reason"),
                "thinking_chars": meta.get("thinking_chars"),
                "response_chars": len(text)}
            print(f"  [{i:>2}/{len(tasks)}] {task['id']:<22} "
                  f"{'PASS' if ok else 'FAIL':<4} {detail[:60]}", flush=True)
            report["models"][model] = {
                "overall": {"passed": sum(1 for r in results.values() if r["passed"]),
                            "total": len(results)},
                "tasks": [results[t["id"]] for t in tasks if t["id"] in results],
            }
            save()
            if consecutive_errors >= 3:
                print(f"  ABORT {model}: {consecutive_errors} consecutive harness "
                      f"errors -- the backend is probably gone. Partial results are "
                      f"checkpointed in {args.out}; rerun with --resume.", flush=True)
                break

        rows = [results[t["id"]] for t in tasks if t["id"] in results]
        by_cat: dict[str, list] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r["passed"])
        summary = {c: {"passed": sum(v), "total": len(v),
                       "rate": sum(v) / len(v),
                       "truncated": sum(1 for r in rows if r["category"] == c
                                        and row_is_truncated(r))}
                   for c, v in sorted(by_cat.items())}
        total_pass = sum(sum(v) for v in by_cat.values())
        truncated = sum(1 for r in rows if row_is_truncated(r))
        errors = sum(1 for r in rows if "harness error" in (r.get("detail") or ""))
        scored_rows = [r for r in rows if not row_is_truncated(r)]
        scored_pass = sum(1 for r in scored_rows if r["passed"])
        report["models"][model] = {
            "summary": summary,
            "overall": {"passed": total_pass, "total": len(rows),
                        "rate": (total_pass / len(rows)) if rows else 0.0},
            "overall_untruncated": {
                "passed": scored_pass, "total": len(scored_rows),
                "rate": (scored_pass / len(scored_rows)) if scored_rows else 0.0},
            "tasks_run": len(rows),
            "tasks_selected": len(tasks),
            "truncated_tasks": truncated,
            "harness_errors": errors,
            "num_predict": args.num_predict,
            "think": args.think,
            "wall_s": (prev_run.get("wall_s") or 0.0) + (time.perf_counter() - t0),
            "tasks": rows,
        }
        save()
        print(f"  -- {model}: {total_pass}/{len(rows)} = "
              f"{100 * total_pass / max(len(rows), 1):.1f}% overall")
        for c, s in summary.items():
            note = f"  ({s['truncated']} truncated)" if s["truncated"] else ""
            print(f"     {c:<12} {s['passed']}/{s['total']} = {100*s['rate']:.0f}%{note}")
        if truncated:
            print(f"     WARNING: {truncated}/{len(rows)} task(s) hit the "
                  f"num_predict={args.num_predict} cap; a harness setting, not a "
                  f"model result")
            print(f"     excluding capped rows: {scored_pass}/{len(scored_rows)} = "
                  f"{100 * scored_pass / max(len(scored_rows), 1):.1f}%")
        if errors:
            print(f"     WARNING: {errors} task(s) hit a harness error (backend, "
                  f"not the model)")
        print()

    save()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
