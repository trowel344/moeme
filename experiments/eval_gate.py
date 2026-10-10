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
API = "http://127.0.0.1:11434"
NUM_PREDICT = 1536
SEED = 1234

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
       "number", expect=0.10, tol=1e-6),
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
       "number", expect=1, tol=1e-6),
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


def ask(model: str, prompt: str, num_predict: int = NUM_PREDICT) -> tuple[str, dict]:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.0, "seed": SEED,
                           "num_predict": num_predict}}
    req = urllib.request.Request(f"{API}/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.load(r)
    ec, ed = resp.get("eval_count") or 0, resp.get("eval_duration") or 0
    return resp.get("response", ""), {
        "eval_count": ec,
        "decode_tok_s": (ec / (ed / 1e9)) if ed else None,
        "done_reason": resp.get("done_reason"),
        "thinking_chars": len(resp.get("thinking") or ""),
        "num_predict": num_predict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--no-execute", action="store_true",
                    help="do not run model-generated code (coding tasks auto-fail)")
    ap.add_argument("--only", default=None, help="comma-separated categories")
    ap.add_argument("--num-predict", type=int, default=NUM_PREDICT)
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("give at least one --model")
    if not ensure_server():
        raise SystemExit("could not reach or start the ollama server")

    tasks = TASKS
    if args.only:
        cats = {c.strip() for c in args.only.split(",")}
        tasks = [t for t in tasks if t["category"] in cats]

    if not args.no_execute:
        print("NOTE: coding tasks execute model-generated Python in a subprocess "
              "with a 10 s timeout.\n")

    report = {"tasks_total": len(tasks), "models": {}}
    for model in args.model:
        print(f"=== {model} ({len(tasks)} tasks) ===", flush=True)
        rows, by_cat = [], {}
        t0 = time.perf_counter()
        truncated = 0
        for i, task in enumerate(tasks, 1):
            try:
                text, meta = ask(model, task["prompt"], args.num_predict)
                ok, detail = evaluate(task, text, execute=not args.no_execute)
                # An empty answer from a token-capped run is a HARNESS problem,
                # not evidence about the model. Say so instead of reporting a
                # bare "no number found".
                if not ok and not text.strip() and meta.get("done_reason") == "length":
                    detail = (f"TRUNCATED during reasoning ({meta['eval_count']} tokens, "
                              f"empty answer; raise --num-predict)")
                    truncated += 1
            except Exception as e:
                text, meta, ok, detail = "", {}, False, f"harness error: {e}"
            rows.append({"id": task["id"], "category": task["category"],
                         "passed": ok, "detail": detail,
                         "decode_tok_s": meta.get("decode_tok_s"),
                         "done_reason": meta.get("done_reason"),
                         "thinking_chars": meta.get("thinking_chars"),
                         "response_chars": len(text)})
            by_cat.setdefault(task["category"], []).append(ok)
            print(f"  [{i:>2}/{len(tasks)}] {task['id']:<22} "
                  f"{'PASS' if ok else 'FAIL':<4} {detail[:60]}", flush=True)

        summary = {c: {"passed": sum(v), "total": len(v),
                       "rate": sum(v) / len(v)} for c, v in sorted(by_cat.items())}
        total_pass = sum(sum(v) for v in by_cat.values())
        report["models"][model] = {
            "summary": summary,
            "overall": {"passed": total_pass, "total": len(tasks),
                        "rate": total_pass / len(tasks)},
            "truncated_tasks": truncated,
            "num_predict": args.num_predict,
            "wall_s": time.perf_counter() - t0,
            "tasks": rows,
        }
        print(f"  -- {model}: {total_pass}/{len(tasks)} = "
              f"{100 * total_pass / len(tasks):.1f}% overall")
        for c, s in summary.items():
            print(f"     {c:<12} {s['passed']}/{s['total']} = {100*s['rate']:.0f}%")
        if truncated:
            print(f"     WARNING: {truncated}/{len(tasks)} tasks truncated during "
                  f"reasoning at num_predict={args.num_predict}; those failures "
                  f"are a harness setting, not a model result")
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
