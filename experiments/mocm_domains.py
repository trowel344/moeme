#!/usr/bin/env python3
"""moeme domain suite: fixed offline eval sets + executable spotlight graders.

Eval sources (fetched once, deterministic forever):
  .cache/moeme/eval_src/{code,math,chat,wiki,web}.txt  (~300k chars each)
  .cache/moeme/eval_src/mbpp_test.json   (500 coding tasks + exec tests)
  .cache/moeme/eval_src/gsm8k_test.json  (1319 reasoning tasks + answers)

Spotlights rotate per generation so no one overfits a single bench:
  even gens -> coding (mbpp exec pass rate), odd gens -> reasoning (gsm8k grade).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", ".cache", "moeme", "eval_src")
sys.path.insert(0, os.path.join(HERE, "..", "src"))

DOMAINS = ["code", "math", "chat", "wiki", "web"]


def domain_text(name: str) -> str:
    with open(os.path.join(SRC, f"{name}.txt")) as fh:
        return fh.read()


def domain_blocks(tokenizer, name: str, seq_len=512, n_blocks=4):
    """Fixed eval blocks for a domain (deterministic: head of fixed file)."""
    sys.path.insert(0, HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_d_poc", os.path.join(HERE, "upcycle_poc.py"))
    poc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = poc
    spec.loader.exec_module(poc)
    ids = tokenizer(domain_text(name), return_tensors="pt").input_ids[0]
    return poc.make_blocks(ids, seq_len)[:n_blocks]


@torch.no_grad()
def domain_ppls(model, blocks_per_domain, device, amp_dtype):
    """Val ppl per domain. Blocks: {domain: [(x, y), ...]}."""
    import math
    out = {}
    model.eval()
    for dom, blocks in blocks_per_domain.items():
        total, ntok = 0.0, 0
        for x, y in blocks:
            x, y = x.to(device), y.to(device)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = model(x, labels=y).loss
            else:
                loss = model(x, labels=y).loss
            total += loss.item() * y.numel()
            ntok += y.numel()
        out[dom] = math.exp(total / max(ntok, 1))
    return out


# ---------------------------------------------------------------------------
# coding spotlight: mbpp, executed (not string-matched)
# ---------------------------------------------------------------------------

def mbpp_tasks():
    with open(os.path.join(SRC, "mbpp_test.json")) as fh:
        return json.load(fh)


def mbpp_subset(k=16, seed=7):
    import random
    tasks = mbpp_tasks()
    rng = random.Random(seed)
    idx = rng.sample(range(len(tasks)), k)
    return [tasks[i] for i in idx]


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def mbpp_pass(task, code: str, timeout_s=15) -> bool:
    """Run setup + candidate + tests in a subprocess. True iff exit 0."""
    body = (task.get("test_setup_code") or "") + "\n" + code + "\n" \
        + "\n".join(task.get("test_list", [])) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        r = subprocess.run(["timeout", str(timeout_s), sys.executable, path],
                           capture_output=True, timeout=timeout_s + 5)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def mbpp_func_name(task) -> str:
    m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", task.get("test_list", [""])[0])
    return m.group(1) if m else "solution"


MBPP_FEW = (
    "Task: Write a function named add that returns the sum of two numbers.\n"
    "Your code:\n```python\ndef add(a, b):\n    return a + b\n```\n\n"
)


def mbpp_prompt(task) -> str:
    return (MBPP_FEW +
            f"Task: {task['text']}\n"
            f"Write a Python function named {mbpp_func_name(task)}. Output code only.\n"
            f"Your code:\n")


# ---------------------------------------------------------------------------
# reasoning spotlight: gsm8k, numeric grade
# ---------------------------------------------------------------------------

def gsm8k_tasks():
    with open(os.path.join(SRC, "gsm8k_test.json")) as fh:
        return json.load(fh)


def gsm8k_subset(k=50, seed=7):
    import random
    tasks = gsm8k_tasks()
    rng = random.Random(seed)
    return [tasks[i] for i in rng.sample(range(len(tasks)), k)]


GSM8K_FEW = (
    "Q: There are 15 trees in the grove. Workers plant 5 more. How many trees now?\n"
    "A: 20\n\n"
    "Q: Janet has 3 apples and buys 4 more. How many apples?\n"
    "A: 7\n\n"
)


def gsm8k_prompt(task) -> str:
    return GSM8K_FEW + f"Q: {task['question']}\nA:"


def gsm8k_grade(task, text: str) -> bool:
    gold = task["answer"].split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text.replace(",", ""))
    if not nums:
        return False
    return nums[-1].replace(",", "") == gold
