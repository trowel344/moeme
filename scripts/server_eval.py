#!/usr/bin/env python3
"""Run deterministic quality and streaming-latency gates on a resident server."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from moeme.ledger import ExperimentLedger

CASES = (
    {
        "name": "factual",
        "prompt": "What is the capital of France? Answer with only the city.",
        "required": ("paris",),
        "max_tokens": 16,
    },
    {
        "name": "arithmetic",
        "prompt": "Calculate 17 multiplied by 23. Answer with only the integer.",
        "required": ("391",),
        "max_tokens": 16,
    },
    {
        "name": "code_generation",
        "prompt": (
            "Write only Python code for a function add(a, b) that returns their sum. "
            "Do not use a code fence."
        ),
        "required": ("def add", "return"),
        "max_tokens": 64,
    },
    {
        "name": "structured_instruction",
        "prompt": (
            "Output exactly this JSON object with no Markdown or extra text: "
            '{"status":"ok","count":3}'
        ),
        "exact": '{"status":"ok","count":3}',
        "max_tokens": 32,
    },
    {
        "name": "code_repair",
        "prompt": (
            "Repair this Python function using the standard max(low, min(high, x)) clamp "
            "expression. Output only the corrected function, with no code fence: "
            "def clamp(x, low, high): return min(low, max(high, x))"
        ),
        "required": ("def clamp", "max(low", "min(high"),
        "max_tokens": 64,
    },
    {
        "name": "summarization",
        "prompt": (
            "Summarize this in one sentence: The backup began at 02:00. A disk filled at "
            "02:17, so the job stopped before uploading anything. The operator freed 40 GB "
            "and restarted it at 02:31. The second run completed at 03:05."
        ),
        "required": ("disk", "02:17", "03:05"),
        "max_tokens": 80,
    },
    {
        "name": "context_retrieval",
        "performance_class": "long_context",
        "prompt": (
            "Read the records and answer with only the project code associated with Rowan. "
            + " ".join(
                f"Record {index}: owner=Person{index}, project=P{index:04d}."
                for index in range(100)
            )
            + " Record 100: owner=Rowan, project=ZX-417. "
            + " ".join(
                f"Record {index}: owner=Person{index}, project=Q{index:04d}."
                for index in range(101, 200)
            )
        ),
        "exact": "ZX-417",
        "max_tokens": 16,
    },
    {
        "name": "long_decode",
        "prompt": (
            "Write the integers 1 through 30 in order, separated by one space. Output nothing else."
        ),
        "required": ("1 2 3", "28 29 30"),
        "max_tokens": 96,
    },
)


def case_passes(case: dict[str, Any], content: str) -> bool:
    normalized = content.strip()
    if "exact" in case:
        return normalized == case["exact"]
    folded = normalized.casefold()
    return all(required.casefold() in folded for required in case.get("required", ()))


def metric_capability(result: dict[str, Any]) -> str:
    return f"{result['name']}-repetition-{result['repetition']}"


def performance_passes(
    *,
    median_ttft: float,
    maximum_ttft: float,
    maximum_long_context_ttft: float,
    median_decode_rate: float,
    minimum_long_decode_rate: float,
    max_median_ttft: float,
    max_ttft: float,
    max_long_context_ttft: float,
    min_median_decode_rate: float,
    min_long_decode_rate: float,
) -> bool:
    return (
        median_ttft <= max_median_ttft
        and maximum_ttft <= max_ttft
        and maximum_long_context_ttft <= max_long_context_ttft
        and median_decode_rate >= min_median_decode_rate
        and minimum_long_decode_rate >= min_long_decode_rate
    )


def stream_chat(url: str, case: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": "moeme",
        "messages": [{"role": "user", "content": case["prompt"]}],
        "temperature": 0,
        "max_tokens": case["max_tokens"],
        "stream": True,
        # Qwen's template does not honor reasoning_budget=0 by itself.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_token = None
    content = ""
    reasoning = ""
    final: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            event = json.loads(body)
            final = event
            delta = event["choices"][0].get("delta", {})
            token = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if token and first_token is None:
                first_token = time.monotonic()
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
    finished = time.monotonic()
    passed = case_passes(case, content)
    return {
        "name": case["name"],
        "performance_class": case.get("performance_class", "interactive"),
        "passed": passed,
        "content": content,
        "reasoning": reasoning,
        "reasoning_disabled": not reasoning,
        "ttft_seconds": None if first_token is None else first_token - started,
        "wall_seconds": finished - started,
        "timings": final.get("timings", {}),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--ledger", type=Path, default=Path(".moeme/experiments.sqlite3"))
    parser.add_argument("--report", type=Path, default=Path(".moeme/server-eval.json"))
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-median-ttft", type=float, default=3.5)
    parser.add_argument("--max-ttft", type=float, default=15.0)
    parser.add_argument("--max-long-context-ttft", type=float, default=30.0)
    parser.add_argument("--min-median-decode-rate", type=float, default=5.0)
    parser.add_argument("--min-long-decode-rate", type=float, default=4.3)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2 to test stability")

    ledger = ExperimentLedger(args.ledger)
    suite_digest = hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest()
    configuration = {
        "url": args.url,
        "cases": [case["name"] for case in CASES],
        "suite_digest": suite_digest,
        "thinking": False,
        "temperature": 0,
        "repetitions": args.repetitions,
        "performance_gates": {
            "max_median_ttft_seconds": args.max_median_ttft,
            "max_ttft_seconds": args.max_ttft,
            "max_long_context_ttft_seconds": args.max_long_context_ttft,
            "min_median_decode_tokens_per_second": args.min_median_decode_rate,
            "min_long_decode_tokens_per_second": args.min_long_decode_rate,
        },
    }
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    run_id = ledger.start("runtime-server-eval", digest, configuration)
    try:
        results = [
            {**stream_chat(args.url, case), "repetition": repetition}
            for repetition in range(1, args.repetitions + 1)
            for case in CASES
        ]
    except BaseException as error:
        ledger.finish(
            run_id,
            "failed",
            {},
            error=f"{type(error).__name__}: {error}",
        )
        raise

    ttfts = [
        result["ttft_seconds"]
        for result in results
        if result["ttft_seconds"] is not None and result["performance_class"] == "interactive"
    ]
    long_context_ttfts = [
        result["ttft_seconds"]
        for result in results
        if result["ttft_seconds"] is not None and result["performance_class"] == "long_context"
    ]
    decode_rates = [
        result["timings"]["predicted_per_second"]
        for result in results
        if "predicted_per_second" in result["timings"]
    ]
    quality_passed = all(result["passed"] and result["reasoning_disabled"] for result in results)
    case_outputs: dict[str, set[str]] = {}
    for result in results:
        case_outputs.setdefault(result["name"], set()).add(result["content"].strip())
    stability_passed = all(len(outputs) == 1 for outputs in case_outputs.values())
    long_decode_rates = [
        result["timings"]["predicted_per_second"]
        for result in results
        if result["name"] == "long_decode" and "predicted_per_second" in result["timings"]
    ]
    median_ttft = statistics.median(ttfts)
    maximum_ttft = max(ttfts)
    median_decode_rate = statistics.median(decode_rates)
    minimum_long_decode_rate = min(long_decode_rates)
    maximum_long_context_ttft = max(long_context_ttfts)
    performance_passed = performance_passes(
        median_ttft=median_ttft,
        maximum_ttft=maximum_ttft,
        maximum_long_context_ttft=maximum_long_context_ttft,
        median_decode_rate=median_decode_rate,
        minimum_long_decode_rate=minimum_long_decode_rate,
        max_median_ttft=args.max_median_ttft,
        max_ttft=args.max_ttft,
        max_long_context_ttft=args.max_long_context_ttft,
        min_median_decode_rate=args.min_median_decode_rate,
        min_long_decode_rate=args.min_long_decode_rate,
    )
    all_passed = quality_passed and stability_passed and performance_passed
    summary = {
        "passed": all_passed,
        "quality_passed": quality_passed,
        "stability_passed": stability_passed,
        "performance_passed": performance_passed,
        "cases_passed": sum(result["passed"] for result in results),
        "cases_total": len(results),
        "unique_cases": len(CASES),
        "repetitions": args.repetitions,
        "median_ttft_seconds": median_ttft,
        "max_ttft_seconds": maximum_ttft,
        "median_long_context_ttft_seconds": statistics.median(long_context_ttfts),
        "max_long_context_ttft_seconds": maximum_long_context_ttft,
        "median_decode_tokens_per_second": median_decode_rate,
        "minimum_long_decode_tokens_per_second": minimum_long_decode_rate,
        "gates": configuration["performance_gates"],
        "suite_digest": suite_digest,
        "results": results,
    }
    _atomic_json(args.report, summary)
    ledger.artifact(
        run_id, "evaluation-report", args.report, bytes_count=args.report.stat().st_size
    )
    for result in results:
        ledger.metric(
            run_id,
            metric_capability(result),
            "passed",
            float(result["passed"]),
            "boolean",
            gate="deterministic case requirements satisfied",
            passed=result["passed"],
        )
    ledger.metric(
        run_id,
        "runtime",
        "median_ttft",
        summary["median_ttft_seconds"],
        "seconds",
        gate=f"<={args.max_median_ttft}",
        passed=median_ttft <= args.max_median_ttft,
    )
    ledger.metric(
        run_id,
        "runtime",
        "median_decode_rate",
        summary["median_decode_tokens_per_second"],
        "tokens/second",
        gate=f">={args.min_median_decode_rate}",
        passed=median_decode_rate >= args.min_median_decode_rate,
    )
    ledger.finish(
        run_id,
        "passed" if all_passed else "failed",
        summary,
        error=None if all_passed else "one or more resident-server gates failed",
    )
    print(json.dumps(summary, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
