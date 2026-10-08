from scripts.server_eval import CASES, case_passes, metric_capability, performance_passes


def case(name: str) -> dict:
    return next(item for item in CASES if item["name"] == name)


def test_required_fragments_are_case_insensitive() -> None:
    assert case_passes(case("factual"), " Paris\n")
    assert not case_passes(case("factual"), "Lyon")


def test_exact_cases_reject_extra_text() -> None:
    structured = case("structured_instruction")
    assert case_passes(structured, '{"status":"ok","count":3}\n')
    assert not case_passes(structured, 'Here: {"status":"ok","count":3}')


def test_suite_covers_chat_work_and_runtime_behaviors() -> None:
    names = {item["name"] for item in CASES}
    assert {
        "factual",
        "arithmetic",
        "code_generation",
        "structured_instruction",
        "code_repair",
        "summarization",
        "context_retrieval",
        "long_decode",
    } == names


def test_context_retrieval_is_a_separate_latency_class() -> None:
    retrieval = case("context_retrieval")
    assert retrieval["performance_class"] == "long_context"
    assert "Record 199" in retrieval["prompt"]


def test_repeated_cases_have_unique_ledger_capabilities() -> None:
    first = metric_capability({"name": "factual", "repetition": 1})
    second = metric_capability({"name": "factual", "repetition": 2})
    assert first == "factual-repetition-1"
    assert second == "factual-repetition-2"


def test_long_context_ttft_is_a_real_performance_gate() -> None:
    values = {
        "median_ttft": 4.0,
        "maximum_ttft": 6.0,
        "median_decode_rate": 3.3,
        "minimum_long_decode_rate": 2.8,
        "max_median_ttft": 6.0,
        "max_ttft": 10.0,
        "max_long_context_ttft": 30.0,
        "min_median_decode_rate": 3.0,
        "min_long_decode_rate": 2.5,
    }
    assert performance_passes(maximum_long_context_ttft=20.0, **values) is True
    assert performance_passes(maximum_long_context_ttft=31.0, **values) is False
