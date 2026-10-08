from pathlib import Path

from moeme.runtime_contract import audit_llama_runtime


def runtime_source(tmp_path: Path, *, scale: bool, cache: bool) -> Path:
    (tmp_path / "src/models").mkdir(parents=True)
    (tmp_path / "ggml/src").mkdir(parents=True)
    scale_line = (
        "ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);" if scale else ""
    )
    (tmp_path / "src/models/qwen35moe.cpp").write_text(
        "void llama_model_qwen35moe::load_arch_hparams() {\n"
        f"{scale_line}\n"
        "}\n"
        "void llama_model_qwen35moe::load_arch_tensors() {}\n"
    )
    cache_line = "#define GGML_EXPERT_CACHE\n" if cache else "// #define GGML_EXPERT_CACHE\n"
    (tmp_path / "ggml/src/ggml-backend.cpp").write_text(cache_line)
    return tmp_path


def test_runtime_contract_accepts_scale_loader_without_experimental_cache(tmp_path: Path) -> None:
    result = audit_llama_runtime(runtime_source(tmp_path, scale=True, cache=False))
    assert result["compatible"] is True
    assert result["issues"] == []


def test_runtime_contract_reports_both_known_failures(tmp_path: Path) -> None:
    result = audit_llama_runtime(runtime_source(tmp_path, scale=False, cache=True))
    assert result["compatible"] is False
    assert len(result["issues"]) == 2
