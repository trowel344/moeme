from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def audit_llama_runtime(source: str | Path) -> dict[str, Any]:
    """Check the llama.cpp behaviors required by the MoEMe GGUF bridge."""

    source = Path(source)
    model_path = source / "src/models/qwen35moe.cpp"
    backend_path = source / "ggml/src/ggml-backend.cpp"
    model_text = model_path.read_text(encoding="utf-8")
    backend_text = backend_path.read_text(encoding="utf-8")

    loader_start = model_text.index("void llama_model_qwen35moe::load_arch_hparams")
    loader_end = model_text.index("void llama_model_qwen35moe::load_arch_tensors", loader_start)
    loader = model_text[loader_start:loader_end]
    loads_scale = "LLM_KV_EXPERT_WEIGHTS_SCALE" in loader
    expert_cache_enabled = bool(
        re.search(r"^\s*#define\s+GGML_EXPERT_CACHE\s*$", backend_text, re.MULTILINE)
    )

    issues = []
    if not loads_scale:
        issues.append(
            "Qwen35MoE loader ignores qwen35moe.expert_weights_scale; "
            "Top-12 routed output will be divided by 12"
        )
    if expert_cache_enabled:
        issues.append(
            "experimental expert cache is compiled in; its default 110 slots "
            "exceed the model's 12 experts"
        )
    return {
        "source": str(source.resolve()),
        "loads_expert_weights_scale": loads_scale,
        "experimental_expert_cache_enabled": expert_cache_enabled,
        "compatible": not issues,
        "issues": issues,
    }
