from pathlib import Path

from scripts.quantize_gguf import quantize_command


def test_quantize_command_is_plain_by_default() -> None:
    command = quantize_command(
        Path("llama-quantize"), Path("in.gguf"), Path("out.gguf"), "Q4_K_M", 8
    )
    assert command == ["llama-quantize", "in.gguf", "out.gguf", "Q4_K_M", "8"]


def test_quantize_command_places_flags_before_positionals() -> None:
    command = quantize_command(
        Path("llama-quantize"),
        Path("in.gguf"),
        Path("out.gguf"),
        "Q4_K_M",
        8,
        imatrix=Path("imatrix.gguf"),
        tensor_types=["ffn_.*_exps=Q5_K"],
        output_tensor_type="Q6_K",
        token_embedding_type="Q6_K",
    )
    assert command == [
        "llama-quantize",
        "--imatrix",
        "imatrix.gguf",
        "--tensor-type",
        "ffn_.*_exps=Q5_K",
        "--output-tensor-type",
        "Q6_K",
        "--token-embedding-type",
        "Q6_K",
        "in.gguf",
        "out.gguf",
        "Q4_K_M",
        "8",
    ]


def test_quantize_command_repeats_tensor_types_and_allows_requantize() -> None:
    command = quantize_command(
        Path("llama-quantize"),
        Path("in.gguf"),
        Path("out.gguf"),
        "Q4_K_M",
        8,
        tensor_types=["a=Q5_K", "b=Q8_0"],
        allow_requantize=True,
    )
    assert command[1] == "--allow-requantize"
    assert command.count("--tensor-type") == 2
    assert "a=Q5_K" in command and "b=Q8_0" in command
