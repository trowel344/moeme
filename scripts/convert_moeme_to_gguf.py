#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MoEMe staging tensors through llama.cpp's native Qwen35MoE path"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", choices=("bf16", "f16", "q8_0"), default="bf16")
    parser.add_argument("--split-max-size", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # llama.cpp's dry-run does not open the output, so make this precondition
    # explicit here instead of allowing the real conversion to fail late.
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.llama_cpp))
    import convert_hf_to_gguf as converter
    from conversion.qwen import Qwen3_5MoeTextModel

    original_parameters = Qwen3_5MoeTextModel.set_gguf_parameters
    original_modify = Qwen3_5MoeTextModel.modify_tensors

    def set_gguf_parameters(self):
        original_parameters(self)
        self.gguf_writer.add_expert_weights_scale(float(self.hparams["routed_scaling_factor"]))

    def modify_tensors(self, data_torch, name, bid):
        if name.endswith(".mlp.shared_expert.down_proj.weight"):
            data_torch = data_torch * 2
        yield from original_modify(self, data_torch, name, bid)

    Qwen3_5MoeTextModel.set_gguf_parameters = set_gguf_parameters
    Qwen3_5MoeTextModel.modify_tensors = modify_tensors
    sys.argv = [
        str(args.llama_cpp / "convert_hf_to_gguf.py"),
        str(args.model),
        "--outfile",
        str(args.outfile),
        "--outtype",
        args.outtype,
        "--split-max-size",
        args.split_max_size,
        "--no-mtp",
    ]
    if args.dry_run:
        sys.argv.append("--dry-run")
    else:
        # The default writer retains converted arrays until its final write pass.
        # Qwen's 248k-token embedding/output tensors push that path beyond this
        # 16 GiB host. Spooling tensor payloads bounds RAM at the cost of a
        # temporary sequential disk pass.
        sys.argv.append("--use-temp-file")
    converter.main()


if __name__ == "__main__":
    main()
