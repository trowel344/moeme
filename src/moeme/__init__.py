"""moeme -- dense-equivalent FFN partitioning and edge-sparsity feasibility tools."""

from .ffn import (  # noqa: F401
    DenseFFN,
    PartitionPlan,
    PartitionedFFN,
    clustered_plan,
    contiguous_plan,
    contribution_concentration,
    interleaved_plan,
    oracle_topk_residual,
    random_plan,
    router_topk_residual,
    train_linear_router,
)
from .accounting import MemoryTier, Qwen38_27B, bytes_for_params, predicted_decode_tok_s  # noqa: F401

__all__ = [
    "DenseFFN",
    "PartitionPlan",
    "PartitionedFFN",
    "clustered_plan",
    "contiguous_plan",
    "interleaved_plan",
    "random_plan",
    "oracle_topk_residual",
    "contribution_concentration",
    "train_linear_router",
    "router_topk_residual",
    "Qwen38_27B",
    "MemoryTier",
    "bytes_for_params",
    "predicted_decode_tok_s",
]

__version__ = "0.1.0"
