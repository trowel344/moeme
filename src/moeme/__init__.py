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
from .upcycle import (  # noqa: F401
    MoEConfig,
    MoEFFN,
    RoutingMetrics,
    build_plan,
    collect_total_aux_loss,
    expert_parameters,
    iter_moe_modules,
    reconstruction_residual,
    router_parameters,
    set_top_k,
    upcycle_model,
)
from .branch import BranchConfig, BranchMoEFFN  # noqa: F401

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
    "MoEConfig",
    "MoEFFN",
    "RoutingMetrics",
    "build_plan",
    "collect_total_aux_loss",
    "expert_parameters",
    "iter_moe_modules",
    "reconstruction_residual",
    "router_parameters",
    "set_top_k",
    "upcycle_model",
    "BranchConfig",
    "BranchMoEFFN",
]

__version__ = "0.1.0"
