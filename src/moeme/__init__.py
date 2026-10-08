"""MoEMe research primitives with lazy heavyweight imports."""

from typing import Any

__all__ = ["ExpertLayout", "MoEMeMLP", "PartitionedSwiGLU", "dense_swiglu"]


def __getattr__(name: str) -> Any:
    if name == "ExpertLayout":
        from .layout import ExpertLayout

        return ExpertLayout
    if name == "MoEMeMLP":
        from .nn import MoEMeMLP

        return MoEMeMLP
    if name in ("PartitionedSwiGLU", "dense_swiglu"):
        from .partition import PartitionedSwiGLU, dense_swiglu

        return {"PartitionedSwiGLU": PartitionedSwiGLU, "dense_swiglu": dense_swiglu}[name]
    raise AttributeError(name)
