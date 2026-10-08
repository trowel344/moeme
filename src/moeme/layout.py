from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class ExpertLayout:
    """A disjoint partition of a dense SwiGLU intermediate dimension.

    Widths are expressed as groups so parameter accounting remains exact and
    layouts can be rejected before checkpoint conversion starts.
    """

    intermediate_size: int
    groups: int = 16
    shared_groups: int = 4
    routed_experts: int = 12
    top_k: int = 4

    def __post_init__(self) -> None:
        positive_values = {
            "intermediate_size": self.intermediate_size,
            "groups": self.groups,
            "routed_experts": self.routed_experts,
            "top_k": self.top_k,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.shared_groups < 0:
            raise ValueError(f"shared_groups cannot be negative, got {self.shared_groups}")
        if self.intermediate_size % self.groups:
            raise ValueError(
                f"intermediate_size={self.intermediate_size} is not divisible by groups={self.groups}"
            )
        if self.shared_groups + self.routed_experts != self.groups:
            raise ValueError("shared_groups + routed_experts must cover every group exactly once")
        if self.top_k > self.routed_experts:
            raise ValueError("top_k cannot exceed routed_experts")

    @property
    def group_width(self) -> int:
        return self.intermediate_size // self.groups

    @property
    def shared_width(self) -> int:
        return self.shared_groups * self.group_width

    @property
    def expert_width(self) -> int:
        return self.group_width

    @property
    def stored_width(self) -> int:
        return self.shared_width + self.routed_experts * self.expert_width

    @property
    def active_width(self) -> int:
        return self.shared_width + self.top_k * self.expert_width

    @property
    def active_fraction(self) -> Fraction:
        return Fraction(self.active_width, self.intermediate_size)

    def with_top_k(self, top_k: int) -> ExpertLayout:
        return ExpertLayout(
            intermediate_size=self.intermediate_size,
            groups=self.groups,
            shared_groups=self.shared_groups,
            routed_experts=self.routed_experts,
            top_k=top_k,
        )


QWEN38_27B_LAYOUT = ExpertLayout(intermediate_size=17_408)
