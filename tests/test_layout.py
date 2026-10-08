from fractions import Fraction

import pytest

from moeme.layout import QWEN38_27B_LAYOUT, ExpertLayout


def test_qwen_layout_accounting_is_exact() -> None:
    layout = QWEN38_27B_LAYOUT
    assert layout.group_width == 1_088
    assert layout.shared_width == 4_352
    assert layout.expert_width == 1_088
    assert layout.stored_width == 17_408
    assert layout.active_width == 8_704
    assert layout.active_fraction == Fraction(1, 2)


@pytest.mark.parametrize(
    ("shared", "experts", "top_k", "expected"),
    [(4, 12, 4, Fraction(1, 2)), (2, 14, 6, Fraction(1, 2)), (0, 16, 8, Fraction(1, 2))],
)
def test_equal_compute_ablations(shared: int, experts: int, top_k: int, expected: Fraction) -> None:
    layout = ExpertLayout(
        17_408,
        groups=16,
        shared_groups=shared,
        routed_experts=experts,
        top_k=top_k,
    )
    assert layout.active_fraction == expected


def test_invalid_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        ExpertLayout(intermediate_size=127)
    with pytest.raises(ValueError, match="cover"):
        ExpertLayout(intermediate_size=128, shared_groups=3)
    with pytest.raises(ValueError, match="cannot exceed"):
        ExpertLayout(intermediate_size=128, top_k=13)
