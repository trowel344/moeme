import pytest
import torch

from moeme.layout import ExpertLayout
from moeme.partition import PartitionedSwiGLU, dense_swiglu


def weights(hidden: int = 11, intermediate: int = 32):
    generator = torch.Generator().manual_seed(41)
    gate = torch.randn(intermediate, hidden, generator=generator, dtype=torch.float64)
    up = torch.randn(intermediate, hidden, generator=generator, dtype=torch.float64)
    down = torch.randn(hidden, intermediate, generator=generator, dtype=torch.float64)
    return gate, up, down


@pytest.mark.parametrize("shape", [(7, 11), (2, 3, 11)])
def test_all_partitions_exactly_reconstruct_dense(shape: tuple[int, ...]) -> None:
    gate, up, down = weights()
    x = torch.randn(*shape, dtype=torch.float64)
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=2)
    partitioned = PartitionedSwiGLU.from_dense(gate, up, down, layout)
    torch.testing.assert_close(
        partitioned.forward_all(x), dense_swiglu(x, gate, up, down), atol=1e-10, rtol=1e-10
    )


def test_reordered_channels_still_reconstruct_exactly() -> None:
    gate, up, down = weights()
    x = torch.randn(5, 11, dtype=torch.float64)
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=2)
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(9))
    partitioned = PartitionedSwiGLU.from_dense(gate, up, down, layout, permutation=permutation)
    torch.testing.assert_close(
        partitioned.forward_all(x), dense_swiglu(x, gate, up, down), atol=1e-10, rtol=1e-10
    )


def test_k_scaled_uniform_softmax_is_exact_but_plain_softmax_is_not() -> None:
    gate, up, down = weights()
    x = torch.randn(5, 11, dtype=torch.float64)
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=6)
    partitioned = PartitionedSwiGLU.from_dense(gate, up, down, layout)
    indices = range(6)
    normalized = torch.full((6,), 1 / 6, dtype=torch.float64)
    dense = dense_swiglu(x, gate, up, down)
    scaled = partitioned.forward_selected(x, indices, normalized, preserve_sum_scale=True)
    plain = partitioned.forward_selected(x, indices, normalized, preserve_sum_scale=False)
    torch.testing.assert_close(scaled, dense, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(plain, dense, atol=1e-6, rtol=1e-6)


def test_bad_permutation_is_rejected() -> None:
    gate, up, down = weights()
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=2)
    bad = torch.zeros(32, dtype=torch.int64)
    with pytest.raises(ValueError, match="every intermediate index"):
        PartitionedSwiGLU.from_dense(gate, up, down, layout, permutation=bad)
