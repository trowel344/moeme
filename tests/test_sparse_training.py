import pytest
import torch

from scripts.train_sparse_layer import (
    distillation_loss,
    resolve_compute_dtype,
    route_utilization,
    training_exit_code,
)


def test_router_auxiliary_produces_gradient_with_overridden_routes() -> None:
    logits = torch.zeros((2, 12), requires_grad=True)
    oracle_indices = torch.tensor([[1, 3, 5, 7], [0, 2, 4, 6]])
    oracle_mixture = torch.full((2, 4), 1.0)
    reconstruction = torch.tensor(0.25, requires_grad=True)

    loss, router_objective = distillation_loss(
        reconstruction,
        logits,
        oracle_indices,
        oracle_mixture,
        top_k=4,
    )
    loss.backward()

    assert router_objective.item() > 0
    assert logits.grad is not None
    assert logits.grad.count_nonzero().item() > 0


def test_route_utilization_reports_coverage_entropy_and_collapse() -> None:
    uniform = route_utilization(torch.ones(12, dtype=torch.int64))
    assert uniform["route_coverage"] == 1.0
    assert uniform["normalized_route_entropy"] == pytest.approx(1.0)
    assert uniform["maximum_route_fraction"] == pytest.approx(1 / 12)

    collapsed = route_utilization(torch.tensor([12, *([0] * 11)]))
    assert collapsed["route_coverage"] == pytest.approx(1 / 12)
    assert collapsed["normalized_route_entropy"] == 0.0
    assert collapsed["maximum_route_fraction"] == 1.0


def test_auto_compute_dtype_uses_float32_on_cpu() -> None:
    assert resolve_compute_dtype("auto", "cpu") == torch.float32
    assert resolve_compute_dtype("float16", "cpu") == torch.float16


def test_systems_smoke_only_relaxes_process_exit_status() -> None:
    assert training_exit_code(True, False) == 0
    assert training_exit_code(True, True) == 0
    assert training_exit_code(False, False) == 1
    assert training_exit_code(False, True) == 0
