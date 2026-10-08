import pytest
import torch

from scripts.patch_gguf_layer import TRAINED_KEYS, validate_trained_state


def state(router: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: router.clone() if key == "router" else torch.ones((2, 2)) for key in TRAINED_KEYS}


def test_checkpoint_health_rejects_zero_router() -> None:
    with pytest.raises(ValueError, match="router nonzero fraction"):
        validate_trained_state(state(torch.zeros((2, 2))), layer=3, minimum_router_nonzero=0.99)


def test_checkpoint_health_rejects_nonfinite_tensor() -> None:
    tensors = state(torch.ones((2, 2)))
    tensors["expert_down"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite values"):
        validate_trained_state(tensors, layer=4, minimum_router_nonzero=0.99)


def test_checkpoint_health_reports_learned_router() -> None:
    health = validate_trained_state(
        state(torch.full((2, 2), 0.125)), layer=5, minimum_router_nonzero=0.99
    )
    assert health == {"router_nonzero_fraction": 1.0, "router_abs_max": 0.125}
