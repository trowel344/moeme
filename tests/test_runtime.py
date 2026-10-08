from types import SimpleNamespace

import torch
from torch import nn

from moeme.layout import ExpertLayout
from moeme.partition import dense_swiglu
from moeme.runtime import install_moeme_mlps


class DenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(12, 32, bias=False, dtype=torch.float64)
        self.up_proj = nn.Linear(12, 32, bias=False, dtype=torch.float64)
        self.down_proj = nn.Linear(32, 12, bias=False, dtype=torch.float64)


class Container(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.config = SimpleNamespace(hidden_size=12)
        self.model.language_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.language_model.layers:
            layer.mlp = DenseMLP()


def test_install_replaces_only_mlps_and_preserves_dense_top_all() -> None:
    model = Container()
    original_layer_ids = [id(layer) for layer in model.model.language_model.layers]
    weights = [
        (
            layer.mlp.gate_proj.weight.detach().clone(),
            layer.mlp.up_proj.weight.detach().clone(),
            layer.mlp.down_proj.weight.detach().clone(),
        )
        for layer in model.model.language_model.layers
    ]
    layout = ExpertLayout(32, groups=8, shared_groups=2, routed_experts=6, top_k=6)
    install_moeme_mlps(model, layout, copy_dense_weights=True)

    x = torch.randn(4, 12, dtype=torch.float64)
    for index, layer in enumerate(model.model.language_model.layers):
        assert id(layer) == original_layer_ids[index]
        torch.testing.assert_close(
            layer.mlp(x), dense_swiglu(x, *weights[index]), atol=1e-10, rtol=1e-10
        )
        keys = set(layer.mlp.state_dict())
        assert "shared_expert.gate_proj.weight" in keys
        assert "experts.0.gate_proj.weight" in keys
        assert "router.weight" in keys
        assert "shared_expert_gate.weight" in keys
