import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text())


def test_local_smoke_preserves_production_partition_and_training_recipe() -> None:
    production = config("free-cloud-layer63-200k.json")
    smoke = config("local-layer63-seeded-smoke.json")
    controlled = {
        "layer",
        "partition_mode",
        "groups",
        "shared_groups",
        "seed_checkpoint",
        "top_k_schedule",
        "train_projections",
        "train_shared",
        "routing_strategy",
        "learning_rate",
        "feature_learning_rate",
        "router_learning_rate",
    }
    assert {key: smoke[key] for key in controlled} == {key: production[key] for key in controlled}
    assert smoke["compute_dtype"] == "float16"
    assert smoke["batch_size"] == 1
    assert smoke["steps_per_stage"] == 2
    assert production["steps_per_stage"] == 50000
    assert 0 < production["max_runtime_seconds"] < 12 * 60 * 60
    assert production["minimum_free_gib"] >= 10
