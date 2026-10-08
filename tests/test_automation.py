import json
from pathlib import Path

from moeme.automation import PhaseResult, StateStore, run_foundation, status_summary


def write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 16,
                    "intermediate_size": 128,
                    "num_hidden_layers": 4,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                    "max_position_embeddings": 1024,
                    "vocab_size": 256,
                },
                "vision_config": {"depth": 2, "out_hidden_size": 16},
            }
        ),
        encoding="utf-8",
    )


def test_foundation_is_resumable_and_writes_compact_context(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    state_dir = tmp_path / ".moeme"
    write_config(config)

    results, skipped = run_foundation(config, state_dir)
    assert {result.phase for result in results} == {"architecture", "partition"}
    assert skipped == []
    before = StateStore(state_dir).load()

    _, skipped = run_foundation(config, state_dir)
    after = StateStore(state_dir).load()
    assert skipped == ["architecture", "partition"]
    assert before == after
    assert "Next: checkpoint tensor validation" in (state_dir / "CONTEXT.md").read_text()
    assert status_summary(state_dir)["next_phase"] == "checkpoint-validation"


def test_changed_config_invalidates_only_its_phase(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    state_dir = tmp_path / ".moeme"
    write_config(config)
    run_foundation(config, state_dir)
    value = json.loads(config.read_text())
    value["text_config"]["max_position_embeddings"] = 2048
    config.write_text(json.dumps(value), encoding="utf-8")

    _, skipped = run_foundation(config, state_dir)
    assert skipped == ["partition"]


def test_rerun_preserves_later_phase_in_context_and_status(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    state_dir = tmp_path / ".moeme"
    write_config(config)
    run_foundation(config, state_dir)
    state = StateStore(state_dir)
    state.save_result(
        PhaseResult(
            phase="checkpoint",
            status="passed",
            input_digest="digest",
            summary={"commit": "abc", "mlp_tensors_checked": 12},
            completed_at="2026-01-01T00:00:00+00:00",
        )
    )

    run_foundation(config, state_dir)
    context = (state_dir / "CONTEXT.md").read_text()
    assert "Checkpoint commit: abc" in context
    assert status_summary(state_dir)["next_phase"] == "streaming-conversion"


def test_compact_context_routes_to_terminal_release_and_open_training(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    state_dir = tmp_path / ".moeme"
    write_config(config)
    run_foundation(config, state_dir)
    state = StateStore(state_dir)
    state.save_result(
        PhaseResult(
            phase="checkpoint",
            status="passed",
            input_digest="digest",
            summary={"commit": "abc", "mlp_tensors_checked": 12},
            completed_at="2026-01-01T00:00:00+00:00",
        )
    )
    (state_dir / "terminal-spec-decision.json").write_text(
        json.dumps(
            {
                "terminal_artifact": {
                    "path": "artifacts/release.gguf",
                    "sha256": "release-sha",
                }
            }
        )
    )
    (state_dir / "STATUS.json").write_text(
        json.dumps(
            {
                "free_cloud": {
                    "activation_capture": {"ready": False, "service": {"running": True}},
                    "acceptance": {"current_phase": "bf16_parity"},
                    "local_smoke": {"run": {"status": "passed"}},
                    "training": {
                        "curve_points": 3,
                        "result_manifest": {"candidate_ready": True},
                        "run": {"status": "interrupted", "stage": "training", "attempt": 2},
                    },
                }
            }
        )
    )

    run_foundation(config, state_dir)
    context = (state_dir / "CONTEXT.md").read_text()
    assert "Validated release: artifacts/release.gguf" in context
    assert "Free-cloud de-risk: activation capture running" in context
    assert "Free-cloud training: interrupted (stage training, attempt 2)" in context
    assert "Cloud result: candidate ready" in context
    assert "Sparse-candidate acceptance: bf16_parity" in context
    assert "Local seeded smoke: passed" in context
    assert "Learning-curve points: 3" in context
    assert "streaming dense-to-MoE checkpoint conversion" not in context
