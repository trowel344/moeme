from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .architecture import inspect_config_file
from .checkpoint import TensorSpec, validate_mlp_tensors
from .layout import ExpertLayout
from .partition import PartitionedSwiGLU, dense_swiglu

STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class PhaseResult:
    phase: str
    status: str
    input_digest: str
    summary: dict[str, Any]
    completed_at: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PhaseResult:
        return cls(**value)


class StateStore:
    """Small resumable state store; large/raw evidence belongs outside state.json."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.state_path = self.root / "state.json"
        self.reports = self.root / "reports"

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"version": STATE_VERSION, "phases": {}}
        with self.state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("version") != STATE_VERSION:
            raise ValueError(
                f"unsupported state version {state.get('version')}; expected {STATE_VERSION}"
            )
        if not isinstance(state.get("phases"), dict):
            raise TypeError("invalid state: phases must be an object")
        return state

    def save_result(self, result: PhaseResult) -> None:
        state = self.load()
        state["phases"][result.phase] = asdict(result)
        self.reports.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.reports / f"{result.phase}.json", asdict(result))
        _atomic_json(self.state_path, state)

    def cached(self, phase: str, input_digest: str) -> PhaseResult | None:
        value = self.load()["phases"].get(phase)
        if not value or value.get("input_digest") != input_digest:
            return None
        result = PhaseResult.from_dict(value)
        return result if result.status == "passed" else None


def run_foundation(
    config_path: str | Path,
    state_dir: str | Path,
    *,
    force: bool = False,
) -> tuple[list[PhaseResult], list[str]]:
    """Run/cache phases A and B and return results plus skipped phase names."""

    config_path = Path(config_path)
    store = StateStore(state_dir)
    results: list[PhaseResult] = []
    skipped: list[str] = []

    phase_a_digest = digest_inputs(
        [config_path],
        {"phase": "architecture", "validator": 1},
    )
    cached = None if force else store.cached("architecture", phase_a_digest)
    if cached:
        results.append(cached)
        skipped.append("architecture")
    else:
        facts = inspect_config_file(config_path)
        layout = ExpertLayout(facts.intermediate_size)
        result = _passed(
            "architecture",
            phase_a_digest,
            {
                **facts.to_dict(),
                "shared_width": layout.shared_width,
                "expert_width": layout.expert_width,
                "top4_active_fraction": float(layout.active_fraction),
            },
        )
        store.save_result(result)
        results.append(result)

    phase_b_digest = digest_inputs(
        [Path(__file__).with_name("partition.py"), Path(__file__).with_name("layout.py")],
        {"phase": "partition-selfcheck", "seed": 7, "dtype": "float64", "validator": 1},
    )
    cached = None if force else store.cached("partition", phase_b_digest)
    if cached:
        results.append(cached)
        skipped.append("partition")
    else:
        summary = partition_selfcheck()
        if not summary["exact_within_tolerance"]:
            raise RuntimeError(f"partition selfcheck failed: {summary}")
        result = _passed("partition", phase_b_digest, summary)
        store.save_result(result)
        results.append(result)

    write_context_summary_from_store(store)
    return results, skipped


def partition_selfcheck() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(17, 64, generator=generator, dtype=torch.float64)
    gate = torch.randn(128, 64, generator=generator, dtype=torch.float64)
    up = torch.randn(128, 64, generator=generator, dtype=torch.float64)
    down = torch.randn(64, 128, generator=generator, dtype=torch.float64)
    layout = ExpertLayout(128)
    partitioned = PartitionedSwiGLU.from_dense(gate, up, down, layout)
    expected = dense_swiglu(x, gate, up, down)
    actual = partitioned.forward_all(x)
    error = (expected - actual).abs()
    return {
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
        "atol": 1e-10,
        "rtol": 1e-10,
        "exact_within_tolerance": torch.allclose(expected, actual, atol=1e-10, rtol=1e-10),
    }


def status_summary(state_dir: str | Path) -> dict[str, Any]:
    store = StateStore(state_dir)
    state = store.load()
    phases = state["phases"]
    return {
        "version": state["version"],
        "phases": {
            name: {
                "status": value["status"],
                "completed_at": value["completed_at"],
                "summary": value["summary"],
            }
            for name, value in sorted(phases.items())
        },
        "next_phase": _next_phase(phases),
    }


def verify_hub_checkpoint(
    repo_id: str,
    revision: str,
    config_path: str | Path,
    state_dir: str | Path,
    *,
    force: bool = False,
) -> tuple[PhaseResult, bool]:
    """Validate remote safetensors headers and cache by immutable commit SHA."""

    try:
        from huggingface_hub import HfApi, get_safetensors_metadata
    except ImportError as error:
        raise RuntimeError("verify-hub requires the 'huggingface-hub' package") from error

    facts = inspect_config_file(config_path)
    model = HfApi().model_info(repo_id, revision=revision)
    commit = model.sha
    if not commit:
        raise RuntimeError(f"the Hub did not return a commit SHA for {repo_id}@{revision}")
    input_digest = digest_inputs(
        [Path(config_path)],
        {"phase": "checkpoint", "repo_id": repo_id, "commit": commit, "validator": 1},
    )
    store = StateStore(state_dir)
    cached = None if force else store.cached("checkpoint", input_digest)
    if cached:
        write_context_summary_from_store(store)
        return cached, True

    metadata = get_safetensors_metadata(repo_id, revision=commit, timeout=30)
    tensors: dict[str, TensorSpec] = {}
    for file_metadata in metadata.files_metadata.values():
        for name, info in file_metadata.tensors.items():
            tensors[name] = TensorSpec(tuple(info.shape), info.dtype)
    validation = validate_mlp_tensors(
        tensors,
        layers=facts.num_hidden_layers,
        hidden_size=facts.hidden_size,
        intermediate_size=facts.intermediate_size,
        expected_dtype="BF16",
    )
    summary = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "commit": commit,
        "total_parameters": metadata.parameter_count,
        "total_size_bytes": int(metadata.metadata["total_size"]),
        "tensor_count": len(tensors),
        **validation,
    }
    if not validation["valid"]:
        result = PhaseResult(
            phase="checkpoint",
            status="failed",
            input_digest=input_digest,
            summary={**summary, "environment": environment_summary()},
            completed_at=datetime.now(UTC).isoformat(),
        )
        store.save_result(result)
        write_context_summary_from_store(store)
        raise RuntimeError(f"checkpoint MLP validation failed: {summary}")
    result = _passed("checkpoint", input_digest, summary)
    store.save_result(result)
    write_context_summary_from_store(store)
    return result, False


def write_context_summary(store: StateStore, results: list[PhaseResult]) -> None:
    by_phase = {result.phase: result for result in results}
    architecture = by_phase["architecture"].summary
    partition = by_phase["partition"].summary
    lines = [
        "# MoEMe compact state",
        "",
        "This file is generated. Read it before raw reports to minimize context use.",
        "",
        f"- Phase A architecture: {by_phase['architecture'].status}",
        (
            f"- Qwen layers: {architecture['num_hidden_layers']} "
            f"({architecture['linear_attention_layers']} linear, "
            f"{architecture['full_attention_layers']} full)"
        ),
        f"- FFN: hidden={architecture['hidden_size']}, intermediate={architecture['intermediate_size']}",
        f"- MoEMe Top-4 active fraction: {architecture['top4_active_fraction']:.3f}",
        f"- Phase B partition: {by_phase['partition'].status}",
        f"- Exact reconstruction max error: {partition['max_abs_error']:.3e}",
        "- Next: checkpoint tensor validation and streaming conversion (Phase C foundation)",
        "",
    ]
    store.root.mkdir(parents=True, exist_ok=True)
    _atomic_text(store.root / "CONTEXT.md", "\n".join(lines))


def write_context_summary_from_store(store: StateStore) -> None:
    phases = store.load()["phases"]
    required = {"architecture", "partition"}
    if not required.issubset(phases):
        return
    results = [PhaseResult.from_dict(phases[name]) for name in sorted(required)]
    write_context_summary(store, results)
    checkpoint = phases.get("checkpoint")
    if checkpoint:
        path = store.root / "CONTEXT.md"
        text = path.read_text(encoding="utf-8")
        summary = checkpoint["summary"]
        extra = (
            f"- Phase A checkpoint headers: {checkpoint['status']}\n"
            f"- Checkpoint commit: {summary.get('commit', 'unknown')}\n"
            f"- MLP tensors checked: {summary.get('mlp_tensors_checked', 0)}\n"
            "- Next: streaming dense-to-MoE checkpoint conversion and parity validation\n"
        )
        text = text.replace(
            "- Next: checkpoint tensor validation and streaming conversion (Phase C foundation)\n",
            extra,
        )
        decision_path = store.root / "terminal-spec-decision.json"
        status_path = store.root / "STATUS.json"
        if decision_path.exists():
            try:
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                status = (
                    json.loads(status_path.read_text(encoding="utf-8"))
                    if status_path.exists()
                    else {}
                )
            except (json.JSONDecodeError, OSError):
                decision = None
                status = {}
            if isinstance(decision, dict):
                terminal = decision.get("terminal_artifact", {})
                free_cloud = status.get("free_cloud", {})
                capture = free_cloud.get("activation_capture", {})
                training = free_cloud.get("training", {})
                cloud_run = training.get("run") or {}
                cloud_result = training.get("result_manifest") or {}
                acceptance = free_cloud.get("acceptance") or {}
                smoke = free_cloud.get("local_smoke") or {}
                smoke_run = smoke.get("run") or {}
                smoke_service = smoke.get("service") or {}
                cloud_state = (
                    "activation capture ready"
                    if capture.get("ready")
                    else (
                        "activation capture running"
                        if capture.get("service", {}).get("running")
                        else "activation capture not ready"
                    )
                )
                terminal_lines = (
                    "- Terminal spec: docs/terminal-spec.md\n"
                    f"- Validated release: {terminal.get('path', 'unknown')}\n"
                    f"- Release SHA-256: {terminal.get('sha256', 'unknown')}\n"
                    "- Real Top-4 training question: open; selection-only target withdrawn\n"
                    "- Local assembly: direct quantized-layer patch; byte-exact outside the trained layer\n"
                    f"- Free-cloud de-risk: {cloud_state}\n"
                    f"- Free-cloud training: {cloud_run.get('status', 'not started')}"
                    f" (stage {cloud_run.get('stage', 'none')}, attempt "
                    f"{cloud_run.get('attempt', 0)})\n"
                    f"- Cloud result: "
                    f"{'candidate ready' if cloud_result.get('candidate_ready') else 'not ready'}\n"
                    f"- Learning-curve points: {training.get('curve_points', 0)}\n"
                    f"- Sparse-candidate acceptance: "
                    f"{'passed' if acceptance.get('passed') else acceptance.get('current_phase', 'not started')}\n"
                    f"- Local seeded smoke: "
                    f"{smoke_run.get('status', 'running' if smoke_service.get('running') else 'not started')}\n"
                    "- Next: finish layer-63 de-risk, then patch and run unchanged full-model gates\n"
                )
                text = text.replace(
                    "- Next: streaming dense-to-MoE checkpoint conversion and parity validation\n",
                    terminal_lines,
                )
        _atomic_text(path, text)


def digest_inputs(paths: list[Path], parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.resolve()).encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    digest.update(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def environment_summary() -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def _next_phase(phases: dict[str, Any]) -> str:
    if phases.get("architecture", {}).get("status") != "passed":
        return "architecture"
    if phases.get("partition", {}).get("status") != "passed":
        return "partition"
    if phases.get("checkpoint", {}).get("status") != "passed":
        return "checkpoint-validation"
    return "streaming-conversion"


def _passed(phase: str, input_digest: str, summary: dict[str, Any]) -> PhaseResult:
    return PhaseResult(
        phase=phase,
        status="passed",
        input_digest=input_digest,
        summary={**summary, "environment": environment_summary()},
        completed_at=datetime.now(UTC).isoformat(),
    )


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
