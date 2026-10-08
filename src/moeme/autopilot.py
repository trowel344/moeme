from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .layout import ExpertLayout
from .ledger import ExperimentLedger
from .safetensors_stream import (
    FORMAT_VERSION,
    convert_checkpoint_streaming,
    validate_converted_shards,
)


@dataclass(frozen=True, slots=True)
class AutopilotResult:
    status: str
    source_shards: int
    converted_shards: int
    expected_shards: int
    bytes_validated: int
    message: str


class CheckpointAutopilot:
    """Advance download -> conversion -> parity without overlapping workers."""

    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        state_dir: str | Path,
        layout: ExpertLayout,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.state_dir = Path(state_dir)
        self.layout = layout
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ExperimentLedger(self.state_dir / "experiments.sqlite3")

    def advance_once(self) -> AutopilotResult:
        with (self.state_dir / "autopilot.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return self._result("busy", "another autopilot worker holds the lock")

            source_count, expected = self._source_progress()
            converted_count = self._converted_progress()
            if source_count == 0:
                return self._save(self._result("waiting", "no completed source shards yet"))
            if source_count <= converted_count:
                status = "ready" if converted_count == expected else "waiting"
                return self._save(
                    self._result(status, "no newly completed source shards to convert")
                )

            configuration = {
                "source": str(self.source_dir.resolve()),
                "output": str(self.output_dir.resolve()),
                "source_shards": source_count,
                "target_top_k": self.layout.top_k,
            }
            previous = self._latest("streaming-conversion")
            with self.ledger.run(
                "streaming-conversion",
                json.dumps(configuration, sort_keys=True),
                configuration,
                parent_run_id=previous.id if previous else None,
            ) as run_id:
                manifest = convert_checkpoint_streaming(
                    self.source_dir,
                    self.output_dir,
                    self.layout,
                    allow_incomplete=True,
                )
                self.ledger.finish(run_id, "passed", manifest)

            previous = self._latest("conversion-parity")
            with self.ledger.run(
                "conversion-parity",
                json.dumps(configuration, sort_keys=True),
                configuration,
                parent_run_id=previous.id if previous else None,
            ) as run_id:
                validation = validate_converted_shards(
                    self.source_dir,
                    self.output_dir,
                    self.layout,
                    allow_incomplete=True,
                )
                self.ledger.metric(
                    run_id,
                    "conversion",
                    "bytes_exact",
                    validation["bytes_checked"],
                    "bytes",
                    gate="all compared bytes equal",
                    passed=validation["valid"],
                )
                self.ledger.finish(run_id, "passed", validation)

            status = "ready" if manifest["complete"] else "advanced"
            return self._save(
                AutopilotResult(
                    status=status,
                    source_shards=source_count,
                    converted_shards=manifest["converted_shards"],
                    expected_shards=manifest["expected_shards"],
                    bytes_validated=validation["bytes_checked"],
                    message=(
                        "all checkpoint shards converted and validated"
                        if status == "ready"
                        else "new shards converted and validated; waiting for download"
                    ),
                )
            )

    def watch(self, interval_seconds: float = 30.0) -> AutopilotResult:
        if interval_seconds < 1:
            raise ValueError("interval_seconds must be at least 1")
        while True:
            result = self.advance_once()
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            if result.status in {"ready", "busy"}:
                return result
            time.sleep(interval_seconds)

    def _source_progress(self) -> tuple[int, int]:
        index = self.source_dir / "model.safetensors.index.json"
        if not index.exists():
            return 0, 0
        with index.open(encoding="utf-8") as handle:
            shard_names = set(json.load(handle)["weight_map"].values())
        return sum((self.source_dir / name).is_file() for name in shard_names), len(shard_names)

    def _converted_progress(self) -> int:
        progress = self.output_dir / "conversion-progress.json"
        if not progress.exists():
            return 0
        with progress.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("format_version") != FORMAT_VERSION:
            return 0
        return len(value.get("completed", {}))

    def _latest(self, stage: str):
        return next((record for record in self.ledger.history() if record.stage == stage), None)

    def _result(self, status: str, message: str) -> AutopilotResult:
        source, expected = self._source_progress()
        converted = self._converted_progress()
        bytes_validated = 0
        latest = self._latest("conversion-parity")
        if latest:
            bytes_validated = int(latest.summary.get("bytes_checked", 0))
        return AutopilotResult(
            status=status,
            source_shards=source,
            converted_shards=converted,
            expected_shards=expected,
            bytes_validated=bytes_validated,
            message=message,
        )

    def _save(self, result: AutopilotResult) -> AutopilotResult:
        _atomic_json(self.state_dir / "autopilot.json", asdict(result))
        return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
