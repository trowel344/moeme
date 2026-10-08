from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: int
    stage: str
    attempt: int
    status: str
    input_digest: str
    started_at: str
    finished_at: str | None
    parent_run_id: int | None
    summary: dict[str, Any]
    error: str | None


class ExperimentLedger:
    """Append-oriented experiment history for regression localization."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    stage TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running','passed','failed','blocked')),
                    input_digest TEXT NOT NULL,
                    parent_run_id INTEGER REFERENCES runs(id),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    configuration_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    UNIQUE(stage, attempt)
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    capability TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    gate TEXT,
                    passed INTEGER,
                    PRIMARY KEY(run_id, capability, name)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT,
                    bytes INTEGER,
                    PRIMARY KEY(run_id, kind, path)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if existing and int(existing["value"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported ledger schema {existing['value']}; expected {SCHEMA_VERSION}"
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def start(
        self,
        stage: str,
        input_digest: str,
        configuration: dict[str, Any],
        *,
        parent_run_id: int | None = None,
    ) -> int:
        with self._connect() as connection:
            # Take the write lock before reading the next attempt. Without this
            # the read-modify-write races across concurrent campaigns and the
            # UNIQUE(stage, attempt) constraint rejects one of them.
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:  # a transaction is already open
                pass
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt), 0) + 1 AS attempt FROM runs WHERE stage=?",
                (stage,),
            ).fetchone()
            cursor = connection.execute(
                """
                INSERT INTO runs(
                    stage, attempt, status, input_digest, parent_run_id,
                    started_at, configuration_json
                ) VALUES(?, ?, 'running', ?, ?, ?, ?)
                """,
                (
                    stage,
                    row["attempt"],
                    input_digest,
                    parent_run_id,
                    _now(),
                    _json(configuration),
                ),
            )
            run_id = int(cursor.lastrowid)
        self.write_compact_summary()
        return run_id

    def finish(
        self,
        run_id: int,
        status: str,
        summary: dict[str, Any],
        *,
        error: str | None = None,
    ) -> None:
        if status not in {"passed", "failed", "blocked"}:
            raise ValueError(f"invalid terminal status: {status}")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status=?, finished_at=?, summary_json=?, error=?
                WHERE id=? AND status='running'
                """,
                (status, _now(), _json(summary), error, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"run {run_id} does not exist or is already finished")
        self.write_compact_summary()

    def compact_summary(self, *, limit: int = 20) -> dict[str, Any]:
        """Return a small recovery view; the database remains authoritative."""

        records = self.history(limit=limit)
        latest_by_stage: dict[str, dict[str, Any]] = {}
        for record in records:
            if record.stage in latest_by_stage:
                continue
            latest_by_stage[record.stage] = {
                "id": record.id,
                "attempt": record.attempt,
                "status": record.status,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "summary": {
                    key: value
                    for key, value in record.summary.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
                "error": record.error,
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _now(),
            "latest_by_stage": latest_by_stage,
            "diagnosis": self.diagnosis(),
        }

    def write_compact_summary(self, path: str | Path | None = None) -> Path:
        """Atomically refresh the bounded handoff snapshot beside the ledger."""

        destination = Path(path) if path is not None else self.path.with_name("ledger-summary.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.compact_summary(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return destination

    @contextmanager
    def run(
        self,
        stage: str,
        input_digest: str,
        configuration: dict[str, Any],
        *,
        parent_run_id: int | None = None,
    ) -> Iterator[int]:
        run_id = self.start(stage, input_digest, configuration, parent_run_id=parent_run_id)
        try:
            yield run_id
        except BaseException as error:
            self.finish(run_id, "failed", {}, error=f"{type(error).__name__}: {error}")
            raise

    def metric(
        self,
        run_id: int,
        capability: str,
        name: str,
        value: float,
        unit: str,
        *,
        gate: str | None = None,
        passed: bool | None = None,
    ) -> None:
        if (gate is None) != (passed is None):
            raise ValueError("gate and passed must either both be provided or both be omitted")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO metrics(run_id, capability, name, value, unit, gate, passed)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, capability, name, value, unit, gate, passed),
            )

    def artifact(
        self,
        run_id: int,
        kind: str,
        path: str | Path,
        *,
        sha256: str | None = None,
        bytes_count: int | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(run_id, kind, path, sha256, bytes)
                VALUES(?, ?, ?, ?, ?)
                """,
                (run_id, kind, str(path), sha256, bytes_count),
            )

    def event(
        self,
        run_id: int,
        level: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events(run_id, created_at, level, message, data_json)
                VALUES(?, ?, ?, ?, ?)
                """,
                (run_id, _now(), level, message, _json(data or {})),
            )

    def history(self, *, limit: int = 50) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_record(row) for row in rows]

    def find(self, stage: str, input_digest: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM runs WHERE stage=? AND input_digest=?
                ORDER BY id DESC LIMIT 1
                """,
                (stage, input_digest),
            ).fetchone()
        return _record(row) if row else None

    def import_phase(
        self,
        stage: str,
        input_digest: str,
        status: str,
        summary: dict[str, Any],
    ) -> int:
        """Idempotently import a cached pipeline phase into experiment history."""

        existing = self.find(stage, input_digest)
        if existing is not None:
            return existing.id
        run_id = self.start(stage, input_digest, {"source": "phase-state-import"})
        self.finish(run_id, status, summary)
        return run_id

    def diagnosis(self) -> dict[str, Any]:
        """Return unresolved failures without losing the historical trail.

        A failure is resolved when a newer attempt of the same stage passes.  This
        keeps unattended recovery focused on the stage that still needs work,
        while the append-only historical fields preserve the first failure for
        post-mortem analysis.
        """

        with self._connect() as connection:
            failed_gate = connection.execute(
                """
                WITH latest AS (
                    SELECT stage, MAX(id) AS id FROM runs GROUP BY stage
                )
                SELECT r.id, r.stage, r.attempt, m.capability, m.name,
                       m.value, m.unit, m.gate
                FROM latest l
                JOIN runs r ON r.id=l.id
                JOIN metrics m ON m.run_id=r.id
                WHERE m.passed=0
                ORDER BY r.id, m.capability, m.name LIMIT 1
                """
            ).fetchone()
            failed_run = connection.execute(
                """
                WITH latest AS (
                    SELECT stage, MAX(id) AS id FROM runs GROUP BY stage
                )
                SELECT r.* FROM latest l JOIN runs r ON r.id=l.id
                WHERE r.status IN ('failed','blocked') ORDER BY r.id LIMIT 1
                """
            ).fetchone()
            historical_gate = connection.execute(
                """
                SELECT r.id, r.stage, r.attempt, m.capability, m.name,
                       m.value, m.unit, m.gate
                FROM runs r JOIN metrics m ON m.run_id=r.id
                WHERE m.passed=0
                ORDER BY r.id, m.capability, m.name LIMIT 1
                """
            ).fetchone()
            historical_run = connection.execute(
                """
                SELECT * FROM runs WHERE status IN ('failed','blocked')
                ORDER BY id LIMIT 1
                """
            ).fetchone()
            cutoff = failed_gate["id"] if failed_gate else failed_run["id"] if failed_run else None
            predecessor = None
            if cutoff is not None:
                predecessor = connection.execute(
                    "SELECT * FROM runs WHERE status='passed' AND id < ? ORDER BY id DESC LIMIT 1",
                    (cutoff,),
                ).fetchone()
        return {
            "first_failed_gate": dict(failed_gate) if failed_gate else None,
            "first_failed_run": asdict(_record(failed_run)) if failed_run else None,
            "last_passing_predecessor": asdict(_record(predecessor)) if predecessor else None,
            "all_clear": failed_gate is None and failed_run is None,
            "historical_first_failed_gate": (dict(historical_gate) if historical_gate else None),
            "historical_first_failed_run": (
                asdict(_record(historical_run)) if historical_run else None
            ),
        }


def _record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        stage=row["stage"],
        attempt=row["attempt"],
        status=row["status"],
        input_digest=row["input_digest"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        parent_run_id=row["parent_run_id"],
        summary=json.loads(row["summary_json"]),
        error=row["error"],
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
