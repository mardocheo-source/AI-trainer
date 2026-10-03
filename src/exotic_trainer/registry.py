from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .paths import registry_path
from .schema import DatasetProbe, ModelProbe, utc_now


class Registry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or registry_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    recipe_json TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def upsert_model(self, probe: ModelProbe) -> int:
        now = utc_now()
        payload = probe.model_dump_json()
        with self.connect() as db:
            db.execute(
                """INSERT INTO models(name,path,fingerprint,metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                name=excluded.name, fingerprint=excluded.fingerprint,
                metadata_json=excluded.metadata_json, updated_at=excluded.updated_at""",
                (probe.name, probe.path, probe.fingerprint, payload, now, now),
            )
            row = db.execute("SELECT id FROM models WHERE path=?", (probe.path,)).fetchone()
            return int(row["id"])

    def upsert_dataset(self, probe: DatasetProbe) -> int:
        now = utc_now()
        payload = probe.model_dump_json()
        with self.connect() as db:
            db.execute(
                """INSERT INTO datasets(name,source,fingerprint,metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(source) DO UPDATE SET
                name=excluded.name, fingerprint=excluded.fingerprint,
                metadata_json=excluded.metadata_json, updated_at=excluded.updated_at""",
                (probe.name, probe.source, probe.fingerprint, payload, now, now),
            )
            row = db.execute("SELECT id FROM datasets WHERE source=?", (probe.source,)).fetchone()
            return int(row["id"])

    def list_models(self) -> list[dict[str, Any]]:
        return self._list("models")

    def list_datasets(self) -> list[dict[str, Any]]:
        return self._list("datasets")

    def list_runs(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM runs ORDER BY id DESC")]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(f"run not found: {run_id}")
        return dict(row)

    def assign_run_to_sheet(self, run_id: str, sheet_id: str) -> None:
        """Attach an existing historical run to a workbook sheet without altering metrics."""
        run = self.get_run(run_id)
        recipe = json.loads(run.get("recipe_json") or "{}")
        recipe["workbook_sheet_id"] = sheet_id
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET recipe_json=?, updated_at=? WHERE run_id=?",
                (json.dumps(recipe), now, run_id),
            )

    def _list(self, table: str) -> list[dict[str, Any]]:
        if table not in {"models", "datasets"}:
            raise ValueError("invalid registry table")
        with self.connect() as db:
            rows = db.execute(f"SELECT * FROM {table} ORDER BY id DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def resolve_model(self, reference: str) -> str:
        candidate = Path(reference).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        with self.connect() as db:
            row = db.execute(
                "SELECT path FROM models WHERE name=? OR CAST(id AS TEXT)=?", (reference, reference)
            ).fetchone()
        if not row:
            raise KeyError(f"model not found in registry: {reference}")
        return str(row["path"])

    def resolve_dataset(self, reference: str) -> str:
        candidate = Path(reference).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        with self.connect() as db:
            row = db.execute(
                "SELECT source FROM datasets WHERE name=? OR CAST(id AS TEXT)=?", (reference, reference)
            ).fetchone()
        return str(row["source"]) if row else reference

    def create_run(self, run_id: str, recipe: dict[str, Any], output_dir: Path) -> None:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """INSERT INTO runs(run_id,status,recipe_json,output_dir,created_at,updated_at)
                VALUES(?,?,?,?,?,?)""",
                (run_id, "created", json.dumps(recipe), str(output_dir), now, now),
            )

    def update_run(
        self, run_id: str, status: str, metrics: dict[str, Any] | None = None
    ) -> None:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET status=?, metrics_json=?, updated_at=? WHERE run_id=?",
                (status, json.dumps(metrics or {}), now, run_id),
            )
