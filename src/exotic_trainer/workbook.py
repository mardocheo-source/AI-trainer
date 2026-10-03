from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from .paths import project_root
from .schema import ExperimentSheet, ExplorationConfig, TrainingRecipe, utc_now

_LOCK = threading.Lock()


def workbook_path() -> Path:
    path = project_root() / "data" / "experiment-workbook.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:36] or "experiment"


class WorkbookStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or workbook_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[ExperimentSheet]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [ExperimentSheet.model_validate(item) for item in payload if isinstance(item, dict)]

    def _save(self, sheets: list[ExperimentSheet]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps([sheet.model_dump(mode="json") for sheet in sheets], indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def list(self) -> list[ExperimentSheet]:
        with _LOCK:
            return sorted(self._load(), key=lambda sheet: sheet.updated_at, reverse=True)

    def get(self, sheet_id: str) -> ExperimentSheet:
        for sheet in self.list():
            if sheet.id == sheet_id:
                return sheet
        raise KeyError(f"workbook sheet not found: {sheet_id}")

    def save(
        self,
        name: str,
        recipe: TrainingRecipe,
        exploration: ExplorationConfig,
        sheet_id: str | None = None,
    ) -> ExperimentSheet:
        with _LOCK:
            sheets = self._load()
            now = utc_now()
            existing = next((item for item in sheets if item.id == sheet_id), None)
            identifier = sheet_id or f"{_slug(name)}-{uuid.uuid4().hex[:8]}"
            recipe = recipe.model_copy(update={"workbook_sheet_id": identifier})
            saved = ExperimentSheet(
                id=identifier,
                name=name.strip() or "Untitled experiment",
                recipe=recipe,
                exploration=exploration,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            sheets = [item for item in sheets if item.id != identifier]
            sheets.append(saved)
            self._save(sheets)
            return saved

    def duplicate(self, sheet_id: str, name: str | None = None) -> ExperimentSheet:
        source = self.get(sheet_id)
        return self.save(
            name=name or f"{source.name} copy",
            recipe=source.recipe.model_copy(
                update={
                    "workbook_sheet_id": None,
                    "exploration_trial": None,
                    "exploration_signature": None,
                }
            ),
            exploration=source.exploration.model_copy(deep=True),
        )

    def delete(self, sheet_id: str) -> None:
        with _LOCK:
            sheets = [item for item in self._load() if item.id != sheet_id]
            self._save(sheets)


def sheet_choices() -> list[tuple[str, str]]:
    return [(f"{sheet.name} — {sheet.updated_at}", sheet.id) for sheet in WorkbookStore().list()]


def sheet_run_map(registry_runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for run in registry_runs:
        try:
            recipe = json.loads(run.get("recipe_json") or "{}")
        except json.JSONDecodeError:
            continue
        sheet_id = recipe.get("workbook_sheet_id")
        if sheet_id:
            output.setdefault(str(sheet_id), []).append(run)
    return output
