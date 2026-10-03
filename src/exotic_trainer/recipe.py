from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import TrainingRecipe


def load_recipe(path: str | Path) -> TrainingRecipe:
    recipe_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"recipe must be a YAML mapping: {recipe_path}")
    return TrainingRecipe.model_validate(payload)


def save_recipe(recipe: TrainingRecipe, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = recipe.model_dump(mode="json")
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destination


def default_recipe(model: str, datasets: list[str]) -> TrainingRecipe:
    return TrainingRecipe(model=model, datasets=datasets)
