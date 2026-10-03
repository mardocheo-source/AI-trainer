from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dev_eval import run_dev_batch
from .paths import runs_dir
from .preflight import preflight_recipe
from .registry import Registry
from .schema import (
    ExplorationConfig,
    GeometryConfig,
    NoiseConfig,
    RoleLossConfig,
    TrainingRecipe,
)
from .sealed import get_sealed_suite, run_sealed_batch, sealed_guard
from .tool_schema import KNOWN_TOOLS
from .trainer import run_training
from .workbook import WorkbookStore

ROUTING_SCREEN_CELLS = (
    "A-baseline",
    "B-pi-noise",
    "C-role-ce",
    "D-pi-role-ce",
)

GEOMETRY_SCREEN_CELLS = (
    "A-baseline",
    "B-pi-noise",
    "E-geometry-only",
    "F-pi-geometry",
)

LITERAL_LOCK_CELLS = (
    "A-baseline",
    "B-pi-noise",
    "F-pi-geometry",
    "G-pi-literal-lock",
)

HUMAN_AGENTIC_CELLS = (
    "A-baseline",
    "B-pi-noise",
    "C-geometry-only",
    "D-pi-geometry",
    "E-human-intent",
)


def pipeline_results_dir() -> Path:
    path = runs_dir() / "pair-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def list_pair_pipeline_results() -> list[dict[str, Any]]:
    results = []
    for path in pipeline_results_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_path"] = str(path)
            results.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(results, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_pair_pipeline_result() -> dict[str, Any] | None:
    results = list_pair_pipeline_results()
    return results[0] if results else None


def dev_pair_pipeline_results_dir() -> Path:
    path = runs_dir() / "dev-pair-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_dev_pair_pipeline_results() -> list[dict[str, Any]]:
    results = []
    for path in dev_pair_pipeline_results_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_path"] = str(path)
            results.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(results, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_dev_pair_pipeline_result() -> dict[str, Any] | None:
    results = list_dev_pair_pipeline_results()
    return results[0] if results else None


def routing_screening_results_dir() -> Path:
    path = runs_dir() / "routing-screening-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_routing_screening_results() -> list[dict[str, Any]]:
    results = []
    for path in routing_screening_results_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_path"] = str(path)
            results.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(results, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_routing_screening_result() -> dict[str, Any] | None:
    results = list_routing_screening_results()
    return results[0] if results else None


def geometry_screening_results_dir() -> Path:
    path = runs_dir() / "geometry-screening-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_geometry_screening_results() -> list[dict[str, Any]]:
    results = []
    for path in geometry_screening_results_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_path"] = str(path)
            results.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(results, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_geometry_screening_result() -> dict[str, Any] | None:
    results = list_geometry_screening_results()
    return results[0] if results else None


def literal_lock_results_dir() -> Path:
    path = runs_dir() / "literal-lock-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_literal_lock_results() -> list[dict[str, Any]]:
    results = []
    for path in literal_lock_results_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_path"] = str(path)
            results.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(results, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_literal_lock_result() -> dict[str, Any] | None:
    results = list_literal_lock_results()
    return results[0] if results else None


def human_agentic_results_dir() -> Path:
    path = runs_dir() / "human-agentic-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_human_agentic_results() -> list[dict[str, Any]]:
    results = []
    for path in human_agentic_results_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_path"] = str(path)
            results.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(results, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_human_agentic_result() -> dict[str, Any] | None:
    results = list_human_agentic_results()
    return results[0] if results else None


def prepare_human_agentic_sheets(
    template: TrainingRecipe,
    max_steps: int = 400,
) -> dict[str, Any]:
    """Create a frozen five-cell, human-instruction agentic experiment."""
    from .human_agentic_data import ensure_human_agentic_data

    if max_steps < 100:
        raise ValueError("human-agentic comparison requires at least 100 optimizer steps")
    data = ensure_human_agentic_data()
    group = f"human-agentic-v1-{uuid.uuid4().hex[:8]}"
    common = template.model_copy(update={
        "datasets": [data["paths"]["train"]],
        "validation_mode": "external",
        "validation_datasets": [data["paths"]["dev"]],
        "benchmark_profile": "mixed",
        "comparison_group": group,
        "budget_mode": "steps",
        "max_steps": int(max_steps),
        "max_source_rows": max(5_200, int(template.max_source_rows)),
        "max_training_samples": max(7_800, int(template.max_training_samples)),
        "max_validation_samples": 520,
        "agent_eval_samples": 0,
        "agent_eval_max_new_tokens": 192,
        "allowed_tools": sorted(KNOWN_TOOLS),
        "filter_unknown_tools": True,
        "tool_menu_conditioning": True,
        "sequence_length": max(2048, int(template.sequence_length)),
        "logging_steps": 10,
        "workbook_sheet_id": None,
        "output_dir": None,
    })
    noise_off = NoiseConfig(
        enabled=False,
        source="pi",
        digit_order="natural",
        alpha=2.0,
        modulation=0.1,
        scope="prompt",
        envelope="constant",
        clean_tail_fraction=0.3,
    )
    noise_on = noise_off.model_copy(update={"enabled": True})
    geometry_off = GeometryConfig(
        enabled=False,
        mode="relational",
        scope="structured",
        weight=0.005,
        layer=-1,
        margin=0.2,
        sample_tokens=32,
    )
    geometry_on = geometry_off.model_copy(update={"enabled": True})
    role_off = RoleLossConfig(enabled=False)
    human_role = RoleLossConfig(
        enabled=True,
        ordinary_weight=1.0,
        delimiter_weight=0.75,
        tool_name_weight=1.25,
        argument_key_weight=1.5,
        argument_value_weight=2.0,
    )
    intent_noise = noise_on.model_copy(update={
        "protect_prompt_literals": True,
        "value_consistency_weight": 0.01,
        "intent_consistency_weight": 0.01,
    })
    recipes = {
        "A-baseline": common.model_copy(update={
            "name": f"{group}-a-clean-rslora-s{max_steps}",
            "experiment_variant": "baseline",
            "noise": noise_off,
            "geometry": geometry_off,
            "role_loss": role_off,
        }),
        "B-pi-noise": common.model_copy(update={
            "name": f"{group}-b-pi-natural-a2-m01-clean30-s{max_steps}",
            "experiment_variant": "exotic",
            "noise": noise_on,
            "geometry": geometry_off,
            "role_loss": role_off,
        }),
        "C-geometry-only": common.model_copy(update={
            "name": f"{group}-c-relational-geo005-s{max_steps}",
            "experiment_variant": "exotic",
            "noise": noise_off,
            "geometry": geometry_on,
            "role_loss": role_off,
        }),
        "D-pi-geometry": common.model_copy(update={
            "name": f"{group}-d-pi-relational-geo005-s{max_steps}",
            "experiment_variant": "exotic",
            "noise": noise_on,
            "geometry": geometry_on,
            "role_loss": role_off,
        }),
        "E-human-intent": common.model_copy(update={
            "name": f"{group}-e-human-intent-pi-geo-ce-kl01-s{max_steps}",
            "experiment_variant": "exotic",
            "noise": intent_noise,
            "geometry": geometry_on,
            "role_loss": human_role,
        }),
    }
    store = WorkbookStore()
    exploration = ExplorationConfig(enabled=False, trial_count=1)
    sheets = {
        label: store.save(recipe.name, recipe, exploration)
        for label, recipe in recipes.items()
    }
    return {
        "comparison_group": group,
        "max_steps": int(max_steps),
        "data_audit": data,
        "dev_source": data["paths"]["dev"],
        "suite_id": data["suite_id"],
        "sheet_ids": {label: sheet.id for label, sheet in sheets.items()},
        "sheet_names": {label: sheet.name for label, sheet in sheets.items()},
    }


def prepare_literal_lock_sheets(
    template: TrainingRecipe,
    max_steps: int = 600,
) -> dict[str, Any]:
    """Build grounded train/DEV/final data and save the frozen A/B/F/G cells."""
    from .literal_data import ensure_literal_lock_data

    if max_steps < 100:
        raise ValueError("LiteralLock comparison requires at least 100 optimizer steps")
    data = ensure_literal_lock_data()
    group = f"literal-lock-v2-{uuid.uuid4().hex[:8]}"
    common = template.model_copy(
        update={
            "datasets": [data["paths"]["train"]],
            "validation_mode": "external",
            "validation_datasets": [data["paths"]["dev"]],
            "benchmark_profile": "mixed",
            "comparison_group": group,
            "budget_mode": "steps",
            "max_steps": int(max_steps),
            "max_source_rows": max(5_200, int(template.max_source_rows)),
            "max_training_samples": max(5_200, int(template.max_training_samples)),
            "max_validation_samples": 520,
            "agent_eval_samples": 0,
            "allowed_tools": sorted(KNOWN_TOOLS),
            "filter_unknown_tools": True,
            "tool_menu_conditioning": True,
            "sequence_length": max(2048, int(template.sequence_length)),
            "logging_steps": 10,
            "workbook_sheet_id": None,
            "output_dir": None,
        }
    )
    noise_off = NoiseConfig(
        enabled=False,
        source="pi",
        digit_order="natural",
        alpha=2.0,
        modulation=0.1,
        scope="prompt",
        envelope="constant",
        clean_tail_fraction=0.3,
    )
    noise_on = noise_off.model_copy(update={"enabled": True})
    geometry_off = GeometryConfig(
        enabled=False,
        mode="relational",
        scope="structured",
        weight=0.005,
        layer=-1,
        margin=0.2,
        sample_tokens=32,
    )
    structured_geometry = geometry_off.model_copy(update={"enabled": True})
    anchor_geometry = structured_geometry.model_copy(update={"scope": "anchors"})
    role_off = RoleLossConfig(enabled=False)
    literal_role = RoleLossConfig(
        enabled=True,
        ordinary_weight=1.0,
        delimiter_weight=0.75,
        tool_name_weight=1.25,
        argument_key_weight=1.5,
        argument_value_weight=3.0,
    )
    literal_noise = noise_on.model_copy(
        update={
            "protect_prompt_literals": True,
            "value_consistency_weight": 0.02,
        }
    )
    recipes = {
        "A-baseline": common.model_copy(update={
            "name": f"{group}-a-baseline-s{max_steps}",
            "experiment_variant": "baseline",
            "noise": noise_off,
            "geometry": geometry_off,
            "role_loss": role_off,
        }),
        "B-pi-noise": common.model_copy(update={
            "name": f"{group}-b-pi-natural-a2-m01-s{max_steps}",
            "experiment_variant": "exotic",
            "noise": noise_on,
            "geometry": geometry_off,
            "role_loss": role_off,
        }),
        "F-pi-geometry": common.model_copy(update={
            "name": f"{group}-f-pi-structured-geo005-s{max_steps}",
            "experiment_variant": "exotic",
            "noise": noise_on,
            "geometry": structured_geometry,
            "role_loss": role_off,
        }),
        "G-pi-literal-lock": common.model_copy(update={
            "name": f"{group}-g-pi-literal-lock-anchor-geo005-vkl02-s{max_steps}",
            "experiment_variant": "exotic",
            "noise": literal_noise,
            "geometry": anchor_geometry,
            "role_loss": literal_role,
        }),
    }
    store = WorkbookStore()
    exploration = ExplorationConfig(enabled=False, trial_count=1)
    sheets = {
        label: store.save(recipe.name, recipe, exploration)
        for label, recipe in recipes.items()
    }
    return {
        "comparison_group": group,
        "max_steps": int(max_steps),
        "data_audit": data,
        "dev_source": data["paths"]["dev"],
        "suite_id": data["suite_id"],
        "sheet_ids": {label: sheet.id for label, sheet in sheets.items()},
        "sheet_names": {label: sheet.name for label, sheet in sheets.items()},
    }


def prepare_routing_screening_sheets(
    template: TrainingRecipe,
    supplement_source: str,
    max_steps: int = 600,
) -> dict[str, Any]:
    """Save the fixed A/B/C/D cells used for routing regularizer screening."""
    if template.validation_mode != "external" or len(template.validation_datasets) != 1:
        raise ValueError("routing screening requires exactly one external DEV dataset")
    if max_steps < 50:
        raise ValueError("routing screening requires at least 50 optimizer steps")
    group = f"routing-screen-v1-{uuid.uuid4().hex[:8]}"
    common = template.model_copy(
        update={
            "datasets": [supplement_source],
            "validation_mode": "external",
            "comparison_group": group,
            "budget_mode": "steps",
            "max_steps": int(max_steps),
            "max_source_rows": max(5_200, int(template.max_source_rows)),
            "max_training_samples": max(5_200, int(template.max_training_samples)),
            "max_validation_samples": max(520, int(template.max_validation_samples)),
            "allowed_tools": sorted(KNOWN_TOOLS),
            "filter_unknown_tools": True,
            "tool_menu_conditioning": True,
            "agent_eval_samples": 0,
            "logging_steps": 10,
            "geometry": GeometryConfig(
                enabled=False,
                mode="relational",
                scope="structured",
                weight=0.001,
                layer=-1,
                margin=0.2,
                sample_tokens=32,
            ),
            "workbook_sheet_id": None,
            "output_dir": None,
        }
    )
    noise_off = NoiseConfig(
        enabled=False,
        source="pi",
        digit_order="natural",
        alpha=2.0,
        modulation=0.1,
        scope="prompt",
        envelope="constant",
        clean_tail_fraction=0.3,
    )
    noise_on = noise_off.model_copy(update={"enabled": True})
    role_off = RoleLossConfig(
        enabled=False,
        delimiter_weight=0.5,
        tool_name_weight=1.5,
        argument_key_weight=2.0,
        argument_value_weight=2.5,
    )
    role_on = role_off.model_copy(update={"enabled": True})
    recipes = {
        "A-baseline": common.model_copy(
            update={
                "name": f"{group}-a-baseline-s{max_steps}",
                "experiment_variant": "baseline",
                "noise": noise_off,
                "role_loss": role_off,
            }
        ),
        "B-pi-noise": common.model_copy(
            update={
                "name": f"{group}-b-pi-a2-m01-prompt-clean30-s{max_steps}",
                "experiment_variant": "exotic",
                "noise": noise_on,
                "role_loss": role_off,
            }
        ),
        "C-role-ce": common.model_copy(
            update={
                "name": f"{group}-c-role-ce-05-15-20-25-s{max_steps}",
                "experiment_variant": "exotic",
                "noise": noise_off,
                "role_loss": role_on,
            }
        ),
        "D-pi-role-ce": common.model_copy(
            update={
                "name": f"{group}-d-pi-role-ce-s{max_steps}",
                "experiment_variant": "exotic",
                "noise": noise_on,
                "role_loss": role_on,
            }
        ),
    }
    store = WorkbookStore()
    exploration = ExplorationConfig(enabled=False, trial_count=1)
    sheets = {
        label: store.save(recipe.name, recipe, exploration)
        for label, recipe in recipes.items()
    }
    return {
        "comparison_group": group,
        "max_steps": int(max_steps),
        "supplement_source": supplement_source,
        "sheet_ids": {label: sheet.id for label, sheet in sheets.items()},
        "sheet_names": {label: sheet.name for label, sheet in sheets.items()},
    }


def _completed_routing_screen(pipeline_id: str | None = None) -> dict[str, Any]:
    result = next(
        (
            item
            for item in list_routing_screening_results()
            if (pipeline_id is None or str(item.get("pipeline_id")) == pipeline_id)
            and item.get("status") == "complete"
        ),
        None,
    )
    if result is None:
        raise KeyError(
            f"completed routing screen not found: {pipeline_id or 'latest completed'}"
        )
    return result


def _run_recipe(run: dict[str, Any]) -> TrainingRecipe:
    return TrainingRecipe.model_validate(json.loads(run.get("recipe_json") or "{}"))


def _geometry_screen_recipes(
    baseline_recipe: TrainingRecipe,
    noise_recipe: TrainingRecipe,
    geometry_weight: float = 0.005,
    sample_tokens: int = 32,
) -> dict[str, TrainingRecipe]:
    """Build E/F while preserving the complete A/B training and DEV contract."""
    if (
        baseline_recipe.noise.enabled
        or baseline_recipe.geometry.enabled
        or baseline_recipe.role_loss.enabled
    ):
        raise ValueError("A source run must be a clean baseline")
    if (
        not noise_recipe.noise.enabled
        or noise_recipe.geometry.enabled
        or noise_recipe.role_loss.enabled
    ):
        raise ValueError("B source run must be π-noise only")
    if baseline_recipe.comparison_group != noise_recipe.comparison_group:
        raise ValueError("A and B source runs must share one comparison group")
    if _screening_core(baseline_recipe) != _screening_core(noise_recipe):
        raise ValueError("A and B source runs differ outside π-noise controls")
    noise = noise_recipe.noise
    if (
        noise.source != "pi"
        or noise.digit_order != "natural"
        or noise.scope != "prompt"
        or noise.envelope != "constant"
        or noise.alpha != 2.0
        or noise.modulation != 0.1
        or noise.clean_tail_fraction != 0.3
    ):
        raise ValueError("B must use natural π pairs, alpha=2, m=.1 and clean-tail=.3")
    if baseline_recipe.budget_mode != "steps":
        raise ValueError("geometry screening requires a fixed optimizer-step budget")
    if baseline_recipe.validation_mode != "external" or len(
        baseline_recipe.validation_datasets
    ) != 1:
        raise ValueError("geometry screening requires one shared external DEV dataset")

    geometry = GeometryConfig(
        enabled=True,
        mode="relational",
        scope="structured",
        weight=float(geometry_weight),
        layer=-1,
        margin=0.2,
        sample_tokens=int(sample_tokens),
    )
    step_count = int(baseline_recipe.max_steps)
    weight_tag = f"{float(geometry_weight):.4g}".replace(".", "p")
    group = baseline_recipe.comparison_group
    common_updates = {
        "experiment_variant": "exotic",
        "geometry": geometry,
        "workbook_sheet_id": None,
        "output_dir": None,
    }
    return {
        "E-geometry-only": baseline_recipe.model_copy(
            update={
                **common_updates,
                "name": f"{group}-e-rel-geometry-g{weight_tag}-s{step_count}",
            }
        ),
        "F-pi-geometry": noise_recipe.model_copy(
            update={
                **common_updates,
                "name": f"{group}-f-pi-rel-geometry-g{weight_tag}-s{step_count}",
            }
        ),
    }


def prepare_geometry_screening_sheets(
    routing_pipeline_id: str | None = None,
    geometry_weight: float = 0.005,
    sample_tokens: int = 32,
) -> dict[str, Any]:
    """Create only the missing E/F sheets from a completed A/B/C/D screen."""
    source = _completed_routing_screen(routing_pipeline_id)
    cells = source.get("cells") or {}
    baseline_run_id = str((cells.get("A-baseline") or {}).get("run_id") or "")
    noise_run_id = str((cells.get("B-pi-noise") or {}).get("run_id") or "")
    if not baseline_run_id or not noise_run_id:
        raise ValueError("completed routing screen does not contain reusable A and B runs")
    registry = Registry()
    baseline_run = registry.get_run(baseline_run_id)
    noise_run = registry.get_run(noise_run_id)
    if baseline_run.get("status") != "complete" or noise_run.get("status") != "complete":
        raise ValueError("A and B source runs must both be complete")
    recipes = _geometry_screen_recipes(
        _run_recipe(baseline_run),
        _run_recipe(noise_run),
        geometry_weight=geometry_weight,
        sample_tokens=sample_tokens,
    )
    store = WorkbookStore()
    exploration = ExplorationConfig(enabled=False, trial_count=1)
    sheets = {
        label: store.save(recipe.name, recipe, exploration)
        for label, recipe in recipes.items()
    }
    return {
        "source_pipeline_id": source["pipeline_id"],
        "comparison_group": recipes["E-geometry-only"].comparison_group,
        "baseline_run_id": baseline_run_id,
        "noise_run_id": noise_run_id,
        "max_steps": recipes["E-geometry-only"].max_steps,
        "dev_source": recipes["E-geometry-only"].validation_datasets[0],
        "dev_samples_requested": int(source.get("dev_samples_requested") or 520),
        "geometry_weight": float(geometry_weight),
        "sample_tokens": int(sample_tokens),
        "sheet_ids": {label: sheet.id for label, sheet in sheets.items()},
        "sheet_names": {label: sheet.name for label, sheet in sheets.items()},
    }


def _screening_core(recipe: TrainingRecipe) -> dict[str, Any]:
    payload = recipe.model_dump(mode="json")
    for key in (
        "name",
        "output_dir",
        "workbook_sheet_id",
        "experiment_variant",
        "noise",
        "geometry",
        "role_loss",
    ):
        payload.pop(key, None)
    return payload


def _load_screening_sheets(sheet_ids: dict[str, str]) -> dict[str, Any]:
    if set(sheet_ids) != set(ROUTING_SCREEN_CELLS):
        raise ValueError(f"screening requires exactly these cells: {ROUTING_SCREEN_CELLS}")
    store = WorkbookStore()
    sheets = {label: store.get(sheet_ids[label]) for label in ROUTING_SCREEN_CELLS}
    recipes = {label: sheet.recipe for label, sheet in sheets.items()}
    group = recipes["A-baseline"].comparison_group
    if any(recipe.comparison_group != group for recipe in recipes.values()):
        raise ValueError("all routing-screen cells must share one comparison group")
    core = _screening_core(recipes["A-baseline"])
    if any(_screening_core(recipe) != core for recipe in recipes.values()):
        raise ValueError("routing-screen recipes differ outside π noise and role-weighted CE")
    expected = {
        "A-baseline": (False, False),
        "B-pi-noise": (True, False),
        "C-role-ce": (False, True),
        "D-pi-role-ce": (True, True),
    }
    for label, recipe in recipes.items():
        if recipe.geometry.enabled:
            raise ValueError("geometry must remain disabled during the A/B/C/D screening")
        if (recipe.noise.enabled, recipe.role_loss.enabled) != expected[label]:
            raise ValueError(f"invalid regularizer controls for {label}")
        if recipe.budget_mode != "steps":
            raise ValueError("routing screening requires fixed optimizer steps")
        if recipe.validation_mode != "external" or len(recipe.validation_datasets) != 1:
            raise ValueError("routing screening requires one shared external DEV dataset")
    noise = recipes["B-pi-noise"].noise
    if (
        noise.source != "pi"
        or noise.digit_order != "natural"
        or noise.scope != "prompt"
        or noise.envelope != "constant"
        or noise.alpha != 2.0
        or noise.modulation != 0.1
        or noise.clean_tail_fraction != 0.3
    ):
        raise ValueError("π cell must use natural digit pairs, alpha=2, m=.1 and clean-tail=.3")
    return sheets


def _screening_summary(
    results: list[dict[str, Any]],
    labels: tuple[str, ...] = ROUTING_SCREEN_CELLS,
) -> dict[str, Any]:
    ordered_labels = list(labels)
    cells: dict[str, dict[str, Any]] = {}
    for label, result in zip(ordered_labels, results, strict=False):
        metrics = result.get("adapter") or {}
        cells[label] = {
            "balanced_routing": metrics.get("agent_balanced_tool_decision_accuracy"),
            "tool_recall": metrics.get("agent_tool_recall"),
            "no_tool_specificity": metrics.get("agent_no_tool_specificity"),
            "routing_mcc": metrics.get("agent_tool_decision_mcc"),
            "tool_name": metrics.get("agent_selected_tool_name_accuracy"),
            "argument_keys": metrics.get("agent_tool_argument_keys_accuracy"),
            "exact_arguments": metrics.get("agent_tool_argument_exact_accuracy"),
            "protocol_id": metrics.get("agent_protocol_id"),
        }
    baseline = cells.get("A-baseline", {})
    baseline_tool = float(baseline.get("tool_name") or 0.0)
    baseline_exact = float(baseline.get("exact_arguments") or 0.0)
    eligible = []
    for label, metrics in cells.items():
        values = {
            key: float(metrics.get(key) or 0.0)
            for key in (
                "balanced_routing",
                "tool_recall",
                "no_tool_specificity",
                "routing_mcc",
                "tool_name",
                "exact_arguments",
            )
        }
        metrics["passes_gates"] = bool(
            values["no_tool_specificity"] >= 0.60
            and values["tool_recall"] >= 0.85
            and values["balanced_routing"] >= 0.70
            and values["routing_mcc"] > 0.0
            and values["tool_name"] >= baseline_tool - 0.02
            and values["exact_arguments"] >= baseline_exact
        )
        metrics["selection_score"] = (
            0.30 * values["balanced_routing"]
            + 0.15 * values["no_tool_specificity"]
            + 0.15 * values["tool_recall"]
            + 0.20 * values["tool_name"]
            + 0.20 * values["exact_arguments"]
        )
        if metrics["passes_gates"]:
            eligible.append(label)
    recommended = max(
        eligible,
        key=lambda label: float(cells[label]["selection_score"]),
        default=None,
    )
    return {"cells": cells, "eligible_cells": eligible, "recommended_cell": recommended}


def _continue_routing_screening(state: dict[str, Any], result_path: Path) -> dict[str, Any]:
    sheets = _load_screening_sheets(dict(state["sheet_ids"]))
    registry = Registry()
    preflight = state.get("preflight") or {}
    run_ids: list[str] = []
    try:
        for label in ROUTING_SCREEN_CELLS:
            cell = state["cells"].setdefault(label, {})
            run_id = str(cell.get("run_id") or "")
            if run_id:
                try:
                    complete = registry.get_run(run_id).get("status") == "complete"
                except KeyError:
                    complete = False
                if complete:
                    cell["status"] = "complete"
                    run_ids.append(run_id)
                    continue
            state.update(status="running", stage=f"training-{label}")
            cell["status"] = "training"
            _write_state(result_path, state)
            metrics = run_training(sheets[label].recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(f"{label} did not reach the fixed-step target")
            run_id = str(metrics["run_id"])
            cell.update(status="complete", run_id=run_id, metrics=metrics)
            run_ids.append(run_id)
            state["run_ids"] = list(run_ids)
            _write_state(result_path, state)
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state

        state.update(
            status="running",
            stage="external-dev-generation-evaluation-4-way",
            run_ids=run_ids,
        )
        _write_state(result_path, state)
        source = sheets["A-baseline"].recipe.validation_datasets[0]
        evaluation = run_dev_batch(
            run_ids=run_ids,
            source=source,
            max_samples=int(state.get("dev_samples_requested") or 520),
            compare_base=bool(state.get("compare_base", True)),
        )
        state["dev_evaluation"] = evaluation
        if evaluation.get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-dev-evaluation")
        else:
            state["screening_summary"] = _screening_summary(evaluation.get("results") or [])
            state.update(status="complete", stage="compare-ready", compare_run_ids=run_ids)
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        state["preflight"] = preflight
        _write_state(result_path, state)
        raise


def run_routing_screening_pipeline(
    sheet_ids: dict[str, str],
    max_samples: int = 520,
    compare_base: bool = True,
) -> dict[str, Any]:
    """Train baseline, π-only, role-CE-only and combined cells, then run one DEV."""
    sheets = _load_screening_sheets(sheet_ids)
    sealed_hashes, sealed_sources = sealed_guard()
    # Data, model, token limit and tool contract are intentionally identical
    # across A/B/F/G. Probing the 5,200-row source four times only inflates the
    # GUI worker's RAM without adding another safety check.
    common_preflight = preflight_recipe(
        sheets["A-baseline"].recipe,
        sealed_hashes,
        sealed_sources,
    )
    preflight = {label: common_preflight for label in LITERAL_LOCK_CELLS}
    blocked = [label for label, report in preflight.items() if report.get("status") == "blocked"]
    if blocked:
        raise ValueError(f"routing screening blocked by preflight: {blocked}")
    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    result_path = routing_screening_results_dir() / f"{pipeline_id}.json"
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "pipeline_type": "routing-regularizer-abcd-screen",
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_group": sheets["A-baseline"].recipe.comparison_group,
        "sheet_ids": dict(sheet_ids),
        "dev_source": sheets["A-baseline"].recipe.validation_datasets[0],
        "dev_samples_requested": int(max_samples),
        "compare_base": bool(compare_base),
        "preflight": preflight,
        "cells": {label: {"status": "pending"} for label in ROUTING_SCREEN_CELLS},
        "run_ids": [],
    }
    _write_state(result_path, state)
    return _continue_routing_screening(state, result_path)


def resume_routing_screening_pipeline(pipeline_id: str | None = None) -> dict[str, Any]:
    """Resume a screening from completed cells; completed adapters are never retrained."""
    state = next(
        (
            item
            for item in list_routing_screening_results()
            if pipeline_id is None or str(item.get("pipeline_id")) == pipeline_id
        ),
        None,
    )
    if state is None:
        raise KeyError(f"routing screening pipeline not found: {pipeline_id or 'latest'}")
    if state.get("status") == "complete":
        return state
    result_path = Path(str(state["result_path"]))
    state.pop("error", None)
    state.update(status="running", stage="resuming")
    _write_state(result_path, state)
    return _continue_routing_screening(state, result_path)


def _load_geometry_screen_inputs(
    baseline_run_id: str,
    noise_run_id: str,
    geometry_sheet_id: str,
    combined_sheet_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    registry = Registry()
    baseline_run = registry.get_run(baseline_run_id)
    noise_run = registry.get_run(noise_run_id)
    if baseline_run.get("status") != "complete" or noise_run.get("status") != "complete":
        raise ValueError("reused A and B runs must both be complete")
    baseline_recipe = _run_recipe(baseline_run)
    noise_recipe = _run_recipe(noise_run)
    _geometry_screen_recipes(baseline_recipe, noise_recipe)

    store = WorkbookStore()
    geometry_sheet = store.get(geometry_sheet_id)
    combined_sheet = store.get(combined_sheet_id)
    geometry_recipe = geometry_sheet.recipe
    combined_recipe = combined_sheet.recipe
    recipes = (baseline_recipe, noise_recipe, geometry_recipe, combined_recipe)
    if len({recipe.comparison_group for recipe in recipes}) != 1:
        raise ValueError("A/B/E/F must share one comparison group")
    core = _screening_core(baseline_recipe)
    if any(_screening_core(recipe) != core for recipe in recipes[1:]):
        raise ValueError("A/B/E/F differ outside noise and geometry controls")
    if geometry_recipe.noise.enabled or not geometry_recipe.geometry.enabled:
        raise ValueError("E must disable noise and enable geometry")
    if not combined_recipe.noise.enabled or not combined_recipe.geometry.enabled:
        raise ValueError("F must enable both natural π noise and geometry")
    if any(recipe.role_loss.enabled for recipe in recipes):
        raise ValueError("role-weighted CE must remain off in the geometry screen")
    if geometry_recipe.geometry != combined_recipe.geometry:
        raise ValueError("E and F must use the same geometry configuration")
    geometry = geometry_recipe.geometry
    if (
        geometry.mode != "relational"
        or geometry.scope != "structured"
        or geometry.layer != -1
        or geometry.margin != 0.2
    ):
        raise ValueError("E/F must use last-layer structured relational geometry")
    if combined_recipe.noise != noise_recipe.noise:
        raise ValueError("F must preserve the exact natural-π configuration from B")
    return baseline_run, noise_run, geometry_sheet, combined_sheet


def _load_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(result.get("adapter"), dict):
        return result
    result_path = result.get("result_path")
    if not result_path:
        raise ValueError("DEV result reference has no result_path")
    return json.loads(Path(str(result_path)).read_text(encoding="utf-8"))


def _result_reference(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "evaluation_kind",
            "run_id",
            "suite_id",
            "suite_name",
            "source",
            "evaluated_at",
            "max_samples",
            "result_path",
        )
        if key in result
    }


def _geometry_source_results(
    source_pipeline_id: str,
    baseline_run_id: str,
    noise_run_id: str,
    max_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _completed_routing_screen(source_pipeline_id)
    if int(source.get("dev_samples_requested") or 0) != int(max_samples):
        raise ValueError(
            "geometry screen must reuse the exact A/B DEV size from the routing screen"
        )
    evaluation = source.get("dev_evaluation") or {}
    by_run = {
        str(result.get("run_id")): _load_result_payload(result)
        for result in evaluation.get("results") or []
    }
    missing = [run_id for run_id in (baseline_run_id, noise_run_id) if run_id not in by_run]
    if missing:
        raise ValueError(f"source routing screen has no reusable DEV results for: {missing}")
    return source, [by_run[baseline_run_id], by_run[noise_run_id]]


def _continue_geometry_screening(state: dict[str, Any], result_path: Path) -> dict[str, Any]:
    baseline_run_id = str(state["baseline_run_id"])
    noise_run_id = str(state["noise_run_id"])
    _baseline, _noise, geometry_sheet, combined_sheet = _load_geometry_screen_inputs(
        baseline_run_id,
        noise_run_id,
        str(state["sheet_ids"]["E-geometry-only"]),
        str(state["sheet_ids"]["F-pi-geometry"]),
    )
    registry = Registry()
    run_ids = [baseline_run_id, noise_run_id]
    try:
        for label, sheet in (
            ("E-geometry-only", geometry_sheet),
            ("F-pi-geometry", combined_sheet),
        ):
            cell = state["cells"].setdefault(label, {})
            run_id = str(cell.get("run_id") or "")
            if run_id:
                try:
                    complete = registry.get_run(run_id).get("status") == "complete"
                except KeyError:
                    complete = False
                if complete:
                    cell["status"] = "complete"
                    run_ids.append(run_id)
                    continue
            state.update(status="running", stage=f"training-{label}")
            cell["status"] = "training"
            _write_state(result_path, state)
            metrics = run_training(sheet.recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(f"{label} did not reach the fixed-step target")
            run_id = str(metrics["run_id"])
            cell.update(status="complete", run_id=run_id, metrics=metrics)
            run_ids.append(run_id)
            state["run_ids"] = list(run_ids)
            _write_state(result_path, state)
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state

        state.update(
            status="running",
            stage="external-dev-generation-evaluation-E-F",
            run_ids=run_ids,
        )
        _write_state(result_path, state)
        source_state, reused_results = _geometry_source_results(
            str(state["source_pipeline_id"]),
            baseline_run_id,
            noise_run_id,
            int(state["dev_samples_requested"]),
        )
        new_evaluation = run_dev_batch(
            run_ids=run_ids[2:],
            source=str(state["dev_source"]),
            max_samples=int(state["dev_samples_requested"]),
            compare_base=False,
        )
        if new_evaluation.get("status") == "stopped":
            state["dev_evaluation"] = {
                **new_evaluation,
                "results": [_result_reference(item) for item in new_evaluation.get("results", [])],
            }
            state.update(status="stopped", stage="stopped-during-dev-evaluation")
        else:
            new_results = [
                _load_result_payload(result) for result in new_evaluation.get("results") or []
            ]
            all_results = [*reused_results, *new_results]
            if len(all_results) != len(GEOMETRY_SCREEN_CELLS):
                raise RuntimeError("geometry screen did not produce all four DEV results")
            state["dev_evaluation"] = {
                "status": "complete",
                "evaluation_kind": "external-dev",
                "source": state["dev_source"],
                "runs_requested": len(GEOMETRY_SCREEN_CELLS),
                "runs_evaluated": len(all_results),
                "reused_from_pipeline": source_state["pipeline_id"],
                "results": [_result_reference(item) for item in all_results],
            }
            state["screening_summary"] = _screening_summary(
                all_results,
                labels=GEOMETRY_SCREEN_CELLS,
            )
            state.update(status="complete", stage="compare-ready", compare_run_ids=run_ids)
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        _write_state(result_path, state)
        raise


def run_geometry_screening_pipeline(
    baseline_run_id: str,
    noise_run_id: str,
    geometry_sheet_id: str,
    combined_sheet_id: str,
    source_pipeline_id: str,
    max_samples: int = 520,
) -> dict[str, Any]:
    """Reuse A/B, train E/F and compare all four on DEV without touching sealed data."""
    baseline, noise, geometry_sheet, combined_sheet = _load_geometry_screen_inputs(
        baseline_run_id,
        noise_run_id,
        geometry_sheet_id,
        combined_sheet_id,
    )
    source_state, _results = _geometry_source_results(
        source_pipeline_id,
        baseline_run_id,
        noise_run_id,
        max_samples,
    )
    dev_source = geometry_sheet.recipe.validation_datasets[0]
    if dev_source != str(source_state.get("dev_source") or ""):
        raise ValueError("E/F external DEV source differs from the reusable A/B screen")
    sealed_hashes, sealed_sources = sealed_guard()
    preflight = {
        "E-geometry-only": preflight_recipe(
            geometry_sheet.recipe, sealed_hashes, sealed_sources
        ),
        "F-pi-geometry": preflight_recipe(
            combined_sheet.recipe, sealed_hashes, sealed_sources
        ),
    }
    blocked = [label for label, report in preflight.items() if report.get("status") == "blocked"]
    if blocked:
        raise ValueError(f"geometry screening blocked by preflight: {blocked}")
    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    result_path = geometry_screening_results_dir() / f"{pipeline_id}.json"
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "pipeline_type": "geometry-regularizer-ABEF-screen",
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "source_pipeline_id": source_pipeline_id,
        "comparison_group": geometry_sheet.recipe.comparison_group,
        "baseline_run_id": baseline_run_id,
        "noise_run_id": noise_run_id,
        "sheet_ids": {
            "E-geometry-only": geometry_sheet_id,
            "F-pi-geometry": combined_sheet_id,
        },
        "dev_source": dev_source,
        "dev_samples_requested": int(max_samples),
        "preflight": preflight,
        "cells": {
            "A-baseline": {"status": "complete", "run_id": baseline["run_id"]},
            "B-pi-noise": {"status": "complete", "run_id": noise["run_id"]},
            "E-geometry-only": {"status": "pending", "sheet_id": geometry_sheet_id},
            "F-pi-geometry": {"status": "pending", "sheet_id": combined_sheet_id},
        },
        "run_ids": [baseline_run_id, noise_run_id],
    }
    _write_state(result_path, state)
    return _continue_geometry_screening(state, result_path)


def resume_geometry_screening_pipeline(pipeline_id: str | None = None) -> dict[str, Any]:
    """Resume E/F training or DEV evaluation without repeating completed geometry cells."""
    state = next(
        (
            item
            for item in list_geometry_screening_results()
            if pipeline_id is None or str(item.get("pipeline_id")) == pipeline_id
        ),
        None,
    )
    if state is None:
        raise KeyError(f"geometry screening pipeline not found: {pipeline_id or 'latest'}")
    if state.get("status") == "complete":
        return state
    result_path = Path(str(state["result_path"]))
    state.pop("error", None)
    state.update(status="running", stage="resuming")
    _write_state(result_path, state)
    return _continue_geometry_screening(state, result_path)


def resume_dev_pair_pipeline(pipeline_id: str | None = None) -> dict[str, Any]:
    """Resume only the DEV evaluation for an already trained fair pair."""
    candidates = list_dev_pair_pipeline_results()
    state = next(
        (
            item
            for item in candidates
            if pipeline_id is None or str(item.get("pipeline_id")) == pipeline_id
        ),
        None,
    )
    if state is None:
        raise KeyError(f"DEV pair pipeline not found: {pipeline_id or 'latest'}")
    if state.get("status") == "complete":
        return state
    run_ids = [str(value) for value in state.get("run_ids") or []]
    if len(run_ids) != 2:
        raise ValueError("both baseline and exotic training runs must exist before DEV resume")
    registry = Registry()
    incomplete = [run_id for run_id in run_ids if registry.get_run(run_id)["status"] != "complete"]
    if incomplete:
        raise ValueError(f"cannot resume DEV evaluation; incomplete runs: {incomplete}")
    source = str(state.get("dev_source") or "")
    if not source:
        raise ValueError("pipeline state has no external DEV source")
    result_path = Path(str(state["result_path"]))
    state.update(status="running", stage="external-dev-generation-evaluation-resumed")
    state.pop("error", None)
    _write_state(result_path, state)
    try:
        state["dev_evaluation"] = run_dev_batch(
            run_ids=run_ids,
            source=source,
            max_samples=int(state.get("dev_samples_requested") or 520),
            compare_base=bool(state.get("compare_base", True)),
        )
        if state["dev_evaluation"].get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-dev-evaluation")
        else:
            state.update(status="complete", stage="compare-ready", compare_run_ids=run_ids)
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        _write_state(result_path, state)
        raise


def ablation_results_dir() -> Path:
    path = runs_dir() / "ablation-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_ablation_pipeline_result() -> dict[str, Any] | None:
    results = []
    for path in ablation_results_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_path"] = str(path)
            results.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return max(results, key=lambda item: str(item.get("created_at") or ""), default=None)


def _recipe_core(recipe: Any) -> dict[str, Any]:
    payload = recipe.model_dump(mode="json")
    for key in (
        "name",
        "output_dir",
        "workbook_sheet_id",
        "experiment_variant",
        "noise",
        "geometry",
        "role_loss",
    ):
        payload.pop(key, None)
    return payload


def run_ablation_pipeline(
    baseline_run_id: str,
    noise_run_id: str,
    geometry_sheet_id: str,
    combined_sheet_id: str,
    suite_id: str,
    max_samples: int,
    compare_base: bool = True,
) -> dict[str, Any]:
    """Train geometry-only and combined variants, then seal all four ablation cells."""
    registry = Registry()
    store = WorkbookStore()
    existing = {
        "baseline": registry.get_run(baseline_run_id),
        "noise-only": registry.get_run(noise_run_id),
    }
    if any(run.get("status") != "complete" for run in existing.values()):
        raise ValueError("baseline and noise-only runs must both be complete")
    existing_recipes = {
        name: TrainingRecipe.model_validate(json.loads(run.get("recipe_json") or "{}"))
        for name, run in existing.items()
    }
    baseline_recipe = existing_recipes["baseline"]
    noise_recipe = existing_recipes["noise-only"]
    geometry = store.get(geometry_sheet_id)
    combined = store.get(combined_sheet_id)
    if baseline_recipe.noise.enabled or baseline_recipe.geometry.enabled:
        raise ValueError("selected baseline run is not a clean baseline")
    if not noise_recipe.noise.enabled or noise_recipe.geometry.enabled:
        raise ValueError("selected noise-only run must enable noise and disable geometry")
    if geometry.recipe.noise.enabled or not geometry.recipe.geometry.enabled:
        raise ValueError("geometry-only sheet must disable noise and enable geometry")
    if not (combined.recipe.noise.enabled and combined.recipe.geometry.enabled):
        raise ValueError("combined sheet must enable both noise and geometry")
    recipes = [baseline_recipe, noise_recipe, geometry.recipe, combined.recipe]
    if len({recipe.comparison_group for recipe in recipes}) != 1:
        raise ValueError("all four ablation cells must share one comparison group")
    core = _recipe_core(baseline_recipe)
    if any(_recipe_core(recipe) != core for recipe in recipes[1:]):
        raise ValueError("ablation recipes differ outside noise/geometry controls")

    sealed_hashes, sealed_sources = sealed_guard()
    preflight = {
        "geometry-only": preflight_recipe(geometry.recipe, sealed_hashes, sealed_sources),
        "noise+geometry": preflight_recipe(combined.recipe, sealed_hashes, sealed_sources),
    }
    blocked = [name for name, report in preflight.items() if report.get("status") == "blocked"]
    if blocked:
        raise ValueError(f"ablation blocked by preflight: {blocked}")

    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    result_path = ablation_results_dir() / f"{pipeline_id}.json"
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "pipeline_type": "2x2-ablation",
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_group": baseline_recipe.comparison_group,
        "suite_id": suite_id,
        "sealed_samples_requested": int(max_samples),
        "compare_base": bool(compare_base),
        "preflight": preflight,
        "run_ids": [baseline_run_id, noise_run_id],
        "baseline_run_id": baseline_run_id,
        "noise_only_run_id": noise_run_id,
        "geometry_sheet_id": geometry_sheet_id,
        "combined_sheet_id": combined_sheet_id,
    }
    _write_state(result_path, state)
    try:
        for label, sheet in (("geometry-only", geometry), ("noise+geometry", combined)):
            state["stage"] = f"training-{label}"
            _write_state(result_path, state)
            metrics = run_training(sheet.recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(
                    f"{label} did not reach its fixed-step target; sealed comparison aborted"
                )
            run_id = str(metrics["run_id"])
            state["run_ids"].append(run_id)
            state[f"{label.replace('+', '_').replace('-', '_')}_run_id"] = run_id
            state[f"{label.replace('+', '_').replace('-', '_')}_metrics"] = metrics
            _write_state(result_path, state)
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state
        state["stage"] = "sealed-evaluation-4-way"
        _write_state(result_path, state)
        state["sealed"] = run_sealed_batch(
            run_ids=list(state["run_ids"]),
            suite_id=suite_id,
            max_samples=int(max_samples),
            compare_base=bool(compare_base),
        )
        if state["sealed"].get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-sealed")
        else:
            state.update(status="complete", stage="compare-ready", compare_run_ids=list(state["run_ids"]))
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        _write_state(result_path, state)
        raise


def run_pair_pipeline(
    baseline_sheet_id: str,
    exotic_sheet_id: str,
    suite_id: str,
    max_samples: int,
    compare_base: bool = True,
) -> dict[str, Any]:
    """Run a saved fair pair, sealed evaluation and compare handoff sequentially."""
    store = WorkbookStore()
    baseline = store.get(baseline_sheet_id)
    exotic = store.get(exotic_sheet_id)
    if baseline.recipe.experiment_variant != "baseline":
        raise ValueError("the first sheet must be a baseline variant")
    if exotic.recipe.experiment_variant != "exotic":
        raise ValueError("the second sheet must be an exotic variant")
    if baseline.recipe.comparison_group != exotic.recipe.comparison_group:
        raise ValueError("baseline and exotic must share one comparison group")
    if (
        baseline.recipe.noise.enabled
        or baseline.recipe.geometry.enabled
        or baseline.recipe.role_loss.enabled
    ):
        raise ValueError("baseline sheet must keep all exotic controls disabled")
    if not (
        exotic.recipe.noise.enabled
        or exotic.recipe.geometry.enabled
        or exotic.recipe.role_loss.enabled
    ):
        raise ValueError("exotic sheet must enable noise, geometry and/or role loss")

    sealed_hashes, sealed_sources = sealed_guard()
    preflight = {
        "baseline": preflight_recipe(baseline.recipe, sealed_hashes, sealed_sources),
        "exotic": preflight_recipe(exotic.recipe, sealed_hashes, sealed_sources),
    }
    blocked = {name: report for name, report in preflight.items() if report["status"] == "blocked"}
    if blocked:
        raise ValueError(f"pipeline blocked by preflight: {list(blocked)}")

    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    result_path = pipeline_results_dir() / f"{pipeline_id}.json"
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_group": baseline.recipe.comparison_group,
        "baseline_sheet_id": baseline.id,
        "exotic_sheet_id": exotic.id,
        "suite_id": suite_id,
        "sealed_samples_requested": int(max_samples),
        "compare_base": bool(compare_base),
        "preflight": preflight,
        "run_ids": [],
    }
    _write_state(result_path, state)
    try:
        for label, sheet in (("baseline", baseline), ("exotic", exotic)):
            state["stage"] = f"training-{label}"
            _write_state(result_path, state)
            metrics = run_training(sheet.recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(
                    f"{label} did not reach its fixed-step target; fair comparison aborted"
                )
            run_id = str(metrics["run_id"])
            state["run_ids"].append(run_id)
            state[f"{label}_run_id"] = run_id
            state[f"{label}_metrics"] = metrics
            _write_state(result_path, state)
            if metrics.get("stopped_by_user"):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if stop_file and Path(stop_file).exists():
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state

        state["stage"] = "sealed-evaluation"
        _write_state(result_path, state)
        state["sealed"] = run_sealed_batch(
            run_ids=list(state["run_ids"]),
            suite_id=suite_id,
            max_samples=int(max_samples),
            compare_base=bool(compare_base),
        )
        if state["sealed"].get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-sealed")
        else:
            state.update(
                status="complete",
                stage="compare-ready",
                compare_run_ids=list(state["run_ids"]),
            )
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(
            status="failed",
            stage="failed",
            error=f"{type(error).__name__}: {error}",
        )
        _write_state(result_path, state)
        raise


def run_dev_pair_pipeline(
    baseline_sheet_id: str,
    exotic_sheet_id: str,
    max_samples: int,
    compare_base: bool = True,
) -> dict[str, Any]:
    """Train a fair pair and evaluate both on their shared external DEV source."""
    store = WorkbookStore()
    baseline = store.get(baseline_sheet_id)
    exotic = store.get(exotic_sheet_id)
    if baseline.recipe.experiment_variant != "baseline":
        raise ValueError("the first sheet must be a baseline variant")
    if exotic.recipe.experiment_variant != "exotic":
        raise ValueError("the second sheet must be an exotic variant")
    if baseline.recipe.comparison_group != exotic.recipe.comparison_group:
        raise ValueError("baseline and exotic must share one comparison group")
    if (
        baseline.recipe.noise.enabled
        or baseline.recipe.geometry.enabled
        or baseline.recipe.role_loss.enabled
    ):
        raise ValueError("baseline sheet must keep all exotic controls disabled")
    if not (
        exotic.recipe.noise.enabled
        or exotic.recipe.geometry.enabled
        or exotic.recipe.role_loss.enabled
    ):
        raise ValueError("exotic sheet must enable noise, geometry and/or role loss")
    if baseline.recipe.validation_mode != "external" or exotic.recipe.validation_mode != "external":
        raise ValueError("overnight DEV pipeline requires external validation mode")
    if baseline.recipe.validation_datasets != exotic.recipe.validation_datasets:
        raise ValueError("fair-pair sheets must use the same external DEV datasets")
    if len(baseline.recipe.validation_datasets) != 1:
        raise ValueError("select exactly one external DEV dataset for the overnight pipeline")
    if baseline.recipe.budget_mode != "steps" or exotic.recipe.budget_mode != "steps":
        raise ValueError("overnight fair comparison requires a fixed optimizer-step budget")

    sealed_hashes, sealed_sources = sealed_guard()
    preflight = {
        "baseline": preflight_recipe(baseline.recipe, sealed_hashes, sealed_sources),
        "exotic": preflight_recipe(exotic.recipe, sealed_hashes, sealed_sources),
    }
    blocked = {name: report for name, report in preflight.items() if report["status"] == "blocked"}
    if blocked:
        raise ValueError(f"DEV pipeline blocked by preflight: {list(blocked)}")

    source = baseline.recipe.validation_datasets[0]
    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    result_path = dev_pair_pipeline_results_dir() / f"{pipeline_id}.json"
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "pipeline_type": "external-dev-fair-pair",
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_group": baseline.recipe.comparison_group,
        "baseline_sheet_id": baseline.id,
        "exotic_sheet_id": exotic.id,
        "dev_source": source,
        "dev_samples_requested": int(max_samples),
        "compare_base": bool(compare_base),
        "preflight": preflight,
        "run_ids": [],
    }
    _write_state(result_path, state)
    try:
        for label, sheet in (("baseline", baseline), ("exotic", exotic)):
            state["stage"] = f"training-{label}"
            _write_state(result_path, state)
            metrics = run_training(sheet.recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(
                    f"{label} did not reach its fixed-step target; DEV comparison aborted"
                )
            run_id = str(metrics["run_id"])
            state["run_ids"].append(run_id)
            state[f"{label}_run_id"] = run_id
            state[f"{label}_metrics"] = metrics
            _write_state(result_path, state)
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state

        state["stage"] = "external-dev-generation-evaluation"
        _write_state(result_path, state)
        state["dev_evaluation"] = run_dev_batch(
            run_ids=list(state["run_ids"]),
            source=source,
            max_samples=int(max_samples),
            compare_base=bool(compare_base),
        )
        if state["dev_evaluation"].get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-dev-evaluation")
        else:
            state.update(
                status="complete",
                stage="compare-ready",
                compare_run_ids=list(state["run_ids"]),
            )
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        _write_state(result_path, state)
        raise


def _load_literal_lock_sheets(sheet_ids: dict[str, str]) -> dict[str, Any]:
    if set(sheet_ids) != set(LITERAL_LOCK_CELLS):
        raise ValueError(f"LiteralLock requires exactly these cells: {LITERAL_LOCK_CELLS}")
    store = WorkbookStore()
    sheets = {label: store.get(str(sheet_ids[label])) for label in LITERAL_LOCK_CELLS}
    recipes = {label: sheet.recipe for label, sheet in sheets.items()}
    group = recipes["A-baseline"].comparison_group
    if any(recipe.comparison_group != group for recipe in recipes.values()):
        raise ValueError("all LiteralLock cells must share one comparison group")
    core = _screening_core(recipes["A-baseline"])
    if any(_screening_core(recipe) != core for recipe in recipes.values()):
        raise ValueError("LiteralLock cells differ outside regularization controls")
    a, b, f, g = (recipes[label] for label in LITERAL_LOCK_CELLS)
    if a.experiment_variant != "baseline" or any(
        (a.noise.enabled, a.geometry.enabled, a.role_loss.enabled)
    ):
        raise ValueError("A must be the clean rsLoRA baseline")
    if not b.noise.enabled or b.geometry.enabled or b.role_loss.enabled:
        raise ValueError("B must be natural π-noise only")
    if not f.noise.enabled or not f.geometry.enabled or f.role_loss.enabled:
        raise ValueError("F must be π-noise plus structured geometry")
    if not (g.noise.enabled and g.geometry.enabled and g.role_loss.enabled):
        raise ValueError("G must enable π LiteralLock, anchor geometry and value weighting")
    for label, recipe in recipes.items():
        if recipe.budget_mode != "steps":
            raise ValueError(f"{label} must use a fixed optimizer-step budget")
        if recipe.validation_mode != "external" or len(recipe.validation_datasets) != 1:
            raise ValueError(f"{label} must use exactly one shared external DEV source")
        if recipe.allowed_tools != sorted(KNOWN_TOOLS):
            raise ValueError(f"{label} must use the complete 13-tool menu")
        if not recipe.tool_menu_conditioning:
            raise ValueError(f"{label} must train with the inference tool menu")
    if any(
        recipe.validation_datasets != a.validation_datasets
        or recipe.datasets != a.datasets
        or recipe.max_steps != a.max_steps
        or recipe.seed != a.seed
        for recipe in recipes.values()
    ):
        raise ValueError("LiteralLock cells must share data, step count and seed")
    for label in ("B-pi-noise", "F-pi-geometry", "G-pi-literal-lock"):
        noise = recipes[label].noise
        if (
            noise.source != "pi"
            or noise.digit_order != "natural"
            or noise.alpha != 2.0
            or noise.modulation != 0.1
            or noise.scope != "prompt"
            or noise.envelope != "constant"
            or noise.clean_tail_fraction != 0.3
        ):
            raise ValueError(f"{label} changed the frozen natural π-pair schedule")
    if g.geometry.scope != "anchors" or not g.noise.protect_prompt_literals:
        raise ValueError("G must protect prompt literals and use anchor-only geometry")
    if g.noise.value_consistency_weight <= 0:
        raise ValueError("G must enable argument-value clean/noisy consistency")
    return sheets


def _literal_lock_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for label, result in zip(
        LITERAL_LOCK_CELLS,
        evaluation.get("results") or [],
        strict=False,
    ):
        metrics = result.get("adapter") or {}
        cells[label] = {
            "run_id": result.get("run_id"),
            "balanced_routing": metrics.get("agent_balanced_tool_decision_accuracy"),
            "tool_name": metrics.get("agent_selected_tool_name_accuracy"),
            "required_arguments_present": metrics.get("agent_tool_arguments_valid_rate"),
            "strict_whole_call_exact": metrics.get("agent_tool_argument_exact_accuracy"),
            "required_call_exact": metrics.get("agent_required_argument_call_exact_accuracy"),
            "required_field_micro": metrics.get("agent_required_argument_field_micro_accuracy"),
            "all_field_micro": metrics.get("agent_argument_field_micro_accuracy"),
            "newline_tolerant_required_diagnostic": metrics.get(
                "agent_required_argument_call_normalized_accuracy"
            ),
            "protocol_id": metrics.get("agent_protocol_id"),
        }
    baseline = cells.get("A-baseline") or {}
    for metrics in cells.values():
        metrics["delta_required_exact_vs_baseline"] = (
            float(metrics.get("required_call_exact") or 0.0)
            - float(baseline.get("required_call_exact") or 0.0)
        )
    recommended = max(
        cells,
        key=lambda label: (
            float(cells[label].get("required_call_exact") or 0.0),
            float(cells[label].get("required_field_micro") or 0.0),
            float(cells[label].get("balanced_routing") or 0.0),
        ),
        default=None,
    )
    return {"cells": cells, "recommended_cell": recommended}


def _continue_literal_lock_pipeline(state: dict[str, Any], result_path: Path) -> dict[str, Any]:
    sheets = _load_literal_lock_sheets(dict(state["sheet_ids"]))
    registry = Registry()
    run_ids: list[str] = []
    try:
        for label in LITERAL_LOCK_CELLS:
            cell = state["cells"].setdefault(label, {})
            run_id = str(cell.get("run_id") or "")
            if run_id:
                try:
                    complete = registry.get_run(run_id).get("status") == "complete"
                except KeyError:
                    complete = False
                if complete:
                    cell["status"] = "complete"
                    run_ids.append(run_id)
                    continue
            state.update(status="running", stage=f"training-{label}")
            cell["status"] = "training"
            _write_state(result_path, state)
            metrics = run_training(sheets[label].recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(f"{label} did not reach the fixed-step target")
            run_id = str(metrics["run_id"])
            cell.update(status="complete", run_id=run_id, metrics=metrics)
            run_ids.append(run_id)
            state["run_ids"] = list(run_ids)
            _write_state(result_path, state)
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state

        state["run_ids"] = run_ids
        if (state.get("dev_evaluation") or {}).get("status") != "complete":
            state.update(status="running", stage="external-dev-evaluation-4-way")
            _write_state(result_path, state)
            state["dev_evaluation"] = run_dev_batch(
                run_ids=run_ids,
                source=str(state["dev_source"]),
                max_samples=int(state["dev_samples_requested"]),
                compare_base=bool(state.get("compare_base", True)),
            )
            _write_state(result_path, state)
        if state["dev_evaluation"].get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-dev")
            _write_state(result_path, state)
            return state
        state["dev_summary"] = _literal_lock_summary(state["dev_evaluation"])
        _write_state(result_path, state)

        if (state.get("sealed_evaluation") or {}).get("status") != "complete":
            state.update(status="running", stage="sealed-final-evaluation-4-way")
            _write_state(result_path, state)
            state["sealed_evaluation"] = run_sealed_batch(
                run_ids=run_ids,
                suite_id=str(state["suite_id"]),
                max_samples=int(state["sealed_samples_requested"]),
                compare_base=bool(state.get("compare_base", True)),
            )
            _write_state(result_path, state)
        if state["sealed_evaluation"].get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-sealed")
        else:
            state["sealed_summary"] = _literal_lock_summary(state["sealed_evaluation"])
            state.update(
                status="complete",
                stage="compare-ready-final",
                compare_run_ids=run_ids,
            )
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        _write_state(result_path, state)
        raise


def run_literal_lock_pipeline(
    sheet_ids: dict[str, str],
    suite_id: str,
    dev_samples: int = 520,
    sealed_samples: int = 520,
    compare_base: bool = True,
) -> dict[str, Any]:
    """Run fresh A/B/F/G training, shared DEV, then one frozen final sealed suite."""
    sheets = _load_literal_lock_sheets(sheet_ids)
    get_sealed_suite(suite_id)
    sealed_hashes, sealed_sources = sealed_guard()
    preflight = {
        label: preflight_recipe(sheet.recipe, sealed_hashes, sealed_sources)
        for label, sheet in sheets.items()
    }
    blocked = [label for label, report in preflight.items() if report.get("status") == "blocked"]
    if blocked:
        raise ValueError(f"LiteralLock pipeline blocked by preflight: {blocked}")
    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    result_path = literal_lock_results_dir() / f"{pipeline_id}.json"
    first = sheets["A-baseline"].recipe
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "pipeline_type": "literal-lock-ABFG-dev-sealed",
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_group": first.comparison_group,
        "sheet_ids": dict(sheet_ids),
        "dev_source": first.validation_datasets[0],
        "suite_id": suite_id,
        "dev_samples_requested": int(dev_samples),
        "sealed_samples_requested": int(sealed_samples),
        "compare_base": bool(compare_base),
        "preflight": preflight,
        "cells": {
            label: {"status": "pending", "sheet_id": sheet_ids[label]}
            for label in LITERAL_LOCK_CELLS
        },
        "run_ids": [],
    }
    _write_state(result_path, state)
    return _continue_literal_lock_pipeline(state, result_path)


def resume_literal_lock_pipeline(pipeline_id: str | None = None) -> dict[str, Any]:
    state = next(
        (
            item for item in list_literal_lock_results()
            if pipeline_id is None or str(item.get("pipeline_id")) == pipeline_id
        ),
        None,
    )
    if state is None:
        raise KeyError(f"LiteralLock pipeline not found: {pipeline_id or 'latest'}")
    if state.get("status") == "complete":
        return state
    result_path = Path(str(state["result_path"]))
    state.pop("result_path", None)
    state.pop("error", None)
    state.update(status="running", stage="resuming")
    _write_state(result_path, state)
    return _continue_literal_lock_pipeline(state, result_path)


def _load_human_agentic_sheets(sheet_ids: dict[str, str]) -> dict[str, Any]:
    if set(sheet_ids) != set(HUMAN_AGENTIC_CELLS):
        raise ValueError(
            f"human-agentic pipeline requires exactly these cells: {HUMAN_AGENTIC_CELLS}"
        )
    store = WorkbookStore()
    sheets = {label: store.get(str(sheet_ids[label])) for label in HUMAN_AGENTIC_CELLS}
    recipes = {label: sheet.recipe for label, sheet in sheets.items()}
    baseline = recipes["A-baseline"]
    if any(recipe.comparison_group != baseline.comparison_group for recipe in recipes.values()):
        raise ValueError("all human-agentic cells must share one comparison group")
    core = _screening_core(baseline)
    if any(_screening_core(recipe) != core for recipe in recipes.values()):
        raise ValueError("human-agentic cells differ outside declared regularizers")
    expected = {
        "A-baseline": (False, False, False, False),
        "B-pi-noise": (True, False, False, False),
        "C-geometry-only": (False, True, False, False),
        "D-pi-geometry": (True, True, False, False),
        "E-human-intent": (True, True, True, True),
    }
    for label, recipe in recipes.items():
        intent_enabled = recipe.noise.intent_consistency_weight > 0
        actual = (
            recipe.noise.enabled,
            recipe.geometry.enabled,
            recipe.role_loss.enabled,
            intent_enabled,
        )
        if actual != expected[label]:
            raise ValueError(f"invalid regularizer controls for {label}: {actual}")
        if recipe.budget_mode != "steps":
            raise ValueError(f"{label} must use a fixed optimizer-step budget")
        if recipe.validation_mode != "external" or len(recipe.validation_datasets) != 1:
            raise ValueError(f"{label} must use one shared external DEV source")
        if recipe.allowed_tools != sorted(KNOWN_TOOLS):
            raise ValueError(f"{label} does not use the complete 13-tool menu")
        if not recipe.tool_menu_conditioning:
            raise ValueError(f"{label} must train with the inference tool menu")
    for label in ("B-pi-noise", "D-pi-geometry", "E-human-intent"):
        noise = recipes[label].noise
        if (
            noise.source != "pi"
            or noise.digit_order != "natural"
            or noise.alpha != 2.0
            or noise.modulation != 0.1
            or noise.scope != "prompt"
            or noise.envelope != "constant"
            or noise.clean_tail_fraction != 0.3
        ):
            raise ValueError(f"{label} changed the frozen natural π decimal-pair schedule")
    if recipes["C-geometry-only"].geometry != recipes["D-pi-geometry"].geometry:
        raise ValueError("C and D must use identical relational geometry")
    if recipes["E-human-intent"].noise.value_consistency_weight <= 0:
        raise ValueError("E must protect exact values with clean/noisy consistency")
    return sheets


def _human_agentic_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for label, result in zip(
        HUMAN_AGENTIC_CELLS,
        evaluation.get("results") or [],
        strict=False,
    ):
        metrics = result.get("adapter") or {}
        cells[label] = {
            "run_id": result.get("run_id"),
            "human_task_success": metrics.get("agent_human_task_success_rate"),
            "strict_human_task_success": metrics.get(
                "agent_strict_human_task_success_rate"
            ),
            "balanced_routing": metrics.get("agent_balanced_tool_decision_accuracy"),
            "routing_mcc": metrics.get("agent_tool_decision_mcc"),
            "tool_recall": metrics.get("agent_tool_recall"),
            "no_tool_specificity": metrics.get("agent_no_tool_specificity"),
            "direct_criteria": metrics.get("agent_direct_required_terms_accuracy"),
            "selected_tool": metrics.get("agent_selected_tool_name_accuracy"),
            "required_call_exact": metrics.get(
                "agent_required_argument_call_exact_accuracy"
            ),
            "whole_call_exact": metrics.get("agent_tool_argument_exact_accuracy"),
            "required_field_micro": metrics.get(
                "agent_required_argument_field_micro_accuracy"
            ),
            "protocol_id": metrics.get("agent_protocol_id"),
        }
    baseline = cells.get("A-baseline") or {}
    for metrics in cells.values():
        metrics["delta_human_success_vs_baseline"] = (
            float(metrics.get("human_task_success") or 0.0)
            - float(baseline.get("human_task_success") or 0.0)
        )
    recommended = max(
        cells,
        key=lambda label: (
            float(cells[label].get("human_task_success") or 0.0),
            float(cells[label].get("strict_human_task_success") or 0.0),
            float(cells[label].get("balanced_routing") or 0.0),
            float(cells[label].get("required_call_exact") or 0.0),
        ),
        default=None,
    )
    return {"cells": cells, "recommended_cell": recommended}


def _continue_human_agentic_pipeline(
    state: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    sheets = _load_human_agentic_sheets(dict(state["sheet_ids"]))
    registry = Registry()
    run_ids: list[str] = []
    try:
        for label in HUMAN_AGENTIC_CELLS:
            cell = state["cells"].setdefault(label, {})
            run_id = str(cell.get("run_id") or "")
            if run_id:
                try:
                    complete = registry.get_run(run_id).get("status") == "complete"
                except KeyError:
                    complete = False
                if complete:
                    cell["status"] = "complete"
                    run_ids.append(run_id)
                    continue
            state.update(status="running", stage=f"training-{label}")
            cell["status"] = "training"
            _write_state(result_path, state)
            metrics = run_training(sheets[label].recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(f"{label} did not reach the fixed-step target")
            run_id = str(metrics["run_id"])
            cell.update(status="complete", run_id=run_id, metrics=metrics)
            run_ids.append(run_id)
            state["run_ids"] = list(run_ids)
            _write_state(result_path, state)
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state

        if (state.get("dev_evaluation") or {}).get("status") != "complete":
            state.update(status="running", stage="human-dev-autoregressive-evaluation")
            _write_state(result_path, state)
            state["dev_evaluation"] = run_dev_batch(
                run_ids=run_ids,
                source=str(state["dev_source"]),
                max_samples=int(state["dev_samples_requested"]),
                compare_base=bool(state.get("compare_base", True)),
            )
            _write_state(result_path, state)
        if state["dev_evaluation"].get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-human-dev")
            _write_state(result_path, state)
            return state
        state["dev_summary"] = _human_agentic_summary(state["dev_evaluation"])
        _write_state(result_path, state)

        if (state.get("sealed_evaluation") or {}).get("status") != "complete":
            state.update(status="running", stage="human-sealed-final-evaluation")
            _write_state(result_path, state)
            state["sealed_evaluation"] = run_sealed_batch(
                run_ids=run_ids,
                suite_id=str(state["suite_id"]),
                max_samples=int(state["sealed_samples_requested"]),
                compare_base=bool(state.get("compare_base", True)),
            )
            _write_state(result_path, state)
        if state["sealed_evaluation"].get("status") == "stopped":
            state.update(status="stopped", stage="stopped-during-human-sealed")
        else:
            state["sealed_summary"] = _human_agentic_summary(state["sealed_evaluation"])
            state.update(
                status="complete",
                stage="compare-ready-human-final",
                compare_run_ids=run_ids,
            )
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        _write_state(result_path, state)
        raise


def run_human_agentic_pipeline(
    sheet_ids: dict[str, str],
    suite_id: str,
    dev_samples: int = 520,
    sealed_samples: int = 520,
    compare_base: bool = True,
) -> dict[str, Any]:
    sheets = _load_human_agentic_sheets(sheet_ids)
    get_sealed_suite(suite_id)
    sealed_hashes, sealed_sources = sealed_guard()
    common_preflight = preflight_recipe(
        sheets["A-baseline"].recipe, sealed_hashes, sealed_sources
    )
    preflight = {label: common_preflight for label in HUMAN_AGENTIC_CELLS}
    if common_preflight.get("status") == "blocked":
        raise ValueError("human-agentic pipeline blocked by preflight")
    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    result_path = human_agentic_results_dir() / f"{pipeline_id}.json"
    first = sheets["A-baseline"].recipe
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "pipeline_type": "human-agentic-ABCDE-dev-sealed-v1",
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_group": first.comparison_group,
        "sheet_ids": dict(sheet_ids),
        "dev_source": first.validation_datasets[0],
        "suite_id": suite_id,
        "dev_samples_requested": int(dev_samples),
        "sealed_samples_requested": int(sealed_samples),
        "compare_base": bool(compare_base),
        "preflight": preflight,
        "cells": {
            label: {"status": "pending", "sheet_id": sheet_ids[label]}
            for label in HUMAN_AGENTIC_CELLS
        },
        "run_ids": [],
    }
    _write_state(result_path, state)
    return _continue_human_agentic_pipeline(state, result_path)


def resume_human_agentic_pipeline(pipeline_id: str | None = None) -> dict[str, Any]:
    state = next(
        (
            item for item in list_human_agentic_results()
            if pipeline_id is None or str(item.get("pipeline_id")) == pipeline_id
        ),
        None,
    )
    if state is None:
        raise KeyError(f"human-agentic pipeline not found: {pipeline_id or 'latest'}")
    if state.get("status") == "complete":
        return state
    result_path = Path(str(state["result_path"]))
    state.pop("result_path", None)
    state.pop("error", None)
    state.update(status="running", stage="resuming")
    _write_state(result_path, state)
    return _continue_human_agentic_pipeline(state, result_path)
