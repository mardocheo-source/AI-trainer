from __future__ import annotations

import hashlib
import html
import inspect
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_eval import routing_metrics_from_predictions
from .bfcl import (
    BFCL_PROFILES,
    BFCL_SPECS,
    bfcl_environment_status,
    build_bfcl_config,
    list_bfcl_runs,
    save_bfcl_config,
)
from .bfcl_pair_pipeline import latest_result as latest_bfcl_pair_result
from .dataset_probe import inspect_dataset
from .decimal_control_pipeline import latest_decimal_control_result
from .dev_eval import dev_results
from .doctor import system_report
from .downloads import fetch_huggingface_dataset
from .exploration import generate_trial_recipes
from .focused_pipeline import latest_focused_result
from .jobs import (
    clear_finished_jobs,
    job_status,
    resume_dev_pair_pipeline_job,
    resume_focused_human_pipeline_job,
    resume_geometry_screening_job,
    resume_literal_lock_pipeline_job,
    resume_routing_screening_job,
    start_ablation_pipeline_job,
    start_bfcl_run_job,
    start_bfcl_setup_job,
    start_bfcl_v3_pair_pipeline_job,
    start_dev_pair_pipeline_job,
    start_exploration_job,
    start_focused_human_pipeline_job,
    start_geometry_screening_job,
    start_literal_lock_pipeline_job,
    start_pair_pipeline_job,
    start_routing_screening_job,
    start_sealed_eval_batch_job,
    start_sealed_eval_job,
    start_training_job,
    stop_latest_training_job,
)
from .model_probe import inspect_model
from .paths import project_root
from .pipeline import (
    latest_ablation_pipeline_result,
    latest_dev_pair_pipeline_result,
    latest_geometry_screening_result,
    latest_human_agentic_result,
    latest_literal_lock_result,
    latest_pair_pipeline_result,
    latest_routing_screening_result,
    prepare_geometry_screening_sheets,
    prepare_literal_lock_sheets,
    prepare_routing_screening_sheets,
)
from .preflight import preflight_recipe
from .progress import read_events
from .recipe import save_recipe
from .registry import Registry
from .routing_data import ensure_routing_supplement
from .schema import (
    ExplorationConfig,
    GeometryConfig,
    NoiseConfig,
    RoleLossConfig,
    TrainingRecipe,
)
from .sealed import (
    list_sealed_suites,
    register_sealed_suite,
    sealed_guard,
    sealed_results,
)
from .tool_schema import KNOWN_TOOLS
from .workbook import WorkbookStore, sheet_choices


def _model_rows() -> list[list[Any]]:
    return [
        [
            item["id"],
            item["name"],
            item["metadata"]["architecture"],
            round(item["metadata"]["parameter_count"] / 1e9, 3),
            item["path"],
        ]
        for item in Registry().list_models()
    ]


def _dataset_rows() -> list[list[Any]]:
    return [
        [
            item["id"],
            item["name"],
            item["metadata"]["source_type"],
            item["metadata"].get("row_count"),
            item["metadata"].get("secret_hits", 0),
            item["source"],
        ]
        for item in Registry().list_datasets()
    ]


def _model_choices() -> list[tuple[str, str]]:
    return [
        (
            f"#{item['id']}  {item['name']}  —  {item['metadata']['architecture']}",
            item["path"],
        )
        for item in Registry().list_models()
    ]


def _dataset_choices() -> list[tuple[str, str]]:
    return [
        (
            (f"#{item['id']}  {item['name']}  —  {item['metadata'].get('row_count', '?')} rows"),
            item["source"],
        )
        for item in Registry().list_datasets()
    ]


def _run_rows() -> list[list[Any]]:
    return [
        [item["run_id"], item["status"], item["output_dir"], _local_time(item["updated_at"])]
        for item in Registry().list_runs()
    ]


def _sealed_choices() -> list[tuple[str, str]]:
    usage: dict[str, int] = {}
    for result in sealed_results():
        suite_id = str(result.get("suite_id") or "")
        usage[suite_id] = usage.get(suite_id, 0) + 1
    return [
        (
            "".join(
                (
                    f"{item['name']} — {item['normalized_samples']} samples — ",
                    (
                        "DEV/USED ×" + str(usage[item["id"]])
                        if usage.get(item["id"])
                        else "UNOPENED"
                    ),
                    f" — {item['created_at']}",
                )
            ),
            item["id"],
        )
        for item in list_sealed_suites()
    ]


def _completed_run_choices() -> list[tuple[str, str]]:
    sheets = {sheet.id: sheet.name for sheet in WorkbookStore().list()}
    choices = []
    for run in Registry().list_runs():
        if run["status"] != "complete":
            continue
        try:
            recipe = json.loads(run.get("recipe_json") or "{}")
        except json.JSONDecodeError:
            recipe = {}
        sheet_name = sheets.get(recipe.get("workbook_sheet_id"), "legacy / unsaved")
        choices.append((f"{run['run_id']} — {sheet_name}", run["run_id"]))
    return choices


def _bfcl_target_choices() -> list[tuple[str, str]]:
    choices = [
        (f"BASE — {label}", f"model:{value}") for label, value in _model_choices()
    ]
    choices.extend(
        (f"ADAPTER — {label}", f"run:{value}") for label, value in _completed_run_choices()
    )
    return choices


def _bfcl_rows() -> list[list[Any]]:
    rows = []
    for item in list_bfcl_runs():
        config = item.get("config") or {}
        rows.append(
            [
                item.get("bfcl_run_id"),
                item.get("status"),
                item.get("stage"),
                config.get("version"),
                config.get("profile"),
                config.get("handler"),
                config.get("target_name"),
                "YES" if item.get("publishable") else "NO",
                config.get("bfcl_commit"),
                config.get("output_dir"),
            ]
        )
    return rows


def _latest_bfcl_state() -> dict[str, Any]:
    runs = list_bfcl_runs()
    return runs[0] if runs else {"status": "no-bfcl-runs"}


def _bfcl_export_files() -> list[str]:
    latest = _latest_bfcl_state()
    output_value = (latest.get("config") or {}).get("output_dir")
    if not output_value:
        return []
    output = Path(str(output_value))
    candidates = [
        output / "state.json",
        output / "config.json",
        output / "official-result.json",
        output / "bfcl.log",
        *sorted((output / "score").glob("*.csv")),
    ]
    return [str(path) for path in candidates if path.is_file()]


def _start_bfcl_setup(version: str) -> dict[str, Any]:
    return start_bfcl_setup_job(str(version))


def _start_bfcl_gui(
    version: str,
    profile: str,
    handler: str,
    transport: str,
    target: str,
    endpoint: str,
    api_model: str,
    smoke_samples: int,
) -> dict[str, Any]:
    environment = bfcl_environment_status(str(version))
    if environment["status"] != "ready":
        return {
            "status": "setup-required",
            "message": f"Install the pinned BFCL {version} environment first.",
            "environment": environment,
        }
    try:
        config = build_bfcl_config(
            version=str(version),
            profile=str(profile),
            handler=str(handler),
            transport=str(transport),
            target=str(target or ""),
            endpoint=str(endpoint or ""),
            api_model=str(api_model or ""),
            smoke_samples=int(smoke_samples),
        )
    except (RuntimeError, ValueError) as error:
        return {"status": "refused", "error": str(error), "environment": environment}
    path = save_bfcl_config(config)
    return start_bfcl_run_job(path)


def _refresh_bfcl_gui() -> tuple[dict[str, Any], list[list[Any]], dict[str, Any], list[str]]:
    return (
        bfcl_environment_status(),
        _bfcl_rows(),
        _latest_bfcl_state(),
        _bfcl_export_files(),
    )


def _start_bfcl_v3_pair(steps: int) -> dict[str, Any]:
    environment = bfcl_environment_status("v3")
    if environment["status"] != "ready":
        raise RuntimeError("Install/update compact BFCL v3 before starting the fair-pair run")
    return start_bfcl_v3_pair_pipeline_job(int(steps))


def _refresh_bfcl_pair() -> dict[str, Any]:
    result = latest_bfcl_pair_result()
    if result is None:
        return {"status": "not-started"}
    current: dict[str, Any] | None = None
    for item in reversed(list((result.get("bfcl") or {}).values())):
        state_path = Path(str(item.get("state_path") or ""))
        if not state_path.is_file():
            continue
        try:
            current = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        break
    compact = {
        key: result.get(key)
        for key in (
            "pipeline_id",
            "pipeline_type",
            "status",
            "stage",
            "progress_percent",
            "created_at",
            "updated_at",
            "resources",
            "training",
            "bfcl",
            "compare_run_ids",
            "error",
        )
        if key in result
    }
    if current:
        compact["current_bfcl_target"] = current
    return compact


def _latest_compare_ready_pipeline() -> dict[str, Any] | None:
    """Read the newest persisted compare-ready pipeline across all generations."""

    candidates = [
        candidate
        for candidate in (
            latest_decimal_control_result(),
            latest_routing_screening_result(),
            latest_focused_result(),
            latest_human_agentic_result(),
            latest_literal_lock_result(),
            latest_dev_pair_pipeline_result(),
            latest_ablation_pipeline_result(),
            latest_pair_pipeline_result(),
        )
        if candidate and candidate.get("status") == "complete" and candidate.get("compare_run_ids")
    ]
    return max(candidates, key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), default=None)


def _build_recipe(
    sheet_id: str | None,
    model: str,
    datasets: list[str] | str,
    minutes: int,
    sequence_length: int,
    rank: int,
    max_source_rows: int,
    target_profile: str,
    adapter_method: str,
    learning_rate: float,
    budget_mode: str,
    max_steps: int,
    noise_enabled: bool,
    noise_amplitude_mode: str,
    noise_source: str,
    noise_digit_order: str,
    noise_alpha: float,
    noise_modulation: float,
    noise_scope: str,
    noise_envelope: str,
    noise_clean_tail: float,
    geometry_enabled: bool,
    geometry_mode: str,
    geometry_scope: str,
    geometry_weight: float,
    geometry_layer: int,
    geometry_margin: float,
    geometry_sample_tokens: int,
    role_loss_enabled: bool,
    delimiter_weight: float,
    tool_name_weight: float,
    argument_key_weight: float,
    argument_value_weight: float,
    benchmark_profile: str,
    comparison_group: str,
    experiment_variant: str,
    validation_mode: str,
    validation_datasets: list[str] | str,
    eval_ratio: float,
    max_validation_samples: int,
    allowed_tools: str | list[str],
    filter_unknown_tools: bool,
) -> TrainingRecipe:
    registry = Registry()
    dataset_references = (
        datasets
        if isinstance(datasets, list)
        else [item.strip() for item in datasets.splitlines() if item.strip()]
    )
    dataset_list = [
        registry.resolve_dataset(str(item).strip())
        for item in dataset_references
        if str(item).strip()
    ]
    if not dataset_list:
        raise ValueError("Select at least one training dataset")
    validation_references = (
        validation_datasets
        if isinstance(validation_datasets, list)
        else [item.strip() for item in validation_datasets.splitlines() if item.strip()]
    )
    validation_list = [
        registry.resolve_dataset(str(item).strip())
        for item in validation_references
        if str(item).strip()
    ]
    if validation_mode == "group_holdout":
        validation_list = []
    tool_list = (
        [item.strip() for item in allowed_tools.split(",") if item.strip()]
        if isinstance(allowed_tools, str)
        else [str(item).strip() for item in allowed_tools if str(item).strip()]
    )
    return TrainingRecipe(
        model=registry.resolve_model(model.strip()),
        datasets=dataset_list,
        time_limit_minutes=int(minutes),
        reserve_minutes=max(5, min(20, int(minutes) // 5)),
        sequence_length=int(sequence_length),
        lora_rank=int(rank),
        lora_alpha=int(rank) * 2,
        max_source_rows=int(max_source_rows),
        target_profile=target_profile,
        use_rslora=adapter_method == "rslora",
        learning_rate=float(learning_rate),
        budget_mode=budget_mode,
        max_steps=int(max_steps),
        noise=NoiseConfig(
            enabled=noise_enabled,
            amplitude_mode=noise_amplitude_mode,
            source=noise_source,
            digit_order=noise_digit_order,
            alpha=noise_alpha,
            modulation=float(noise_modulation),
            scope=noise_scope,
            envelope=noise_envelope,
            clean_tail_fraction=float(noise_clean_tail),
        ),
        geometry=GeometryConfig(
            enabled=geometry_enabled,
            mode=geometry_mode,
            scope=geometry_scope,
            weight=float(geometry_weight),
            layer=int(geometry_layer),
            margin=float(geometry_margin),
            sample_tokens=int(geometry_sample_tokens),
        ),
        role_loss=RoleLossConfig(
            enabled=role_loss_enabled,
            delimiter_weight=float(delimiter_weight),
            tool_name_weight=float(tool_name_weight),
            argument_key_weight=float(argument_key_weight),
            argument_value_weight=float(argument_value_weight),
        ),
        validation_mode=validation_mode,
        validation_datasets=validation_list,
        eval_ratio=float(eval_ratio),
        max_validation_samples=int(max_validation_samples),
        benchmark_profile=benchmark_profile,
        comparison_group=comparison_group.strip() or "default",
        experiment_variant=experiment_variant,
        allowed_tools=tool_list,
        filter_unknown_tools=bool(filter_unknown_tools),
        workbook_sheet_id=sheet_id or None,
    )


def _preflight_recipe(recipe: TrainingRecipe) -> dict[str, Any]:
    sealed_hashes, sealed_sources = sealed_guard()
    return preflight_recipe(
        recipe,
        sealed_hashes=sealed_hashes,
        sealed_sources=sealed_sources,
    )


def _technique_preset(name: str) -> tuple[Any, ...]:
    common = {
        "noise_enabled": True,
        "amplitude_mode": "digit_pairs",
        "source": "pi",
        "order": "natural",
        "alpha": 2.0,
        "modulation": 0.10,
        "noise_scope": "prompt",
        "envelope": "constant",
        "clean_tail": 0.30,
        "geometry_enabled": False,
        "geometry_mode": "relational",
        "geometry_scope": "structured",
        "geometry_weight": 0.005,
        "geometry_layer": -1,
        "geometry_margin": 0.2,
        "geometry_samples": 32,
        "role_enabled": False,
        "delimiter_weight": 0.5,
        "tool_weight": 1.5,
        "key_weight": 2.0,
        "value_weight": 2.5,
    }
    if name == "Original π controls — alpha 5 / all tokens":
        common.update(alpha=5.0, modulation=0.35, noise_scope="all", clean_tail=0.0)
    elif name == "Constant NEFTune control — no digit modulation":
        common.update(
            amplitude_mode="constant", modulation=0.0, noise_scope="all", clean_tail=0.0
        )
    elif name == "Shuffled π control — matched decimal pairs":
        common.update(order="shuffled")
    elif name == "Relational geometry only":
        common.update(noise_enabled=False, geometry_enabled=True)
    elif name == "Route-then-Resolve — π + role CE + relational geometry":
        common.update(geometry_enabled=True, role_enabled=True)
    # The remaining preset is π routing-only and therefore already equals common.
    return tuple(common.values())


def _local_time(value: str | float) -> str:
    try:
        if isinstance(value, (float, int)):
            moment = datetime.fromtimestamp(float(value)).astimezone()
        else:
            moment = datetime.fromisoformat(value).astimezone()
        return moment.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (OSError, TypeError, ValueError):
        return str(value)


def _add_model(path: str, name: str) -> tuple[str, list[list[Any]]]:
    try:
        probe = inspect_model(path)
        if name.strip():
            probe.name = name.strip()
        identifier = Registry().upsert_model(probe)
        return f"Registered model #{identifier}: {probe.name}", _model_rows()
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return f"ERROR: {type(error).__name__}: {error}", _model_rows()


def _add_dataset(source: str, name: str) -> tuple[str, list[list[Any]]]:
    try:
        probe = inspect_dataset(source, name.strip() or None)
        identifier = Registry().upsert_dataset(probe)
        warning = f"; possible secrets: {probe.secret_hits}" if probe.secret_hits else ""
        return f"Registered dataset #{identifier}: {probe.name}{warning}", _dataset_rows()
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return f"ERROR: {type(error).__name__}: {error}", _dataset_rows()


def _fetch_dataset(source: str, name: str) -> tuple[str, list[list[Any]]]:
    try:
        path = fetch_huggingface_dataset(source)
        probe = inspect_dataset(str(path), name.strip() or None)
        identifier = Registry().upsert_dataset(probe)
        return f"Downloaded dataset #{identifier} to {path}", _dataset_rows()
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return f"ERROR: {type(error).__name__}: {error}", _dataset_rows()


def _start_run(*values: Any) -> str:
    try:
        if not values:
            raise ValueError("Missing training configuration")
        *recipe_values, warnings_acknowledged = values
        recipe = _build_recipe(*recipe_values)
        preflight = _preflight_recipe(recipe)
        if preflight["status"] == "blocked":
            return json.dumps({"status": "BLOCKED_BY_PREFLIGHT", "preflight": preflight}, indent=2)
        if preflight["warnings"] and not warnings_acknowledged:
            return json.dumps(
                {
                    "status": "REVIEW_PREFLIGHT_WARNINGS",
                    "message": "Review the preflight report, then enable the acknowledgement checkbox.",
                    "preflight": preflight,
                },
                indent=2,
            )
        recipe_path = project_root() / "recipes" / f"gui-{int(time.time())}.yaml"
        save_recipe(recipe, recipe_path)
        job = start_training_job(recipe_path)
        return json.dumps(job, indent=2)
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return f"ERROR: {type(error).__name__}: {error}"


def _preflight_run(*values: Any) -> dict[str, Any]:
    try:
        recipe = _build_recipe(*values)
        return _preflight_recipe(recipe)
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def _start_exploration(sheet_id: str | None, warnings_acknowledged: bool = False) -> str:
    try:
        if not sheet_id:
            raise ValueError("Save the workbook sheet before starting exploration")
        sheet = WorkbookStore().get(sheet_id)
        trial_recipes = generate_trial_recipes(sheet.recipe, sheet.exploration)
        preflight = {
            sequence: _preflight_recipe(
                next(recipe for recipe in trial_recipes if recipe.sequence_length == sequence)
            )
            for sequence in sorted({recipe.sequence_length for recipe in trial_recipes})
        }
        blocked = {
            sequence: report
            for sequence, report in preflight.items()
            if report["status"] == "blocked"
        }
        if blocked:
            return json.dumps({"status": "BLOCKED_BY_PREFLIGHT", "preflight": blocked}, indent=2)
        has_warnings = any(report.get("warnings") for report in preflight.values())
        if has_warnings and not warnings_acknowledged:
            return json.dumps(
                {
                    "status": "REVIEW_PREFLIGHT_WARNINGS",
                    "message": (
                        "Review the reports for every explored sequence length, then enable "
                        "the acknowledgement checkbox."
                    ),
                    "preflight": preflight,
                },
                indent=2,
            )
        return json.dumps(start_exploration_job(sheet_id), indent=2)
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return f"ERROR: {type(error).__name__}: {error}"


def _save_workbook_sheet(
    sheet_id: str | None,
    sheet_name: str,
    *values: Any,
) -> dict[str, Any]:
    recipe_value_count = len(inspect.signature(_build_recipe).parameters) - 1
    recipe_values = values[:recipe_value_count]
    exploration_values = values[recipe_value_count:]
    recipe = _build_recipe(sheet_id, *recipe_values)
    if len(exploration_values) != 28:
        raise ValueError(f"Expected 28 exploration controls, received {len(exploration_values)}")
    (
        exploration_strategy,
        exploration_trials,
        exploration_total_minutes,
        exploration_seed,
        adapter_methods,
        lora_ranks,
        target_profiles,
        sequence_lengths,
        learning_rate_min,
        learning_rate_max,
        noise_modes,
        noise_alpha_min,
        noise_alpha_max,
        noise_digit_orders,
        noise_scopes,
        noise_envelopes,
        noise_modulation_min,
        noise_modulation_max,
        noise_clean_tail_min,
        noise_clean_tail_max,
        geometry_modes,
        geometry_scopes,
        geometry_layers,
        geometry_weight_min,
        geometry_weight_max,
        geometry_margin_min,
        geometry_margin_max,
        role_loss_modes,
    ) = exploration_values
    exploration = ExplorationConfig(
        enabled=True,
        strategy=exploration_strategy,
        trial_count=int(exploration_trials),
        total_time_limit_minutes=int(exploration_total_minutes),
        seed=int(exploration_seed),
        adapter_methods=adapter_methods,
        lora_ranks=[int(value) for value in lora_ranks],
        target_profiles=target_profiles,
        sequence_lengths=[int(value) for value in sequence_lengths],
        learning_rate_min=float(learning_rate_min),
        learning_rate_max=float(learning_rate_max),
        noise_modes=noise_modes,
        noise_alpha_min=float(noise_alpha_min),
        noise_alpha_max=float(noise_alpha_max),
        noise_digit_orders=noise_digit_orders,
        noise_scopes=noise_scopes,
        noise_envelopes=noise_envelopes,
        noise_modulation_min=float(noise_modulation_min),
        noise_modulation_max=float(noise_modulation_max),
        noise_clean_tail_min=float(noise_clean_tail_min),
        noise_clean_tail_max=float(noise_clean_tail_max),
        geometry_modes=geometry_modes,
        geometry_scopes=geometry_scopes,
        geometry_layers=[int(value) for value in geometry_layers],
        geometry_weight_min=float(geometry_weight_min),
        geometry_weight_max=float(geometry_weight_max),
        geometry_margin_min=float(geometry_margin_min),
        geometry_margin_max=float(geometry_margin_max),
        role_loss_modes=role_loss_modes,
    )
    saved = WorkbookStore().save(sheet_name, recipe, exploration, sheet_id=sheet_id or None)
    return saved.model_dump(mode="json")


def _create_fair_pair(save_values: list[Any]) -> tuple[Any, Any]:
    saved_payload = _save_workbook_sheet(*save_values)
    store = WorkbookStore()
    source = store.get(saved_payload["id"])
    if not (
        source.recipe.noise.enabled
        or source.recipe.geometry.enabled
        or source.recipe.role_loss.enabled
    ):
        raise ValueError("Enable noise, geometry and/or role loss before creating an exotic pair")
    base_name = source.name.removesuffix(" — exotic").removesuffix(" — baseline")
    exotic_recipe = source.recipe.model_copy(update={"experiment_variant": "exotic"})
    exotic = store.save(
        f"{base_name} — exotic",
        exotic_recipe,
        source.exploration,
        sheet_id=source.id,
    )
    baseline_payload = source.recipe.model_dump(mode="json")
    baseline_payload["experiment_variant"] = "baseline"
    baseline_payload["workbook_sheet_id"] = None
    baseline_payload["noise"]["enabled"] = False
    baseline_payload["geometry"]["enabled"] = False
    baseline_payload["role_loss"]["enabled"] = False
    baseline_recipe = TrainingRecipe.model_validate(baseline_payload)
    baseline_exploration = source.exploration.model_copy(
        update={
            "trial_count": 1,
            "adapter_methods": ["rslora" if baseline_recipe.use_rslora else "lora"],
            "lora_ranks": [baseline_recipe.lora_rank],
            "target_profiles": [baseline_recipe.target_profile],
            "sequence_lengths": [baseline_recipe.sequence_length],
            "learning_rate_min": baseline_recipe.learning_rate,
            "learning_rate_max": baseline_recipe.learning_rate,
            "noise_modes": ["off"],
            "geometry_modes": ["off"],
            "role_loss_modes": ["off"],
        }
    )
    baseline = store.save(
        f"{base_name} — baseline",
        baseline_recipe,
        baseline_exploration,
    )
    return baseline, exotic


def _sheet_payload(sheet_id: str) -> dict[str, Any]:
    sheet = WorkbookStore().get(sheet_id)
    recipe = sheet.recipe
    exploration = sheet.exploration
    return {
        "sheet_name": sheet.name,
        "model": recipe.model,
        "datasets": recipe.datasets,
        "minutes": recipe.time_limit_minutes,
        "sequence": recipe.sequence_length,
        "rank": recipe.lora_rank,
        "max_source_rows": recipe.max_source_rows,
        "target_profile": recipe.target_profile,
        "adapter_method": "rslora" if recipe.use_rslora else "lora",
        "learning_rate": recipe.learning_rate,
        "budget_mode": recipe.budget_mode,
        "max_steps": recipe.max_steps,
        "noise_enabled": recipe.noise.enabled,
        "noise_amplitude_mode": recipe.noise.amplitude_mode,
        "noise_source": recipe.noise.source,
        "noise_digit_order": recipe.noise.digit_order,
        "noise_alpha": recipe.noise.alpha,
        "noise_modulation": recipe.noise.modulation,
        "noise_scope": recipe.noise.scope,
        "noise_envelope": recipe.noise.envelope,
        "noise_clean_tail": recipe.noise.clean_tail_fraction,
        "geometry_enabled": recipe.geometry.enabled,
        "geometry_mode": recipe.geometry.mode,
        "geometry_scope": recipe.geometry.scope,
        "geometry_weight": recipe.geometry.weight,
        "geometry_layer": recipe.geometry.layer,
        "geometry_margin": recipe.geometry.margin,
        "geometry_sample_tokens": recipe.geometry.sample_tokens,
        "role_loss_enabled": recipe.role_loss.enabled,
        "delimiter_weight": recipe.role_loss.delimiter_weight,
        "tool_name_weight": recipe.role_loss.tool_name_weight,
        "argument_key_weight": recipe.role_loss.argument_key_weight,
        "argument_value_weight": recipe.role_loss.argument_value_weight,
        "benchmark_profile": recipe.benchmark_profile,
        "comparison_group": recipe.comparison_group,
        "experiment_variant": recipe.experiment_variant,
        "validation_mode": recipe.validation_mode,
        "validation_datasets": recipe.validation_datasets,
        "eval_ratio": recipe.eval_ratio,
        "max_validation_samples": recipe.max_validation_samples,
        "allowed_tools": ", ".join(recipe.allowed_tools),
        "filter_unknown_tools": recipe.filter_unknown_tools,
        "exploration_strategy": exploration.strategy,
        "exploration_trials": exploration.trial_count,
        "exploration_total_minutes": exploration.total_time_limit_minutes,
        "exploration_seed": exploration.seed,
        "adapter_methods": exploration.adapter_methods,
        "lora_ranks": exploration.lora_ranks,
        "target_profiles": exploration.target_profiles,
        "sequence_lengths": exploration.sequence_lengths,
        "learning_rate_min": exploration.learning_rate_min,
        "learning_rate_max": exploration.learning_rate_max,
        "noise_modes": exploration.noise_modes,
        "noise_alpha_min": exploration.noise_alpha_min,
        "noise_alpha_max": exploration.noise_alpha_max,
        "noise_digit_orders": exploration.noise_digit_orders,
        "noise_scopes": exploration.noise_scopes,
        "noise_envelopes": exploration.noise_envelopes,
        "noise_modulation_min": exploration.noise_modulation_min,
        "noise_modulation_max": exploration.noise_modulation_max,
        "noise_clean_tail_min": exploration.noise_clean_tail_min,
        "noise_clean_tail_max": exploration.noise_clean_tail_max,
        "geometry_modes": exploration.geometry_modes,
        "geometry_scopes": exploration.geometry_scopes,
        "geometry_layers": exploration.geometry_layers,
        "geometry_weight_min": exploration.geometry_weight_min,
        "geometry_weight_max": exploration.geometry_weight_max,
        "geometry_margin_min": exploration.geometry_margin_min,
        "geometry_margin_max": exploration.geometry_margin_max,
        "role_loss_modes": exploration.role_loss_modes,
    }


def _register_sealed_gui(source: str, name: str, max_rows: int) -> dict[str, Any]:
    try:
        return register_sealed_suite(source.strip(), name.strip(), int(max_rows))
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def _start_sealed_eval(
    run_id: list[str] | str | None,
    suite_id: str | None,
    max_samples: int,
    compare_base: bool,
) -> str:
    try:
        run_ids = run_id if isinstance(run_id, list) else [run_id] if run_id else []
        if not run_ids or not suite_id:
            raise ValueError("Select both a completed run and a sealed suite")
        if len(run_ids) > 1:
            return json.dumps(
                start_sealed_eval_batch_job(
                    run_ids, suite_id, int(max_samples), bool(compare_base)
                ),
                indent=2,
            )
        return json.dumps(
            start_sealed_eval_job(run_ids[0], suite_id, int(max_samples), bool(compare_base)),
            indent=2,
        )
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return f"ERROR: {type(error).__name__}: {error}"


def _sealed_result_rows() -> list[list[Any]]:
    rows = []
    for result in sealed_results():
        adapter = result.get("adapter", {})
        base = result.get("base", {})
        routing = _display_routing_metrics(adapter)
        base_selection = _display_tool_selection_accuracy(base)
        adapter_selection = _display_tool_selection_accuracy(adapter)
        rows.append(
            [
                result.get("evaluated_at"),
                result.get("run_id"),
                result.get("suite_name"),
                adapter.get("agent_eval_samples"),
                adapter.get("agent_tool_name_expected"),
                adapter.get("agent_expected_format_accuracy"),
                routing.get("balanced_accuracy"),
                base_selection,
                adapter_selection,
                (
                    adapter_selection - base_selection
                    if isinstance(adapter_selection, (float, int))
                    and isinstance(base_selection, (float, int))
                    else None
                ),
                adapter.get("agent_tool_arguments_valid_rate"),
                adapter.get("agent_tool_argument_exact_accuracy"),
                result.get("result_path"),
            ]
        )
    return rows


def _display_routing_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return decision-only metrics, repairing schema 3/4 result files when possible."""
    predictions = metrics.get("agent_predictions")
    if isinstance(predictions, list) and predictions:
        return routing_metrics_from_predictions(predictions)
    if int(metrics.get("agent_metric_schema_version") or 0) >= 5:
        return {
            "balanced_accuracy": metrics.get("agent_balanced_tool_decision_accuracy"),
            "tool_recall": metrics.get("agent_tool_recall"),
            "no_tool_specificity": metrics.get("agent_no_tool_specificity"),
            "mcc": metrics.get("agent_tool_decision_mcc"),
        }
    # Schema 2 did not store class counts or predictions, so its raw,
    # prevalence-sensitive accuracy cannot honestly be relabeled as balanced.
    return {
        "balanced_accuracy": None,
        "legacy_raw_accuracy": metrics.get("agent_tool_decision_accuracy"),
    }


def _display_tool_selection_accuracy(metrics: dict[str, Any]) -> float | None:
    """Tool choice independent of whether the argument expression parsed."""
    predictions = metrics.get("agent_predictions")
    if isinstance(predictions, list) and predictions:
        expected = [item for item in predictions if item.get("expected_tool") is not None]
        return sum(
            item.get("regex_predicted_tool") == item.get("expected_tool") for item in expected
        ) / max(1, len(expected))
    value = metrics.get("agent_selected_tool_name_accuracy")
    return float(value) if isinstance(value, (float, int)) else None


def _display_parseable_tool_recall(metrics: dict[str, Any]) -> float | None:
    predictions = metrics.get("agent_predictions")
    if isinstance(predictions, list) and predictions:
        expected = [item for item in predictions if item.get("expected_tool") is not None]
        return sum(bool(item.get("predicted_tool")) for item in expected) / max(1, len(expected))
    value = metrics.get("agent_parseable_tool_recall")
    return float(value) if isinstance(value, (float, int)) else None


def _protocol_signature(recipe: dict[str, Any]) -> str:
    controlled = {
        key: recipe.get(key)
        for key in (
            "model",
            "datasets",
            "validation_mode",
            "validation_datasets",
            "eval_ratio",
            "max_validation_samples",
            "seed",
            "sequence_length",
            "max_source_rows",
            "max_training_samples",
            "time_limit_minutes",
            "reserve_minutes",
            "budget_mode",
            "max_steps",
            "use_rslora",
            "lora_rank",
            "lora_alpha",
            "target_profile",
            "learning_rate",
            "allowed_tools",
        )
    }
    return hashlib.sha256(json.dumps(controlled, sort_keys=True, default=str).encode()).hexdigest()[
        :10
    ]


def _compact_preflight(preflight: Any) -> Any:
    """Keep large source samples out of Gradio's JSON tree renderer."""
    if not isinstance(preflight, dict):
        return preflight
    if "status" in preflight:
        warnings = preflight.get("warnings") or []
        return {
            "status": preflight.get("status"),
            "warning_count": len(warnings) if isinstance(warnings, list) else bool(warnings),
        }
    return {name: _compact_preflight(value) for name, value in preflight.items()}


def _compact_metrics(metrics: Any) -> Any:
    if not isinstance(metrics, dict):
        return metrics
    keys = (
        "status",
        "run_id",
        "global_step",
        "train_loss",
        "eval_loss",
        "eval_perplexity",
        "agent_eval_samples",
        "agent_balanced_tool_decision_accuracy",
        "agent_tool_recall",
        "agent_no_tool_specificity",
        "agent_tool_decision_mcc",
        "agent_selected_tool_name_accuracy",
        "agent_tool_argument_keys_accuracy",
        "agent_tool_argument_exact_accuracy",
        "agent_required_argument_call_exact_accuracy",
        "agent_required_argument_call_normalized_accuracy",
        "agent_required_argument_call_elastic_accuracy",
        "agent_argument_field_micro_accuracy",
        "agent_required_argument_field_micro_accuracy",
        "agent_direct_required_terms_accuracy",
        "agent_human_task_success_rate",
        "agent_strict_human_task_success_rate",
        "agent_elastic_human_task_success_rate",
        "agent_protocol_id",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _compact_pipeline_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a browser-safe pipeline summary without predictions or full preflight rows."""
    if not result:
        return None
    scalar_keys = (
        "pipeline_id",
        "pipeline_type",
        "status",
        "stage",
        "created_at",
        "updated_at",
        "comparison_group",
        "dev_source",
        "suite_id",
        "dev_samples_requested",
        "sealed_samples_requested",
        "samples_requested",
        "max_samples",
        "compare_base",
        "development_suite_id",
        "final_suite_id",
        "baseline_sheet_id",
        "exotic_sheet_id",
        "baseline_run_id",
        "exotic_run_id",
        "geometry_only_run_id",
        "noise_geometry_run_id",
        "error",
    )
    compact = {key: result[key] for key in scalar_keys if key in result}
    for key in ("run_ids", "compare_run_ids", "sheet_ids", "sheet_names"):
        if key in result:
            compact[key] = result[key]
    if "preflight" in result:
        compact["preflight"] = _compact_preflight(result["preflight"])
    for key in (
        "baseline_metrics",
        "exotic_metrics",
        "geometry_only_metrics",
        "noise_geometry_metrics",
    ):
        if key in result:
            compact[key] = _compact_metrics(result[key])
    cells = result.get("cells")
    if isinstance(cells, dict):
        compact["cells"] = {
            name: {
                key: value
                for key, value in cell.items()
                if key in {"status", "sheet_id", "run_id", "error"}
            }
            for name, cell in cells.items()
            if isinstance(cell, dict)
        }
    summary = result.get("screening_summary")
    if isinstance(summary, dict):
        compact["screening_summary"] = {
            key: summary[key]
            for key in ("recommended_cell", "gates_passed", "ranking")
            if key in summary
        }
    for evaluation_key in (
        "dev_evaluation",
        "sealed",
        "sealed_evaluation",
        "development_evaluation",
        "final_evaluation",
    ):
        evaluation = result.get(evaluation_key)
        if isinstance(evaluation, dict):
            compact[evaluation_key] = {
                key: evaluation[key]
                for key in (
                    "status",
                    "evaluation_kind",
                    "source",
                    "suite_id",
                    "runs_requested",
                    "runs_evaluated",
                )
                if key in evaluation
            }
    for summary_key in (
        "dev_summary",
        "sealed_summary",
        "development_summary",
        "final_summary",
    ):
        summary_value = result.get(summary_key)
        if isinstance(summary_value, dict):
            compact[summary_key] = {
                "recommended_cell": summary_value.get("recommended_cell"),
                "cells": summary_value.get("cells"),
            }
    compact["full_result_path"] = result.get("result_path")
    compact["display_note"] = "Large predictions omitted from the GUI; full JSON remains on disk."
    return compact


def _compare_runs(run_ids: list[str] | None) -> tuple[list[list[Any]], str]:
    selected = set(run_ids or [])
    sheets = {sheet.id: sheet.name for sheet in WorkbookStore().list()}
    sealed_by_run_suite: dict[tuple[str, str], dict[str, Any]] = {}
    for result in [*sealed_results(), *dev_results()]:
        run_id = str(result.get("run_id"))
        suite_id = str(result.get("suite_id"))
        key = (run_id, suite_id)
        current = sealed_by_run_suite.get(key)
        candidate_key = (
            int((result.get("adapter") or {}).get("agent_eval_samples") or 0),
            int(bool(result.get("base"))),
            str(result.get("evaluated_at") or ""),
        )
        current_key = (
            int((current or {}).get("adapter", {}).get("agent_eval_samples") or 0),
            int(bool((current or {}).get("base"))),
            str((current or {}).get("evaluated_at") or ""),
        )
        if current is None or candidate_key > current_key:
            sealed_by_run_suite[key] = result
    rows = []
    for run in Registry().list_runs():
        if run["run_id"] not in selected:
            continue
        recipe = json.loads(run.get("recipe_json") or "{}")
        metrics = json.loads(run.get("metrics_json") or "{}")
        events = read_events(Path(run["output_dir"]) / "events.jsonl", limit=5000)
        latest_periodic_eval = next(
            (
                event
                for event in reversed(events)
                if event.get("event") == "log" and event.get("eval_loss") is not None
            ),
            {},
        )
        validation_loss = metrics.get("eval_loss")
        if validation_loss is None:
            validation_loss = latest_periodic_eval.get("eval_loss")
        validation_perplexity = metrics.get("eval_perplexity")
        if validation_perplexity is None and isinstance(validation_loss, (float, int)):
            validation_perplexity = math.exp(min(20.0, float(validation_loss)))
        noise_enabled = bool((recipe.get("noise") or {}).get("enabled"))
        geometry_enabled = bool((recipe.get("geometry") or {}).get("enabled"))
        role_loss_enabled = bool((recipe.get("role_loss") or {}).get("enabled"))
        literal_lock_enabled = bool(
            (recipe.get("noise") or {}).get("protect_prompt_literals")
            and (recipe.get("noise") or {}).get("value_consistency_weight", 0) > 0
        )
        human_intent_enabled = bool(
            (recipe.get("noise") or {}).get("intent_consistency_weight", 0) > 0
        )
        derived_variant = (
            "human-intent"
            if human_intent_enabled
            else "literal-lock"
            if literal_lock_enabled
            else "route-resolve"
            if noise_enabled and geometry_enabled and role_loss_enabled
            else "noise+geometry"
            if noise_enabled and geometry_enabled
            else "noise+role-loss"
            if noise_enabled and role_loss_enabled
            else "noise-only"
            if noise_enabled
            else "geometry+role-loss"
            if geometry_enabled and role_loss_enabled
            else "geometry-only"
            if geometry_enabled
            else "role-loss"
            if role_loss_enabled
            else "baseline"
        )
        run_sealed = [
            result
            for (sealed_run_id, _suite_id), result in sealed_by_run_suite.items()
            if sealed_run_id == run["run_id"]
        ] or [{}]
        for sealed in sorted(run_sealed, key=lambda item: str(item.get("suite_name") or "")):
            sealed_base = sealed.get("base", {})
            sealed_adapter = sealed.get("adapter", {})
            sealed_routing = _display_routing_metrics(sealed_adapter)
            sealed_base_selection = _display_tool_selection_accuracy(sealed_base)
            sealed_adapter_selection = _display_tool_selection_accuracy(sealed_adapter)
            rows.append(
                [
                    run["run_id"],
                    sheets.get(recipe.get("workbook_sheet_id"), "legacy / unsaved"),
                    recipe.get("benchmark_profile", "general"),
                    recipe.get("comparison_group", "legacy"),
                    derived_variant,
                    _protocol_signature(recipe),
                    recipe.get("exploration_trial"),
                    "rsLoRA" if recipe.get("use_rslora", True) else "LoRA",
                    recipe.get("lora_rank"),
                    recipe.get("target_profile"),
                    recipe.get("sequence_length"),
                    recipe.get("learning_rate"),
                    (recipe.get("noise") or {}).get("source")
                    if (recipe.get("noise") or {}).get("enabled")
                    else "off",
                    (recipe.get("geometry") or {}).get("weight")
                    if (recipe.get("geometry") or {}).get("enabled")
                    else "off",
                    metrics.get("train_loss"),
                    validation_loss,
                    validation_perplexity,
                    metrics.get("agent_tool_name_accuracy"),
                    metrics.get("agent_tool_arguments_valid_rate"),
                    sealed.get("suite_name"),
                    sealed_adapter.get("agent_eval_samples"),
                    sealed_adapter.get("agent_tool_name_expected"),
                    sealed_adapter.get("agent_expected_format_accuracy"),
                    sealed_routing.get("balanced_accuracy"),
                    sealed_base_selection,
                    sealed_adapter_selection,
                    (
                        sealed_adapter_selection - sealed_base_selection
                        if isinstance(sealed_adapter_selection, (float, int))
                        and isinstance(sealed_base_selection, (float, int))
                        else None
                    ),
                    sealed_adapter.get("agent_tool_arguments_valid_rate"),
                    sealed_adapter.get("agent_tool_argument_exact_accuracy"),
                    sealed_adapter.get("agent_metric_schema_version"),
                    sealed_adapter.get("agent_protocol_id"),
                    sealed_routing.get("tool_recall"),
                    sealed_routing.get("no_tool_specificity"),
                    sealed_routing.get("mcc"),
                    _display_parseable_tool_recall(sealed_adapter),
                    sealed_adapter.get("agent_required_argument_call_exact_accuracy"),
                    sealed_adapter.get("agent_required_argument_call_normalized_accuracy"),
                    sealed_adapter.get("agent_argument_field_micro_accuracy"),
                    sealed_adapter.get("agent_required_argument_field_micro_accuracy"),
                    sealed_adapter.get("agent_direct_required_terms_accuracy"),
                    sealed_adapter.get("agent_human_task_success_rate"),
                    sealed_adapter.get("agent_strict_human_task_success_rate"),
                    sealed_adapter.get("agent_required_argument_call_elastic_accuracy"),
                    sealed_adapter.get("agent_elastic_human_task_success_rate"),
                    sealed.get("evaluated_at"),
                ]
            )
    rows.sort(
        key=lambda row: (
            row[25] if isinstance(row[25], (float, int)) else -1,
            -(row[15] if isinstance(row[15], (float, int)) else 1e9),
        ),
        reverse=True,
    )
    chart = _comparison_chart(rows)
    return rows, chart


def _comparison_chart(rows: list[list[Any]]) -> str:
    if not rows:
        return "<p>Select at least one completed run to compare.</p>"
    variant_order = {
        "baseline": 0,
        "noise-only": 1,
        "geometry-only": 2,
        "noise+geometry": 3,
        "role-loss": 4,
        "noise+role-loss": 5,
        "geometry+role-loss": 6,
        "route-resolve": 7,
        "exotic": 8,
        "literal-lock": 9,
        "human-intent": 10,
    }
    variant_colors = {
        "baseline": "#3b82f6",
        "noise-only": "#a855f7",
        "geometry-only": "#22c55e",
        "noise+geometry": "#f97316",
        "role-loss": "#06b6d4",
        "noise+role-loss": "#ec4899",
        "geometry+role-loss": "#84cc16",
        "route-resolve": "#facc15",
        "exotic": "#a855f7",
        "literal-lock": "#eab308",
        "human-intent": "#14b8a6",
    }
    groups: dict[str, list[list[Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[19] or "Internal validation"), []).append(row)

    group_items = sorted(
        groups.items(),
        key=lambda item: max(
            (str(row[44] or "") for row in item[1] if len(row) > 44),
            default="",
        ),
        reverse=True,
    )
    cards: list[str] = []
    for group_index, (suite_name, suite_rows) in enumerate(group_items):
        suite_updated = max(
            (str(row[44] or "") for row in suite_rows if len(row) > 44),
            default="not dated",
        )
        suite_rows.sort(
            key=lambda row: (variant_order.get(str(row[4]), 9), str(row[4]), str(row[0]))
        )
        baseline = next((row for row in suite_rows if row[4] == "baseline"), None)

        # Keep the visual recommendation aligned with the screening pipeline:
        # favour routing, but do not let a routing-only win hide regressions in
        # tool selection or literal argument realization.
        baseline_tool = (
            float(baseline[25])
            if baseline and isinstance(baseline[25], (float, int))
            else 0.0
        )
        baseline_exact = (
            float(baseline[28])
            if baseline and isinstance(baseline[28], (float, int))
            else 0.0
        )

        def compromise_score(
            row: list[Any],
            baseline_tool: float = baseline_tool,
            baseline_exact: float = baseline_exact,
        ) -> float | None:
            indices = (23, 32, 31, 25, 28)
            if any(index >= len(row) or not isinstance(row[index], (float, int)) for index in indices):
                return None
            balanced, specificity, recall, tool_name, exact = (
                float(row[index]) for index in indices
            )
            passes_gates = (
                specificity >= 0.60
                and recall >= 0.85
                and balanced >= 0.70
                and tool_name >= baseline_tool - 0.02
                and exact >= baseline_exact
            )
            if not passes_gates:
                return None
            return (
                0.30 * balanced
                + 0.15 * specificity
                + 0.15 * recall
                + 0.20 * tool_name
                + 0.20 * exact
            )

        scored_rows = [(row, compromise_score(row)) for row in suite_rows]
        scored_rows = [(row, score) for row, score in scored_rows if score is not None]
        best_compromise = max(scored_rows, key=lambda item: item[1], default=(None, None))
        best_compromise_row, best_compromise_score = best_compromise
        if best_compromise_row is None and baseline is not None:
            # Older stored comparison rows predate the routing recall/specificity
            # columns required by the current compromise gate. Keep their chart
            # deltas visible without pretending that a modern compromise score
            # was computed.
            legacy_candidates = [row for row in suite_rows if row is not baseline]
            best_compromise_row = max(
                legacy_candidates,
                key=lambda row: (
                    float(row[28]) if len(row) > 28 and isinstance(row[28], (int, float)) else -1.0,
                    float(row[25]) if len(row) > 25 and isinstance(row[25], (int, float)) else -1.0,
                ),
                default=None,
            )
        metrics = [
            ("Parseable expected response format (raw)", 22),
            ("Balanced tool / no-tool routing", 23),
            ("Tool-attempt recall", 31),
            ("No-tool abstention specificity", 32),
            ("Parseable call recall", 34),
            ("Selected tool name (syntax-independent)", 25),
            ("Required arguments present", 27),
            ("Exact argument values", 28),
            ("Required values exact (whole call)", 35),
            ("Required values normalized diagnostic", 36),
            ("All argument fields exact (micro)", 37),
            ("Required argument fields exact (micro)", 38),
            ("Direct-answer criteria satisfied", 39),
            ("Human task success", 40),
            ("Strict human task success", 41),
            ("Required values elastic (semantic-safe)", 42),
            ("Elastic human task success", 43),
        ]
        metrics = [
            metric
            for metric in metrics
            if any(metric[1] < len(row) and row[metric[1]] is not None for row in suite_rows)
        ]
        columns = f"minmax(220px,1.3fr) repeat({len(suite_rows)},minmax(180px,1fr)) 90px"
        header = ['<div style="font-weight:700;color:#d4d4d8">Metric</div>']
        for row in suite_rows:
            variant = str(row[4] or "custom")
            color = variant_colors.get(variant, "#22c55e")
            label = html.escape(variant.upper())
            sheet = html.escape(str(row[1]))
            run_id = html.escape(str(row[0]))
            samples = row[20]
            required = row[21]
            is_best = row is best_compromise_row
            best_style = (
                "background:rgba(250,204,21,.09);border:2px solid #facc15;"
                "border-bottom-width:1px;border-radius:9px 9px 0 0;padding:8px;"
                if is_best
                else ""
            )
            best_badge = (
                '<div style="display:inline-block;background:#facc15;color:#18181b;'
                'font-size:10px;font-weight:900;padding:3px 6px;border-radius:999px;'
                'margin-bottom:5px">BEST COMPROMISE</div>'
                if is_best
                else ""
            )
            header.append(
                f'<div style="border-top:4px solid {color};padding-top:6px;{best_style}">'
                f'{best_badge}'
                f'<div style="font-weight:800;color:{color};font-size:15px">{label}</div>'
                f'<div style="font-size:11px;color:#d4d4d8">{sheet}</div>'
                f'<div style="font-size:10px;color:#71717a">{run_id}</div>'
                f'<div style="font-size:10px;color:#a1a1aa">samples {samples or "n/a"}; '
                f"tool-required {required or 'n/a'}</div>"
                + (
                    f'<div style="font-size:10px;color:#fde047;font-weight:800">'
                    f'compromise score {float(best_compromise_score) * 100:.2f}%</div>'
                    if is_best and isinstance(best_compromise_score, (float, int))
                    else ""
                )
                + "</div>"
            )
        header.append(
            '<div style="font-weight:800;color:#fde047">BEST COMPROMISE<br>'
            '<span style="font-size:10px;color:#d4d4d8">Δ vs BASE</span></div>'
        )

        def comparison_bar(value: float, color: str, label: str) -> str:
            score = max(0.0, min(1.0, float(value)))
            return (
                '<div style="display:grid;grid-template-columns:28px 1fr 48px;gap:5px;'
                'align-items:center">'
                f'<div style="font-size:8px;font-weight:800;color:{color}">{label}</div>'
                '<div style="height:9px;background:#27272a;border-radius:5px;overflow:hidden">'
                f'<div style="height:100%;width:{score * 100:.2f}%;background:{color}"></div>'
                '</div>'
                f'<div style="font-variant-numeric:tabular-nums;color:{color};font-weight:700;'
                f'font-size:11px">{score * 100:.1f}%</div></div>'
            )

        body: list[str] = []
        for metric_name, metric_index in metrics:
            body.append(
                f'<div style="color:#e4e4e7;font-weight:600">{html.escape(metric_name)}</div>'
            )
            base_value = (
                baseline[metric_index] if baseline and metric_index < len(baseline) else None
            )
            for row in suite_rows:
                variant = str(row[4] or "custom")
                color = variant_colors.get(variant, "#22c55e")
                value = row[metric_index] if metric_index < len(row) else None
                is_best = row is best_compromise_row
                best_cell_style = (
                    "background:rgba(250,204,21,.09);"
                    "box-shadow:inset 2px 0 #facc15,inset -2px 0 #facc15;"
                    + "padding:5px 7px;"
                    if is_best
                    else ""
                )
                if not isinstance(value, (float, int)):
                    body.append(
                        f'<div style="color:#71717a;{best_cell_style}">n/a</div>'
                    )
                    continue
                bars = comparison_bar(float(value), color, "BASE" if row is baseline else "NEW")
                if row is not baseline and isinstance(base_value, (float, int)):
                    bars += comparison_bar(float(base_value), variant_colors["baseline"], "BASE")
                body.append(f'<div style="display:grid;gap:3px;{best_cell_style}">{bars}</div>')
            best_value = (
                best_compromise_row[metric_index]
                if best_compromise_row is not None
                and metric_index < len(best_compromise_row)
                else None
            )
            if isinstance(base_value, (float, int)) and isinstance(best_value, (float, int)):
                delta = (float(best_value) - float(base_value)) * 100
                delta_color = "#4ade80" if delta >= 0 else "#fb7185"
                winner_label = html.escape(str(best_compromise_row[4]).upper())
                body.append(
                    f'<div style="color:{delta_color};font-weight:800;'
                    f'font-variant-numeric:tabular-nums">{delta:+.1f} pp'
                    f'<div style="font-size:9px">{winner_label}</div></div>'
                )
            else:
                body.append('<div style="color:#71717a">n/a</div>')

        # Descriptive, unweighted mean of the metrics visible in this card.
        # It is intentionally separate from the weighted Best Compromise score.
        body.append(
            '<div style="color:#fde047;font-weight:900;border-top:2px solid #52525b;'
            'padding-top:9px">AVERAGE OF DISPLAYED METRICS'
            '<div style="font-size:9px;color:#a1a1aa">descriptive; unweighted</div></div>'
        )
        best_average_delta: float | None = None
        for row in suite_rows:
            variant = str(row[4] or "custom")
            color = variant_colors.get(variant, "#22c55e")
            common_values = [
                (float(row[index]), float(baseline[index]))
                for _name, index in metrics
                if baseline is not None
                and index < len(row)
                and index < len(baseline)
                and isinstance(row[index], (float, int))
                and isinstance(baseline[index], (float, int))
            ]
            is_best = row is best_compromise_row
            best_average_style = (
                "background:rgba(250,204,21,.09);"
                "box-shadow:inset 2px 0 #facc15,inset -2px 0 #facc15;"
                "border-bottom:2px solid #facc15;border-radius:0 0 9px 9px;padding:7px;"
                if is_best
                else "border-top:2px solid #52525b;padding-top:7px;"
            )
            if not common_values:
                body.append(f'<div style="color:#71717a;{best_average_style}">n/a</div>')
                continue
            selected_average = sum(value for value, _base in common_values) / len(common_values)
            base_average = sum(base for _value, base in common_values) / len(common_values)
            average_delta = (selected_average - base_average) * 100
            if is_best:
                best_average_delta = average_delta
            average_delta_color = "#4ade80" if average_delta >= 0 else "#fb7185"
            bars = comparison_bar(
                selected_average,
                color,
                "BASE" if row is baseline else "NEW",
            )
            if row is not baseline:
                bars += comparison_bar(base_average, variant_colors["baseline"], "BASE")
            delta_label = (
                '<div style="font-size:10px;font-weight:900;margin-bottom:4px;'
                f'color:{average_delta_color}">AVG Δ {average_delta:+.2f} pp</div>'
                if row is not baseline
                else '<div style="font-size:10px;color:#93c5fd;margin-bottom:4px">REFERENCE</div>'
            )
            body.append(
                f'<div style="display:grid;gap:3px;{best_average_style}">'
                f'{delta_label}{bars}</div>'
            )
        if best_compromise_row is not None and best_average_delta is not None:
            delta_color = "#4ade80" if best_average_delta >= 0 else "#fb7185"
            body.append(
                f'<div style="border-top:2px solid #52525b;padding-top:9px;color:{delta_color};'
                f'font-weight:900;font-variant-numeric:tabular-nums">{best_average_delta:+.2f} pp'
                f'<div style="font-size:9px">{html.escape(str(best_compromise_row[4]).upper())}'
                '</div></div>'
            )
        else:
            body.append('<div style="border-top:2px solid #52525b;color:#71717a">n/a</div>')

        validation = []
        for row in suite_rows:
            value = row[15]
            variant = html.escape(str(row[4] or "custom").upper())
            validation.append(
                f"{variant}: {float(value):.5f}"
                if isinstance(value, (float, int))
                else f"{variant}: n/a"
            )
        test_sizes = sorted(
            {int(row[20]) for row in suite_rows if isinstance(row[20], (float, int))}
        )
        size_label = "/".join(str(value) for value in test_sizes) or "n/a"
        distributions = sorted(
            {
                (int(row[21]), int(row[20]) - int(row[21]))
                for row in suite_rows
                if isinstance(row[20], (float, int)) and isinstance(row[21], (float, int))
            }
        )
        distribution_badges = "".join(
            (
                '<div style="background:#dc2626;color:white;font-weight:900;font-size:12px;'
                'padding:6px 10px;border-radius:999px;white-space:nowrap">'
                f"IMBALANCED: {tool_count} TOOL / {no_tool_count} NO-TOOL</div>"
            )
            if tool_count != no_tool_count
            else (
                '<div style="background:#166534;color:white;font-weight:900;font-size:12px;'
                'padding:6px 10px;border-radius:999px;white-space:nowrap">'
                f"BALANCED: {tool_count} / {no_tool_count}</div>"
            )
            for tool_count, no_tool_count in distributions
        )
        card = (
            '<section style="background:#18181b;border:1px solid #3f3f46;border-radius:10px;'
            'padding:16px;margin:12px 0">'
            '<div style="display:flex;justify-content:space-between;align-items:center;gap:12px">'
            f'<h3 style="margin:0 0 4px;color:#fafafa">{html.escape(suite_name)}</h3>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap">{distribution_badges}'
            '<div style="background:#f59e0b;color:#111827;font-weight:900;font-size:14px;'
            'padding:6px 10px;border-radius:999px;white-space:nowrap">'
            f"TEST SIZE: {html.escape(size_label)} / ADAPTER</div></div></div>"
            '<div style="font-size:11px;color:#a1a1aa;margin-bottom:14px">'
            "Blue = baseline; purple = noise-only; green = geometry-only; "
            "orange = noise+geometry; yellow = π-LiteralLock; teal = Human Intent. Higher is better. "
            "Raw format depends on tool/no-tool prevalence; balanced routing is recomputed "
            "from actual tool attempts when per-sample evidence exists. Legacy schema v2 "
            "routing is n/a because it cannot be reconstructed honestly. "
            f"Validation loss (lower is better): {html.escape(' · '.join(validation))}</div>"
            f'<div style="display:grid;grid-template-columns:{columns};gap:10px 14px;'
            'align-items:center;overflow-x:auto">' + "".join(header + body) + "</div></section>"
        )
        latest_label = "LATEST RESULT" if group_index == 0 else "HISTORICAL RESULT"
        open_attribute = " open" if group_index == 0 else ""
        cards.append(
            f'<details{open_attribute} style="margin:10px 0">'
            '<summary style="cursor:pointer;padding:12px 14px;background:#27272a;'
            'border:1px solid #3f3f46;border-radius:9px;color:#fafafa;font-weight:800">'
            f'{latest_label} · {html.escape(suite_name)} · {len(suite_rows)} run(s) · '
            f'{html.escape(suite_updated)}</summary>{card}</details>'
        )
    return "".join(cards)


def _export_comparison(run_ids: list[str] | None) -> list[str]:
    """Export the selected comparison as reproducible JSON, Markdown and SVG."""

    rows, _chart = _compare_runs(run_ids)
    if not rows:
        raise ValueError("Select at least one completed run before exporting")
    generated_at = datetime.now().astimezone().isoformat()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = project_root() / "runs" / "exports" / f"comparison-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "run_id": row[0],
            "sheet": row[1],
            "comparison_group": row[3],
            "variant": row[4],
            "protocol_id": row[5],
            "suite": row[19],
            "samples": row[20],
            "tool_required": row[21],
            "expected_format": row[22],
            "balanced_routing": row[23],
            "selected_tool": row[25],
            "exact_arguments": row[35],
            "human_task_success": row[40],
            "elastic_human_task_success": row[43],
            "evaluated_at": row[44],
        }
        for row in rows
    ]
    json_path = output_dir / "comparison.json"
    json_path.write_text(
        json.dumps(
            {"generated_at": generated_at, "run_ids": run_ids or [], "results": records},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def percentage(value: Any) -> str:
        return f"{float(value) * 100:.2f}%" if isinstance(value, (float, int)) else "n/a"

    lines = [
        "# Exotic Agent Trainer comparison",
        "",
        f"Generated: `{generated_at}`",
        "",
        "| Suite | Variant | Run | Protocol | Routing | Tool | Exact args | Human success |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {suite} | {variant} | `{run_id}` | `{protocol_id}` | {routing} | {tool} | "
            "{exact} | {human} |".format(
                **record,
                routing=percentage(record["balanced_routing"]),
                tool=percentage(record["selected_tool"]),
                exact=percentage(record["exact_arguments"]),
                human=percentage(record["human_task_success"]),
            )
        )
    lines.extend(
        (
            "",
            "Scores are comparable only within an identical evaluation suite and protocol ID.",
            "This report is an experiment artifact, not an external certification.",
        )
    )
    markdown_path = output_dir / "comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest_suite = max(records, key=lambda item: str(item.get("evaluated_at") or ""))["suite"]
    picture_records = [record for record in records if record["suite"] == latest_suite]
    metric_specs = (
        ("Routing", "balanced_routing"),
        ("Tool selection", "selected_tool"),
        ("Exact arguments", "exact_arguments"),
        ("Human success", "human_task_success"),
    )
    colors = ("#3b82f6", "#a855f7", "#22c55e", "#f97316", "#eab308", "#14b8a6")
    row_height = 34
    height = 150 + len(picture_records) * len(metric_specs) * row_height
    svg: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" '
            f'viewBox="0 0 1400 {height}">'
        ),
        '<rect width="100%" height="100%" fill="#18181b"/>',
        (
            '<text x="36" y="42" fill="#fafafa" font-family="sans-serif" font-size="24" '
            f'font-weight="700">{html.escape(str(latest_suite))}</text>'
        ),
        (
            '<text x="36" y="68" fill="#a1a1aa" font-family="sans-serif" font-size="13">'
            f'Generated {html.escape(generated_at)} · higher is better</text>'
        ),
    ]
    y = 110
    for record_index, record in enumerate(picture_records):
        color = colors[record_index % len(colors)]
        label = f"{record['variant']} · {record['run_id']}"
        svg.append(
            f'<text x="36" y="{y}" fill="{color}" font-family="sans-serif" '
            f'font-size="17" font-weight="700">{html.escape(label)}</text>'
        )
        y += 24
        for metric_label, key in metric_specs:
            value = record.get(key)
            score = max(0.0, min(1.0, float(value))) if isinstance(value, (float, int)) else 0.0
            svg.extend(
                (
                    (
                        f'<text x="56" y="{y + 16}" fill="#e4e4e7" '
                        f'font-family="sans-serif" font-size="13">'
                        f'{html.escape(metric_label)}</text>'
                    ),
                    f'<rect x="250" y="{y}" width="900" height="20" rx="7" fill="#27272a"/>',
                    (
                        f'<rect x="250" y="{y}" width="{900 * score:.2f}" height="20" '
                        f'rx="7" fill="{color}"/>'
                    ),
                    (
                        f'<text x="1170" y="{y + 16}" fill="{color}" '
                        f'font-family="sans-serif" font-size="14" font-weight="700">'
                        f'{percentage(value)}</text>'
                    ),
                )
            )
            y += row_height
        y += 16
    svg.append("</svg>")
    svg_path = output_dir / "comparison.svg"
    svg_path.write_text("".join(svg), encoding="utf-8")
    return [str(markdown_path), str(json_path), str(svg_path)]


def _stop_run() -> str:
    try:
        return json.dumps(stop_latest_training_job(), indent=2)
    except Exception as error:  # noqa: BLE001 - UI boundary returns a visible error
        return f"ERROR: {type(error).__name__}: {error}"


def _geometry_svg(events: list[dict[str, Any]]) -> str:
    geometries = [item for item in events if item.get("event") == "geometry" and item.get("points")]
    if not geometries:
        return "<p>Enable geometric loss to generate a live latent projection.</p>"
    points = geometries[-1]["points"]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(value: float, low: float, high: float, size: float) -> float:
        return 20 + (value - low) / max(1e-9, high - low) * size

    circles = "".join(
        f'<circle cx="{scale(x, min_x, max_x, 360):.1f}" '
        f'cy="{260 - scale(y, min_y, max_y, 220):.1f}" r="4" fill="#7c3aed" opacity="0.72" />'
        for x, y in zip(xs, ys, strict=True)
    )
    return (
        '<svg viewBox="0 0 400 280" style="width:100%;max-height:320px;background:#111827;'
        f'border-radius:8px">{circles}</svg><p>PCA layer projection at step '
        f"{geometries[-1].get('step', 0)}</p>"
    )


def _metrics_chart(events: list[dict[str, Any]]) -> str:
    samples = [
        {
            "epoch": float(item["epoch"]),
            "loss": float(item["loss"]) if item.get("loss") is not None else None,
            "accuracy": (
                float(item["mean_token_accuracy"])
                if item.get("mean_token_accuracy") is not None
                else None
            ),
        }
        for item in events
        if item.get("event") == "log"
        and item.get("epoch") is not None
        and (item.get("loss") is not None or item.get("mean_token_accuracy") is not None)
    ]
    if not samples:
        return "<p>Loss and mean token accuracy will appear after the first training log.</p>"
    if len(samples) > 300:
        stride = max(1, len(samples) // 300)
        samples = samples[::stride]

    width, height = 820.0, 340.0
    left, right, top, bottom = 68.0, 72.0, 28.0, 58.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    epochs = [float(item["epoch"]) for item in samples]
    loss_values = [float(item["loss"]) for item in samples if item["loss"] is not None]
    accuracy_values = [float(item["accuracy"]) for item in samples if item["accuracy"] is not None]
    epoch_min, epoch_max = min(epochs), max(epochs)
    if epoch_max <= epoch_min:
        epoch_max = epoch_min + 1.0
    loss_min = min(loss_values) if loss_values else 0.0
    loss_max = max(loss_values) if loss_values else 1.0
    if loss_max <= loss_min:
        loss_max = loss_min + 1.0
    loss_padding = max(1e-6, (loss_max - loss_min) * 0.08)
    loss_min = max(0.0, loss_min - loss_padding)
    loss_max += loss_padding

    def x_position(epoch: float) -> float:
        return left + (epoch - epoch_min) / (epoch_max - epoch_min) * plot_width

    def loss_y(value: float) -> float:
        return top + (loss_max - value) / (loss_max - loss_min) * plot_height

    def accuracy_y(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * plot_height

    loss_points = " ".join(
        f"{x_position(float(item['epoch'])):.1f},{loss_y(float(item['loss'])):.1f}"
        for item in samples
        if item["loss"] is not None
    )
    accuracy_points = " ".join(
        f"{x_position(float(item['epoch'])):.1f},{accuracy_y(float(item['accuracy'])):.1f}"
        for item in samples
        if item["accuracy"] is not None
    )
    grid = []
    for index in range(6):
        ratio = index / 5
        x = left + ratio * plot_width
        epoch = epoch_min + ratio * (epoch_max - epoch_min)
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" '
            'stroke="#3f3f46" stroke-width="1" />'
            f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" '
            f'fill="#d4d4d8" font-size="12">{epoch:.2f}</text>'
        )
    for index in range(6):
        ratio = index / 5
        y = top + ratio * plot_height
        loss_label = loss_max - ratio * (loss_max - loss_min)
        accuracy_label = 1.0 - ratio
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
            'stroke="#3f3f46" stroke-width="1" />'
            f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#fb923c" font-size="12">{loss_label:.3f}</text>'
            f'<text x="{left + plot_width + 9}" y="{y + 4:.1f}" text-anchor="start" '
            f'fill="#4ade80" font-size="12">{accuracy_label:.1f}</text>'
        )
    lines = ""
    if loss_points:
        lines += (
            f'<polyline points="{loss_points}" fill="none" stroke="#f97316" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        )
    if accuracy_points:
        lines += (
            f'<polyline points="{accuracy_points}" fill="none" stroke="#22c55e" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        )
    latest_loss = f"{loss_values[-1]:.5f}" if loss_values else "n/a"
    latest_accuracy = f"{accuracy_values[-1]:.5f}" if accuracy_values else "n/a"
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" '
        'style="width:100%;max-height:420px;background:#18181b;border-radius:8px">'
        + "".join(grid)
        + lines
        + f'<text x="{width / 2:.1f}" y="18" text-anchor="middle" fill="#fafafa" '
        'font-size="14" font-weight="700">Training metrics by epoch</text>'
        f'<text x="{width / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        'fill="#d4d4d8" font-size="13">Epoch</text>'
        f'<text x="{left}" y="{height - 6}" fill="#fb923c" font-size="12">'
        f"Loss (left axis), latest {latest_loss}</text>"
        f'<text x="{left + plot_width}" y="{height - 6}" text-anchor="end" '
        f'fill="#4ade80" font-size="12">Accuracy (right axis), latest {latest_accuracy}</text>'
        "</svg>"
    )


def _jobs_for_display() -> list[dict[str, Any]]:
    displayed = []
    for raw in job_status():
        item = dict(raw)
        for field in ("started_at", "stop_requested_at", "finished_at"):
            if item.get(field):
                item[field.replace("_at", "_local")] = _local_time(item[field])
        displayed.append(item)
    return displayed


def _event_for_display(event: dict[str, Any]) -> dict[str, Any]:
    displayed = {"local_time": _local_time(event.get("time", ""))}
    displayed.update({key: value for key, value in event.items() if key != "time"})
    return displayed


def _progress_summary(
    run: dict[str, Any] | None,
    events: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    latest_event = events[-1] if events else {}
    event_name = str(latest_event.get("event", ""))
    status = str(run.get("status", "idle")) if run else "idle"
    phase_by_event = {
        "run_created": "Loading model",
        "model_loaded": "Preparing dataset",
        "dataset_ready": "Preparing trainer",
        "train_begin": "Training",
        "log": "Training",
        "geometry": "Training + latent geometry",
        "geometry_error": "Training; geometry monitor reported an error",
        "checkpoint": "Saving checkpoint",
        "time_budget_reached": "Training budget reached; saving",
        "stop_requested": "Stopping safely and saving adapter",
        "adapter_saved": "Adapter saved",
        "validation_eval_begin": "Validation/Dev evaluation",
        "validation_eval_complete": "Agentic evaluation",
        "validation_eval_failed": "Agentic evaluation after validation error",
        "agent_eval_begin": "Agentic evaluation",
        "agent_eval_complete": "Finalizing results",
        "agent_eval_failed": "Finalizing after evaluation error",
        "run_complete": "Complete",
        "run_stopped": "Stopped safely",
        "run_failed": "Failed",
    }
    phase = phase_by_event.get(event_name, status.replace("_", " ").title())

    terminal = event_name in {"run_complete", "run_stopped", "run_failed"} or status in {
        "complete",
        "stopped",
        "failed",
    }
    percent = 100.0 if terminal else 0.0
    if not terminal and events:
        created = next((item for item in events if item.get("event") == "run_created"), {})
        train_begin = next((item for item in events if item.get("event") == "train_begin"), {})
        budget_seconds = max(60.0, float(created.get("training_budget_minutes", 1)) * 60)
        if train_begin:
            elapsed = max(0.0, time.time() - float(train_begin.get("time", time.time())))
            percent = min(90.0, 10.0 + 80.0 * elapsed / budget_seconds)
        elif event_name == "dataset_ready":
            percent = 8.0
        elif event_name == "model_loaded":
            percent = 5.0
        else:
            percent = 2.0
        if event_name == "adapter_saved":
            percent = 93.0
        elif event_name in {"validation_eval_begin", "validation_eval_failed"}:
            percent = 95.0
        elif event_name == "validation_eval_complete":
            percent = 97.0
        elif event_name in {"agent_eval_begin", "agent_eval_failed"}:
            percent = 98.0
        elif event_name == "agent_eval_complete":
            percent = 99.0
    elif not events and jobs and jobs[0].get("status") in {"running", "stopping"}:
        phase = "Launching training process"
        percent = 1.0

    color = "#f97316"
    if status == "complete" or event_name == "run_complete":
        color = "#22c55e"
    elif status == "stopped" or event_name == "run_stopped":
        color = "#eab308"
    elif status == "failed" or event_name == "run_failed":
        color = "#ef4444"
    progress_html = (
        '<div style="padding:8px 0">'
        '<div style="height:30px;background:#27272a;border-radius:8px;overflow:hidden">'
        f'<div style="height:100%;width:{percent:.1f}%;background:{color};'
        'transition:width .5s;border-radius:8px"></div></div>'
        f'<div style="text-align:center;margin-top:-27px;color:white;font-weight:700">'
        f"{percent:.1f}%</div></div>"
    )

    result: dict[str, Any] = {
        "run": run.get("run_id") if run else None,
        "status": status,
        "output": run.get("output_dir") if run else None,
    }
    latest_log = next((item for item in reversed(events) if item.get("event") == "log"), {})
    for key in ("step", "epoch", "loss", "train_loss", "num_tokens", "mean_token_accuracy"):
        if key in latest_log:
            result[key] = latest_log[key]
    if run:
        try:
            final_metrics = json.loads(run.get("metrics_json") or "{}")
        except json.JSONDecodeError:
            final_metrics = {}
        for key in ("global_step", "train_loss", "elapsed_total_seconds", "stopped_by_user"):
            if key in final_metrics:
                result[key] = final_metrics[key]
    return progress_html, phase, result


def _monitor(
    log_cutoff: float = 0.0,
) -> tuple[list[list[Any]], str, str, str, str, str, dict[str, Any], str]:
    runs = Registry().list_runs()
    events_text = "No run events yet."
    events: list[dict[str, Any]] = []
    if runs:
        event_path = Path(runs[0]["output_dir"]) / "events.jsonl"
        events = read_events(event_path, limit=500)
        visible_events = [
            item for item in events if float(item.get("time", 0.0)) > float(log_cutoff or 0.0)
        ]
        events_text = "\n".join(
            json.dumps(_event_for_display(item), ensure_ascii=False)
            for item in reversed(visible_events[-20:])
        )
        if not events_text:
            events_text = "No new events since visible logs were reset."
        geometry = _geometry_svg(events)
    else:
        geometry = _geometry_svg([])
    jobs = _jobs_for_display()
    progress, phase, result = _progress_summary(runs[0] if runs else None, events, jobs)
    return (
        _run_rows(),
        json.dumps(jobs, indent=2),
        events_text,
        geometry,
        progress,
        phase,
        result,
        _metrics_chart(events),
    )


def launch_gui(host: str, port: int, share: bool = False) -> None:
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            "GUI dependencies missing; install the project with the [gui] extra"
        ) from error

    report = system_report()
    model_choices = _model_choices()
    dataset_choices = _dataset_choices()
    default_model = (
        model_choices[0][1] if model_choices else "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct"
    )
    default_dataset = [dataset_choices[0][1]] if dataset_choices else []

    with gr.Blocks(title="Exotic Agent Trainer") as demo:
        gr.Markdown("# Exotic Agent Trainer\nLocal rsLoRA workbench for agentic coding models.")
        with gr.Tab("System"):
            gr.JSON(value=report, label="Environment and Intel XPU")
        with gr.Tab("Official BFCL"):
            gr.Markdown(
                "Run the official Berkeley Function Calling Leaderboard datasets and AST "
                "evaluator. **BFCL v3** reproduces the protocol generation used by older model "
                "cards; **BFCL v4** is the current benchmark and adds agentic sections. The "
                "managed local path serves the selected model/adapter on Intel XPU, so it does "
                "not require CUDA/vLLM. Compact setup reuses the project venv and never installs "
                "BFCL's duplicate Torch/CUDA stack. Smoke results are diagnostic and are marked "
                "non-publishable. Disk space is checked before any run; with low free space, "
                "only Smoke is allowed."
            )
            with gr.Accordion(
                "AUTOMATED BFCL v3 FAIR PAIR: base + rsLoRA baseline + pi decimal noise",
                open=True,
            ):
                gr.Markdown(
                    "This bounded pipeline trains two **fixed-step** adapters sequentially on "
                    "the same human-like, schema-conditioned 13-tool data. The only difference "
                    "is natural two-digit pi modulation (`14,15,92,65,...`, cyclic), applied to "
                    "prompt embeddings with a final 30% clean phase. It then evaluates the "
                    "untouched model and both adapters on the pinned **full BFCL v3** protocol. "
                    "Artifacts are written to the configured spacious runs directory."
                )
                with gr.Row():
                    bfcl_pair_steps = gr.Number(
                        value=400,
                        precision=0,
                        minimum=100,
                        maximum=2000,
                        label="Fixed optimizer steps per adapter",
                    )
                    bfcl_pair_start = gr.Button(
                        "START FAIR TRAINING + FULL BFCL v3",
                        variant="primary",
                    )
                    bfcl_pair_refresh = gr.Button("Refresh fair-pair status")
                bfcl_pair_launcher = gr.JSON(label="Fair-pair launcher")
                bfcl_pair_status = gr.JSON(
                    value=_refresh_bfcl_pair(),
                    label="Live fair-pair / current BFCL target",
                )
                bfcl_pair_timer = gr.Timer(value=5.0, active=True)
                bfcl_pair_start.click(
                    _start_bfcl_v3_pair,
                    inputs=[bfcl_pair_steps],
                    outputs=[bfcl_pair_launcher],
                )
                bfcl_pair_refresh.click(_refresh_bfcl_pair, outputs=[bfcl_pair_status])
                bfcl_pair_timer.tick(_refresh_bfcl_pair, outputs=[bfcl_pair_status])
            with gr.Row():
                bfcl_version = gr.Dropdown(
                    choices=[
                        (f"{key.upper()} — {value['label']}", key)
                        for key, value in BFCL_SPECS.items()
                    ],
                    value="v4",
                    label="Official BFCL protocol",
                )
                bfcl_profile = gr.Dropdown(
                    choices=[
                        (value["label"], key) for key, value in BFCL_PROFILES.items()
                    ],
                    value="smoke",
                    label="Test scope",
                )
                bfcl_handler = gr.Dropdown(
                    choices=[
                        (
                            "Liquid AI LFM2 native tools (recommended for this model)",
                            "liquid-lfm2",
                        ),
                        ("Generic OpenAI-compatible native tools", "generic-openai-tools"),
                    ],
                    value="liquid-lfm2",
                    label="Model handler",
                )
            with gr.Row():
                bfcl_transport = gr.Dropdown(
                    choices=[
                        ("Managed local Intel XPU gateway", "managed-local-xpu"),
                        ("Existing OpenAI-compatible endpoint", "existing-openai-endpoint"),
                    ],
                    value="managed-local-xpu",
                    label="Inference transport",
                )
                bfcl_target_choices = _bfcl_target_choices()
                bfcl_target = gr.Dropdown(
                    choices=bfcl_target_choices,
                    value=bfcl_target_choices[0][1] if bfcl_target_choices else None,
                    allow_custom_value=True,
                    label="Base model or completed adapter",
                )
                bfcl_smoke_samples = gr.Number(
                    value=20,
                    precision=0,
                    label="Smoke samples total (ignored by official full sections)",
                )
            with gr.Row():
                bfcl_endpoint = gr.Textbox(
                    value="http://127.0.0.1:8080/v1",
                    label="Existing endpoint (used only when selected above)",
                )
                bfcl_api_model = gr.Textbox(
                    value="local-model",
                    label="API model id (generic endpoints; local gateway ignores it)",
                )
            gr.Markdown(
                "The Liquid handler adds LFM-specific abstention/tool guidance and passes the "
                "BFCL schemas through the tokenizer's native `tools=` path. The generic handler "
                "does not add vendor instructions and can test any compatible model or server."
            )
            with gr.Row():
                bfcl_setup = gr.Button("Install/update compact BFCL evaluator")
                bfcl_start = gr.Button("Start official BFCL evaluation", variant="primary")
                bfcl_refresh = gr.Button("Refresh BFCL status")
            bfcl_launcher = gr.JSON(label="BFCL launcher")
            bfcl_environment = gr.JSON(
                value=bfcl_environment_status(), label="Pinned environments"
            )
            bfcl_latest = gr.JSON(value=_latest_bfcl_state(), label="Latest BFCL result/state")
            bfcl_table = gr.Dataframe(
                headers=[
                    "BFCL run",
                    "Status",
                    "Stage",
                    "Version",
                    "Scope",
                    "Handler",
                    "Target",
                    "Publishable",
                    "Pinned commit",
                    "Output",
                ],
                value=_bfcl_rows(),
                interactive=False,
            )
            bfcl_exports = gr.File(
                value=_bfcl_export_files(),
                label="Latest BFCL report, scores and logs",
                file_count="multiple",
                interactive=False,
            )
            bfcl_setup.click(_start_bfcl_setup, inputs=[bfcl_version], outputs=[bfcl_launcher])
            bfcl_start.click(
                _start_bfcl_gui,
                inputs=[
                    bfcl_version,
                    bfcl_profile,
                    bfcl_handler,
                    bfcl_transport,
                    bfcl_target,
                    bfcl_endpoint,
                    bfcl_api_model,
                    bfcl_smoke_samples,
                ],
                outputs=[bfcl_launcher],
            )
            bfcl_refresh.click(
                _refresh_bfcl_gui,
                outputs=[bfcl_environment, bfcl_table, bfcl_latest, bfcl_exports],
            )
        with gr.Tab("Models"):
            gr.Markdown("Register models here, or choose an existing model for the Train tab.")
            with gr.Row():
                model_path = gr.Textbox(
                    value="/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct",
                    label="Model folder",
                )
                model_name = gr.Textbox(label="Optional name")
            add_model = gr.Button("Inspect and register", variant="primary")
            model_status = gr.Textbox(label="Status")
            with gr.Row():
                model_train_picker = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Choose a registered model",
                )
                use_model = gr.Button("Use this model in Train", variant="secondary")
            model_table = gr.Dataframe(
                headers=["ID", "Name", "Architecture", "B params", "Path"],
                value=_model_rows(),
                interactive=False,
            )
            add_model.click(_add_model, [model_path, model_name], [model_status, model_table])
        with gr.Tab("Datasets"):
            gr.Markdown(
                "Register or download datasets here, or choose an existing dataset for Train."
            )
            with gr.Row():
                dataset_source = gr.Textbox(
                    label="Local path, Hugging Face id, or hf-file:owner/repo::file.jsonl"
                )
                dataset_name = gr.Textbox(label="Optional name")
            add_dataset = gr.Button("Inspect and register", variant="primary")
            fetch_dataset = gr.Button("Download as Parquet and register")
            dataset_status = gr.Textbox(label="Status")
            with gr.Row():
                dataset_train_picker = gr.Dropdown(
                    choices=dataset_choices,
                    value=default_dataset[0] if default_dataset else None,
                    label="Choose a registered dataset",
                )
                use_dataset = gr.Button("Add this dataset to Train", variant="secondary")
            dataset_table = gr.Dataframe(
                headers=["ID", "Name", "Type", "Rows", "Secrets", "Source"],
                value=_dataset_rows(),
                interactive=False,
            )
            add_dataset.click(
                _add_dataset, [dataset_source, dataset_name], [dataset_status, dataset_table]
            )
            fetch_dataset.click(
                _fetch_dataset, [dataset_source, dataset_name], [dataset_status, dataset_table]
            )
        with gr.Tab("Train"):
            gr.Markdown(
                "Each saved sheet is an experiment configuration. Open, duplicate and compare it "
                "like a workbook. Save edits before starting an exploration."
            )
            with gr.Row():
                sheet_selector = gr.Dropdown(
                    choices=sheet_choices(),
                    value=None,
                    allow_custom_value=True,
                    label="Workbook sheet",
                )
                sheet_name = gr.Textbox(value="Quick baseline", label="Sheet name")
            with gr.Row():
                open_sheet = gr.Button("Open sheet")
                save_sheet = gr.Button("Save sheet", variant="primary")
                new_sheet = gr.Button("New sheet")
                duplicate_sheet = gr.Button("Duplicate sheet")
                create_fair_pair = gr.Button("Create baseline + exotic pair")
                import_latest = gr.Button("Import latest completed run")
                reload_sheets = gr.Button("Reload workbook")
            sheet_status = gr.Textbox(label="Workbook status", interactive=False)
            with gr.Row():
                train_model = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    allow_custom_value=True,
                    label="Registered model",
                )
                train_datasets = gr.Dropdown(
                    choices=dataset_choices,
                    value=default_dataset,
                    multiselect=True,
                    allow_custom_value=True,
                    label="Registered training datasets",
                )
            with gr.Row():
                validation_mode = gr.Dropdown(
                    ["group_holdout", "external"],
                    value="group_holdout",
                    label="Validation policy",
                )
                validation_datasets = gr.Dropdown(
                    choices=dataset_choices,
                    value=[],
                    multiselect=True,
                    allow_custom_value=True,
                    label="External Validation/Dev datasets (never optimized on)",
                )
                eval_ratio = gr.Slider(
                    0.01,
                    0.3,
                    value=0.05,
                    step=0.01,
                    label="Group holdout ratio",
                )
                max_validation_samples = gr.Number(
                    value=512, precision=0, label="Max validation samples"
                )
            with gr.Row():
                benchmark_profile = gr.Dropdown(
                    ["general", "pi-agent", "mixed"],
                    value="general",
                    label="Benchmark profile",
                )
                comparison_group = gr.Textbox(
                    value="baseline-vs-exotic-v1",
                    label="Comparison group (same for fair pairs)",
                )
                experiment_variant = gr.Dropdown(
                    ["baseline", "exotic", "custom"],
                    value="custom",
                    label="Experiment variant",
                )
            with gr.Row():
                allowed_tools = gr.Textbox(
                    value=", ".join(sorted(KNOWN_TOOLS)),
                    label="Declared tool names (comma-separated)",
                )
                filter_unknown_tools = gr.Checkbox(
                    value=True,
                    label="Reject training/validation samples using undeclared tools",
                )
            refresh_choices = gr.Button("Reload registered models and datasets")

            def reload_train_choices() -> tuple[Any, Any, Any]:
                refreshed_models = _model_choices()
                refreshed_datasets = _dataset_choices()
                return (
                    gr.Dropdown(
                        choices=refreshed_models,
                        value=refreshed_models[0][1] if refreshed_models else None,
                    ),
                    gr.Dropdown(
                        choices=refreshed_datasets,
                        value=[refreshed_datasets[0][1]] if refreshed_datasets else [],
                    ),
                    gr.Dropdown(choices=refreshed_datasets, value=[]),
                )

            refresh_choices.click(
                reload_train_choices,
                outputs=[train_model, train_datasets, validation_datasets],
            )
            with gr.Row():
                minutes = gr.Slider(15, 180, value=180, step=5, label="Total time limit (minutes)")
                sequence = gr.Dropdown([512, 1024, 2048, 4096], value=2048, label="Sequence")
                rank = gr.Dropdown([4, 8, 16, 32, 64], value=16, label="LoRA rank")
                source_rows = gr.Number(value=25000, precision=0, label="Max rows per source")
                profile = gr.Dropdown(
                    ["auto", "attention", "attention-mlp", "all-linear"],
                    value="auto",
                    label="Target profile",
                )
            with gr.Row():
                adapter_method = gr.Dropdown(
                    ["lora", "rslora"], value="rslora", label="Adapter method"
                )
                learning_rate = gr.Number(value=0.0001, label="Learning rate")
                budget_mode = gr.Dropdown(
                    ["time", "steps"],
                    value="time",
                    label="Fairness budget (steps recommended for papers)",
                )
                max_steps = gr.Number(value=1600, precision=0, label="Max optimizer steps")
            with gr.Accordion("Experimental controls — π routing/realization", open=True):
                gr.Markdown(
                    "The decimal pairs of **π remain the primary noise modulation**. "
                    "Envelope and clean-tail controls only phase that π signal; choose constant "
                    "to reproduce the original behavior."
                )
                with gr.Row():
                    technique_preset = gr.Dropdown(
                        [
                            "Original π controls — alpha 5 / all tokens",
                            "π routing-only — prompt + clean tail",
                            "Constant NEFTune control — no digit modulation",
                            "Shuffled π control — matched decimal pairs",
                            "Relational geometry only",
                            "Route-then-Resolve — π + role CE + relational geometry",
                        ],
                        value="Route-then-Resolve — π + role CE + relational geometry",
                        label="Technique preset",
                    )
                    apply_technique_preset = gr.Button("Apply technique preset")
                with gr.Row():
                    noise_on = gr.Checkbox(label="π/digit embedding noise")
                    noise_amplitude_mode = gr.Dropdown(
                        ["digit_pairs", "constant"],
                        value="digit_pairs",
                        label="Noise amplitude law",
                    )
                    noise_source = gr.Dropdown(
                        ["pi", "e", "sqrt2", "phi"], value="pi", label="Digit source"
                    )
                    noise_digit_order = gr.Dropdown(
                        ["natural", "shuffled"], value="natural", label="Digit-pair order"
                    )
                    noise_alpha = gr.Slider(0, 20, value=2, step=0.25, label="Base noise alpha")
                    noise_modulation = gr.Slider(
                        0, 1, value=0.10, step=0.01, label="Digit-pair modulation"
                    )
                with gr.Row():
                    noise_scope = gr.Dropdown(
                        ["all", "prompt"], value="prompt", label="Noise token scope"
                    )
                    noise_envelope = gr.Dropdown(
                        ["constant", "linear", "cosine"],
                        value="constant",
                        label="Envelope over π amplitudes",
                    )
                    noise_clean_tail = gr.Slider(
                        0,
                        0.75,
                        value=0.30,
                        step=0.05,
                        label="Final clean fraction (noise off)",
                    )
                with gr.Row():
                    geometry_on = gr.Checkbox(label="Geometric representation loss")
                    geometry_mode = gr.Dropdown(
                        ["orthogonal", "margin", "relational"],
                        value="relational",
                        label="Geometry mode",
                    )
                    geometry_scope = gr.Dropdown(
                        ["all_completion", "structured"],
                        value="structured",
                        label="Geometry token scope",
                    )
                    geometry_weight = gr.Slider(
                        0.001, 0.2, value=0.005, step=0.001, label="Geometry loss weight"
                    )
                with gr.Row():
                    geometry_layer = gr.Number(value=-1, precision=0, label="Hidden layer")
                    geometry_margin = gr.Slider(
                        0.01, 0.8, value=0.2, step=0.01, label="Cosine margin"
                    )
                    geometry_sample_tokens = gr.Slider(
                        4, 256, value=32, step=4, label="Geometry tokens per batch"
                    )
                role_loss_on = gr.Checkbox(
                    label="Role-weighted CE (protect tool names, keys and exact values)"
                )
                with gr.Row():
                    delimiter_weight = gr.Number(value=0.5, label="Delimiter CE weight")
                    tool_name_weight = gr.Number(value=1.5, label="Tool-name CE weight")
                    argument_key_weight = gr.Number(value=2.0, label="Argument-key CE weight")
                    argument_value_weight = gr.Number(value=2.5, label="Argument-value CE weight")
                technique_outputs = [
                    noise_on,
                    noise_amplitude_mode,
                    noise_source,
                    noise_digit_order,
                    noise_alpha,
                    noise_modulation,
                    noise_scope,
                    noise_envelope,
                    noise_clean_tail,
                    geometry_on,
                    geometry_mode,
                    geometry_scope,
                    geometry_weight,
                    geometry_layer,
                    geometry_margin,
                    geometry_sample_tokens,
                    role_loss_on,
                    delimiter_weight,
                    tool_name_weight,
                    argument_key_weight,
                    argument_value_weight,
                ]
                apply_technique_preset.click(
                    _technique_preset,
                    inputs=[technique_preset],
                    outputs=technique_outputs,
                )
            with gr.Accordion("Exploration scope", open=False):
                gr.Markdown(
                    "Random samples ranges continuously; Grid builds combinations from endpoints "
                    "and choices. The total exploration limit caps all trials together."
                )
                with gr.Row():
                    exploration_strategy = gr.Dropdown(
                        ["random", "grid"], value="random", label="Search strategy"
                    )
                    exploration_trials = gr.Slider(1, 12, value=3, step=1, label="Number of trials")
                    exploration_total_minutes = gr.Slider(
                        15, 360, value=180, step=5, label="Total exploration limit (minutes)"
                    )
                    exploration_seed = gr.Number(value=42, precision=0, label="Exploration seed")
                with gr.Row():
                    explore_adapter_methods = gr.Dropdown(
                        ["lora", "rslora"],
                        value=["lora", "rslora"],
                        multiselect=True,
                        label="Adapter methods to explore",
                    )
                    explore_ranks = gr.Dropdown(
                        [4, 8, 16, 32, 64],
                        value=[8, 16],
                        multiselect=True,
                        label="LoRA ranks",
                    )
                    explore_profiles = gr.Dropdown(
                        ["auto", "attention", "attention-mlp", "all-linear"],
                        value=["attention", "auto"],
                        multiselect=True,
                        label="Target profiles",
                    )
                    explore_sequences = gr.Dropdown(
                        [512, 1024, 2048, 4096],
                        value=[1024, 2048],
                        multiselect=True,
                        label="Sequence lengths",
                    )
                with gr.Row():
                    explore_lr_min = gr.Number(value=0.00005, label="Learning rate min")
                    explore_lr_max = gr.Number(value=0.0002, label="Learning rate max")
                    explore_noise_modes = gr.Dropdown(
                        ["off", "pi", "e", "sqrt2", "phi"],
                        value=["off", "pi"],
                        multiselect=True,
                        label="Noise choices",
                    )
                    explore_noise_alpha_min = gr.Number(value=1.0, label="Noise alpha min")
                    explore_noise_alpha_max = gr.Number(value=8.0, label="Noise alpha max")
                with gr.Row():
                    explore_noise_digit_orders = gr.Dropdown(
                        ["natural", "shuffled"],
                        value=["natural"],
                        multiselect=True,
                        label="Digit-pair orders",
                    )
                    explore_noise_scopes = gr.Dropdown(
                        ["all", "prompt"],
                        value=["prompt"],
                        multiselect=True,
                        label="Noise scopes",
                    )
                    explore_noise_envelopes = gr.Dropdown(
                        ["constant", "linear", "cosine"],
                        value=["constant"],
                        multiselect=True,
                        label="π amplitude envelopes",
                    )
                    explore_noise_modulation_min = gr.Number(
                        value=0.10, label="Digit modulation min"
                    )
                    explore_noise_modulation_max = gr.Number(
                        value=0.35, label="Digit modulation max"
                    )
                with gr.Row():
                    explore_noise_clean_tail_min = gr.Number(value=0.0, label="Clean tail min")
                    explore_noise_clean_tail_max = gr.Number(value=0.3, label="Clean tail max")
                    explore_geometry_modes = gr.Dropdown(
                        ["off", "orthogonal", "margin", "relational"],
                        value=["off", "relational"],
                        multiselect=True,
                        label="Geometry modes",
                    )
                    explore_geometry_scopes = gr.Dropdown(
                        ["all_completion", "structured"],
                        value=["structured"],
                        multiselect=True,
                        label="Geometry scopes",
                    )
                with gr.Row():
                    explore_geometry_layers = gr.Dropdown(
                        [-1, -2, -4, -8],
                        value=[-1, -4],
                        multiselect=True,
                        label="Geometry layers",
                    )
                    explore_geometry_weight_min = gr.Number(
                        value=0.005, label="Geometry weight min"
                    )
                    explore_geometry_weight_max = gr.Number(value=0.05, label="Geometry weight max")
                    explore_geometry_margin_min = gr.Number(value=0.1, label="Geometry margin min")
                    explore_geometry_margin_max = gr.Number(value=0.3, label="Geometry margin max")
                explore_role_loss_modes = gr.Dropdown(
                    ["off", "on"],
                    value=["off", "on"],
                    multiselect=True,
                    label="Role-weighted CE choices",
                )
            with gr.Accordion(
                "OVERNIGHT DEV FAIR-PAIR: baseline → exotic → full DEV → compare",
                open=True,
            ):
                gr.Markdown(
                    "Creates and saves a fresh fair pair from the fields above, trains baseline "
                    "then exotic sequentially, and runs generation evaluation on the shared "
                    "**External Validation/Dev dataset**. It never opens or evaluates a sealed "
                    "final suite. The detached job keeps running if the browser disconnects."
                )
                with gr.Row():
                    dev_pipeline_samples = gr.Number(
                        value=520,
                        precision=0,
                        label="DEV GENERATION EVAL SIZE (prompts per adapter)",
                    )
                    dev_pipeline_compare_base = gr.Checkbox(
                        value=True,
                        label="Also evaluate untouched instruct base once",
                    )
                with gr.Row():
                    start_dev_pair_pipeline = gr.Button(
                        "SAVE PAIR + RUN OVERNIGHT DEV PIPELINE",
                        variant="primary",
                    )
                    refresh_dev_pair_pipeline = gr.Button("Refresh overnight DEV status")
                resume_dev_pair_pipeline = gr.Button(
                    "RESUME DEV EVALUATION ONLY — DO NOT RETRAIN",
                    variant="secondary",
                )
                dev_pipeline_status = gr.JSON(
                    value=_compact_pipeline_result(latest_dev_pair_pipeline_result()),
                    label="Latest overnight DEV pipeline",
                )
                dev_pipeline_auto_compare = gr.HTML(
                    value="<p>No overnight DEV pipeline has been launched yet.</p>",
                    label="Automatic DEV comparison",
                )
                dev_pipeline_timer = gr.Timer(value=5.0, active=False)

            with gr.Accordion(
                "RECOMMENDED: balanced routing-data repair + A/B/C/D screen",
                open=True,
            ):
                gr.Markdown(
                    "Builds and registers a deterministic **5,200-row training supplement**: "
                    "2,600 tool calls, 2,600 direct answers, 200 examples for each of 13 tools, "
                    "and the full inference tool menu in every prompt. It verifies zero exact "
                    "overlap with the selected DEV set, saves four workbook cells, then runs "
                    "**A baseline → B natural π noise → C role-weighted CE → D π+CE**. Geometry "
                    "is forced off. Completed cells are reusable if the terminal is interrupted."
                )
                gr.Markdown(
                    "**The manual checkboxes above are intentionally not a preview of the four "
                    "cells.** This pipeline always writes the following fixed contract before "
                    "training:\n\n"
                    "| Cell | π noise | Role-weighted CE | Geometry | Tool menu |\n"
                    "|---|---:|---:|---:|---:|\n"
                    "| A — baseline | off | off | off | 13 tools |\n"
                    "| B — π noise | **on** | off | off | 13 tools |\n"
                    "| C — role CE | off | **on** | off | 13 tools |\n"
                    "| D — π + CE | **on** | **on** | off | 13 tools |\n\n"
                    "Noise in B/D is fixed to natural π digit pairs, alpha 2, modulation 0.1, "
                    "prompt-only, constant envelope and final clean 30%. The four cells use "
                    "exactly the same model, data, DEV menu, seed, rank, LR and 600-step budget."
                )
                gr.Textbox(
                    value=", ".join(sorted(KNOWN_TOOLS)),
                    label="LOCKED 13-TOOL CONTRACT USED BY ALL FOUR CELLS",
                    interactive=False,
                )
                with gr.Row():
                    routing_screen_steps = gr.Number(
                        value=600,
                        precision=0,
                        label="FIXED OPTIMIZER STEPS PER CELL",
                    )
                    routing_screen_samples = gr.Number(
                        value=520,
                        precision=0,
                        label="DEV PROMPTS PER CELL",
                    )
                    routing_screen_compare_base = gr.Checkbox(
                        value=True,
                        label="Evaluate untouched instruct base once",
                    )
                with gr.Row():
                    start_routing_screen = gr.Button(
                        "PREPARE DATA + SAVE 4 CELLS + RUN SCREEN",
                        variant="primary",
                    )
                    refresh_routing_screen = gr.Button("Refresh A/B/C/D screen status")
                resume_routing_screen = gr.Button(
                    "RESUME FROM LAST COMPLETED CELL — DO NOT REPEAT FINISHED RUNS",
                    variant="secondary",
                )
                routing_screen_status = gr.JSON(
                    value=_compact_pipeline_result(latest_routing_screening_result()),
                    label="Latest routing A/B/C/D screening pipeline",
                )
                routing_screen_auto_compare = gr.HTML(
                    value="<p>No routing screening pipeline has been launched yet.</p>",
                    label="Automatic A/B/C/D DEV comparison",
                )
                routing_screen_timer = gr.Timer(value=5.0, active=False)

            with gr.Accordion(
                "NEXT: geometry E/F screen — reuse A/B → full DEV → compare",
                open=True,
            ):
                gr.Markdown(
                    "Reuses **A baseline** and **B natural π-noise** from the latest completed "
                    "A/B/C/D screen. It trains only **E relational geometry** and "
                    "**F natural π-noise + relational geometry**, then compares A/B/E/F on "
                    "the identical 520-prompt external DEV. **No sealed suite is opened.**"
                )
                gr.Markdown(
                    "| Cell | π noise | Relational geometry | Role CE |\n"
                    "|---|---:|---:|---:|\n"
                    "| A — reused baseline | off | off | off |\n"
                    "| B — reused π noise | **on** | off | off |\n"
                    "| E — geometry only | off | **on** | off |\n"
                    "| F — π + geometry | **on** | **on** | off |\n\n"
                    "Geometry is restricted to structured completion tokens at the last "
                    "hidden layer. F preserves B's natural π pairs, alpha 2, modulation 0.1, "
                    "prompt-only noise and final clean 30%."
                )
                with gr.Row():
                    geometry_screen_weight = gr.Number(
                        value=0.005,
                        minimum=0.0001,
                        maximum=0.2,
                        label="RELATIONAL GEOMETRY WEIGHT",
                    )
                    geometry_screen_tokens = gr.Number(
                        value=32,
                        minimum=4,
                        maximum=256,
                        precision=0,
                        label="GEOMETRY TOKENS PER BATCH",
                    )
                with gr.Row():
                    start_geometry_screen = gr.Button(
                        "PREPARE E/F + RUN GEOMETRY DEV SCREEN",
                        variant="primary",
                    )
                    refresh_geometry_screen = gr.Button("Refresh A/B/E/F screen status")
                resume_geometry_screen = gr.Button(
                    "RESUME E/F FROM LAST COMPLETED CELL",
                    variant="secondary",
                )
                geometry_screen_status = gr.JSON(
                    value=_compact_pipeline_result(latest_geometry_screening_result()),
                    label="Latest geometry A/B/E/F screening pipeline",
                )
                geometry_screen_auto_compare = gr.HTML(
                    value="<p>No geometry screening pipeline has been launched yet.</p>",
                    label="Automatic A/B/E/F DEV comparison",
                )
                geometry_screen_timer = gr.Timer(value=5.0, active=False)

            with gr.Accordion(
                "FINAL EXPERIMENT: A/B/F/G π-LiteralLock → DEV → sealed → compare",
                open=True,
            ):
                gr.Markdown(
                    "Builds a new grounded corpus where every expected argument value is visible "
                    "inside the prompt, freezes four fresh cells, trains them sequentially, runs "
                    "one common 520-prompt DEV and only then the untouched 520-prompt final sealed "
                    "suite. The detached job is resumable and never repeats completed cells.\n\n"
                    "| Cell | Natural π noise | Geometry | Exact-value protection |\n"
                    "|---|---:|---:|---:|\n"
                    "| A baseline | off | off | off |\n"
                    "| B π-noise | on | off | off |\n"
                    "| F π+geometry | on | structured relational | off |\n"
                    "| G π-LiteralLock | on | tool-name/key anchors | literal mask + value KL + value CE |\n\n"
                    "All cells use identical data, seed, rank, LR, sequence length, step count and "
                    "the locked 13-tool menu. Strict whole-JSON exactness is reported separately "
                    "from required-value and per-field accuracy."
                )
                gr.Textbox(
                    value=", ".join(sorted(KNOWN_TOOLS)),
                    label="LOCKED 13-TOOL CONTRACT",
                    interactive=False,
                )
                with gr.Row():
                    literal_lock_steps = gr.Number(
                        value=600,
                        precision=0,
                        label="FIXED OPTIMIZER STEPS PER CELL",
                    )
                    literal_lock_dev_samples = gr.Number(
                        value=520,
                        precision=0,
                        label="DEV PROMPTS PER ADAPTER",
                    )
                    literal_lock_sealed_samples = gr.Number(
                        value=520,
                        precision=0,
                        label="FINAL SEALED PROMPTS PER ADAPTER",
                    )
                    literal_lock_compare_base = gr.Checkbox(
                        value=True,
                        label="Evaluate untouched instruct base once per protocol",
                    )
                with gr.Row():
                    start_literal_lock = gr.Button(
                        "PREPARE V2 DATA + SAVE A/B/F/G + RUN COMPLETE OVERNIGHT",
                        variant="primary",
                    )
                    refresh_literal_lock = gr.Button("Refresh LiteralLock pipeline status")
                resume_literal_lock = gr.Button(
                    "RESUME FROM LAST COMPLETED STAGE — DO NOT RETRAIN FINISHED CELLS",
                    variant="secondary",
                )
                literal_lock_status = gr.JSON(
                    value=_compact_pipeline_result(latest_literal_lock_result()),
                    label="Latest π-LiteralLock pipeline",
                )
                literal_lock_compare = gr.HTML(
                    value="<p>No π-LiteralLock pipeline has been launched yet.</p>",
                    label="Automatic DEV/final comparison",
                )
                literal_lock_timer = gr.Timer(value=5.0, active=False)

            with gr.Accordion(
                "FOCUSED HUMAN + AGENTIC: π/√2 100% + anchor geometry + π-LiteralLock",
                open=True,
            ):
                gr.Markdown(
                    "Reuses the frozen human-agentic **A baseline** and **B natural-π clean-tail** "
                    "controls. It trains only four new 400-step cells, then evaluates all six on "
                    "an already-consumed OOD suite (DEV) and a fresh untouched final suite. Both "
                    "natural human-task success and strict/elastic structured tool calling are "
                    "reported. State files stay compact and the process is detached/resumable.\n\n"
                    "| Cell | Decimal-pair noise | Targeted geometry | LiteralLock |\n"
                    "|---|---|---|---|\n"
                    "| A | clean baseline (reused) | off | off |\n"
                    "| B | π, final clean 30% (reused) | off | off |\n"
                    "| C | π, **100% of steps** | off | off |\n"
                    "| D | √2, **100% of steps** | off | off |\n"
                    "| E | π, 100% | names + argument keys only | off |\n"
                    "| F | π, 100% | names + keys only | literal mask + value CE/KL |"
                )
                gr.Textbox(
                    value=", ".join(sorted(KNOWN_TOOLS)),
                    label="LOCKED HUMAN-AGENTIC 13-TOOL MENU",
                    interactive=False,
                )
                focused_samples = gr.Number(
                    value=520,
                    precision=0,
                    label="DEV AND FINAL PROMPTS PER ADAPTER",
                )
                with gr.Row():
                    start_focused = gr.Button(
                        "RUN DETACHED FOCUSED PIPELINE",
                        variant="primary",
                    )
                    refresh_focused = gr.Button("Refresh focused status")
                resume_focused = gr.Button(
                    "RESUME WITHOUT REPEATING COMPLETED CELLS/EVALUATIONS",
                    variant="secondary",
                )
                focused_status = gr.JSON(
                    value=_compact_pipeline_result(latest_focused_result()),
                    label="Latest focused pipeline (compact state)",
                )
                focused_compare = gr.HTML(
                    value="<p>No focused human/agentic pipeline has been launched yet.</p>",
                    label="Focused DEV/final comparison",
                )
                focused_timer = gr.Timer(value=10.0, active=False)

            pipeline_choices = _sealed_choices()
            with gr.Accordion(
                "AUTOMATED FAIR-PAIR PIPELINE: baseline → exotic → sealed → compare",
                open=True,
            ):
                gr.Markdown(
                    "Saves both sheets automatically, trains baseline then exotic (never in "
                    "parallel on one GPU), evaluates both on the same sealed suite, and stores "
                    "the two run IDs for one-click final comparison. **Sealed test size is the "
                    "number of prompts evaluated independently for each adapter.**"
                )
                with gr.Row():
                    pipeline_suite = gr.Dropdown(
                        choices=pipeline_choices,
                        value=pipeline_choices[0][1] if pipeline_choices else None,
                        label="Sealed suite for automatic final test",
                    )
                    pipeline_samples = gr.Slider(
                        1,
                        1000,
                        value=500,
                        step=1,
                        label="SEALED TEST SIZE (prompts per adapter)",
                    )
                    pipeline_compare_base = gr.Checkbox(
                        value=True,
                        label="Also evaluate untouched instruct base",
                    )
                with gr.Row():
                    start_pair_pipeline = gr.Button(
                        "Save pair + run training + sealed + prepare compare",
                        variant="primary",
                    )
                    refresh_pair_pipeline = gr.Button("Refresh latest macro status")
                pipeline_status = gr.JSON(
                    value=_compact_pipeline_result(latest_pair_pipeline_result()),
                    label="Latest automated pipeline",
                )
                pipeline_auto_compare = gr.HTML(
                    value="<p>The final baseline/exotic comparison will appear here automatically.</p>",
                    label="Automatic final comparison",
                )
                pipeline_timer = gr.Timer(value=5.0, active=False)
            ablation_run_choices = _completed_run_choices()
            ablation_sheet_choices = sheet_choices()
            baseline_default = next(
                (value for label, value in ablation_run_choices if "20260810-104854-3900" in label),
                ablation_run_choices[0][1] if ablation_run_choices else None,
            )
            noise_default = next(
                (value for label, value in ablation_run_choices if "20260810-114532-3900" in label),
                ablation_run_choices[0][1] if ablation_run_choices else None,
            )
            geometry_default = next(
                (
                    value
                    for label, value in ablation_sheet_choices
                    if "geometry-only-g002-70m-v3" in label
                ),
                None,
            )
            combined_default = next(
                (
                    value
                    for label, value in ablation_sheet_choices
                    if "noise-plus-geometry-pi-a5-g002-70m-v4" in label
                ),
                None,
            )
            with gr.Accordion(
                "OVERNIGHT 2×2 ABLATION: reuse baseline + noise; train geometry + combined; sealed 500",
                open=True,
            ):
                gr.Markdown(
                    "Reuses the completed baseline and noise-only adapters, trains only the two "
                    "missing cells sequentially, then evaluates **all four adapters on one shared "
                    "sealed protocol**. Keep the GUI terminal open overnight."
                )
                with gr.Row():
                    ablation_baseline_run = gr.Dropdown(
                        choices=ablation_run_choices,
                        value=baseline_default,
                        label="Existing BASELINE run (no noise, no geometry)",
                    )
                    ablation_noise_run = gr.Dropdown(
                        choices=ablation_run_choices,
                        value=noise_default,
                        label="Existing NOISE-ONLY run (pi alpha 5)",
                    )
                with gr.Row():
                    ablation_geometry_sheet = gr.Dropdown(
                        choices=ablation_sheet_choices,
                        value=geometry_default,
                        label="Prepared GEOMETRY-ONLY sheet (g=0.02)",
                    )
                    ablation_combined_sheet = gr.Dropdown(
                        choices=ablation_sheet_choices,
                        value=combined_default,
                        label="Prepared NOISE+GEOMETRY sheet (pi a=5, g=0.02)",
                    )
                with gr.Row():
                    ablation_suite = gr.Dropdown(
                        choices=pipeline_choices,
                        value=pipeline_choices[0][1] if pipeline_choices else None,
                        label="Shared sealed suite",
                    )
                    ablation_samples = gr.Slider(
                        1,
                        1000,
                        value=500,
                        step=1,
                        label="SEALED TEST SIZE (prompts per adapter)",
                    )
                    ablation_compare_base = gr.Checkbox(
                        value=True,
                        label="Also evaluate untouched instruct base once",
                    )
                with gr.Row():
                    start_ablation_pipeline = gr.Button(
                        "START OVERNIGHT 2×2 BATCH", variant="primary"
                    )
                    refresh_ablation_pipeline = gr.Button("Refresh 2×2 batch status")
                ablation_status = gr.JSON(
                    value=_compact_pipeline_result(latest_ablation_pipeline_result()),
                    label="Latest 2×2 ablation pipeline",
                )
                ablation_auto_compare = gr.HTML(
                    value="<p>The four-way sealed comparison will appear here when complete.</p>"
                )
                ablation_timer = gr.Timer(value=5.0, active=False)
            recipe_inputs = [
                sheet_selector,
                train_model,
                train_datasets,
                minutes,
                sequence,
                rank,
                source_rows,
                profile,
                adapter_method,
                learning_rate,
                budget_mode,
                max_steps,
                noise_on,
                noise_amplitude_mode,
                noise_source,
                noise_digit_order,
                noise_alpha,
                noise_modulation,
                noise_scope,
                noise_envelope,
                noise_clean_tail,
                geometry_on,
                geometry_mode,
                geometry_scope,
                geometry_weight,
                geometry_layer,
                geometry_margin,
                geometry_sample_tokens,
                role_loss_on,
                delimiter_weight,
                tool_name_weight,
                argument_key_weight,
                argument_value_weight,
                benchmark_profile,
                comparison_group,
                experiment_variant,
                validation_mode,
                validation_datasets,
                eval_ratio,
                max_validation_samples,
                allowed_tools,
                filter_unknown_tools,
            ]
            exploration_inputs = [
                exploration_strategy,
                exploration_trials,
                exploration_total_minutes,
                exploration_seed,
                explore_adapter_methods,
                explore_ranks,
                explore_profiles,
                explore_sequences,
                explore_lr_min,
                explore_lr_max,
                explore_noise_modes,
                explore_noise_alpha_min,
                explore_noise_alpha_max,
                explore_noise_digit_orders,
                explore_noise_scopes,
                explore_noise_envelopes,
                explore_noise_modulation_min,
                explore_noise_modulation_max,
                explore_noise_clean_tail_min,
                explore_noise_clean_tail_max,
                explore_geometry_modes,
                explore_geometry_scopes,
                explore_geometry_layers,
                explore_geometry_weight_min,
                explore_geometry_weight_max,
                explore_geometry_margin_min,
                explore_geometry_margin_max,
                explore_role_loss_modes,
            ]
            preflight_button = gr.Button("Preflight dataset + token + tool constraints")
            preflight_report = gr.JSON(label="Preflight report")
            warnings_acknowledged = gr.Checkbox(
                value=False,
                label="I reviewed non-blocking preflight warnings",
            )
            with gr.Row():
                start = gr.Button("Start this configuration", variant="primary")
                start_exploration = gr.Button("Save, then start exploration", variant="primary")
                stop_train = gr.Button("Stop training safely", variant="stop")
            launch_status = gr.Code(label="Launcher status", language="json")
            preflight_button.click(_preflight_run, recipe_inputs, preflight_report)
            start.click(
                _start_run,
                [*recipe_inputs, warnings_acknowledged],
                launch_status,
            )
            stop_train.click(_stop_run, outputs=launch_status)

            save_inputs = [sheet_selector, sheet_name, *recipe_inputs[1:], *exploration_inputs]

            def save_sheet_ui(*values: Any) -> tuple[str, Any]:
                try:
                    saved = _save_workbook_sheet(*values)
                    return (
                        f"Saved sheet: {saved['name']} ({saved['id']})",
                        gr.Dropdown(choices=sheet_choices(), value=saved["id"]),
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    return (
                        f"ERROR: {type(error).__name__}: {error}",
                        gr.Dropdown(choices=sheet_choices()),
                    )

            def save_and_start_exploration_ui(*values: Any) -> tuple[str, Any, str]:
                try:
                    *save_values, warnings_ack = values
                    saved = _save_workbook_sheet(*save_values)
                    launch = _start_exploration(saved["id"], bool(warnings_ack))
                    return (
                        launch,
                        gr.Dropdown(choices=sheet_choices(), value=saved["id"]),
                        f"Saved exploration sheet: {saved['name']} ({saved['id']})",
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    return (
                        f"ERROR: {type(error).__name__}: {error}",
                        gr.Dropdown(choices=sheet_choices()),
                        "Exploration was not started",
                    )

            def create_fair_pair_ui(*values: Any) -> tuple[Any, str, str]:
                baseline, exotic = _create_fair_pair(list(values))
                return (
                    gr.Dropdown(choices=sheet_choices(), value=exotic.id),
                    exotic.name,
                    (
                        f"Created fair pair {baseline.id} / {exotic.id}. Both share comparison "
                        f"group '{exotic.recipe.comparison_group}' and differ only in exotic controls."
                    ),
                )

            def start_dev_pair_pipeline_ui(*values: Any) -> tuple[Any, str, str, Any]:
                try:
                    *save_values, samples, compare_base, warnings_ack = values
                    baseline, exotic = _create_fair_pair(list(save_values))
                    if baseline.recipe.validation_mode != "external":
                        raise ValueError(
                            "Set Validation policy to external and select one DEV dataset"
                        )
                    if len(baseline.recipe.validation_datasets) != 1:
                        raise ValueError("Select exactly one External Validation/Dev dataset")
                    if baseline.recipe.budget_mode != "steps":
                        raise ValueError("Set Fairness budget to steps for an overnight fair pair")
                    sealed_hashes, sealed_sources = sealed_guard()
                    reports = {
                        "baseline": preflight_recipe(
                            baseline.recipe, sealed_hashes, sealed_sources
                        ),
                        "exotic": preflight_recipe(exotic.recipe, sealed_hashes, sealed_sources),
                    }
                    blocked = {
                        name: report
                        for name, report in reports.items()
                        if report.get("status") == "blocked"
                    }
                    if blocked:
                        raise ValueError(f"DEV pipeline blocked by preflight: {list(blocked)}")
                    if any(report.get("warnings") for report in reports.values()) and not bool(
                        warnings_ack
                    ):
                        raise ValueError(
                            "Review preflight warnings and enable the acknowledgement checkbox"
                        )
                    sample_count = int(samples)
                    if sample_count < 1:
                        raise ValueError("DEV evaluation size must be positive")
                    job = start_dev_pair_pipeline_job(
                        baseline.id,
                        exotic.id,
                        sample_count,
                        bool(compare_base),
                    )
                    return (
                        gr.Dropdown(choices=sheet_choices(), value=exotic.id),
                        exotic.name,
                        (
                            f"Overnight DEV pair saved: {baseline.name} / {exotic.name}. "
                            f"PID {job['pid']} is training sequentially; DEV size={sample_count}."
                        ),
                        {"job": job, "preflight": reports},
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return (
                        gr.Dropdown(choices=sheet_choices()),
                        str(
                            save_values[1]
                            if "save_values" in locals() and len(save_values) > 1
                            else ""
                        ),
                        f"ERROR: {message}",
                        {"status": "error", "error": message},
                    )

            def monitor_dev_pair_pipeline() -> tuple[Any, str]:
                result = latest_dev_pair_pipeline_result()
                if not result:
                    return None, "<p>No overnight DEV pipeline has been launched yet.</p>"
                if result.get("status") == "complete":
                    _rows, chart = _compare_runs(result.get("compare_run_ids", []))
                    return _compact_pipeline_result(result), chart
                stage = html.escape(str(result.get("stage") or result.get("status") or "unknown"))
                if result.get("status") == "failed":
                    error = html.escape(str(result.get("error") or "unknown error"))
                    return _compact_pipeline_result(result), (
                        f"<p style='color:#fb7185'>DEV pipeline failed: {error}</p>"
                    )
                return _compact_pipeline_result(result), (
                    f"<p>Overnight DEV pipeline is running: <strong>{stage}</strong>.</p>"
                )

            def resume_dev_pair_pipeline_ui() -> tuple[Any, str]:
                try:
                    result = latest_dev_pair_pipeline_result()
                    if not result:
                        raise ValueError("No DEV pair pipeline is available to resume")
                    if len(result.get("run_ids") or []) != 2:
                        raise ValueError("Both completed training runs are required before resume")
                    job = resume_dev_pair_pipeline_job(str(result["pipeline_id"]))
                    return {"job": job, "pipeline": _compact_pipeline_result(result)}, (
                        f"Resumed DEV evaluation only for pipeline {result['pipeline_id']}; "
                        f"PID {job['pid']}. Existing adapters will not be retrained."
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return {"status": "error", "error": message}, f"ERROR: {message}"

            def start_routing_screening_ui(*values: Any) -> tuple[Any, str, Any]:
                try:
                    *workbook_values, steps, samples, compare_base, warnings_ack = values
                    recipe_value_count = len(inspect.signature(_build_recipe).parameters) - 1
                    recipe_values = workbook_values[2 : 2 + recipe_value_count]
                    template = _build_recipe(None, *recipe_values)
                    if template.validation_mode != "external":
                        raise ValueError(
                            "Select external validation and the 520-row routing DEV dataset"
                        )
                    if len(template.validation_datasets) != 1:
                        raise ValueError("Select exactly one External Validation/Dev dataset")
                    step_count = int(steps)
                    sample_count = int(samples)
                    if step_count < 50:
                        raise ValueError("Use at least 50 optimizer steps per screening cell")
                    if sample_count < 1:
                        raise ValueError("DEV evaluation size must be positive")
                    dataset_audit = ensure_routing_supplement(
                        dev_source=template.validation_datasets[0]
                    )
                    prepared = prepare_routing_screening_sheets(
                        template=template,
                        supplement_source=str(dataset_audit["source"]),
                        max_steps=step_count,
                    )
                    store = WorkbookStore()
                    common_report = _preflight_recipe(
                        store.get(prepared["sheet_ids"]["A-baseline"]).recipe
                    )
                    reports = {label: common_report for label in prepared["sheet_ids"]}
                    blocked = [
                        label
                        for label, report in reports.items()
                        if report.get("status") == "blocked"
                    ]
                    if blocked:
                        raise ValueError(f"routing screening blocked by preflight: {blocked}")
                    if any(report.get("warnings") for report in reports.values()) and not bool(
                        warnings_ack
                    ):
                        raise ValueError(
                            "Review the screening preflight warnings and enable acknowledgement"
                        )
                    job = start_routing_screening_job(
                        sheet_ids=prepared["sheet_ids"],
                        max_samples=sample_count,
                        compare_base=bool(compare_base),
                    )
                    selected_sheet = prepared["sheet_ids"]["D-pi-role-ce"]
                    return (
                        {
                            "job": job,
                            "dataset_audit": dataset_audit,
                            "screen": prepared,
                            "preflight": reports,
                        },
                        (
                            f"Routing screen launched as PID {job['pid']}: four × "
                            f"{step_count} fixed-step cells, followed by {sample_count} DEV "
                            "prompts per adapter. Geometry is disabled."
                        ),
                        gr.Dropdown(choices=sheet_choices(), value=selected_sheet),
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return (
                        {"status": "error", "error": message},
                        f"ERROR: {message}",
                        gr.Dropdown(choices=sheet_choices()),
                    )

            def monitor_routing_screening() -> tuple[Any, str]:
                result = latest_routing_screening_result()
                if not result:
                    return None, "<p>No routing screening pipeline has been launched yet.</p>"
                if result.get("status") == "complete":
                    _rows, chart = _compare_runs(result.get("compare_run_ids", []))
                    recommendation = result.get("screening_summary", {}).get("recommended_cell")
                    note = (
                        f"<p><strong>Automatic gates winner:</strong> "
                        f"{html.escape(str(recommendation))}</p>"
                        if recommendation
                        else (
                            "<p style='color:#fbbf24'><strong>No cell passed every routing, "
                            "selection and exact-argument gate.</strong></p>"
                        )
                    )
                    return _compact_pipeline_result(result), note + chart
                stage = html.escape(str(result.get("stage") or result.get("status") or "unknown"))
                completed = sum(
                    cell.get("status") == "complete"
                    for cell in (result.get("cells") or {}).values()
                )
                if result.get("status") == "failed":
                    error = html.escape(str(result.get("error") or "unknown error"))
                    return _compact_pipeline_result(result), (
                        f"<p style='color:#fb7185'>Screen failed after {completed}/4 cells: "
                        f"{error}</p>"
                    )
                return _compact_pipeline_result(result), (
                    f"<p>Routing screen: <strong>{stage}</strong>; completed cells "
                    f"<strong>{completed}/4</strong>.</p>"
                )

            def resume_routing_screening_ui() -> tuple[Any, str]:
                try:
                    result = latest_routing_screening_result()
                    if not result:
                        raise ValueError("No routing screening pipeline is available to resume")
                    job = resume_routing_screening_job(str(result["pipeline_id"]))
                    return {"job": job, "pipeline": _compact_pipeline_result(result)}, (
                        f"Resumed routing screen {result['pipeline_id']} as PID {job['pid']}. "
                        "Completed cells will be reused."
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return {"status": "error", "error": message}, f"ERROR: {message}"

            def start_geometry_screening_ui(
                weight: float,
                sample_tokens: int,
                warnings_ack: bool,
            ) -> tuple[Any, str, Any]:
                try:
                    geometry_weight = float(weight)
                    token_count = int(sample_tokens)
                    if not 0.0 < geometry_weight <= 0.2:
                        raise ValueError("Geometry weight must be in (0, 0.2]")
                    if not 4 <= token_count <= 256:
                        raise ValueError("Geometry tokens per batch must be between 4 and 256")
                    prepared = prepare_geometry_screening_sheets(
                        geometry_weight=geometry_weight,
                        sample_tokens=token_count,
                    )
                    store = WorkbookStore()
                    reports = {
                        label: _preflight_recipe(store.get(sheet_id).recipe)
                        for label, sheet_id in prepared["sheet_ids"].items()
                    }
                    blocked = [
                        label
                        for label, report in reports.items()
                        if report.get("status") == "blocked"
                    ]
                    if blocked:
                        raise ValueError(f"geometry screening blocked by preflight: {blocked}")
                    if any(report.get("warnings") for report in reports.values()) and not bool(
                        warnings_ack
                    ):
                        raise ValueError(
                            "Review the geometry preflight warnings and enable acknowledgement"
                        )
                    job = start_geometry_screening_job(
                        baseline_run_id=str(prepared["baseline_run_id"]),
                        noise_run_id=str(prepared["noise_run_id"]),
                        geometry_sheet_id=str(prepared["sheet_ids"]["E-geometry-only"]),
                        combined_sheet_id=str(prepared["sheet_ids"]["F-pi-geometry"]),
                        source_pipeline_id=str(prepared["source_pipeline_id"]),
                        max_samples=int(prepared["dev_samples_requested"]),
                    )
                    return (
                        {"job": job, "geometry_screen": prepared, "preflight": reports},
                        (
                            f"Geometry DEV screen launched as PID {job['pid']}: reusing A/B, "
                            f"training E/F at {prepared['max_steps']} steps, then evaluating "
                            f"{prepared['dev_samples_requested']} shared DEV prompts. Sealed "
                            "data remains unopened."
                        ),
                        gr.Dropdown(
                            choices=sheet_choices(),
                            value=prepared["sheet_ids"]["F-pi-geometry"],
                        ),
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return (
                        {"status": "error", "error": message},
                        f"ERROR: {message}",
                        gr.Dropdown(choices=sheet_choices()),
                    )

            def monitor_geometry_screening() -> tuple[Any, str]:
                result = latest_geometry_screening_result()
                if not result:
                    return None, "<p>No geometry screening pipeline has been launched yet.</p>"
                if result.get("status") == "complete":
                    _rows, chart = _compare_runs(result.get("compare_run_ids", []))
                    recommendation = result.get("screening_summary", {}).get("recommended_cell")
                    note = (
                        f"<p><strong>Automatic geometry-screen winner:</strong> "
                        f"{html.escape(str(recommendation))}</p>"
                        if recommendation
                        else (
                            "<p style='color:#fbbf24'><strong>No A/B/E/F cell passed every "
                            "routing, selection and exact-argument gate.</strong></p>"
                        )
                    )
                    return _compact_pipeline_result(result), note + chart
                stage = html.escape(str(result.get("stage") or result.get("status") or "unknown"))
                completed = sum(
                    cell.get("status") == "complete"
                    for cell in (result.get("cells") or {}).values()
                )
                if result.get("status") == "failed":
                    error = html.escape(str(result.get("error") or "unknown error"))
                    return _compact_pipeline_result(result), (
                        f"<p style='color:#fb7185'>Geometry screen failed after "
                        f"{completed}/4 cells: {error}</p>"
                    )
                return _compact_pipeline_result(result), (
                    f"<p>Geometry screen: <strong>{stage}</strong>; available cells "
                    f"<strong>{completed}/4</strong> (A/B are reused).</p>"
                )

            def resume_geometry_screening_ui() -> tuple[Any, str]:
                try:
                    result = latest_geometry_screening_result()
                    if not result:
                        raise ValueError("No geometry screening pipeline is available to resume")
                    job = resume_geometry_screening_job(str(result["pipeline_id"]))
                    return {"job": job, "pipeline": _compact_pipeline_result(result)}, (
                        f"Resumed geometry screen {result['pipeline_id']} as PID {job['pid']}. "
                        "Completed E/F cells will be reused."
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return {"status": "error", "error": message}, f"ERROR: {message}"

            def start_literal_lock_ui(
                steps: int,
                dev_samples: int,
                sealed_samples: int,
                compare_base: bool,
                *values: Any,
            ) -> tuple[Any, str, Any]:
                try:
                    recipe_value_count = len(inspect.signature(_build_recipe).parameters) - 1
                    recipe_values = list(values)[2 : 2 + recipe_value_count]
                    template = _build_recipe(None, *recipe_values)
                    prepared = prepare_literal_lock_sheets(template, max_steps=int(steps))
                    store = WorkbookStore()
                    reports = {
                        label: _preflight_recipe(store.get(sheet_id).recipe)
                        for label, sheet_id in prepared["sheet_ids"].items()
                    }
                    blocked = [
                        label
                        for label, report in reports.items()
                        if report.get("status") == "blocked"
                    ]
                    if blocked:
                        raise ValueError(f"LiteralLock pipeline blocked by preflight: {blocked}")
                    job = start_literal_lock_pipeline_job(
                        sheet_ids=prepared["sheet_ids"],
                        suite_id=str(prepared["suite_id"]),
                        dev_samples=int(dev_samples),
                        sealed_samples=int(sealed_samples),
                        compare_base=bool(compare_base),
                    )
                    return (
                        {
                            "job": job,
                            "pipeline": prepared,
                            "data_audit": prepared["data_audit"],
                            "preflight": reports,
                        },
                        (
                            f"π-LiteralLock pipeline PID {job['pid']} started: A/B/F/G × "
                            f"{int(steps)} steps, {int(dev_samples)} DEV prompts, then "
                            f"{int(sealed_samples)} untouched sealed prompts per adapter."
                        ),
                        gr.Dropdown(
                            choices=sheet_choices(),
                            value=prepared["sheet_ids"]["G-pi-literal-lock"],
                        ),
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return (
                        {"status": "error", "error": message},
                        f"ERROR: {message}",
                        gr.Dropdown(choices=sheet_choices()),
                    )

            def monitor_literal_lock() -> tuple[Any, str]:
                result = latest_literal_lock_result()
                if not result:
                    return None, "<p>No π-LiteralLock pipeline has been launched yet.</p>"
                if result.get("status") == "complete":
                    _rows, chart = _compare_runs(result.get("compare_run_ids", []))
                    winner = html.escape(
                        str((result.get("sealed_summary") or {}).get("recommended_cell") or "n/a")
                    )
                    return _compact_pipeline_result(result), (
                        f"<p><strong>Final required-value winner:</strong> {winner}</p>" + chart
                    )
                stage = html.escape(str(result.get("stage") or result.get("status") or "unknown"))
                completed = sum(
                    cell.get("status") == "complete"
                    for cell in (result.get("cells") or {}).values()
                    if isinstance(cell, dict)
                )
                if result.get("status") == "failed":
                    error = html.escape(str(result.get("error") or "unknown error"))
                    return _compact_pipeline_result(result), (
                        f"<p style='color:#fb7185'>LiteralLock failed after {completed}/4 "
                        f"training cells: {error}</p>"
                    )
                return _compact_pipeline_result(result), (
                    f"<p>π-LiteralLock: <strong>{stage}</strong>; completed training cells "
                    f"<strong>{completed}/4</strong>.</p>"
                )

            def resume_literal_lock_ui() -> tuple[Any, str]:
                try:
                    result = latest_literal_lock_result()
                    if not result:
                        raise ValueError("No π-LiteralLock pipeline is available to resume")
                    job = resume_literal_lock_pipeline_job(str(result["pipeline_id"]))
                    return {"job": job, "pipeline": _compact_pipeline_result(result)}, (
                        f"Resumed π-LiteralLock {result['pipeline_id']} as PID {job['pid']}; "
                        "completed cells and evaluations will be reused."
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return {"status": "error", "error": message}, f"ERROR: {message}"

            def start_focused_ui(samples: int) -> tuple[Any, str]:
                try:
                    sample_count = int(samples)
                    if not 1 <= sample_count <= 520:
                        raise ValueError(
                            "Focused DEV/final size must be between 1 and 520 prompts per adapter"
                        )
                    job = start_focused_human_pipeline_job(sample_count)
                    return {"job": job}, (
                        f"Focused detached pipeline PID {job['pid']} started. It reuses A/B, "
                        "trains C/D/E/F sequentially, then evaluates human, strict-agentic "
                        f"and elastic-agentic scores on {sample_count} DEV and final prompts."
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return {"status": "error", "error": message}, f"ERROR: {message}"

            def monitor_focused() -> tuple[Any, str]:
                result = latest_focused_result()
                if not result:
                    return None, "<p>No focused human/agentic pipeline has been launched yet.</p>"
                compact = _compact_pipeline_result(result)
                if result.get("status") == "complete":
                    _rows, chart = _compare_runs(result.get("compare_run_ids", []))
                    winner = html.escape(
                        str((result.get("final_summary") or {}).get("recommended_cell") or "n/a")
                    )
                    return compact, (
                        "<p><strong>Fresh final winner (elastic human-task ranking):</strong> "
                        f"{winner}. Strict and elastic metrics remain visible separately.</p>"
                        + chart
                    )
                stage = html.escape(str(result.get("stage") or result.get("status") or "unknown"))
                cells = result.get("cells") or {}
                completed = sum(
                    str(cell.get("status", "")).startswith("complete")
                    for cell in cells.values()
                    if isinstance(cell, dict)
                )
                if result.get("status") == "failed":
                    error = html.escape(str(result.get("error") or "unknown error"))
                    return compact, (
                        f"<p style='color:#fb7185'>Focused pipeline failed after "
                        f"{completed}/6 cells: {error}</p>"
                    )
                return compact, (
                    f"<p>Focused pipeline: <strong>{stage}</strong>; reusable/completed cells "
                    f"<strong>{completed}/6</strong>. Full predictions stay on disk.</p>"
                )

            def resume_focused_ui() -> tuple[Any, str]:
                try:
                    result = latest_focused_result()
                    if not result:
                        raise ValueError("No focused pipeline is available to resume")
                    job = resume_focused_human_pipeline_job(str(result["pipeline_id"]))
                    return {"job": job, "pipeline": _compact_pipeline_result(result)}, (
                        f"Resumed focused pipeline {result['pipeline_id']} as PID {job['pid']}; "
                        "completed training cells and schema-v8 evaluations will be reused."
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return {"status": "error", "error": message}, f"ERROR: {message}"

            def start_pair_pipeline_ui(*values: Any) -> tuple[Any, str, str, Any]:
                try:
                    *save_values, suite_id, samples, compare_base, warnings_ack = values
                    if not suite_id:
                        raise ValueError("Register and select a sealed suite first")
                    baseline, exotic = _create_fair_pair(list(save_values))
                    sealed_hashes, sealed_sources = sealed_guard()
                    reports = {
                        "baseline": preflight_recipe(
                            baseline.recipe, sealed_hashes, sealed_sources
                        ),
                        "exotic": preflight_recipe(exotic.recipe, sealed_hashes, sealed_sources),
                    }
                    blocked = {
                        name: report
                        for name, report in reports.items()
                        if report.get("status") == "blocked"
                    }
                    if blocked:
                        raise ValueError(f"Pipeline blocked by preflight: {list(blocked)}")
                    if any(report.get("warnings") for report in reports.values()) and not bool(
                        warnings_ack
                    ):
                        raise ValueError(
                            "Review preflight warnings and enable the acknowledgement checkbox"
                        )
                    job = start_pair_pipeline_job(
                        baseline.id,
                        exotic.id,
                        str(suite_id),
                        int(samples),
                        bool(compare_base),
                    )
                    return (
                        gr.Dropdown(choices=sheet_choices(), value=exotic.id),
                        exotic.name,
                        (
                            f"Saved automated pair {baseline.name} / {exotic.name}; "
                            f"sealed size={int(samples)}. Pipeline PID {job['pid']} started."
                        ),
                        {"job": job, "preflight": reports},
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    return (
                        gr.Dropdown(choices=sheet_choices()),
                        str(
                            save_values[1]
                            if "save_values" in locals() and len(save_values) > 1
                            else ""
                        ),
                        f"ERROR: {type(error).__name__}: {error}",
                        {"status": "error", "error": f"{type(error).__name__}: {error}"},
                    )

            def monitor_pair_pipeline() -> tuple[Any, str]:
                result = latest_pair_pipeline_result()
                if not result:
                    return None, "<p>No automated pipeline has been launched yet.</p>"
                if result.get("status") == "complete":
                    _rows, chart = _compare_runs(result.get("compare_run_ids", []))
                    return _compact_pipeline_result(result), chart
                stage = html.escape(str(result.get("stage") or result.get("status") or "unknown"))
                return _compact_pipeline_result(result), (
                    f"<p>Automated pipeline is running: <strong>{stage}</strong>.</p>"
                )

            def start_ablation_pipeline_ui(
                baseline_run_id: str,
                noise_run_id: str,
                geometry_sheet_id: str,
                combined_sheet_id: str,
                suite_id: str,
                samples: int,
                compare_base: bool,
            ) -> tuple[Any, str]:
                try:
                    if not all(
                        (
                            baseline_run_id,
                            noise_run_id,
                            geometry_sheet_id,
                            combined_sheet_id,
                            suite_id,
                        )
                    ):
                        raise ValueError(
                            "Select both existing runs, both prepared sheets and a sealed suite"
                        )
                    job = start_ablation_pipeline_job(
                        str(baseline_run_id),
                        str(noise_run_id),
                        str(geometry_sheet_id),
                        str(combined_sheet_id),
                        str(suite_id),
                        int(samples),
                        bool(compare_base),
                    )
                    return {"job": job}, (
                        f"Overnight 2×2 batch PID {job['pid']} started: two sequential trainings, "
                        f"then {int(samples)} sealed prompts for all four adapters."
                    )
                except Exception as error:  # noqa: BLE001 - UI boundary
                    message = f"{type(error).__name__}: {error}"
                    return {"status": "error", "error": message}, f"ERROR: {message}"

            def monitor_ablation_pipeline() -> tuple[Any, str]:
                result = latest_ablation_pipeline_result()
                if not result:
                    return None, "<p>No 2×2 ablation pipeline has been launched yet.</p>"
                if result.get("status") == "complete":
                    _rows, chart = _compare_runs(result.get("compare_run_ids", []))
                    return _compact_pipeline_result(result), chart
                stage = html.escape(str(result.get("stage") or result.get("status") or "unknown"))
                if result.get("status") == "failed":
                    error = html.escape(str(result.get("error") or "unknown error"))
                    return _compact_pipeline_result(result), (
                        f"<p style='color:#fb7185'>2×2 batch failed: {error}</p>"
                    )
                return _compact_pipeline_result(result), (
                    f"<p>2×2 ablation is running: <strong>{stage}</strong>.</p>"
                )

            save_sheet.click(
                save_sheet_ui,
                inputs=save_inputs,
                outputs=[sheet_status, sheet_selector],
            )
            start_exploration.click(
                save_and_start_exploration_ui,
                inputs=[*save_inputs, warnings_acknowledged],
                outputs=[launch_status, sheet_selector, sheet_status],
            )
            create_fair_pair.click(
                create_fair_pair_ui,
                inputs=save_inputs,
                outputs=[sheet_selector, sheet_name, sheet_status],
            )
            start_dev_pair_pipeline.click(
                start_dev_pair_pipeline_ui,
                inputs=[
                    *save_inputs,
                    dev_pipeline_samples,
                    dev_pipeline_compare_base,
                    warnings_acknowledged,
                ],
                outputs=[
                    sheet_selector,
                    sheet_name,
                    sheet_status,
                    dev_pipeline_status,
                ],
            )
            refresh_dev_pair_pipeline.click(
                monitor_dev_pair_pipeline,
                outputs=[dev_pipeline_status, dev_pipeline_auto_compare],
            )
            resume_dev_pair_pipeline.click(
                resume_dev_pair_pipeline_ui,
                outputs=[dev_pipeline_status, sheet_status],
            )
            dev_pipeline_timer.tick(
                monitor_dev_pair_pipeline,
                outputs=[dev_pipeline_status, dev_pipeline_auto_compare],
            )
            start_routing_screen.click(
                start_routing_screening_ui,
                inputs=[
                    *save_inputs,
                    routing_screen_steps,
                    routing_screen_samples,
                    routing_screen_compare_base,
                    warnings_acknowledged,
                ],
                outputs=[routing_screen_status, sheet_status, sheet_selector],
            )
            refresh_routing_screen.click(
                monitor_routing_screening,
                outputs=[routing_screen_status, routing_screen_auto_compare],
            )
            resume_routing_screen.click(
                resume_routing_screening_ui,
                outputs=[routing_screen_status, sheet_status],
            )
            routing_screen_timer.tick(
                monitor_routing_screening,
                outputs=[routing_screen_status, routing_screen_auto_compare],
            )
            start_geometry_screen.click(
                start_geometry_screening_ui,
                inputs=[
                    geometry_screen_weight,
                    geometry_screen_tokens,
                    warnings_acknowledged,
                ],
                outputs=[geometry_screen_status, sheet_status, sheet_selector],
            )
            refresh_geometry_screen.click(
                monitor_geometry_screening,
                outputs=[geometry_screen_status, geometry_screen_auto_compare],
            )
            resume_geometry_screen.click(
                resume_geometry_screening_ui,
                outputs=[geometry_screen_status, sheet_status],
            )
            geometry_screen_timer.tick(
                monitor_geometry_screening,
                outputs=[geometry_screen_status, geometry_screen_auto_compare],
            )
            start_literal_lock.click(
                start_literal_lock_ui,
                inputs=[
                    literal_lock_steps,
                    literal_lock_dev_samples,
                    literal_lock_sealed_samples,
                    literal_lock_compare_base,
                    *save_inputs,
                ],
                outputs=[literal_lock_status, sheet_status, sheet_selector],
            )
            refresh_literal_lock.click(
                monitor_literal_lock,
                outputs=[literal_lock_status, literal_lock_compare],
            )
            resume_literal_lock.click(
                resume_literal_lock_ui,
                outputs=[literal_lock_status, sheet_status],
            )
            literal_lock_timer.tick(
                monitor_literal_lock,
                outputs=[literal_lock_status, literal_lock_compare],
            )
            start_focused.click(
                start_focused_ui,
                inputs=[focused_samples],
                outputs=[focused_status, sheet_status],
            )
            refresh_focused.click(
                monitor_focused,
                outputs=[focused_status, focused_compare],
            )
            resume_focused.click(
                resume_focused_ui,
                outputs=[focused_status, sheet_status],
            )
            focused_timer.tick(
                monitor_focused,
                outputs=[focused_status, focused_compare],
            )
            start_pair_pipeline.click(
                start_pair_pipeline_ui,
                inputs=[
                    *save_inputs,
                    pipeline_suite,
                    pipeline_samples,
                    pipeline_compare_base,
                    warnings_acknowledged,
                ],
                outputs=[sheet_selector, sheet_name, sheet_status, pipeline_status],
            )
            refresh_pair_pipeline.click(
                monitor_pair_pipeline,
                outputs=[pipeline_status, pipeline_auto_compare],
            )
            pipeline_timer.tick(
                monitor_pair_pipeline,
                outputs=[pipeline_status, pipeline_auto_compare],
            )
            start_ablation_pipeline.click(
                start_ablation_pipeline_ui,
                inputs=[
                    ablation_baseline_run,
                    ablation_noise_run,
                    ablation_geometry_sheet,
                    ablation_combined_sheet,
                    ablation_suite,
                    ablation_samples,
                    ablation_compare_base,
                ],
                outputs=[ablation_status, sheet_status],
            )
            refresh_ablation_pipeline.click(
                monitor_ablation_pipeline,
                outputs=[ablation_status, ablation_auto_compare],
            )
            ablation_timer.tick(
                monitor_ablation_pipeline,
                outputs=[ablation_status, ablation_auto_compare],
            )

            sheet_load_outputs = [
                sheet_name,
                train_model,
                train_datasets,
                minutes,
                sequence,
                rank,
                source_rows,
                profile,
                adapter_method,
                learning_rate,
                budget_mode,
                max_steps,
                noise_on,
                noise_amplitude_mode,
                noise_source,
                noise_digit_order,
                noise_alpha,
                noise_modulation,
                noise_scope,
                noise_envelope,
                noise_clean_tail,
                geometry_on,
                geometry_mode,
                geometry_scope,
                geometry_weight,
                geometry_layer,
                geometry_margin,
                geometry_sample_tokens,
                role_loss_on,
                delimiter_weight,
                tool_name_weight,
                argument_key_weight,
                argument_value_weight,
                benchmark_profile,
                comparison_group,
                experiment_variant,
                validation_mode,
                validation_datasets,
                eval_ratio,
                max_validation_samples,
                allowed_tools,
                filter_unknown_tools,
                *exploration_inputs,
                sheet_status,
            ]

            def open_sheet_ui(sheet_id: str | None) -> tuple[Any, ...]:
                if not sheet_id:
                    raise ValueError("Select a workbook sheet first")
                payload = _sheet_payload(sheet_id)
                keys = [
                    "sheet_name",
                    "model",
                    "datasets",
                    "minutes",
                    "sequence",
                    "rank",
                    "max_source_rows",
                    "target_profile",
                    "adapter_method",
                    "learning_rate",
                    "budget_mode",
                    "max_steps",
                    "noise_enabled",
                    "noise_amplitude_mode",
                    "noise_source",
                    "noise_digit_order",
                    "noise_alpha",
                    "noise_modulation",
                    "noise_scope",
                    "noise_envelope",
                    "noise_clean_tail",
                    "geometry_enabled",
                    "geometry_mode",
                    "geometry_scope",
                    "geometry_weight",
                    "geometry_layer",
                    "geometry_margin",
                    "geometry_sample_tokens",
                    "role_loss_enabled",
                    "delimiter_weight",
                    "tool_name_weight",
                    "argument_key_weight",
                    "argument_value_weight",
                    "benchmark_profile",
                    "comparison_group",
                    "experiment_variant",
                    "validation_mode",
                    "validation_datasets",
                    "eval_ratio",
                    "max_validation_samples",
                    "allowed_tools",
                    "filter_unknown_tools",
                    "exploration_strategy",
                    "exploration_trials",
                    "exploration_total_minutes",
                    "exploration_seed",
                    "adapter_methods",
                    "lora_ranks",
                    "target_profiles",
                    "sequence_lengths",
                    "learning_rate_min",
                    "learning_rate_max",
                    "noise_modes",
                    "noise_alpha_min",
                    "noise_alpha_max",
                    "noise_digit_orders",
                    "noise_scopes",
                    "noise_envelopes",
                    "noise_modulation_min",
                    "noise_modulation_max",
                    "noise_clean_tail_min",
                    "noise_clean_tail_max",
                    "geometry_modes",
                    "geometry_scopes",
                    "geometry_layers",
                    "geometry_weight_min",
                    "geometry_weight_max",
                    "geometry_margin_min",
                    "geometry_margin_max",
                    "role_loss_modes",
                ]
                return (*[payload[key] for key in keys], f"Opened sheet: {payload['sheet_name']}")

            open_sheet.click(
                open_sheet_ui,
                inputs=[sheet_selector],
                outputs=sheet_load_outputs,
            )

            def new_sheet_ui() -> tuple[Any, str, str]:
                return (
                    gr.Dropdown(choices=sheet_choices(), value=None),
                    "New experiment",
                    "Editing a new unsaved sheet",
                )

            new_sheet.click(
                new_sheet_ui,
                outputs=[sheet_selector, sheet_name, sheet_status],
            )

            def duplicate_sheet_ui(sheet_id: str | None) -> tuple[Any, str, str]:
                if not sheet_id:
                    raise ValueError("Select a workbook sheet first")
                duplicate = WorkbookStore().duplicate(sheet_id)
                return (
                    gr.Dropdown(choices=sheet_choices(), value=duplicate.id),
                    duplicate.name,
                    f"Duplicated as {duplicate.name}",
                )

            duplicate_sheet.click(
                duplicate_sheet_ui,
                inputs=[sheet_selector],
                outputs=[sheet_selector, sheet_name, sheet_status],
            )

            def reload_sheets_ui(current: str | None) -> Any:
                values = {value for _, value in sheet_choices()}
                return gr.Dropdown(
                    choices=sheet_choices(), value=current if current in values else None
                )

            reload_sheets.click(
                reload_sheets_ui,
                inputs=[sheet_selector],
                outputs=[sheet_selector],
            )

            def import_latest_ui() -> tuple[Any, str, str]:
                latest = next(
                    (run for run in Registry().list_runs() if run["status"] == "complete"),
                    None,
                )
                if not latest:
                    raise ValueError("No completed run is available")
                recipe = TrainingRecipe.model_validate(json.loads(latest["recipe_json"]))
                recipe = recipe.model_copy(
                    update={
                        "workbook_sheet_id": None,
                        "exploration_trial": None,
                        "exploration_signature": None,
                    }
                )
                saved = WorkbookStore().save(
                    f"Imported {latest['run_id']}", recipe, ExplorationConfig()
                )
                Registry().assign_run_to_sheet(latest["run_id"], saved.id)
                return (
                    gr.Dropdown(choices=sheet_choices(), value=saved.id),
                    saved.name,
                    (
                        f"Imported and linked run {latest['run_id']}; press Open sheet to "
                        "load its fields"
                    ),
                )

            import_latest.click(
                import_latest_ui,
                outputs=[sheet_selector, sheet_name, sheet_status],
            )
        with gr.Tab("Live progress"):
            initial_live = _monitor()
            log_cutoff = gr.State(0.0)
            live_timer = gr.Timer(value=2.0, active=False)
            with gr.Row():
                refresh = gr.Button("Refresh")
                live_refresh = gr.Checkbox(
                    value=False,
                    label="Live auto-refresh (every 2 seconds)",
                )
                reset_logs = gr.Button("Reset visible logs")
                stop_live = gr.Button("Stop training safely", variant="stop")
            progress_view = gr.HTML(value=initial_live[4], label="Progress")
            phase_view = gr.Textbox(value=initial_live[5], label="Current phase", interactive=False)
            result_view = gr.JSON(value=initial_live[6], label="Current / latest result")
            metrics_view = gr.HTML(value=initial_live[7], label="Loss and token accuracy")
            run_table = gr.Dataframe(
                headers=["Run", "Status", "Output", "Updated (local time)"],
                value=initial_live[0],
                interactive=False,
            )
            with gr.Accordion("Technical logs (newest first)", open=False):
                jobs_view = gr.Code(value=initial_live[1], label="GUI jobs", language="json")
                events_view = gr.Textbox(
                    value=initial_live[2], label="Latest events in local time", lines=16
                )
            geometry_view = gr.HTML(label="Latent geometry")
            live_outputs = [
                run_table,
                jobs_view,
                events_view,
                geometry_view,
                progress_view,
                phase_view,
                result_view,
                metrics_view,
            ]
            refresh.click(_monitor, inputs=[log_cutoff], outputs=live_outputs)
            live_timer.tick(_monitor, inputs=[log_cutoff], outputs=live_outputs)

            def set_live_refresh(enabled: bool) -> Any:
                return gr.Timer(value=2.0, active=bool(enabled))

            def reset_visible_logs() -> tuple[float, str, str]:
                jobs = clear_finished_jobs()
                return (
                    time.time(),
                    "Visible logs reset. New events will appear here.",
                    json.dumps(jobs, indent=2),
                )

            live_refresh.change(set_live_refresh, inputs=[live_refresh], outputs=[live_timer])
            reset_logs.click(
                reset_visible_logs,
                outputs=[log_cutoff, events_view, jobs_view],
            )
            stop_live.click(_stop_run, outputs=jobs_view)

        with gr.Tab("Sealed test area"):
            gr.Markdown(
                "A sealed suite is fingerprinted and permanently blocked from training. The "
                "evaluator compares the selected adapter with the untouched instruct base on "
                "format, tool choice, required arguments, argument keys and exact values. For a "
                "broader official function-calling benchmark use "
                "[BFCL](https://huggingface.co/datasets/gorilla-llm/"
                "Berkeley-Function-Calling-Leaderboard) with its dedicated AST evaluator."
            )
            gr.Markdown(
                "Metric denominators: expected-format and tool-decision use every evaluated "
                "sample; tool-name and argument metrics use only samples whose reference "
                "requires a tool. They are not conditioned on the model producing valid format."
            )
            with gr.Row():
                sealed_source = gr.Textbox(
                    value=str(project_root() / "examples" / "sealed-cli-tools-v2.jsonl"),
                    label="Local path or Hugging Face dataset ID",
                )
                sealed_name = gr.Textbox(value="CLI tools sealed v2", label="Suite name")
                sealed_max_rows = gr.Number(value=1000, precision=0, label="Max sealed rows")
            register_sealed = gr.Button("Register as sealed — never train", variant="primary")
            sealed_register_status = gr.JSON(label="Sealed registration report")
            with gr.Row():
                sealed_suite = gr.Dropdown(
                    choices=_sealed_choices(), allow_custom_value=True, label="Sealed suite"
                )
                sealed_run = gr.Dropdown(
                    choices=_completed_run_choices(),
                    multiselect=True,
                    allow_custom_value=True,
                    label="Completed adapter runs (same sealed suite for every selection)",
                )
                sealed_samples = gr.Slider(
                    1,
                    1000,
                    value=500,
                    step=1,
                    label="SEALED TEST SIZE (prompts per adapter)",
                )
                sealed_compare_base = gr.Checkbox(
                    value=True, label="Compare against untouched instruct base"
                )
            with gr.Row():
                start_sealed = gr.Button("Start sealed evaluation", variant="primary")
                refresh_sealed = gr.Button("Refresh suites, runs and results")
            sealed_job_status = gr.Code(label="Sealed evaluator launcher", language="json")
            sealed_results_table = gr.Dataframe(
                headers=[
                    "Evaluated",
                    "Run",
                    "Suite",
                    "Adapter samples",
                    "Tool-required samples",
                    "Parseable format (raw/all)",
                    "Balanced tool/no-tool routing",
                    "Base selected tool",
                    "Adapter selected tool",
                    "Selected-tool delta",
                    "Adapter valid args",
                    "Adapter exact args",
                    "Result file",
                ],
                value=_sealed_result_rows(),
                interactive=False,
            )
            sealed_event = register_sealed.click(
                _register_sealed_gui,
                inputs=[sealed_source, sealed_name, sealed_max_rows],
                outputs=[sealed_register_status],
            )

            def refresh_sealed_choices() -> tuple[Any, Any, list[list[Any]], str]:
                suites = _sealed_choices()
                runs = _completed_run_choices()
                return (
                    gr.Dropdown(choices=suites, value=suites[0][1] if suites else None),
                    gr.Dropdown(choices=runs, value=[runs[0][1]] if runs else []),
                    _sealed_result_rows(),
                    json.dumps(_jobs_for_display(), indent=2),
                )

            sealed_event.then(
                refresh_sealed_choices,
                outputs=[sealed_suite, sealed_run, sealed_results_table, sealed_job_status],
            )
            refresh_sealed.click(
                refresh_sealed_choices,
                outputs=[sealed_suite, sealed_run, sealed_results_table, sealed_job_status],
            )
            start_sealed.click(
                _start_sealed_eval,
                inputs=[sealed_run, sealed_suite, sealed_samples, sealed_compare_base],
                outputs=[sealed_job_status],
            )

        with gr.Tab("Compare experiments") as compare_tab:
            gr.Markdown(
                "Select completed runs from any workbook sheet. External DEV or sealed generation "
                "metrics take priority when available; otherwise the table shows the internal "
                "validation metrics. A fair "
                "baseline/exotic comparison must have the same Comparison group and Protocol ID; "
                "otherwise more than the exotic intervention changed."
            )
            compare_run_selector = gr.Dropdown(
                choices=_completed_run_choices(),
                multiselect=True,
                allow_custom_value=True,
                label="Completed runs",
            )
            with gr.Row():
                compare_button = gr.Button("Compare selected runs", variant="primary")
                refresh_compare = gr.Button("Reload completed runs")
                load_pipeline_compare = gr.Button(
                    "Load latest automated pipeline comparison",
                    variant="primary",
                )
                export_compare = gr.Button("Export report + picture")
            comparison_table = gr.Dataframe(
                headers=[
                    "Run",
                    "Sheet",
                    "Profile",
                    "Comparison group",
                    "Variant",
                    "Protocol ID",
                    "Trial",
                    "Adapter",
                    "Rank",
                    "Target",
                    "Sequence",
                    "LR",
                    "Noise",
                    "Geometry",
                    "Train loss",
                    "Validation loss",
                    "Validation perplexity",
                    "Internal tool name",
                    "Internal valid args",
                    "Evaluation suite",
                    "Evaluation samples",
                    "Tool-required samples",
                    "Parseable format (raw/all)",
                    "Balanced tool/no-tool routing",
                    "Base selected tool",
                    "Adapter selected tool",
                    "Selected-tool delta",
                    "Evaluation valid args",
                    "Evaluation exact args",
                    "Metric schema",
                    "Evaluation protocol",
                    "Tool-attempt recall",
                    "No-tool specificity",
                    "Routing MCC",
                    "Parseable call recall",
                    "Required values exact",
                    "Required values normalized diagnostic",
                    "All argument fields exact (micro)",
                    "Required argument fields exact (micro)",
                    "Direct-answer criteria satisfied",
                    "Human task success",
                    "Strict human task success",
                    "Required values elastic (semantic-safe)",
                    "Elastic human task success",
                    "Evaluated at",
                ],
                interactive=False,
            )
            comparison_chart = gr.HTML(label="Tool comparison")
            comparison_exports = gr.File(
                label="Exported comparison artifacts",
                file_count="multiple",
                interactive=False,
            )
            compare_button.click(
                _compare_runs,
                inputs=[compare_run_selector],
                outputs=[comparison_table, comparison_chart],
            )

            def load_latest_pipeline_compare() -> tuple[Any, list[list[Any]], str]:
                result = _latest_compare_ready_pipeline()
                choices = _completed_run_choices()
                if not result:
                    status = (result or {}).get("status", "missing")
                    return (
                        gr.Dropdown(choices=choices, value=[]),
                        [],
                        f"<p>Latest automated pipeline is not compare-ready: {html.escape(str(status))}.</p>",
                    )
                run_ids = [str(value) for value in result.get("compare_run_ids", [])]
                rows, chart = _compare_runs(run_ids)
                return gr.Dropdown(choices=choices, value=run_ids), rows, chart

            load_pipeline_compare.click(
                load_latest_pipeline_compare,
                outputs=[compare_run_selector, comparison_table, comparison_chart],
            )
            compare_tab.select(
                load_latest_pipeline_compare,
                outputs=[compare_run_selector, comparison_table, comparison_chart],
            )
            export_compare.click(
                _export_comparison,
                inputs=[compare_run_selector],
                outputs=[comparison_exports],
            )

            def refresh_compare_choices(current: list[str] | None) -> Any:
                choices = _completed_run_choices()
                valid = {value for _, value in choices}
                return gr.Dropdown(
                    choices=choices,
                    value=[value for value in (current or []) if value in valid],
                )

            refresh_compare.click(
                refresh_compare_choices,
                inputs=[compare_run_selector],
                outputs=[compare_run_selector],
            )

        def use_model_for_train(selected: str | None) -> tuple[str, str]:
            selected_path = str(selected or "").strip()
            if not selected_path:
                return default_model, "Select a registered model first"
            return selected_path, f"Selected for Train: {selected_path}"

        def use_dataset_for_train(
            selected_source: str | None, current: list[str] | None
        ) -> tuple[list[str], str]:
            selected = list(current or [])
            selected_source = str(selected_source or "").strip()
            if not selected_source:
                return selected, "Select a registered dataset first"
            if selected_source not in selected:
                selected.append(selected_source)
            return selected, f"Added to Train: {selected_source}"

        use_model.click(
            use_model_for_train,
            inputs=[model_train_picker],
            outputs=[train_model, model_status],
        )
        use_dataset.click(
            use_dataset_for_train,
            inputs=[dataset_train_picker, train_datasets],
            outputs=[train_datasets, dataset_status],
        )
    demo.launch(server_name=host, server_port=port, share=share)
