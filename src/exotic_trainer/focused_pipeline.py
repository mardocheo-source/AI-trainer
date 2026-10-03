from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .human_agentic_focus_data import ensure_focused_human_agentic_suite
from .human_agentic_ood import ensure_human_agentic_ood_suite
from .paths import runs_dir
from .preflight import preflight_recipe
from .registry import Registry
from .schema import ExplorationConfig, GeometryConfig, RoleLossConfig, TrainingRecipe
from .sealed import run_sealed_evaluation, sealed_guard, sealed_results
from .tool_schema import KNOWN_TOOLS
from .trainer import run_training
from .workbook import WorkbookStore

FOCUSED_CELLS = (
    "A-baseline-control",
    "B-pi-clean30-control",
    "C-pi-full100",
    "D-sqrt2-full100",
    "E-pi-anchor-geometry",
    "F-pi-literal-lock",
)
NEW_FOCUSED_CELLS = FOCUSED_CELLS[2:]


def focused_results_dir() -> Path:
    path = runs_dir() / "focused-human-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _focused_pipeline_lock():
    """Prevent a resume from duplicating an already-running focused pipeline.

    ``flock`` is owned by the kernel and is released even when the worker is
    killed, so it remains reliable across PID namespaces and stale PID files.
    """
    lock_path = focused_results_dir() / ".pipeline.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.seek(0)
            owner = lock_file.read().strip() or "an existing focused worker"
            raise RuntimeError(
                f"focused pipeline already running ({owner}); do not start or resume it twice"
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} started={datetime.now(UTC).isoformat()}")
        lock_file.flush()
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def list_focused_results() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in focused_results_dir().glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        item["result_path"] = str(path)
        output.append(item)
    return sorted(output, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_focused_result() -> dict[str, Any] | None:
    results = list_focused_results()
    return results[0] if results else None


def _recipe_from_run(run: dict[str, Any]) -> TrainingRecipe:
    return TrainingRecipe.model_validate(json.loads(run.get("recipe_json") or "{}"))


def _core(recipe: TrainingRecipe) -> dict[str, Any]:
    return recipe.model_dump(
        exclude={
            "name",
            "experiment_variant",
            "output_dir",
            "workbook_sheet_id",
            "noise",
            "geometry",
            "role_loss",
        }
    )


def _sheet_recipe_payload(recipe: TrainingRecipe) -> dict[str, Any]:
    return recipe.model_dump(exclude={"workbook_sheet_id"})


def _save_or_reuse_sheet(
    store: WorkbookStore,
    recipe: TrainingRecipe,
    exploration: ExplorationConfig,
) -> Any:
    target = _sheet_recipe_payload(recipe)
    existing = next(
        (
            sheet
            for sheet in store.list()
            if sheet.name == recipe.name and _sheet_recipe_payload(sheet.recipe) == target
        ),
        None,
    )
    return existing or store.save(recipe.name, recipe, exploration)


def _compact_training(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "run_id",
        "output_dir",
        "device",
        "global_step",
        "fixed_step_target_reached",
        "steps_shortfall",
        "train_runtime",
        "train_loss",
        "eval_loss",
        "eval_perplexity",
        "elapsed_total_seconds",
        "stopped_by_user",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _compact_preflight(report: dict[str, Any]) -> dict[str, Any]:
    training = report.get("training") or {}
    return {
        "status": report.get("status"),
        "warnings": report.get("warnings") or [],
        "training_status": training.get("status"),
        "summary": training.get("summary") or {},
        "validation": report.get("validation") or {},
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "agent_metric_schema_version",
        "agent_eval_samples",
        "agent_tool_schema_applied",
        "agent_protocol_id",
        "agent_expected_format_accuracy",
        "agent_balanced_tool_decision_accuracy",
        "agent_tool_decision_mcc",
        "agent_tool_recall",
        "agent_no_tool_specificity",
        "agent_selected_tool_name_accuracy",
        "agent_tool_arguments_typed_rate",
        "agent_required_argument_call_exact_accuracy",
        "agent_required_argument_call_normalized_accuracy",
        "agent_required_argument_call_elastic_accuracy",
        "agent_required_argument_field_micro_accuracy",
        "agent_direct_required_terms_accuracy",
        "agent_human_task_success_rate",
        "agent_strict_human_task_success_rate",
        "agent_elastic_human_task_success_rate",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _latest_control_runs() -> tuple[dict[str, Any], dict[str, Any]]:
    source = next(
        (item for item in list_focused_source_results() if item.get("status") == "complete"),
        None,
    )
    if source is None:
        raise KeyError("no completed human-agentic A/B control pipeline is available")
    cells = source.get("cells") or {}
    registry = Registry()
    baseline = registry.get_run(str(cells["A-baseline"]["run_id"]))
    pi_noise = registry.get_run(str(cells["B-pi-noise"]["run_id"]))
    if baseline.get("status") != "complete" or pi_noise.get("status") != "complete":
        raise ValueError("the reusable human-agentic A/B controls are not complete")
    return baseline, pi_noise


def list_focused_source_results() -> list[dict[str, Any]]:
    # Kept separate for tests and to make the dependency on a frozen control
    # experiment explicit.
    from .pipeline import list_human_agentic_results

    return list_human_agentic_results()


def prepare_focused_sheets() -> dict[str, Any]:
    """Freeze four new variants around the existing fair human-agentic A/B pair."""

    baseline_run, pi_run = _latest_control_runs()
    baseline = _recipe_from_run(baseline_run)
    pi_control = _recipe_from_run(pi_run)
    if _core(baseline) != _core(pi_control):
        raise ValueError("existing human-agentic A/B controls differ outside regularizers")
    if baseline.budget_mode != "steps" or baseline.max_steps != pi_control.max_steps:
        raise ValueError("focused pipeline requires equal fixed-step controls")
    if baseline.allowed_tools != sorted(KNOWN_TOOLS) or not baseline.tool_menu_conditioning:
        raise ValueError("focused controls must use the complete schema-conditioned 13-tool menu")
    expected_pi = pi_control.noise
    if not (
        expected_pi.enabled
        and expected_pi.source == "pi"
        and expected_pi.digit_order == "natural"
        and expected_pi.alpha == 2.0
        and expected_pi.modulation == 0.1
        and expected_pi.scope == "prompt"
        and expected_pi.envelope == "constant"
        and expected_pi.clean_tail_fraction == 0.3
        and not pi_control.geometry.enabled
        and not pi_control.role_loss.enabled
    ):
        raise ValueError("the reusable π control is not the frozen natural-pair schedule")

    full_pi = expected_pi.model_copy(
        update={
            "clean_tail_fraction": 0.0,
            "protect_prompt_literals": False,
            "value_consistency_weight": 0.0,
            "intent_consistency_weight": 0.0,
        }
    )
    full_sqrt2 = full_pi.model_copy(update={"source": "sqrt2"})
    geometry_off = GeometryConfig(
        enabled=False,
        mode="relational",
        scope="anchors",
        weight=0.002,
        layer=-1,
        margin=0.2,
        sample_tokens=32,
    )
    anchor_geometry = geometry_off.model_copy(update={"enabled": True})
    role_off = RoleLossConfig(enabled=False)
    literal_role = RoleLossConfig(
        enabled=True,
        ordinary_weight=1.0,
        delimiter_weight=0.75,
        tool_name_weight=1.15,
        argument_key_weight=1.35,
        argument_value_weight=2.5,
    )
    literal_noise = full_pi.model_copy(
        update={
            "protect_prompt_literals": True,
            "value_consistency_weight": 0.01,
            "intent_consistency_weight": 0.005,
        }
    )
    step_count = int(baseline.max_steps)
    group = baseline.comparison_group
    common = baseline.model_copy(
        update={
            "workbook_sheet_id": None,
            "output_dir": None,
            "experiment_variant": "exotic",
            "logging_steps": 10,
        }
    )
    recipes = {
        "C-pi-full100": common.model_copy(
            update={
                "name": f"{group}-c-pi-natural-full100-a2-m01-s{step_count}",
                "noise": full_pi,
                "geometry": geometry_off,
                "role_loss": role_off,
            }
        ),
        "D-sqrt2-full100": common.model_copy(
            update={
                "name": f"{group}-d-sqrt2-natural-full100-a2-m01-s{step_count}",
                "noise": full_sqrt2,
                "geometry": geometry_off,
                "role_loss": role_off,
            }
        ),
        "E-pi-anchor-geometry": common.model_copy(
            update={
                "name": f"{group}-e-pi-full100-anchor-relgeo002-s{step_count}",
                "noise": full_pi,
                "geometry": anchor_geometry,
                "role_loss": role_off,
            }
        ),
        "F-pi-literal-lock": common.model_copy(
            update={
                "name": (
                    f"{group}-f-pi-literal-lock-full100-anchor002-vce25-kl01-ikl005-s{step_count}"
                ),
                "noise": literal_noise,
                "geometry": anchor_geometry,
                "role_loss": literal_role,
            }
        ),
    }
    store = WorkbookStore()
    exploration = ExplorationConfig(enabled=False, trial_count=1)
    sheets = {
        label: _save_or_reuse_sheet(store, recipe, exploration) for label, recipe in recipes.items()
    }
    development = ensure_human_agentic_ood_suite()
    final = ensure_focused_human_agentic_suite()
    return {
        "comparison_group": group,
        "max_steps": step_count,
        "control_run_ids": {
            "A-baseline-control": baseline_run["run_id"],
            "B-pi-clean30-control": pi_run["run_id"],
        },
        "sheet_ids": {label: sheet.id for label, sheet in sheets.items()},
        "sheet_names": {label: sheet.name for label, sheet in sheets.items()},
        "development_suite_id": development["suite_id"],
        "development_audit": development,
        "final_suite_id": final["suite_id"],
        "final_audit": final,
    }


def _load_sheets(sheet_ids: dict[str, str]) -> dict[str, Any]:
    if set(sheet_ids) != set(NEW_FOCUSED_CELLS):
        raise ValueError(f"focused pipeline requires exactly: {NEW_FOCUSED_CELLS}")
    store = WorkbookStore()
    sheets = {label: store.get(sheet_ids[label]) for label in NEW_FOCUSED_CELLS}
    recipes = {label: sheet.recipe for label, sheet in sheets.items()}
    reference = recipes[NEW_FOCUSED_CELLS[0]]
    if any(_core(recipe) != _core(reference) for recipe in recipes.values()):
        raise ValueError("focused cells differ outside declared regularizers")
    for label, recipe in recipes.items():
        if recipe.budget_mode != "steps" or recipe.max_steps != reference.max_steps:
            raise ValueError(f"{label} changed the fixed optimizer-step budget")
        if recipe.allowed_tools != sorted(KNOWN_TOOLS) or not recipe.tool_menu_conditioning:
            raise ValueError(f"{label} does not use the complete 13-tool menu")
    pi_full = recipes["C-pi-full100"].noise
    if not (
        pi_full.enabled
        and pi_full.source == "pi"
        and pi_full.digit_order == "natural"
        and pi_full.scope == "prompt"
        and pi_full.envelope == "constant"
        and pi_full.clean_tail_fraction == 0.0
    ):
        raise ValueError("C must use natural π pairs over 100% of optimizer steps")
    sqrt2 = recipes["D-sqrt2-full100"].noise
    if sqrt2.model_copy(update={"source": "pi"}) != pi_full:
        raise ValueError("D must differ from C only by sqrt(2) decimal-pair source")
    for label in ("E-pi-anchor-geometry", "F-pi-literal-lock"):
        geometry = recipes[label].geometry
        if not (
            geometry.enabled
            and geometry.mode == "relational"
            and geometry.scope == "anchors"
            and geometry.weight == 0.002
        ):
            raise ValueError(f"{label} must use targeted name/key relational geometry")
    literal = recipes["F-pi-literal-lock"]
    if not (
        literal.noise.protect_prompt_literals
        and literal.noise.value_consistency_weight > 0
        and literal.role_loss.enabled
        and literal.role_loss.argument_value_weight > literal.role_loss.argument_key_weight
    ):
        raise ValueError("F does not implement the complete π-LiteralLock contract")
    return sheets


def _cached_eval(run_id: str, suite_id: str, samples: int) -> dict[str, Any] | None:
    candidates = [
        item
        for item in sealed_results()
        if str(item.get("run_id")) == run_id
        and str(item.get("suite_id")) == suite_id
        and int(item.get("max_samples") or 0) >= samples
        and int((item.get("adapter") or {}).get("agent_metric_schema_version") or 0) >= 8
        and int((item.get("adapter") or {}).get("agent_eval_samples") or 0) >= samples
    ]
    return candidates[0] if candidates else None


def _summary(labels: tuple[str, ...], evaluations: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for label in labels:
        metrics = (evaluations.get(label) or {}).get("adapter") or {}
        cells[label] = {
            "run_id": (evaluations.get(label) or {}).get("run_id"),
            **_compact_metrics(metrics),
        }
    winner = max(
        cells,
        key=lambda label: (
            float(cells[label].get("agent_elastic_human_task_success_rate") or 0.0),
            float(cells[label].get("agent_strict_human_task_success_rate") or 0.0),
            float(cells[label].get("agent_balanced_tool_decision_accuracy") or 0.0),
            float(cells[label].get("agent_required_argument_call_exact_accuracy") or 0.0),
        ),
        default=None,
    )
    return {"cells": cells, "recommended_cell": winner}


def _evaluate_suite(
    state: dict[str, Any],
    result_path: Path,
    key: str,
    suite_id: str,
    samples: int,
    run_ids: dict[str, str],
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    shared_base: dict[str, Any] | None = None
    for label in FOCUSED_CELLS:
        run_id = run_ids[label]
        cached = _cached_eval(run_id, suite_id, samples)
        result = cached or run_sealed_evaluation(
            run_id=run_id,
            suite_id=suite_id,
            max_samples=samples,
            compare_base=shared_base is None,
            base_metrics=shared_base,
        )
        if shared_base is None and result.get("base"):
            shared_base = result["base"]
        evaluations[label] = result
        state[key] = {
            name: {
                "run_id": item.get("run_id"),
                "result_path": item.get("result_path"),
                "adapter": _compact_metrics(item.get("adapter") or {}),
            }
            for name, item in evaluations.items()
        }
        state[f"{key}_summary"] = _summary(FOCUSED_CELLS, evaluations)
        _write_state(result_path, state)
        stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
        if stop_file and Path(stop_file).exists():
            state.update(status="stopped", stage=f"stopped-during-{key}")
            _write_state(result_path, state)
            return evaluations
    return evaluations


def _continue_focused_locked(state: dict[str, Any], result_path: Path) -> dict[str, Any]:
    sheets = _load_sheets(dict(state["sheet_ids"]))
    registry = Registry()
    run_ids: dict[str, str] = {}
    for label in FOCUSED_CELLS:
        candidate = str((state.get("cells", {}).get(label) or {}).get("run_id") or "")
        if not candidate:
            continue
        try:
            if registry.get_run(candidate).get("status") == "complete":
                run_ids[label] = candidate
        except KeyError:
            continue
    try:
        if any(label not in run_ids for label in NEW_FOCUSED_CELLS):
            import torch

            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError(
                    "focused B580 pipeline requires an available XPU; the Intel GPU/driver "
                    "is currently unavailable, so CPU fallback was blocked"
                )
        for label in NEW_FOCUSED_CELLS:
            if label in run_ids:
                state["cells"][label]["status"] = "complete"
                continue
            state.update(status="running", stage=f"training-{label}")
            state["cells"][label]["status"] = "training"
            _write_state(result_path, state)
            metrics = run_training(sheets[label].recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(f"{label} did not reach its fixed-step target")
            run_id = str(metrics["run_id"])
            run_ids[label] = run_id
            state["cells"][label].update(
                status="complete", run_id=run_id, metrics=_compact_training(metrics)
            )
            state["run_ids"] = [run_ids[name] for name in FOCUSED_CELLS if name in run_ids]
            _write_state(result_path, state)
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(result_path, state)
                return state

        if set(run_ids) != set(FOCUSED_CELLS):
            raise RuntimeError("focused controls or completed training cells are missing")
        samples = int(state["samples_requested"])
        state.update(status="running", stage="development-human-strict-elastic")
        _write_state(result_path, state)
        development = _evaluate_suite(
            state,
            result_path,
            "development_evaluation",
            str(state["development_suite_id"]),
            samples,
            run_ids,
        )
        if state.get("status") == "stopped":
            return state
        state["development_summary"] = _summary(FOCUSED_CELLS, development)
        state.update(stage="final-sealed-human-strict-elastic")
        _write_state(result_path, state)
        final = _evaluate_suite(
            state,
            result_path,
            "final_evaluation",
            str(state["final_suite_id"]),
            samples,
            run_ids,
        )
        if state.get("status") == "stopped":
            return state
        state["final_summary"] = _summary(FOCUSED_CELLS, final)
        state.update(
            status="complete",
            stage="compare-ready-focused-final",
            compare_run_ids=[run_ids[label] for label in FOCUSED_CELLS],
        )
        _write_state(result_path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        _write_state(result_path, state)
        raise


def _continue_focused(state: dict[str, Any], result_path: Path) -> dict[str, Any]:
    with _focused_pipeline_lock():
        return _continue_focused_locked(state, result_path)


def run_focused_pipeline(prepared: dict[str, Any], samples: int = 520) -> dict[str, Any]:
    sheets = _load_sheets(dict(prepared["sheet_ids"]))
    registry = Registry()
    run_ids = {str(key): str(value) for key, value in prepared["control_run_ids"].items()}
    baseline_recipe = _recipe_from_run(registry.get_run(run_ids["A-baseline-control"]))
    if any(_core(sheet.recipe) != _core(baseline_recipe) for sheet in sheets.values()):
        raise ValueError("new focused cells do not share the frozen A/B training contract")
    sealed_hashes, sealed_sources = sealed_guard()
    preflight = {
        label: preflight_recipe(sheet.recipe, sealed_hashes, sealed_sources)
        for label, sheet in sheets.items()
    }
    blocked = [label for label, report in preflight.items() if report.get("status") == "blocked"]
    if blocked:
        raise ValueError(f"focused pipeline blocked by preflight: {blocked}")
    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    result_path = focused_results_dir() / f"{pipeline_id}.json"
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "pipeline_type": "focused-human-strict-elastic-v1",
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_group": baseline_recipe.comparison_group,
        "samples_requested": int(samples),
        "development_suite_id": prepared["development_suite_id"],
        "final_suite_id": prepared["final_suite_id"],
        "sheet_ids": dict(prepared["sheet_ids"]),
        "preflight": {label: _compact_preflight(report) for label, report in preflight.items()},
        "cells": {
            "A-baseline-control": {
                "status": "complete-reused",
                "run_id": run_ids["A-baseline-control"],
            },
            "B-pi-clean30-control": {
                "status": "complete-reused",
                "run_id": run_ids["B-pi-clean30-control"],
            },
            **{
                label: {"status": "pending", "sheet_id": prepared["sheet_ids"][label]}
                for label in NEW_FOCUSED_CELLS
            },
        },
        "run_ids": list(run_ids.values()),
    }
    _write_state(result_path, state)
    return _continue_focused(state, result_path)


def resume_focused_pipeline(pipeline_id: str | None = None) -> dict[str, Any]:
    state = next(
        (
            item
            for item in list_focused_results()
            if pipeline_id is None or item.get("pipeline_id") == pipeline_id
        ),
        None,
    )
    if state is None:
        raise KeyError(f"focused pipeline not found: {pipeline_id or 'latest'}")
    if state.get("status") == "complete":
        return state
    result_path = Path(str(state.pop("result_path")))
    state.pop("error", None)
    state.update(status="running", stage="resuming")
    _write_state(result_path, state)
    return _continue_focused(state, result_path)
