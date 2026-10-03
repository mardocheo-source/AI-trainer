from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .decimal_control_data import ensure_decimal_control_suite
from .focused_pipeline import (
    _cached_eval,
    _compact_metrics,
    _compact_preflight,
    _compact_training,
    _core,
    _latest_control_runs,
    _recipe_from_run,
    _save_or_reuse_sheet,
)
from .human_agentic_ood import ensure_human_agentic_ood_suite
from .paths import runs_dir
from .preflight import preflight_recipe
from .registry import Registry
from .schema import ExplorationConfig, NoiseConfig
from .sealed import run_sealed_evaluation, sealed_guard
from .tool_schema import KNOWN_TOOLS
from .trainer import run_training
from .workbook import WorkbookStore

CONTROL_CELLS = (
    "A-baseline-control",
    "B-pi-clean30-control",
    "G-sqrt2-clean30",
    "H-standard-constant-clean30",
)
NEW_CONTROL_CELLS = CONTROL_CELLS[2:]


def decimal_control_results_dir() -> Path:
    path = runs_dir() / "decimal-noise-control-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _pipeline_lock():
    lock_path = decimal_control_results_dir() / ".pipeline.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("decimal-noise control pipeline is already running") from error
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


def list_decimal_control_results() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in decimal_control_results_dir().glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        item["result_path"] = str(path)
        output.append(item)
    return sorted(output, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_decimal_control_result() -> dict[str, Any] | None:
    results = list_decimal_control_results()
    return results[0] if results else None


def _summary(evaluations: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for label in CONTROL_CELLS:
        item = evaluations.get(label) or {}
        cells[label] = {"run_id": item.get("run_id"), **_compact_metrics(item.get("adapter") or {})}
    winner = max(
        cells,
        key=lambda label: (
            float(cells[label].get("agent_elastic_human_task_success_rate") or 0.0),
            float(cells[label].get("agent_balanced_tool_decision_accuracy") or 0.0),
            float(cells[label].get("agent_selected_tool_name_accuracy") or 0.0),
            float(cells[label].get("agent_required_argument_call_exact_accuracy") or 0.0),
        ),
    )
    return {"cells": cells, "recommended_cell": winner}


def prepare_decimal_control_sheets() -> dict[str, Any]:
    baseline_run, pi_run = _latest_control_runs()
    baseline = _recipe_from_run(baseline_run)
    pi_control = _recipe_from_run(pi_run)
    if _core(baseline) != _core(pi_control):
        raise ValueError("frozen A/B controls differ outside regularizers")
    if baseline.budget_mode != "steps" or baseline.max_steps != pi_control.max_steps:
        raise ValueError("decimal controls require the same fixed optimizer-step budget")
    if baseline.allowed_tools != sorted(KNOWN_TOOLS) or not baseline.tool_menu_conditioning:
        raise ValueError("frozen controls do not use the complete conditioned 13-tool menu")
    noise = pi_control.noise
    if not (
        noise.enabled
        and noise.amplitude_mode == "digit_pairs"
        and noise.source == "pi"
        and noise.digit_order == "natural"
        and noise.alpha == 2.0
        and noise.modulation == 0.1
        and noise.scope == "prompt"
        and noise.envelope == "constant"
        and noise.clean_tail_fraction == 0.3
        and not pi_control.geometry.enabled
        and not pi_control.role_loss.enabled
    ):
        raise ValueError("run B is not the frozen natural π clean30 control")

    common = pi_control.model_copy(
        update={
            "workbook_sheet_id": None,
            "output_dir": None,
            "experiment_variant": "exotic",
            "logging_steps": 10,
        }
    )
    sqrt2_noise = noise.model_copy(update={"source": "sqrt2"})
    standard_noise = NoiseConfig(
        enabled=True,
        amplitude_mode="constant",
        source="pi",
        digit_order="natural",
        alpha=2.0,
        modulation=0.0,
        scope="prompt",
        envelope="constant",
        clean_tail_fraction=0.3,
        seed_offset=noise.seed_offset,
    )
    steps = int(baseline.max_steps)
    # Reuse the frozen comparison group so the GUI can recognise A/B/G/H as
    # one fair protocol; the sheet names still identify this control round.
    group = baseline.comparison_group
    recipes = {
        "G-sqrt2-clean30": common.model_copy(
            update={
                "name": f"{group}-decimal-v1-g-sqrt2-natural-a2-m01-prompt-clean30-s{steps}",
                "comparison_group": group,
                "noise": sqrt2_noise,
            }
        ),
        "H-standard-constant-clean30": common.model_copy(
            update={
                "name": f"{group}-decimal-v1-h-standard-constant-a2-prompt-clean30-s{steps}",
                "comparison_group": group,
                "noise": standard_noise,
            }
        ),
    }
    store = WorkbookStore()
    exploration = ExplorationConfig(enabled=False, trial_count=1)
    sheets = {
        label: _save_or_reuse_sheet(store, recipe, exploration) for label, recipe in recipes.items()
    }
    development = ensure_human_agentic_ood_suite()
    final = ensure_decimal_control_suite()
    return {
        "comparison_group": group,
        "max_steps": steps,
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
    if set(sheet_ids) != set(NEW_CONTROL_CELLS):
        raise ValueError(f"control pipeline requires exactly {NEW_CONTROL_CELLS}")
    store = WorkbookStore()
    sheets = {label: store.get(sheet_ids[label]) for label in NEW_CONTROL_CELLS}
    sqrt2 = sheets["G-sqrt2-clean30"].recipe
    standard = sheets["H-standard-constant-clean30"].recipe
    if _core(sqrt2) != _core(standard):
        raise ValueError("G and H differ outside the declared noise amplitude law")
    if sqrt2.noise.model_copy(update={"source": "pi"}) != standard.noise.model_copy(
        update={"amplitude_mode": "digit_pairs", "modulation": 0.1}
    ):
        raise ValueError("G/H must differ only by natural √2 pairs versus constant amplitude")
    if sqrt2.geometry.enabled or standard.geometry.enabled:
        raise ValueError("geometry must remain disabled in decimal-source controls")
    if sqrt2.role_loss.enabled or standard.role_loss.enabled:
        raise ValueError("role-weighted CE must remain disabled in decimal-source controls")
    return sheets


def _evaluate(
    state: dict[str, Any],
    path: Path,
    key: str,
    suite_id: str,
    samples: int,
    run_ids: dict[str, str],
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    shared_base: dict[str, Any] | None = None
    for label in CONTROL_CELLS:
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
        state[f"{key}_summary"] = _summary(evaluations)
        _write_state(path, state)
    return evaluations


def _continue_locked(state: dict[str, Any], path: Path) -> dict[str, Any]:
    sheets = _load_sheets(dict(state["sheet_ids"]))
    registry = Registry()
    run_ids: dict[str, str] = {}
    for label in CONTROL_CELLS:
        candidate = str((state.get("cells", {}).get(label) or {}).get("run_id") or "")
        if candidate:
            try:
                if registry.get_run(candidate).get("status") == "complete":
                    run_ids[label] = candidate
            except KeyError:
                pass
    try:
        if any(label not in run_ids for label in NEW_CONTROL_CELLS):
            import torch

            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError("B580/XPU is unavailable; CPU fallback was blocked")
        for label in NEW_CONTROL_CELLS:
            if label in run_ids:
                state["cells"][label]["status"] = "complete"
                continue
            state.update(status="running", stage=f"training-{label}")
            state["cells"][label]["status"] = "training"
            _write_state(path, state)
            metrics = run_training(sheets[label].recipe)
            if metrics.get("fixed_step_target_reached") is False:
                raise RuntimeError(f"{label} did not reach its fixed-step target")
            run_ids[label] = str(metrics["run_id"])
            state["cells"][label].update(
                status="complete", run_id=run_ids[label], metrics=_compact_training(metrics)
            )
            state["run_ids"] = [run_ids[name] for name in CONTROL_CELLS if name in run_ids]
            _write_state(path, state)
            stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                state.update(status="stopped", stage=f"stopped-after-{label}")
                _write_state(path, state)
                return state

        state.update(status="running", stage="development-evaluation")
        _write_state(path, state)
        development = _evaluate(
            state,
            path,
            "development_evaluation",
            str(state["development_suite_id"]),
            int(state["samples_requested"]),
            run_ids,
        )
        state["development_summary"] = _summary(development)
        state.update(stage="fresh-final-sealed-evaluation")
        _write_state(path, state)
        final = _evaluate(
            state,
            path,
            "final_evaluation",
            str(state["final_suite_id"]),
            int(state["samples_requested"]),
            run_ids,
        )
        state["final_summary"] = _summary(final)
        state.update(
            status="complete",
            stage="compare-ready-decimal-controls",
            compare_run_ids=[run_ids[label] for label in CONTROL_CELLS],
        )
        _write_state(path, state)
        return state
    except Exception as error:
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        _write_state(path, state)
        raise


def run_decimal_control_pipeline(prepared: dict[str, Any], samples: int = 520) -> dict[str, Any]:
    sheets = _load_sheets(dict(prepared["sheet_ids"]))
    sealed_hashes, sealed_sources = sealed_guard()
    preflight = {
        label: preflight_recipe(sheet.recipe, sealed_hashes, sealed_sources)
        for label, sheet in sheets.items()
    }
    blocked = [label for label, report in preflight.items() if report.get("status") == "blocked"]
    if blocked:
        raise ValueError(f"decimal-control pipeline blocked by preflight: {blocked}")
    pipeline_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = decimal_control_results_dir() / f"{pipeline_id}.json"
    controls = dict(prepared["control_run_ids"])
    state: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "pipeline_type": "decimal-pair-vs-standard-control-v1",
        "status": "running",
        "stage": "preflight-complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_group": prepared["comparison_group"],
        "samples_requested": int(samples),
        "development_suite_id": prepared["development_suite_id"],
        "final_suite_id": prepared["final_suite_id"],
        "sheet_ids": dict(prepared["sheet_ids"]),
        "preflight": {label: _compact_preflight(report) for label, report in preflight.items()},
        "cells": {
            "A-baseline-control": {"status": "complete-reused", "run_id": controls["A-baseline-control"]},
            "B-pi-clean30-control": {"status": "complete-reused", "run_id": controls["B-pi-clean30-control"]},
            **{
                label: {"status": "pending", "sheet_id": prepared["sheet_ids"][label]}
                for label in NEW_CONTROL_CELLS
            },
        },
        "run_ids": list(controls.values()),
    }
    _write_state(path, state)
    with _pipeline_lock():
        return _continue_locked(state, path)


def resume_decimal_control_pipeline(pipeline_id: str | None = None) -> dict[str, Any]:
    state = next(
        (
            item
            for item in list_decimal_control_results()
            if pipeline_id is None or item.get("pipeline_id") == pipeline_id
        ),
        None,
    )
    if state is None:
        raise KeyError(f"decimal-control pipeline not found: {pipeline_id or 'latest'}")
    if state.get("status") == "complete":
        return state
    path = Path(str(state.pop("result_path")))
    state.pop("error", None)
    state.update(status="running", stage="resuming")
    _write_state(path, state)
    with _pipeline_lock():
        return _continue_locked(state, path)
