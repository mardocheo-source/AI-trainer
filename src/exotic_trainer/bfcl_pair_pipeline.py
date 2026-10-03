from __future__ import annotations

import fcntl
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bfcl import (
    BFCL_PROFILE_DISK_ESTIMATES,
    GIB,
    bfcl_environment_status,
    build_bfcl_config,
    run_bfcl,
    save_bfcl_config,
    seed_bfcl_results,
)
from .paths import runs_dir
from .preflight import preflight_recipe
from .registry import Registry
from .schema import ExplorationConfig, GeometryConfig, NoiseConfig, RoleLossConfig, TrainingRecipe
from .sealed import sealed_guard
from .tool_schema import KNOWN_TOOLS
from .trainer import run_training
from .workbook import WorkbookStore

MODEL_PATH = "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct"
TRAIN_PATH = (
    "/mnt/git0/git/repository/AI-trainier/downloads/generated/"
    "human-agentic-train-v1-5200.jsonl"
)
DEV_PATH = "/mnt/git0/git/repository/AI-trainier/examples/dev-human-agentic-v1-520.jsonl"
MODEL_CARD_BFCL_V3 = 49.12
DEFAULT_STEPS = 400
TRAINING_OUTPUT_ESTIMATE = 384 * 1024 * 1024
PIPELINE_RESERVE = 5 * GIB


def pipeline_dir() -> Path:
    path = runs_dir() / "bfcl-v3-pair-pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _lock():
    path = pipeline_dir() / ".pipeline.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("a BFCL v3 fair-pair pipeline is already running") from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _write(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def list_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in pipeline_dir().glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        item["result_path"] = str(path)
        results.append(item)
    return sorted(results, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def latest_result() -> dict[str, Any] | None:
    results = list_results()
    return results[0] if results else None


def _memory_available_bytes() -> int:
    try:
        fields = {
            key.rstrip(":"): int(value) * 1024
            for key, value, *_ in (
                line.split() for line in Path("/proc/meminfo").read_text().splitlines()
            )
        }
        return fields.get("MemAvailable", 0)
    except (OSError, ValueError):
        return 0


def resource_preflight() -> dict[str, Any]:
    output = runs_dir()
    disk = shutil.disk_usage(output)
    required = (
        2 * TRAINING_OUTPUT_ESTIMATE
        + 3 * BFCL_PROFILE_DISK_ESTIMATES["full"]
        + PIPELINE_RESERVE
    )
    memory_available = _memory_available_bytes()
    return {
        "status": "ready"
        if disk.free >= required and memory_available >= 6 * GIB
        else "blocked",
        "runs_dir": str(output),
        "disk_free_gib": round(disk.free / GIB, 2),
        "disk_required_gib": round(required / GIB, 2),
        "ram_available_gib": round(memory_available / GIB, 2),
        "ram_required_gib": 6.0,
        "vram_policy": {
            "device": "Intel XPU",
            "micro_batch_size": 1,
            "gradient_accumulation": 8,
            "sequence_length": 2048,
            "dtype": "bf16",
            "gradient_checkpointing": True,
            "kv_cache_during_training": False,
            "expected_peak": "approximately 8-11 GiB; abort rather than fall back to CPU",
        },
    }


def _base_recipe(steps: int) -> TrainingRecipe:
    group = f"bfclv3-lfm25-12b-pi-clean30-s{steps}-seed42"
    return TrainingRecipe(
        name=f"{group}-baseline",
        model=MODEL_PATH,
        datasets=[TRAIN_PATH],
        validation_mode="external",
        validation_datasets=[DEV_PATH],
        benchmark_profile="mixed",
        comparison_group=group,
        experiment_variant="baseline",
        seed=42,
        dtype="bf16",
        device="xpu",
        sequence_length=2048,
        micro_batch_size=1,
        gradient_accumulation=8,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_steps=steps,
        budget_mode="steps",
        time_limit_minutes=180,
        reserve_minutes=20,
        save_every_minutes=30,
        logging_steps=10,
        eval_ratio=0.05,
        max_source_rows=5_200,
        max_training_samples=5_200,
        max_validation_samples=520,
        agent_eval_samples=0,
        agent_eval_max_new_tokens=192,
        allowed_tools=sorted(KNOWN_TOOLS),
        filter_unknown_tools=True,
        tool_menu_conditioning=True,
        packing=False,
        gradient_checkpointing=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        use_rslora=True,
        target_profile="auto",
        noise=NoiseConfig(enabled=False),
        geometry=GeometryConfig(enabled=False, mode="relational", scope="anchors", weight=0.002),
        role_loss=RoleLossConfig(enabled=False),
    )


def prepare_pair(steps: int = DEFAULT_STEPS) -> dict[str, Any]:
    if steps < 100 or steps > 2_000:
        raise ValueError("BFCL screening steps must be between 100 and 2000")
    base = _base_recipe(steps)
    pi_noise = NoiseConfig(
        enabled=True,
        amplitude_mode="digit_pairs",
        source="pi",
        digit_order="natural",
        alpha=2.0,
        modulation=0.1,
        scope="prompt",
        envelope="constant",
        clean_tail_fraction=0.3,
        seed_offset=10_007,
    )
    exotic = base.model_copy(
        update={
            "name": f"{base.comparison_group}-pi-natural-a2-m01-prompt-clean30",
            "experiment_variant": "exotic",
            "noise": pi_noise,
        }
    )
    store = WorkbookStore()
    exploration = ExplorationConfig(enabled=False, trial_count=1)
    def save_or_reuse(recipe: TrainingRecipe):
        target = recipe.model_dump(exclude={"workbook_sheet_id"})
        existing = next(
            (
                sheet
                for sheet in store.list()
                if sheet.name == recipe.name
                and sheet.recipe.model_dump(exclude={"workbook_sheet_id"}) == target
            ),
            None,
        )
        return existing or store.save(recipe.name, recipe, exploration)

    baseline_sheet = save_or_reuse(base)
    exotic_sheet = save_or_reuse(exotic)
    return {
        "comparison_group": base.comparison_group,
        "steps": steps,
        "baseline_sheet_id": baseline_sheet.id,
        "exotic_sheet_id": exotic_sheet.id,
        "noise_contract": {
            "decimal_pairs": "14,15,92,65,... from the reviewed 100-digit pi window",
            "advance": "one natural pair per optimizer step",
            "wrap": "after pair 50, restart from 14",
            "alpha": 2.0,
            "modulation": 0.1,
            "scope": "prompt only",
            "clean_tail_fraction": 0.3,
        },
    }


def _compact_training(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "output_dir",
        "device",
        "global_step",
        "fixed_step_target_reached",
        "train_loss",
        "eval_loss",
        "elapsed_total_seconds",
        "noise",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _score_hint(bfcl_state: dict[str, Any]) -> dict[str, Any]:
    """Expose official CSV rows without inventing or renaming BFCL metrics."""
    summary = bfcl_state.get("score_summary") or {}
    return {
        "overall": summary.get("overall") or [],
        "csv_files": [item.get("path") for item in summary.get("csv_files") or []],
    }


def run_pipeline(steps: int = DEFAULT_STEPS) -> dict[str, Any]:
    with _lock():
        resources = resource_preflight()
        if resources["status"] != "ready":
            raise RuntimeError(f"BFCL pipeline resource preflight blocked: {resources}")
        environment = bfcl_environment_status("v3")
        if environment["status"] != "ready":
            raise RuntimeError("compact BFCL v3 is not installed; run bfcl-setup v3 first")

        prepared = prepare_pair(steps)
        store = WorkbookStore()
        sheets = {
            "baseline": store.get(prepared["baseline_sheet_id"]),
            "pi-noise": store.get(prepared["exotic_sheet_id"]),
        }
        sealed_hashes, sealed_sources = sealed_guard()
        dataset_preflight = {
            label: preflight_recipe(sheet.recipe, sealed_hashes, sealed_sources)
            for label, sheet in sheets.items()
        }
        blocked = [name for name, report in dataset_preflight.items() if report["status"] == "blocked"]
        if blocked:
            raise RuntimeError(f"training dataset preflight blocked: {blocked}")

        identifier = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        result_path = pipeline_dir() / f"{identifier}.json"
        state: dict[str, Any] = {
            "pipeline_id": identifier,
            "pipeline_type": "bfcl-v3-rslora-pi-fair-pair",
            "status": "running",
            "stage": "preflight-complete",
            "created_at": datetime.now(UTC).isoformat(),
            "progress_percent": 0.0,
            "prepared": prepared,
            "resources": resources,
            "dataset_preflight": dataset_preflight,
            "official_protocol": {
                "version": "v3",
                "scope": "full",
                "card_reference_score": MODEL_CARD_BFCL_V3,
                "card_model": "LFM2.5-1.2B-Instruct",
                "handler": "liquid-lfm2 compatibility handler",
                "warning": (
                    "The model card used Liquid's custom handler. This local handler follows "
                    "the public Liquid/Gorilla format but is not claimed byte-identical."
                ),
            },
            "training": {},
            "bfcl": {},
        }
        _write(result_path, state)
        try:
            import torch

            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError("Intel XPU unavailable; CPU fallback is disabled")

            run_ids: dict[str, str] = {}
            for index, (label, sheet) in enumerate(sheets.items()):
                state.update(stage=f"training-{label}", progress_percent=index * 20.0)
                _write(result_path, state)
                metrics = run_training(sheet.recipe)
                if metrics.get("fixed_step_target_reached") is not True:
                    raise RuntimeError(f"{label} did not reach the fixed {steps}-step budget")
                run_ids[label] = str(metrics["run_id"])
                state["training"][label] = _compact_training(metrics)
                state["run_ids"] = run_ids
                state["progress_percent"] = (index + 1) * 20.0
                _write(result_path, state)
                stop_file = os.environ.get("EXOTIC_TRAINER_STOP_FILE")
                if metrics.get("stopped_by_user") or (stop_file and Path(stop_file).exists()):
                    state.update(status="stopped", stage=f"stopped-after-{label}")
                    _write(result_path, state)
                    return state

            targets = {
                "untouched-base": f"model:{MODEL_PATH}",
                "rslora-baseline": f"run:{run_ids['baseline']}",
                "rslora-pi-noise": f"run:{run_ids['pi-noise']}",
            }
            for index, (label, target) in enumerate(targets.items()):
                state.update(stage=f"bfcl-v3-full-{label}", progress_percent=40.0 + index * 20.0)
                _write(result_path, state)
                config = build_bfcl_config(
                    version="v3",
                    profile="full",
                    handler="liquid-lfm2",
                    transport="managed-local-xpu",
                    target=target,
                    endpoint="",
                    api_model="local-model",
                    smoke_samples=20,
                    device="xpu",
                )
                config_path = save_bfcl_config(config)
                state["bfcl"][label] = {
                    "status": "running",
                    "config_path": str(config_path),
                    "state_path": str(config_path.parent / "state.json"),
                }
                _write(result_path, state)
                result = run_bfcl(config_path)
                state["bfcl"][label] = {
                    "status": result.get("status"),
                    "config_path": str(config_path),
                    "state_path": str(config_path.parent / "state.json"),
                    "output_dir": str(config_path.parent),
                    "score": _score_hint(result),
                }
                state["progress_percent"] = 40.0 + (index + 1) * 20.0
                _write(result_path, state)

            state.update(
                status="complete",
                stage="compare-ready",
                progress_percent=100.0,
                compare_run_ids=[run_ids["baseline"], run_ids["pi-noise"]],
                finished_at=datetime.now(UTC).isoformat(),
            )
            _write(result_path, state)
            return state
        except Exception as error:
            state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
            _write(result_path, state)
            raise


def resume_pipeline(
    pipeline_id: str | None = None,
    target_order: str = "pi-noise,baseline,base",
) -> dict[str, Any]:
    """Resume only the BFCL evaluations of an already trained fair pair."""

    with _lock():
        if pipeline_id:
            result_path = pipeline_dir() / f"{pipeline_id}.json"
            if not result_path.exists():
                raise ValueError(f"BFCL v3 pipeline not found: {pipeline_id}")
            state = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            latest = latest_result()
            if not latest:
                raise ValueError("No BFCL v3 fair-pair pipeline is available to resume")
            result_path = Path(str(latest["result_path"]))
            state = json.loads(result_path.read_text(encoding="utf-8"))

        run_ids = dict(state.get("run_ids") or {})
        if not run_ids.get("baseline") or not run_ids.get("pi-noise"):
            raise RuntimeError("Both completed training runs are required before BFCL resume")
        for run_id in run_ids.values():
            run = Registry().get_run(str(run_id))
            if not run or run.get("status") != "complete":
                raise RuntimeError(f"Completed training run not found: {run_id}")

        environment = bfcl_environment_status("v3")
        if environment["status"] != "ready":
            raise RuntimeError("compact BFCL v3 is not installed; run bfcl-setup v3 first")
        state.pop("error", None)
        state.update(status="running", stage="bfcl-resume", progress_percent=40.0)
        state.setdefault("bfcl", {})
        _write(result_path, state)

        available_targets = {
            "untouched-base": f"model:{MODEL_PATH}",
            "rslora-baseline": f"run:{run_ids['baseline']}",
            "rslora-pi-noise": f"run:{run_ids['pi-noise']}",
        }
        aliases = {
            "pi-noise": "rslora-pi-noise",
            "rslora-pi-noise": "rslora-pi-noise",
            "baseline": "rslora-baseline",
            "rslora-baseline": "rslora-baseline",
            "base": "untouched-base",
            "untouched-base": "untouched-base",
        }
        labels = [aliases.get(item.strip()) for item in target_order.split(",") if item.strip()]
        if not labels or any(label is None for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(
                "target order must contain unique values from: pi-noise, baseline, base"
            )
        targets = {label: available_targets[label] for label in labels if label is not None}
        state["bfcl_target_order"] = list(targets)
        state["bfcl_runtime_policy"] = {
            "one_model_at_a_time": True,
            "cpu_fallback": False,
            "resume_valid_rows": True,
            "retry_failed_rows": True,
        }
        _write(result_path, state)
        try:
            import torch

            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError("Intel XPU unavailable; CPU fallback is disabled")
            target_count = len(targets)
            for index, (label, target) in enumerate(targets.items()):
                previous = dict(state["bfcl"].get(label) or {})
                previous_state = Path(str(previous.get("state_path") or ""))
                if previous_state.is_file():
                    child = json.loads(previous_state.read_text(encoding="utf-8"))
                    if child.get("status") == "complete":
                        state["progress_percent"] = 40.0 + (index + 1) * 60.0 / target_count
                        _write(result_path, state)
                        continue
                state.update(
                    stage=f"bfcl-v3-full-{label}",
                    progress_percent=40.0 + index * 60.0 / target_count,
                )
                config = build_bfcl_config(
                    version="v3",
                    profile="full",
                    handler="liquid-lfm2",
                    transport="managed-local-xpu",
                    target=target,
                    endpoint="",
                    api_model="local-model",
                    smoke_samples=20,
                    device="xpu",
                )
                config_path = save_bfcl_config(config)
                recovery = None
                previous_output_text = str(previous.get("output_dir") or "")
                if not previous_output_text and previous.get("config_path"):
                    previous_output_text = str(Path(str(previous["config_path"])).parent)
                previous_output = Path(previous_output_text)
                if previous_output.is_dir():
                    recovery = seed_bfcl_results(config_path, previous_output)
                state["bfcl"][label] = {
                    "status": "running",
                    "config_path": str(config_path),
                    "state_path": str(config_path.parent / "state.json"),
                    "output_dir": str(config_path.parent),
                    "recovery": recovery,
                }
                _write(result_path, state)
                result = run_bfcl(config_path)
                state["bfcl"][label] = {
                    "status": result.get("status"),
                    "config_path": str(config_path),
                    "state_path": str(config_path.parent / "state.json"),
                    "output_dir": str(config_path.parent),
                    "score": _score_hint(result),
                    "recovery": recovery,
                }
                state["progress_percent"] = 40.0 + (index + 1) * 60.0 / target_count
                _write(result_path, state)

            state.update(
                status="complete",
                stage="compare-ready",
                progress_percent=100.0,
                compare_run_ids=[run_ids["baseline"], run_ids["pi-noise"]],
                finished_at=datetime.now(UTC).isoformat(),
            )
            _write(result_path, state)
            return state
        except Exception as error:
            state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
            _write(result_path, state)
            raise
