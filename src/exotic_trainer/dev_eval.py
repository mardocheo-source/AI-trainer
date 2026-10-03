from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent_eval import evaluate_agent_behavior
from .data_loading import _load_one, _normalize_dataset
from .paths import project_root, runs_dir
from .registry import Registry


def dev_results_dir() -> Path:
    path = runs_dir() / "dev-results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dev_results() -> list[dict[str, Any]]:
    results = []
    # Older/detached pipelines may have been started without
    # EXOTIC_TRAINER_RUNS_DIR and therefore persisted their DEV artifacts in
    # <project>/runs even while the GUI uses the configured XDG runs directory.
    # Read both stores so the registry entry and its evaluation cannot become
    # disconnected in Compare experiments.
    directories = [dev_results_dir(), project_root() / "runs" / "dev-results"]
    seen_paths: set[Path] = set()
    for directory in directories:
        resolved_directory = directory.expanduser().resolve()
        if resolved_directory in seen_paths:
            continue
        seen_paths.add(resolved_directory)
        for path in resolved_directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["result_path"] = str(path)
                results.append(payload)
            except (OSError, json.JSONDecodeError):
                continue
    return sorted(results, key=lambda item: str(item.get("evaluated_at") or ""), reverse=True)


def _dataset_name(source: str) -> str:
    for item in Registry().list_datasets():
        if str(item.get("source")) == source:
            return str(item.get("name") or Path(source).stem)
    return Path(source).stem or "External DEV"


def run_dev_variant(
    run_id: str,
    source: str,
    max_samples: int,
    variant: str,
    result_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one adapter/base variant on its declared external DEV source."""
    if variant not in {"base", "adapter"}:
        raise ValueError(f"unsupported DEV evaluation variant: {variant}")
    registry = Registry()
    run = registry.get_run(run_id)
    recipe = json.loads(run["recipe_json"])
    resolved_source = registry.resolve_dataset(source)
    resolved_validation = {
        registry.resolve_dataset(value) for value in recipe.get("validation_datasets", [])
    }
    resolved_training = {registry.resolve_dataset(value) for value in recipe.get("datasets", [])}
    if recipe.get("validation_mode") != "external" or resolved_source not in resolved_validation:
        raise ValueError("DEV source must be declared as external validation by the run")
    if resolved_source in resolved_training:
        raise ValueError("DEV source also appears in training datasets")

    dataset = _normalize_dataset(
        _load_one(resolved_source, max_rows=max_samples, seed=1776),
        allowed_tools=None,
    )
    if len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=1776).select(range(max_samples))

    from .server import LocalModel

    runtime = LocalModel(
        model_path=registry.resolve_model(recipe["model"]),
        adapter_path=run["output_dir"] if variant == "adapter" else None,
        # Long batch evaluations must never silently fall back to a large
        # FP32 CPU model when the caller explicitly requires the accelerator.
        # This matters for detached/sandboxed launchers where /dev/dri may not
        # be visible even though the host has a working XPU.
        device=os.environ.get("EXOTIC_TRAINER_EVAL_DEVICE", "auto"),
    )
    try:
        stop_file = (
            Path(os.environ["EXOTIC_TRAINER_STOP_FILE"])
            if os.environ.get("EXOTIC_TRAINER_STOP_FILE")
            else None
        )
        metrics = evaluate_agent_behavior(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            dataset=dataset,
            max_samples=max_samples,
            max_new_tokens=192,
            deadline=time.monotonic() + max(300, max_samples * 45),
            stop_file=stop_file,
            tool_names=list(recipe.get("allowed_tools") or []),
            include_predictions=True,
        )
    finally:
        del runtime
        gc.collect()
        try:
            import torch

            if torch.xpu.is_available():
                torch.xpu.empty_cache()
        except (ImportError, RuntimeError):
            pass
    destination = Path(result_path)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _evaluate_variant_process(
    run_id: str,
    source: str,
    max_samples: int,
    variant: str,
    directory: Path,
) -> dict[str, Any]:
    temporary = directory / f".{uuid.uuid4().hex}_{variant}.worker.json"
    command = [
        sys.executable,
        "-m",
        "exotic_trainer.cli",
        "dev-eval-variant",
        run_id,
        source,
        str(max_samples),
        variant,
        str(temporary),
    ]
    try:
        completed = subprocess.run(command, check=False, cwd=project_root(), env=os.environ.copy())
        if completed.returncode != 0:
            signal = -completed.returncode if completed.returncode < 0 else None
            detail = f"; termination signal {signal}" if signal else ""
            raise RuntimeError(
                f"DEV {variant} evaluator exited with code {completed.returncode}{detail}"
            )
        if not temporary.exists():
            raise RuntimeError(f"DEV {variant} evaluator produced no result file")
        return json.loads(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def run_dev_evaluation(
    run_id: str,
    source: str,
    max_samples: int,
    compare_base: bool = True,
    base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = dev_results_dir()
    output: dict[str, Any] = {
        "evaluation_kind": "external-dev",
        "run_id": run_id,
        "suite_id": f"dev:{Path(source).name}",
        "suite_name": f"DEV — {_dataset_name(source)}",
        "source": source,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "max_samples": int(max_samples),
    }
    if compare_base:
        output["base"] = base_metrics or _evaluate_variant_process(
            run_id, source, max_samples, "base", directory
        )
    stop_file = (
        Path(os.environ["EXOTIC_TRAINER_STOP_FILE"])
        if os.environ.get("EXOTIC_TRAINER_STOP_FILE")
        else None
    )
    if not (stop_file and stop_file.exists()):
        output["adapter"] = _evaluate_variant_process(
            run_id, source, max_samples, "adapter", directory
        )
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    destination = directory / f"{stamp}_{run_id}_dev.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["result_path"] = str(destination)
    return output


def run_dev_batch(
    run_ids: list[str],
    source: str,
    max_samples: int,
    compare_base: bool = True,
) -> dict[str, Any]:
    if not run_ids:
        raise ValueError("select at least one completed run")
    stop_file = (
        Path(os.environ["EXOTIC_TRAINER_STOP_FILE"])
        if os.environ.get("EXOTIC_TRAINER_STOP_FILE")
        else None
    )
    results = []
    shared_base: dict[str, Any] | None = None
    for run_id in run_ids:
        if stop_file and stop_file.exists():
            break
        result = run_dev_evaluation(
            run_id=run_id,
            source=source,
            max_samples=max_samples,
            compare_base=compare_base,
            base_metrics=shared_base,
        )
        if compare_base and shared_base is None:
            shared_base = result.get("base")
        results.append(result)
    return {
        "status": "stopped" if stop_file and stop_file.exists() else "complete",
        "evaluation_kind": "external-dev",
        "source": source,
        "runs_requested": len(run_ids),
        "runs_evaluated": len(results),
        "results": results,
    }
