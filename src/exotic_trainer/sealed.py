from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent_eval import evaluate_agent_behavior
from .data_loading import _load_one, _normalize_dataset
from .dataset_probe import tool_names_in_example
from .paths import project_root, runs_dir
from .preflight import example_hash
from .registry import Registry

_LOCK = threading.Lock()


def sealed_store_path() -> Path:
    path = project_root() / "data" / "sealed-suites.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_suites() -> list[dict[str, Any]]:
    path = sealed_store_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _save_suites(suites: list[dict[str, Any]]) -> None:
    path = sealed_store_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(suites, indent=2), encoding="utf-8")
    temporary.replace(path)


def list_sealed_suites() -> list[dict[str, Any]]:
    with _LOCK:
        return sorted(_load_suites(), key=lambda item: item["created_at"], reverse=True)


def get_sealed_suite(suite_id: str) -> dict[str, Any]:
    for suite in list_sealed_suites():
        if suite["id"] == suite_id:
            return suite
    raise KeyError(f"sealed suite not found: {suite_id}")


def register_sealed_suite(source: str, name: str, max_rows: int = 1000) -> dict[str, Any]:
    raw = _load_one(source, max_rows=max_rows, seed=1776)
    normalized = _normalize_dataset(raw, allowed_tools=None)
    ordered_hashes = [example_hash(example) for example in normalized]
    hashes = sorted(set(ordered_hashes))
    tools: Counter[str] = Counter()
    tool_required_samples = 0
    prompt_skeletons: set[str] = set()
    for example in normalized:
        sample_tools = tool_names_in_example(
            {"prompt": [], "completion": example.get("completion", [])}
        )
        tools.update(sample_tools)
        tool_required_samples += int(bool(sample_tools))
        prompt_text = json.dumps(example.get("prompt", []), ensure_ascii=False, sort_keys=True)
        skeleton = re.sub(r"\d+", "<N>", prompt_text)
        skeleton = re.sub(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", "<PATH>", skeleton)
        prompt_skeletons.add(skeleton)
    no_tool_samples = len(normalized) - tool_required_samples
    warnings = []
    if min(tool_required_samples, no_tool_samples) < 0.5 * max(
        1, tool_required_samples, no_tool_samples
    ):
        warnings.append("tool/no-tool labels are imbalanced; use balanced accuracy and MCC")
    if len(prompt_skeletons) < max(10, len(normalized) // 5):
        warnings.append(
            "low prompt-template diversity; cluster-aware confidence intervals are required"
        )
    suite = {
        "id": f"sealed-{uuid.uuid4().hex[:10]}",
        "name": name.strip() or Path(source).stem or "Sealed suite",
        "source": source,
        "raw_rows": len(raw),
        "normalized_samples": len(normalized),
        "hashes": hashes,
        "ordered_hashes": ordered_hashes,
        "content_fingerprint": hashlib.sha256(
            json.dumps(ordered_hashes, separators=(",", ":")).encode()
        ).hexdigest(),
        "tool_distribution": dict(tools.most_common()),
        "tool_required_samples": tool_required_samples,
        "no_tool_samples": no_tool_samples,
        "prompt_skeletons": len(prompt_skeletons),
        "warnings": warnings,
        "created_at": datetime.now(UTC).isoformat(),
        "policy": "never-use-for-training",
    }
    with _LOCK:
        suites = _load_suites()
        if any(item.get("source") == source for item in suites):
            raise ValueError("this source is already registered as a sealed suite")
        suites.append(suite)
        _save_suites(suites)
    return suite


def sealed_guard() -> tuple[set[str], set[str]]:
    suites = list_sealed_suites()
    hashes = {value for suite in suites for value in suite.get("hashes", [])}
    sources = {str(suite["source"]) for suite in suites}
    return hashes, sources


def sealed_results() -> list[dict[str, Any]]:
    directory = runs_dir() / "sealed-results"
    if not directory.exists():
        return []
    results = []
    for path in sorted(
        directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item["result_path"] = str(path)
            results.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    return results


def _evaluate_model(
    model_path: str,
    adapter_path: str | None,
    dataset: Any,
    max_samples: int,
    stop_file: Path | None,
    tool_names: list[str] | None = None,
) -> dict[str, Any]:
    from .server import LocalModel

    runtime = LocalModel(
        model_path=model_path,
        adapter_path=adapter_path,
        # Honour an explicit accelerator requirement instead of silently
        # allocating the whole model in FP32 system RAM.
        device=os.environ.get("EXOTIC_TRAINER_EVAL_DEVICE", "auto"),
    )
    try:
        return evaluate_agent_behavior(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            dataset=dataset,
            max_samples=max_samples,
            max_new_tokens=192,
            deadline=time.monotonic() + max(300, max_samples * 45),
            stop_file=stop_file,
            tool_names=tool_names,
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


def run_sealed_variant(
    run_id: str,
    suite_id: str,
    max_samples: int,
    variant: str,
    result_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one model variant in an isolated process to fully release XPU state."""
    if variant not in {"base", "adapter"}:
        raise ValueError(f"unsupported sealed evaluation variant: {variant}")
    registry = Registry()
    run = registry.get_run(run_id)
    recipe = run["recipe_json"]
    if isinstance(recipe, str):
        recipe = json.loads(recipe)
    if isinstance(recipe, str):
        recipe = json.loads(recipe)
    suite = get_sealed_suite(suite_id)
    full_dataset = _normalize_dataset(
        _load_one(
            suite["source"],
            max_rows=max(int(suite.get("raw_rows") or max_samples), max_samples),
            seed=1776,
        ),
        allowed_tools=None,
    )
    actual_ordered_hashes = [example_hash(example) for example in full_dataset]
    registered_ordered_hashes = suite.get("ordered_hashes")
    if registered_ordered_hashes:
        fingerprint = hashlib.sha256(
            json.dumps(actual_ordered_hashes, separators=(",", ":")).encode()
        ).hexdigest()
        if actual_ordered_hashes != registered_ordered_hashes or fingerprint != suite.get(
            "content_fingerprint"
        ):
            raise ValueError(
                "sealed suite content, order or multiplicity changed after registration"
            )
    elif not set(actual_ordered_hashes).issubset(set(suite.get("hashes", []))):
        raise ValueError("sealed suite content changed after registration")
    dataset = full_dataset
    if len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=1776).select(range(max_samples))
    stop_file = (
        Path(os.environ["EXOTIC_TRAINER_STOP_FILE"])
        if os.environ.get("EXOTIC_TRAINER_STOP_FILE")
        else None
    )
    metrics = _evaluate_model(
        model_path=registry.resolve_model(recipe["model"]),
        adapter_path=run["output_dir"] if variant == "adapter" else None,
        dataset=dataset,
        max_samples=max_samples,
        stop_file=stop_file,
        tool_names=sorted((suite.get("tool_distribution") or {}).keys()),
    )
    destination = Path(result_path)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _evaluate_variant_process(
    run_id: str,
    suite_id: str,
    max_samples: int,
    variant: str,
    directory: Path,
) -> dict[str, Any]:
    temporary = directory / f".{uuid.uuid4().hex}_{variant}.worker.json"
    command = [
        sys.executable,
        "-m",
        "exotic_trainer.cli",
        "sealed-eval-variant",
        run_id,
        suite_id,
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
                f"sealed {variant} evaluator exited with code {completed.returncode}{detail}"
            )
        if not temporary.exists():
            raise RuntimeError(f"sealed {variant} evaluator produced no result file")
        return json.loads(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def run_sealed_evaluation(
    run_id: str,
    suite_id: str,
    max_samples: int = 32,
    compare_base: bool = True,
    base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = Registry()
    run = registry.get_run(run_id)
    recipe = run["recipe_json"]
    if isinstance(recipe, str):
        recipe = json.loads(recipe)
    if isinstance(recipe, str):
        recipe = json.loads(recipe)
    suite = get_sealed_suite(suite_id)
    if suite["source"] in [
        *recipe.get("datasets", []),
        *recipe.get("validation_datasets", []),
    ]:
        raise ValueError(
            "sealed suite source appears in this run's training or validation datasets"
        )
    stop_file = (
        Path(os.environ["EXOTIC_TRAINER_STOP_FILE"])
        if os.environ.get("EXOTIC_TRAINER_STOP_FILE")
        else None
    )
    directory = runs_dir() / "sealed-results"
    directory.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "run_id": run_id,
        "sheet_id": recipe.get("workbook_sheet_id"),
        "suite_id": suite_id,
        "suite_name": suite["name"],
        "source": suite["source"],
        "evaluated_at": datetime.now(UTC).isoformat(),
        "max_samples": max_samples,
    }
    if compare_base:
        output["base"] = base_metrics or _evaluate_variant_process(
            run_id, suite_id, max_samples, "base", directory
        )
    if not (stop_file and stop_file.exists()):
        output["adapter"] = _evaluate_variant_process(
            run_id, suite_id, max_samples, "adapter", directory
        )
    if output.get("base") and output.get("adapter"):
        output["delta"] = {
            key: float(output["adapter"].get(key, 0.0)) - float(output["base"].get(key, 0.0))
            for key in (
                "agent_format_valid_rate",
                "agent_expected_format_accuracy",
                "agent_tool_decision_accuracy",
                "agent_balanced_tool_decision_accuracy",
                "agent_tool_decision_mcc",
                "agent_parseable_balanced_tool_decision_accuracy",
                "agent_tool_name_accuracy",
                "agent_selected_tool_name_accuracy",
                "agent_tool_name_accuracy_given_expected_format",
                "agent_tool_arguments_valid_rate",
                "agent_tool_argument_keys_accuracy",
                "agent_tool_argument_exact_accuracy",
                "agent_required_argument_call_exact_accuracy",
                "agent_required_argument_call_normalized_accuracy",
                "agent_required_argument_call_elastic_accuracy",
                "agent_argument_field_micro_accuracy",
                "agent_required_argument_field_micro_accuracy",
                "agent_tool_arguments_typed_rate",
                "agent_human_task_success_rate",
                "agent_strict_human_task_success_rate",
                "agent_elastic_human_task_success_rate",
            )
        }
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    destination = directory / f"{stamp}_{run_id}_{suite_id}.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["result_path"] = str(destination)
    return output


def run_sealed_batch(
    run_ids: list[str],
    suite_id: str,
    max_samples: int = 32,
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
        result = run_sealed_evaluation(
            run_id=run_id,
            suite_id=suite_id,
            max_samples=max_samples,
            compare_base=compare_base,
            base_metrics=shared_base,
        )
        if compare_base and shared_base is None:
            shared_base = result.get("base")
        results.append(result)
    return {
        "status": "stopped" if stop_file and stop_file.exists() else "complete",
        "suite_id": suite_id,
        "runs_requested": len(run_ids),
        "runs_evaluated": len(results),
        "results": results,
    }
