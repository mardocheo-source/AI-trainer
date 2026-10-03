from __future__ import annotations

import gc
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

from .agent_eval import evaluate_agent_behavior
from .data_loading import _load_one, _normalize_dataset
from .paths import project_root
from .schema import (
    Boolean3DConfig,
    GeometryConfig,
    NoiseConfig,
    RoleLossConfig,
    TrainingRecipe,
)
from .tool_schema import KNOWN_TOOLS
from .trainer import run_training
from .v14_data_engine import OUTPUT_DIR, TRAIN_PATH, dev_path

MODEL_PATH = "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct"
OPTUNA_DIR = project_root() / "runs" / "trinity_v14_optuna_prod"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def compute_v14_berkeley_fitness(metrics: dict[str, Any]) -> dict[str, Any]:
    predictions = list(metrics.get("agent_predictions") or [])
    tool_rows = [item for item in predictions if item.get("expected_tool") is not None]

    def success(categories: set[str]) -> float:
        selected = [
            item
            for item in predictions
            if str((item.get("evaluation_metadata") or {}).get("category")) in categories
        ]
        return sum(bool(item.get("strict_human_task_success")) for item in selected) / max(
            1, len(selected)
        )

    strict_exact = sum(bool(item.get("argument_values_exact")) for item in tool_rows) / max(
        1, len(tool_rows)
    )
    elastic_required = sum(
        bool(item.get("required_argument_values_elastic")) for item in tool_rows
    ) / max(1, len(tool_rows))

    polyglot_success = success({"polyglot_java", "polyglot_javascript"})
    multiturn_success = success({"multi_step", "memory", "web_multihop", "glaive_multiturn"})
    rehearsal_success = success({"rehearsal"})
    routing = float(metrics.get("agent_balanced_tool_decision_accuracy") or 0.0)
    mcc = (float(metrics.get("agent_tool_decision_mcc") or 0.0) + 1.0) / 2.0
    format_score = float(metrics.get("agent_expected_format_accuracy") or 0.0)
    gap = max(0.0, elastic_required - strict_exact)
    regression_guard = min(routing, format_score, rehearsal_success)

    # Weighted fitness: Executable exactness (0.30) + Polyglot (0.20) + Multi-Turn (0.15) + Routing (0.15) + MCC (0.10) + Format (0.10)
    fitness = (
        0.30 * strict_exact
        + 0.20 * polyglot_success
        + 0.15 * multiturn_success
        + 0.15 * routing
        + 0.10 * mcc
        + 0.10 * format_score
        + 0.05 * regression_guard
        - 0.10 * gap
    )
    return {
        "fitness": fitness,
        "strict_exact": strict_exact,
        "elastic_required": elastic_required,
        "polyglot_success": polyglot_success,
        "multiturn_success": multiturn_success,
        "balanced_routing": routing,
        "routing_mcc_scaled": mcc,
        "expected_format": format_score,
        "rehearsal_strict": rehearsal_success,
        "regression_guard": regression_guard,
    }


def _evaluate_trial_adapter(
    adapter_path: Path, dev_source: Path, seed: int = 6492
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(seed)
    torch.xpu.empty_cache()
    torch.xpu.set_per_process_memory_fraction(0.70, device=0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path)).to("xpu")
    model.eval()

    raw = _load_one(str(dev_source), max_rows=300, seed=seed + 50_000)
    evaluation = _normalize_dataset(raw, allowed_tools=set(KNOWN_TOOLS))
    metrics = evaluate_agent_behavior(
        model=model,
        tokenizer=tokenizer,
        dataset=evaluation,
        max_samples=len(evaluation),
        max_new_tokens=128,
        deadline=float("inf"),
        tool_names=sorted(KNOWN_TOOLS),
        include_predictions=True,
    )
    fitness_dict = compute_v14_berkeley_fitness(metrics)
    del model, base
    gc.collect()
    torch.xpu.empty_cache()
    return {"metrics": metrics, "fitness": fitness_dict}


def create_recipe_from_trial(
    trial: optuna.Trial,
    trial_dir: Path,
    train_data_path: Path,
    dev_data_path: Path,
    seed: int = 6492,
) -> tuple[TrainingRecipe, dict[str, Any]]:
    lr = trial.suggest_float("learning_rate", 1.0e-4, 1.2e-3, log=True)
    steps = trial.suggest_int("max_steps", 20, 60)
    warmup = trial.suggest_float("warmup_ratio", 0.03, 0.15)
    weight_decay = trial.suggest_float("weight_decay", 1.0e-3, 0.10, log=True)
    grad_accum = trial.suggest_categorical("gradient_accumulation", [2, 4, 8])

    pi_noise_scale = trial.suggest_float("pi_noise_scale", 1.0e-6, 5.0e-4, log=True)
    boolean_weight = trial.suggest_float("boolean_weight", 0.01, 0.25)
    riemann_curvature = trial.suggest_float("riemann_curvature", 0.005, 0.15)
    role_loss_weight = trial.suggest_float("role_loss_weight", 0.05, 0.40)
    state_binding_weight = trial.suggest_float("state_binding_weight", 0.01, 0.20)

    gene_dict = {
        "learning_rate": lr,
        "max_steps": steps,
        "warmup_ratio": warmup,
        "weight_decay": weight_decay,
        "gradient_accumulation": grad_accum,
        "noise": {"enabled": True, "mode": "pi", "alpha": pi_noise_scale * 10000.0, "modulation": 0.35},
        "boolean_3d": {"enabled": True, "weight": boolean_weight, "polytope_margin": 0.35, "polytope_weight": 0.5},
        "geometry": {"enabled": True, "mode": "relational", "weight": riemann_curvature, "margin": 0.2},
        "role_loss": {"enabled": True, "ordinary_weight": 1.0, "tool_name_weight": role_loss_weight * 1.5, "argument_key_weight": role_loss_weight * 2.0, "argument_value_weight": role_loss_weight * 2.5},
        "state_binding_weight": state_binding_weight,
    }

    recipe = TrainingRecipe(
        name=f"v14-trial-{trial.number}",
        model=MODEL_PATH,
        datasets=[str(train_data_path.resolve())],
        validation_mode="external",
        validation_datasets=[str(dev_data_path.resolve())],
        benchmark_profile="mixed",
        comparison_group="trinity-v14-optuna",
        experiment_variant="custom",
        output_dir=str((trial_dir / "sft").resolve()),
        seed=seed,
        dtype="bf16",
        device="xpu",
        xpu_memory_fraction=0.70,
        sequence_length=1408,
        micro_batch_size=1,
        gradient_accumulation=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup,
        weight_decay=weight_decay,
        max_steps=steps,
        budget_mode="steps",
        time_limit_minutes=30,
        reserve_minutes=5,
        noise=NoiseConfig.model_validate(gene_dict["noise"]),
        geometry=GeometryConfig.model_validate(gene_dict["geometry"]),
        role_loss=RoleLossConfig.model_validate(gene_dict["role_loss"]),
        boolean_3d=Boolean3DConfig.model_validate(gene_dict["boolean_3d"]),
        state_binding_weight=state_binding_weight,
    )
    return recipe, gene_dict


def run_optuna_study(
    max_trials: int = 12,
    timeout_seconds: int = 3600,
    seed: int = 20260823,
) -> dict[str, Any]:
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    all_train = _read_jsonl(TRAIN_PATH)
    all_dev = _read_jsonl(dev_path("a"))

    rng = random.Random(seed)
    study = optuna.create_study(
        study_name="trinity_v14_optuna_study",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    trials_log = []

    def objective(trial: optuna.Trial) -> float:
        trial_dir = OPTUNA_DIR / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Sample micro training set (450 rows) for rapid trial iteration
        sample_pool = list(all_train)
        rng.shuffle(sample_pool)
        trial_train_rows = sample_pool[:450]
        trial_train_path = trial_dir / "train_sample.jsonl"
        _write_jsonl(trial_train_path, trial_train_rows)

        trial_dev_path = trial_dir / "dev_sample.jsonl"
        _write_jsonl(trial_dev_path, all_dev)

        recipe, gene = create_recipe_from_trial(
            trial, trial_dir, trial_train_path, trial_dev_path
        )
        (trial_dir / "gene.json").write_text(json.dumps(gene, indent=2), encoding="utf-8")

        start_t = time.time()
        train_metrics = run_training(recipe)
        adapter_path = Path(str(train_metrics["output_dir"])) / "adapter-final"

        eval_res = _evaluate_trial_adapter(adapter_path, trial_dev_path)
        elapsed = time.time() - start_t

        fitness = float(eval_res["fitness"]["fitness"])
        trial_info = {
            "trial_number": trial.number,
            "fitness": fitness,
            "details": eval_res["fitness"],
            "gene": gene,
            "train_loss": train_metrics.get("train_loss"),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (trial_dir / "result.json").write_text(json.dumps(trial_info, indent=2), encoding="utf-8")
        trials_log.append(trial_info)

        print(
            f"[*] Trial {trial.number:03d} | Fitness: {fitness:.4f} | "
            f"Strict: {eval_res['fitness']['strict_exact']:.2%} | "
            f"Polyglot: {eval_res['fitness']['polyglot_success']:.2%} | "
            f"LR: {gene['learning_rate']:.2e} | Steps: {gene['max_steps']} | Time: {elapsed:.1f}s",
            flush=True,
        )
        return fitness

    print(f"=== Starting Trinity V14 Optuna Study (Max Trials: {max_trials}, Timeout: {timeout_seconds}s) ===")
    study.optimize(objective, n_trials=max_trials, timeout=timeout_seconds)

    best_trial = study.best_trial
    champion_gene = None
    for item in trials_log:
        if item["trial_number"] == best_trial.number:
            champion_gene = item["gene"]
            break

    summary = {
        "best_trial_number": best_trial.number,
        "best_fitness": best_trial.value,
        "best_params": best_trial.params,
        "champion_gene": champion_gene,
        "total_trials_completed": len(study.trials),
        "study_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (OPTUNA_DIR / "optuna_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OPTUNA_DIR / "champion_gene.json").write_text(json.dumps(champion_gene, indent=2), encoding="utf-8")
    return summary
