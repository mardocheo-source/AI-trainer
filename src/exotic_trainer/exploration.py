from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import runs_dir
from .preflight import preflight_recipe
from .schema import ExplorationConfig, TrainingRecipe
from .sealed import sealed_guard
from .trainer import run_training
from .workbook import WorkbookStore


def _signature(recipe: TrainingRecipe) -> str:
    selected = {
        "use_rslora": recipe.use_rslora,
        "rank": recipe.lora_rank,
        "target": recipe.target_profile,
        "sequence": recipe.sequence_length,
        "learning_rate": recipe.learning_rate,
        "noise": recipe.noise.model_dump(mode="json"),
        "geometry": recipe.geometry.model_dump(mode="json"),
        "role_loss": recipe.role_loss.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:12]


def _uniform_log(generator: random.Random, low: float, high: float) -> float:
    if low == high:
        return low
    return math.exp(generator.uniform(math.log(low), math.log(high)))


def _apply_trial(
    base: TrainingRecipe,
    config: ExplorationConfig,
    trial_index: int,
    values: tuple[Any, ...],
) -> TrainingRecipe:
    (
        method,
        rank,
        target,
        sequence,
        learning_rate,
        noise_mode,
        noise_alpha,
        noise_digit_order,
        noise_scope,
        noise_envelope,
        noise_modulation,
        noise_clean_tail,
        geometry_mode,
        geometry_scope,
        geometry_layer,
        weight,
        geometry_margin,
        role_loss_mode,
    ) = values
    recipe = base.model_copy(deep=True)
    recipe.name = f"{base.name}-trial-{trial_index:02d}"
    recipe.use_rslora = method == "rslora"
    recipe.lora_rank = int(rank)
    recipe.lora_alpha = int(rank) * 2
    recipe.target_profile = target
    recipe.sequence_length = int(sequence)
    recipe.learning_rate = float(learning_rate)
    recipe.noise.enabled = noise_mode != "off"
    recipe.noise.source = "pi" if noise_mode == "off" else noise_mode
    recipe.noise.alpha = float(noise_alpha)
    recipe.noise.digit_order = noise_digit_order
    recipe.noise.scope = noise_scope
    recipe.noise.envelope = noise_envelope
    recipe.noise.modulation = float(noise_modulation)
    recipe.noise.clean_tail_fraction = float(noise_clean_tail)
    recipe.geometry.enabled = geometry_mode != "off"
    recipe.geometry.mode = "orthogonal" if geometry_mode == "off" else geometry_mode
    recipe.geometry.scope = geometry_scope
    recipe.geometry.layer = int(geometry_layer)
    recipe.geometry.weight = float(weight)
    recipe.geometry.margin = float(geometry_margin)
    recipe.role_loss.enabled = role_loss_mode == "on"
    recipe.experiment_variant = (
        "exotic"
        if recipe.noise.enabled or recipe.geometry.enabled or recipe.role_loss.enabled
        else "baseline"
    )
    recipe.exploration_trial = trial_index
    recipe.exploration_signature = _signature(recipe)
    return recipe


def generate_trial_recipes(
    base: TrainingRecipe, config: ExplorationConfig
) -> list[TrainingRecipe]:
    generator = random.Random(config.seed)
    candidates: list[tuple[Any, ...]] = []
    if config.strategy == "grid":
        learning_rates = sorted({config.learning_rate_min, config.learning_rate_max})
        noise_alphas = sorted({config.noise_alpha_min, config.noise_alpha_max})
        noise_modulations = sorted(
            {config.noise_modulation_min, config.noise_modulation_max}
        )
        noise_clean_tails = sorted(
            {config.noise_clean_tail_min, config.noise_clean_tail_max}
        )
        geometry_weights = sorted({config.geometry_weight_min, config.geometry_weight_max})
        geometry_margins = sorted(
            {config.geometry_margin_min, config.geometry_margin_max}
        )
        domains = (
            config.adapter_methods,
            config.lora_ranks,
            config.target_profiles,
            config.sequence_lengths,
            learning_rates,
            config.noise_modes,
            noise_alphas,
            config.noise_digit_orders,
            config.noise_scopes,
            config.noise_envelopes,
            noise_modulations,
            noise_clean_tails,
            config.geometry_modes,
            config.geometry_scopes,
            config.geometry_layers,
            geometry_weights,
            geometry_margins,
            config.role_loss_modes,
        )
        cardinality = math.prod(len(domain) for domain in domains)
        if cardinality <= 100_000:
            candidates = list(itertools.product(*domains))
            generator.shuffle(candidates)
        else:
            # A full high-dimensional grid can contain millions of cells.
            # Draw reproducible cells from the discrete grid without building
            # the Cartesian product in memory.
            candidates = [
                tuple(generator.choice(domain) for domain in domains)
                for _ in range(config.trial_count * 20)
            ]
    else:
        for _ in range(config.trial_count * 5):
            candidates.append(
                (
                    generator.choice(config.adapter_methods),
                    generator.choice(config.lora_ranks),
                    generator.choice(config.target_profiles),
                    generator.choice(config.sequence_lengths),
                    _uniform_log(
                        generator, config.learning_rate_min, config.learning_rate_max
                    ),
                    generator.choice(config.noise_modes),
                    generator.uniform(config.noise_alpha_min, config.noise_alpha_max),
                    generator.choice(config.noise_digit_orders),
                    generator.choice(config.noise_scopes),
                    generator.choice(config.noise_envelopes),
                    generator.uniform(
                        config.noise_modulation_min, config.noise_modulation_max
                    ),
                    generator.uniform(
                        config.noise_clean_tail_min, config.noise_clean_tail_max
                    ),
                    generator.choice(config.geometry_modes),
                    generator.choice(config.geometry_scopes),
                    generator.choice(config.geometry_layers),
                    generator.uniform(
                        config.geometry_weight_min, config.geometry_weight_max
                    ),
                    generator.uniform(
                        config.geometry_margin_min, config.geometry_margin_max
                    ),
                    generator.choice(config.role_loss_modes),
                )
            )
    recipes: list[TrainingRecipe] = []
    signatures: set[str] = set()
    for values in candidates:
        recipe = _apply_trial(base, config, len(recipes) + 1, values)
        if recipe.exploration_signature in signatures:
            continue
        signatures.add(str(recipe.exploration_signature))
        recipes.append(recipe)
        if len(recipes) >= config.trial_count:
            break
    return recipes


def run_exploration(sheet_id: str) -> dict[str, Any]:
    sheet = WorkbookStore().get(sheet_id)
    recipes = generate_trial_recipes(sheet.recipe, sheet.exploration)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = runs_dir() / "explorations" / f"{stamp}_{sheet.id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    stop_file = Path(os.environ["EXOTIC_TRAINER_STOP_FILE"]) if os.environ.get(
        "EXOTIC_TRAINER_STOP_FILE"
    ) else None
    started = time.monotonic()
    deadline = started + sheet.exploration.total_time_limit_minutes * 60
    results: list[dict[str, Any]] = []
    sealed_hashes, sealed_sources = sealed_guard()
    preflight_reports: dict[int, dict[str, Any]] = {}
    for sequence_length in sorted({recipe.sequence_length for recipe in recipes}):
        representative = next(
            recipe for recipe in recipes if recipe.sequence_length == sequence_length
        )
        report = preflight_recipe(
            representative,
            sealed_hashes=sealed_hashes,
            sealed_sources=sealed_sources,
        )
        preflight_reports[sequence_length] = report
    blocked_sequences = [
        sequence for sequence, report in preflight_reports.items() if report["status"] == "blocked"
    ]
    if blocked_sequences:
        result = {
            "status": "blocked_by_preflight",
            "sheet_id": sheet.id,
            "blocked_sequence_lengths": blocked_sequences,
            "preflight": preflight_reports,
            "output_dir": str(output_dir),
        }
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    for index, recipe in enumerate(recipes, start=1):
        if stop_file and stop_file.exists():
            break
        remaining_minutes = int((deadline - time.monotonic()) // 60)
        if remaining_minutes < 6:
            break
        recipe.time_limit_minutes = min(recipe.time_limit_minutes, remaining_minutes)
        recipe.reserve_minutes = min(recipe.reserve_minutes, recipe.time_limit_minutes - 1)
        try:
            result = run_training(recipe)
        except Exception as error:  # noqa: BLE001 - preserve other exploration trials
            result = {
                "status": "failed",
                "trial": index,
                "signature": recipe.exploration_signature,
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        summary_path.write_text(
            json.dumps(
                {
                    "sheet_id": sheet.id,
                    "sheet_name": sheet.name,
                    "started_at": stamp,
                    "preflight": preflight_reports,
                    "results": results,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    return {
        "status": "stopped" if stop_file and stop_file.exists() else "complete",
        "sheet_id": sheet.id,
        "trials_requested": len(recipes),
        "trials_finished": len(results),
        "output_dir": str(output_dir),
        "preflight": preflight_reports,
        "results": results,
    }
