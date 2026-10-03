from __future__ import annotations

import json
import random

import pytest

from exotic_trainer.strict_genetic_data import CATEGORIES
from exotic_trainer.strict_genetic_pipeline import (
    DUAL_SEEDS,
    _gene_signature,
    aggregate_dual_seed,
    audit_gene_safety,
    build_candidate_mix,
    build_round_dev,
    fitness_from_metrics,
    initial_population,
    mutate_gene,
    resource_preflight,
    sample_gene,
)


@pytest.fixture
def synthetic_genetic_data(tmp_path, monkeypatch):
    """Small synthetic corpora preserve category and split coverage controls."""
    import exotic_trainer.strict_genetic_pipeline as pipeline

    def write(name, rows):
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    rows = [
        {
            "training_metadata": {"case_id": f"synthetic-{category}-{index}", "category": category},
            "prompt": "Synthetic instruction",
            "completion": "Synthetic response",
        }
        for category in CATEGORIES
        for index in range(10)
    ]
    monkeypatch.setattr(pipeline, "TRAIN_PATH", write("train.jsonl", rows))
    pairs = [
        {
            "preference_metadata": {"case_id": row["training_metadata"]["case_id"]},
            "chosen": "Synthetic preferred response",
            "rejected": "Synthetic alternative",
        }
        for row in rows
    ]
    monkeypatch.setattr(pipeline, "PREFERENCE_PATH", write("preferences.jsonl", pairs))
    paths = {}
    for fold in pipeline.DEV_FOLDS:
        dev = [
            {
                "evaluation_metadata": {
                    "case_id": f"synthetic-{fold}-{category}-{index}",
                    "category": category,
                    "split": f"dev-{fold}",
                }
            }
            for category in CATEGORIES
            for index in range(10)
        ]
        paths[fold] = write(f"dev-{fold}.jsonl", dev)
    monkeypatch.setattr(pipeline, "dev_path", lambda fold: paths[fold])


def test_gene_space_is_deterministic_but_mutation_changes_signature() -> None:
    gene = sample_gene(random.Random(17), 1)
    same = sample_gene(random.Random(17), 1)
    child = mutate_gene(gene, random.Random(18), "child")

    assert gene == same
    assert _gene_signature(gene) == _gene_signature(same)
    assert _gene_signature(child) != _gene_signature(gene)
    assert abs(sum(gene["category_weights"].values()) - 1.0) < 1e-9
    assert audit_gene_safety(gene)["status"] == "safe"


def test_xpu_safety_blocks_unsafe_configuration() -> None:
    gene = sample_gene(random.Random(19), 1)
    gene["lora_rank"] = 32
    gene["sequence_length"] = 1792
    gene["target_profile"] = "all_linear"
    safety = audit_gene_safety(gene)

    assert safety["status"] == "blocked"
    assert len(safety["problems"]) >= 3


def test_resource_preflight_blocks_when_external_vram_monitor_is_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("exotic_trainer.strict_genetic_pipeline.xpu_vram_status", lambda: None)

    result = resource_preflight(tmp_path)

    assert result["status"] == "blocked"
    assert any("monitor" in problem for problem in result["problems"])


def test_initial_safe_population_has_all_exotic_regularizers_active() -> None:
    genes = initial_population(random.Random(23), 5)
    assert all(gene["regularizer_family"] == "unified_exotic" for gene in genes)
    assert all(
        gene["noise"]["enabled"]
        and gene["geometry"]["enabled"]
        and gene["boolean_3d"]["enabled"]
        and gene["role_loss"]["enabled"]
        for gene in genes
    )
    assert all(audit_gene_safety(gene)["status"] == "safe" for gene in genes)


def test_candidate_mix_respects_size_and_all_category_coverage(synthetic_genetic_data) -> None:
    gene = sample_gene(random.Random(21), 1)
    gene["repeat_focus"] = "all-weak"
    gene["hard_repeat"] = 3
    rows, preferences, manifest = build_candidate_mix(gene, 6492, total_rows=180)

    assert len(rows) == 180
    assert len(preferences) == 180
    assert set(manifest["actual_categories"]) == set(CATEGORIES)
    assert all(count >= 1 for count in manifest["actual_categories"].values())
    assert manifest["repeated_rows"] > 0
    assert manifest["unique_cases"] < manifest["rows"]


def test_round_rotation_uses_half_old_half_new_after_first_round(synthetic_genetic_data) -> None:
    first, first_label = build_round_dev(0)
    second, second_label = build_round_dev(1)

    assert first_label == "dev-a"
    assert second_label == "half-a-half-b"
    assert len(first) == len(second) == len(CATEGORIES) * 10
    assert all("dev-a" in row["evaluation_metadata"]["split"] for row in first)
    second_splits = {row["evaluation_metadata"]["split"] for row in second}
    assert second_splits == {"dev-a", "dev-b"}


def test_fitness_penalizes_elastic_strict_gap() -> None:
    base_prediction = {
        "expected_tool": "read",
        "evaluation_metadata": {"category": "strict_ast"},
        "argument_values_exact": False,
        "required_argument_values_elastic": True,
        "strict_human_task_success": False,
    }
    metrics = {
        "agent_predictions": [base_prediction],
        "agent_balanced_tool_decision_accuracy": 1.0,
        "agent_tool_decision_mcc": 1.0,
        "agent_expected_format_accuracy": 1.0,
    }
    score = fitness_from_metrics(metrics)
    exact_metrics = {
        **metrics,
        "agent_predictions": [
            {**base_prediction, "argument_values_exact": True, "strict_human_task_success": True}
        ],
    }
    exact_score = fitness_from_metrics(exact_metrics)

    assert score["elastic_strict_gap"] == 1.0
    assert exact_score["fitness"] > score["fitness"]
    assert "regression_guard" in exact_score


def test_dual_seed_aggregate_requires_both_seeds_and_rewards_stability() -> None:
    stable = aggregate_dual_seed([0.6, 0.6])
    unstable = aggregate_dual_seed([0.8, 0.4])

    assert len(DUAL_SEEDS) == 2
    assert stable["robust_fitness"] > unstable["robust_fitness"]
