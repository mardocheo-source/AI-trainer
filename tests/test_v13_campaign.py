from __future__ import annotations

import csv
import json
from pathlib import Path

from exotic_trainer.v13_campaign import (
    REQUIRED_DIRECTORIES,
    create_campaign_layout,
    generate_campaign_reports,
    read_bfcl_category_scores,
    read_sealed_category_scores,
    validate_campaign_layout,
)


def test_campaign_layout_keeps_docs_benchmarks_and_graphs_inside_run(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "trinity_v13_strict_genetic_prod"
    manifest = create_campaign_layout(root)
    validation = validate_campaign_layout(root)

    assert manifest["selection_benchmark_policy"] == "local-dev-only"
    assert validation["status"] == "ready"
    assert all((root / relative).is_dir() for relative in REQUIRED_DIRECTORIES)
    assert (root / "docs" / "EXPERIMENT_DESIGN.md").is_file()
    assert (root / "docs" / "METRICS_AND_PROVENANCE.md").is_file()
    assert (root / "docs" / "GENETIC_TOURNAMENT_PLAN.md").is_file()
    assert (root / "roadmap" / "ROADMAP_AND_REPRODUCIBILITY.md").is_file()
    assert (root / "benchmarks" / "README.md").is_file()
    assert (root / "graphs" / "README.md").is_file()


def test_official_bfcl_score_json_is_normalized_by_category(tmp_path: Path) -> None:
    score = tmp_path / "score" / "model" / "agentic"
    score.mkdir(parents=True)
    (score / "BFCL_v4_web_search_base_score.json").write_text(
        json.dumps({"accuracy": 0.75, "correct_count": 3, "total_count": 4})
        + "\n"
        + json.dumps({"id": "case-1", "valid": True})
        + "\n",
        encoding="utf-8",
    )

    rows = read_bfcl_category_scores(tmp_path, "v4")

    assert rows == [
        {
            "category": "web_search_base",
            "family": "agentic",
            "total": 4,
            "correct": 3,
            "accuracy": 0.75,
            "metric": "official_accuracy",
            "source": str((score / "BFCL_v4_web_search_base_score.json").resolve()),
        }
    ]


def test_sealed_predictions_produce_strict_elastic_routing_and_format_slices(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sealed.json"
    predictions = [
        {
            "evaluation_metadata": {"stratum": "memory"},
            "strict_human_task_success": True,
            "elastic_human_task_success": True,
            "tool_decision_correct": True,
            "expected_format_correct": True,
        },
        {
            "evaluation_metadata": {"stratum": "memory"},
            "strict_human_task_success": False,
            "elastic_human_task_success": True,
            "tool_decision_correct": True,
            "expected_format_correct": False,
        },
    ]
    source.write_text(json.dumps({"metrics": {"agent_predictions": predictions}}), encoding="utf-8")

    rows = read_sealed_category_scores(source)

    assert rows[0]["category"] == "memory"
    assert rows[0]["strict_accuracy"] == 0.5
    assert rows[0]["elastic_accuracy"] == 1.0
    assert rows[0]["routing_accuracy"] == 1.0
    assert rows[0]["format_accuracy"] == 0.5


def test_report_generation_uses_only_present_raw_results(tmp_path: Path) -> None:
    root = tmp_path / "v13"
    create_campaign_layout(root)
    score = root / "benchmarks" / "bfcl_v4" / "raw" / "run" / "score"
    score.mkdir(parents=True)
    (score / "BFCL_v4_simple_python_score.json").write_text(
        json.dumps({"accuracy": 0.8, "correct_count": 8, "total_count": 10}) + "\n",
        encoding="utf-8",
    )

    status = generate_campaign_reports(root)

    assert status["benchmarks"]["bfcl_v4"]["status"] == "ready"
    assert status["benchmarks"]["bfcl_v3"]["status"] == "pending"
    assert (root / "graphs" / "bfcl_v4_by_category.png").is_file()
    csv_path = root / "benchmarks" / "bfcl_v4" / "category_scores.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["category"] == "simple_python"
    assert not (root / "graphs" / "bfcl_v3_by_category.png").exists()
