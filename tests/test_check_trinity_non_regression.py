import json

from scripts.check_trinity_non_regression import (
    compare_bfcl,
    compare_official_bfcl,
    compare_sealed,
)


def _sealed(strict: int) -> dict:
    summary = {
        "agent_protocol_id": "protocol",
        "agent_eval_samples": 2,
    }
    summary.update({metric: 1 for metric in (
        "agent_format_valid",
        "agent_tool_true_positive",
        "agent_no_tool_true_negative",
        "agent_tool_decision_correct",
        "agent_selected_tool_name_correct",
        "agent_required_argument_calls_exact",
        "agent_required_argument_calls_elastic",
        "agent_elastic_human_task_success",
    )})
    summary["agent_strict_human_task_success"] = strict
    return summary


def test_sealed_gate_rejects_one_case_regression() -> None:
    checks = compare_sealed(_sealed(1), _sealed(0))
    assert not all(check["passed"] for check in checks)


def test_bfcl_gate_uses_integer_counts() -> None:
    baseline = {
        "category_scores": {
            "category": {"total": 3, "correct_strict": 2, "correct_elastic": 2}
        }
    }
    candidate = {
        "category_scores": {
            "category": {"total": 3, "correct_strict": 2, "correct_elastic": 3}
        }
    }
    assert all(check["passed"] for check in compare_bfcl(baseline, candidate))


def test_official_bfcl_gate_reads_category_score_counts(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "BFCL_v4_simple_python_score.json").write_text(
        json.dumps({"accuracy": 0.5, "correct_count": 1, "total_count": 2}) + "\n"
    )
    (candidate / "BFCL_v4_simple_python_score.json").write_text(
        json.dumps({"accuracy": 1.0, "correct_count": 2, "total_count": 2}) + "\n"
    )
    assert all(check["passed"] for check in compare_official_bfcl(baseline, candidate))
