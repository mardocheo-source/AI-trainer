#!/usr/bin/env python3
"""Fail unless a Trinity candidate preserves its benchmark baselines.

The gate intentionally compares integer success counts instead of rounded rates.
Sealed summaries must also use the same protocol and sample count.  BFCL requires
the same set of categories and compares aggregate strict and elastic successes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SEALED_COUNT_METRICS = (
    "agent_format_valid",
    "agent_tool_true_positive",
    "agent_no_tool_true_negative",
    "agent_tool_decision_correct",
    "agent_selected_tool_name_correct",
    "agent_required_argument_calls_exact",
    "agent_required_argument_calls_elastic",
    "agent_strict_human_task_success",
    "agent_elastic_human_task_success",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def compare_sealed(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for field in ("agent_protocol_id", "agent_eval_samples"):
        expected = baseline.get(field)
        actual = candidate.get(field)
        checks.append(
            {
                "suite": "sealed",
                "metric": field,
                "baseline": expected,
                "candidate": actual,
                "passed": actual == expected,
                "comparison": "equal",
            }
        )
    for field in SEALED_COUNT_METRICS:
        expected = baseline.get(field)
        actual = candidate.get(field)
        valid = isinstance(expected, (int, float)) and isinstance(actual, (int, float))
        checks.append(
            {
                "suite": "sealed",
                "metric": field,
                "baseline": expected,
                "candidate": actual,
                "delta": actual - expected if valid else None,
                "passed": bool(valid and actual >= expected),
                "comparison": "greater_or_equal",
            }
        )
    return checks


def _bfcl_totals(summary: dict[str, Any]) -> tuple[int, int, int, set[str]]:
    categories = summary.get("category_scores")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("BFCL summary has no category_scores")
    total = sum(int(row["total"]) for row in categories.values())
    strict = sum(int(row["correct_strict"]) for row in categories.values())
    elastic = sum(int(row["correct_elastic"]) for row in categories.values())
    return total, strict, elastic, set(categories)


def compare_bfcl(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    base_total, base_strict, base_elastic, base_categories = _bfcl_totals(baseline)
    cand_total, cand_strict, cand_elastic, cand_categories = _bfcl_totals(candidate)
    return [
        {
            "suite": "bfcl_v4",
            "metric": "categories",
            "baseline": sorted(base_categories),
            "candidate": sorted(cand_categories),
            "passed": cand_categories == base_categories,
            "comparison": "equal",
        },
        {
            "suite": "bfcl_v4",
            "metric": "total_cases_evaluated",
            "baseline": base_total,
            "candidate": cand_total,
            "passed": cand_total == base_total,
            "comparison": "equal",
        },
        {
            "suite": "bfcl_v4",
            "metric": "strict_successes",
            "baseline": base_strict,
            "candidate": cand_strict,
            "delta": cand_strict - base_strict,
            "passed": cand_strict >= base_strict,
            "comparison": "greater_or_equal",
        },
        {
            "suite": "bfcl_v4",
            "metric": "elastic_successes",
            "baseline": base_elastic,
            "candidate": cand_elastic,
            "delta": cand_elastic - base_elastic,
            "passed": cand_elastic >= base_elastic,
            "comparison": "greater_or_equal",
        },
    ]


def _official_bfcl_categories(score_dir: Path) -> dict[str, tuple[int, int]]:
    categories: dict[str, tuple[int, int]] = {}
    for path in sorted(score_dir.rglob("BFCL_v4_*_score.json")):
        first_line = next(
            (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            None,
        )
        if first_line is None:
            continue
        summary = json.loads(first_line)
        if not isinstance(summary, dict) or "total_count" not in summary:
            continue
        category = path.name.removeprefix("BFCL_v4_").removesuffix("_score.json")
        categories[category] = (
            int(summary["total_count"]),
            int(summary["correct_count"]),
        )
    if not categories:
        raise ValueError(f"No official BFCL category scores found below {score_dir}")
    return categories


def compare_official_bfcl(
    baseline_score_dir: Path, candidate_score_dir: Path
) -> list[dict[str, Any]]:
    baseline = _official_bfcl_categories(baseline_score_dir)
    candidate = _official_bfcl_categories(candidate_score_dir)
    same_categories = set(candidate) == set(baseline)
    same_totals = same_categories and all(
        candidate[category][0] == baseline[category][0] for category in baseline
    )
    base_total = sum(total for total, _correct in baseline.values())
    cand_total = sum(total for total, _correct in candidate.values())
    base_correct = sum(correct for _total, correct in baseline.values())
    cand_correct = sum(correct for _total, correct in candidate.values())
    return [
        {
            "suite": "bfcl_v4_official",
            "metric": "categories",
            "baseline": sorted(baseline),
            "candidate": sorted(candidate),
            "passed": same_categories,
            "comparison": "equal",
        },
        {
            "suite": "bfcl_v4_official",
            "metric": "category_sample_counts",
            "baseline": {key: value[0] for key, value in baseline.items()},
            "candidate": {key: value[0] for key, value in candidate.items()},
            "passed": same_totals,
            "comparison": "equal",
        },
        {
            "suite": "bfcl_v4_official",
            "metric": "total_cases_evaluated",
            "baseline": base_total,
            "candidate": cand_total,
            "passed": same_totals and cand_total == base_total,
            "comparison": "equal",
        },
        {
            "suite": "bfcl_v4_official",
            "metric": "official_successes",
            "baseline": base_correct,
            "candidate": cand_correct,
            "delta": cand_correct - base_correct,
            "passed": same_totals and cand_correct >= base_correct,
            "comparison": "greater_or_equal",
            "category_deltas": {
                category: candidate[category][1] - baseline[category][1]
                for category in sorted(set(baseline) & set(candidate))
            },
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sealed", type=Path, required=True)
    parser.add_argument("--candidate-sealed", type=Path, required=True)
    parser.add_argument("--baseline-bfcl", type=Path)
    parser.add_argument("--candidate-bfcl", type=Path)
    parser.add_argument("--baseline-bfcl-official-score-dir", type=Path)
    parser.add_argument("--candidate-bfcl-official-score-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.baseline_bfcl is None) != (args.candidate_bfcl is None):
        raise SystemExit("Both --baseline-bfcl and --candidate-bfcl are required together")
    if (args.baseline_bfcl_official_score_dir is None) != (
        args.candidate_bfcl_official_score_dir is None
    ):
        raise SystemExit(
            "Both official BFCL score-directory arguments are required together"
        )
    if args.baseline_bfcl is not None and args.baseline_bfcl_official_score_dir is not None:
        raise SystemExit("Choose either legacy BFCL summaries or official BFCL score directories")

    checks = compare_sealed(
        _load_json(args.baseline_sealed), _load_json(args.candidate_sealed)
    )
    if args.baseline_bfcl is not None and args.candidate_bfcl is not None:
        checks.extend(
            compare_bfcl(
                _load_json(args.baseline_bfcl), _load_json(args.candidate_bfcl)
            )
        )
    if (
        args.baseline_bfcl_official_score_dir is not None
        and args.candidate_bfcl_official_score_dir is not None
    ):
        checks.extend(
            compare_official_bfcl(
                args.baseline_bfcl_official_score_dir,
                args.candidate_bfcl_official_score_dir,
            )
        )

    passed = all(check["passed"] for check in checks)
    report = {"passed": passed, "checks": checks}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
