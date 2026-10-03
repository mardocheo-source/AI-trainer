from __future__ import annotations

from collections import Counter

from exotic_trainer.server import parse_native_tool_calls
from exotic_trainer.strict_genetic_data import (
    CATEGORIES,
    DEV_FOLDS,
    DEV_PER_CATEGORY,
    TRAIN_PER_CATEGORY,
    audit_corpus,
    build_dev_rows,
    build_preference_rows,
    build_train_rows,
)


def test_strict_genetic_corpus_is_balanced_unique_and_benchmark_free() -> None:
    train = build_train_rows()
    dev = {fold: build_dev_rows(fold) for fold in DEV_FOLDS}
    preferences = build_preference_rows(train)
    audit = audit_corpus(train, preferences, dev)

    assert audit["status"] == "ready"
    assert audit["benchmark_sources_used"] == []
    assert audit["sealed_sources_used"] == []
    assert set(audit["train_categories"].values()) == {TRAIN_PER_CATEGORY}
    assert all(
        item["rows"] == len(CATEGORIES) * DEV_PER_CATEGORY for item in audit["dev_folds"].values()
    )
    assert not any(audit["exact_overlap"].values())


def test_each_dev_fold_has_ten_cases_per_category_and_one_menu() -> None:
    for fold in DEV_FOLDS:
        rows = build_dev_rows(fold)
        assert Counter(row["evaluation_metadata"]["category"] for row in rows) == {
            category: DEV_PER_CATEGORY for category in CATEGORIES
        }
        assert all(row["prompt"][0]["content"].count("List of tools: [") == 1 for row in rows)


def test_preferences_penalize_near_misses_including_overtrigger() -> None:
    train = build_train_rows()
    pairs = build_preference_rows(train)
    mutations = Counter(pair["preference_metadata"]["mutation"] for pair in pairs)

    assert mutations["overtrigger"] > 0
    assert mutations["missing-key"] > 0
    assert mutations["wrong-value"] > 0
    assert mutations["wrong-tool"] > 0
    assert all(pair["chosen"] != pair["rejected"] for pair in pairs)
    tool_pair = next(
        pair for pair in pairs if pair["preference_metadata"]["mutation"] == "wrong-value"
    )
    assert parse_native_tool_calls(tool_pair["chosen"][0]["content"])[1]
    assert parse_native_tool_calls(tool_pair["rejected"][0]["content"])[1]
