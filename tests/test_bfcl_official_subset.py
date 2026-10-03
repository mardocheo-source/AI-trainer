import json
from pathlib import Path

import pytest

from scripts.prepare_bfcl_v3_subset import prepare_subset
from scripts.run_bfcl_v4_official_subset import build_official_id_map


@pytest.fixture
def synthetic_bfcl(tmp_path, monkeypatch):
    """Exercise ID selection with synthetic data, without vendoring BFCL datasets."""
    import sys
    from types import ModuleType

    entries = {
        "simple_python": [{"id": "simple_python_0"}],
        "multi_turn_base": [{"id": "multi_turn_base_0"}],
    }
    for variant in ("base", "no_snippet"):
        entries[f"web_search_{variant}"] = [{"id": f"web_search_{variant}_0"}]
    for backend in ("kv", "vector", "rec_sum"):
        prefix = f"memory_{backend}"
        entries[prefix] = [
            {"id": f"{prefix}_prereq", "depends_on": []},
            {"id": f"{prefix}_0-customer-0", "depends_on": [f"{prefix}_prereq"]},
        ]
    module = ModuleType("bfcl_eval.utils")
    module.load_dataset_entry = lambda category: entries[category]
    parent = ModuleType("bfcl_eval")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "bfcl_eval", parent)
    monkeypatch.setitem(sys.modules, "bfcl_eval.utils", module)
    for category, identifier in {
        "simple_python": "simple_python_0",
        "multi_turn_base": "multi_turn_base_0",
        "web_search": "web_search_0",
        "memory": "memory_0-customer-0",
    }.items():
        (tmp_path / f"BFCL_v4_{category}.json").write_text(json.dumps({"id": identifier}) + "\n")
    return tmp_path


def test_single_turn_scope_excludes_agentic_and_multi_turn(synthetic_bfcl) -> None:
    selected = build_official_id_map(synthetic_bfcl, "single-turn", limit_per_category=1)
    assert "simple_python" in selected
    assert not any(category.startswith("multi_turn") for category in selected)
    assert not any(category.startswith("memory") for category in selected)
    assert not any(category.startswith("web_search") for category in selected)


def test_official_agentic_variants_and_dependencies_are_expanded(synthetic_bfcl) -> None:
    selected = build_official_id_map(synthetic_bfcl, "all", limit_per_category=1)
    assert selected["web_search_base"] == ["web_search_base_0"]
    assert selected["web_search_no_snippet"] == ["web_search_no_snippet_0"]
    assert "memory_kv_0-customer-0" in selected["memory_kv"]
    assert any("prereq" in identifier for identifier in selected["memory_kv"])


def test_bfcl_v3_subset_is_frozen_stratified_and_answer_aligned(tmp_path: Path) -> None:
    source = tmp_path / "source"
    answers = source / "possible_answer"
    answers.mkdir(parents=True)
    rows = [{"id": f"simple_{index}", "question": str(index)} for index in range(8)]
    answer_rows = [{"id": row["id"], "ground_truth": [row["id"]]} for row in rows]
    (source / "BFCL_v3_simple.json").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (answers / "BFCL_v3_simple.json").write_text(
        "".join(json.dumps(row) + "\n" for row in answer_rows), encoding="utf-8"
    )
    (source / "BFCL_v3_chatable.json").write_text(
        "".join(json.dumps({"question": str(index), "function": ""}) + "\n" for index in range(4)),
        encoding="utf-8",
    )

    output = tmp_path / "subset"
    manifest = prepare_subset(source, output, ratio=0.25, seed=20260823)
    repeated = prepare_subset(source, output, ratio=0.25, seed=20260823)

    selected = [
        json.loads(line) for line in (output / "BFCL_v3_simple.json").read_text().splitlines()
    ]
    selected_answers = [
        json.loads(line)
        for line in (output / "possible_answer" / "BFCL_v3_simple.json").read_text().splitlines()
    ]
    assert manifest["partial"] is True
    assert manifest["publishable"] is False
    assert manifest["source_cases"] == 12
    assert manifest["selected_cases"] == 3
    assert repeated["fingerprint"] == manifest["fingerprint"]
    assert [row["id"] for row in selected_answers] == [row["id"] for row in selected]

    with pytest.raises(RuntimeError, match="different frozen"):
        prepare_subset(source, output, ratio=0.50, seed=20260823)
