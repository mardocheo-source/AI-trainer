import json

from exotic_trainer.registry import Registry


def test_assign_run_to_workbook_sheet(tmp_path) -> None:
    registry = Registry(tmp_path / "registry.sqlite3")
    registry.create_run("run-1", {"model": "m", "datasets": ["d"]}, tmp_path / "run")

    registry.assign_run_to_sheet("run-1", "sheet-1")

    recipe = json.loads(registry.get_run("run-1")["recipe_json"])
    assert recipe["workbook_sheet_id"] == "sheet-1"
