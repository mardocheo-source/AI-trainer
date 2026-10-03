from exotic_trainer.gui import _technique_preset
from exotic_trainer.schema import ExplorationConfig, TrainingRecipe
from exotic_trainer.workbook import WorkbookStore


def test_workbook_save_open_and_duplicate(tmp_path) -> None:
    store = WorkbookStore(tmp_path / "workbook.json")
    recipe = TrainingRecipe(model="model", datasets=["dataset"])
    sheet = store.save("Baseline", recipe, ExplorationConfig(trial_count=2))

    loaded = store.get(sheet.id)
    duplicate = store.duplicate(sheet.id)

    assert loaded.recipe.workbook_sheet_id == sheet.id
    assert loaded.exploration.trial_count == 2
    assert duplicate.id != sheet.id
    assert duplicate.recipe.workbook_sheet_id == duplicate.id
    assert len(store.list()) == 2


def test_route_then_resolve_preset_keeps_natural_pi() -> None:
    values = _technique_preset(
        "Route-then-Resolve — π + role CE + relational geometry"
    )

    assert values[0] is True
    assert values[1:9] == (
        "digit_pairs",
        "pi",
        "natural",
        2.0,
        0.10,
        "prompt",
        "constant",
        0.30,
    )
    assert values[9] is True
    assert values[10:13] == ("relational", "structured", 0.005)
    assert values[16] is True
